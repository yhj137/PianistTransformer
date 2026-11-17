from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5GemmaConfig, T5GemmaModuleConfig, T5GemmaPreTrainedModel
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.t5gemma.modeling_t5gemma import (
    GenerationMixin,
    T5GemmaDecoder,
    T5GemmaEncoderLayer,
    T5GemmaLMHead,
    T5GemmaRMSNorm,
    T5GemmaRotaryEmbedding,
    T5GemmaSelfAttention,
    bidirectional_mask_function,
    create_causal_mask,
    create_sliding_window_causal_mask,
    make_default_2d_attention_mask,
    sliding_window_bidirectional_mask_function,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


class PianoT5GemmaConfig(T5GemmaConfig):
    """
    Configuration for PianoT5Gemma.

    The encoder consumes groups of eight MIDI/event tokens and maps each group
    to one encoder timestep. The decoder operates on the ordinary token
    vocabulary and timestep resolution.
    """

    model_type = "piano_t5gemma"

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        encoder_layers_num: int = 8,
        decoder_layers_num: int = 4,
        **kwargs,
    ):
        vocab_size = int(kwargs.pop("vocab_size", 5389))

        # from_pretrained()/from_dict() passes the checkpoint's nested module
        # configurations through kwargs. Preserve them: rebuilding from the
        # constructor defaults silently changes layer counts and architecture.
        serialized_encoder = kwargs.pop("encoder", None)
        serialized_decoder = kwargs.pop("decoder", None)

        # Keep the checkpoint's tie_word_embeddings setting. T5Gemma checkpoints
        # may store the decoder embedding/output head as tied weights.
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("bos_token_id", 2)
        kwargs.setdefault("eos_token_id", 3)
        kwargs.setdefault("decoder_start_token_id", 0)

        if isinstance(serialized_encoder, T5GemmaModuleConfig):
            encoder_config = serialized_encoder
        elif isinstance(serialized_encoder, dict):
            encoder_config = T5GemmaModuleConfig(**serialized_encoder)
        else:
            encoder_config = T5GemmaModuleConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=encoder_layers_num,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                pad_token_id=kwargs["pad_token_id"],
                bos_token_id=kwargs["bos_token_id"],
                eos_token_id=kwargs["eos_token_id"],
            )

        if isinstance(serialized_decoder, T5GemmaModuleConfig):
            decoder_config = serialized_decoder
        elif isinstance(serialized_decoder, dict):
            decoder_config = T5GemmaModuleConfig(**serialized_decoder)
        else:
            decoder_config = T5GemmaModuleConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=decoder_layers_num,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                pad_token_id=kwargs["pad_token_id"],
                bos_token_id=kwargs["bos_token_id"],
                eos_token_id=kwargs["eos_token_id"],
            )

        super().__init__(
            encoder=encoder_config,
            decoder=decoder_config,
            vocab_size=vocab_size,
            **kwargs,
        )

        # Older checkpoints may have been saved with model_type="t5gemma".
        # Identify this custom architecture in newly loaded/saved configs.
        self.model_type = type(self).model_type

        # MIDI vocabulary layout used by the project.
        self.mask_token_id = 1
        self.play_token_id = 4
        self.pitch_start = 5
        self.velocity_start = 5 + 128
        self.timing_start = 5 + 128 + 128
        self.pedal_start = 5 + 128 + 128 + 5000
        self.hidden_size = self.encoder.hidden_size

        self.valid_id_range = [
            (5, 133),
            (261, 5261),
            (133, 261),
            (261, 5261),
            (5261, 5389),
            (5261, 5389),
            (5261, 5389),
            (5261, 5389),
        ]

        self.encoder_group_size = 8


class PianoEncoderEmbeddings(nn.Module):
    """
    Converts eight token embeddings into one hidden-size encoder timestep.

    Each of the eight positions has its own projection. Input sequence lengths
    must therefore be divisible by eight.
    """

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        if config.hidden_size % 8 != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by 8."
            )

        projection_size = config.hidden_size // 8
        self.projection_layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, projection_size) for _ in range(8)]
        )
        self.hidden_size = config.hidden_size
        self.group_size = 8

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            if input_ids.ndim != 2:
                raise ValueError(
                    f"input_ids must have shape (batch, sequence), got {tuple(input_ids.shape)}."
                )
            inputs_embeds = self.word_embeddings(input_ids)

        if inputs_embeds.ndim != 3:
            raise ValueError(
                "inputs_embeds must have shape (batch, sequence, hidden_size)."
            )

        batch_size, sequence_length, hidden_size = inputs_embeds.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected embedding dimension {self.hidden_size}, got {hidden_size}."
            )
        if sequence_length % self.group_size:
            raise ValueError(
                f"Encoder input length ({sequence_length}) must be divisible by "
                f"{self.group_size}; received a remainder of "
                f"{sequence_length % self.group_size}."
            )

        grouped = inputs_embeds.reshape(
            batch_size, sequence_length // self.group_size, self.group_size, hidden_size
        )

        # Each position in a group has a dedicated projection. Stacking avoids
        # repeated concatenation and preserves the original eight-way layout.
        projected = [
            self.projection_layers[i](grouped[:, :, i, :])
            for i in range(self.group_size)
        ]
        return torch.cat(projected, dim=-1)


def _compress_encoder_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    encoder_length: int,
    device: torch.device,
    group_size: int = 8,
) -> torch.Tensor:
    """
    Convert a token-level encoder mask (B, 8*T) to frame-level (B, T).

    A frame is valid if at least one token in its group is valid. Already
    compressed masks (B, T) are accepted unchanged.
    """
    if attention_mask is None:
        return torch.ones(
            (batch_size, encoder_length), dtype=torch.long, device=device
        )

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must be a 2-D token-level or frame-level mask."
        )
    if attention_mask.shape[0] != batch_size:
        raise ValueError(
            f"attention_mask batch size {attention_mask.shape[0]} does not match "
            f"input batch size {batch_size}."
        )

    mask_length = attention_mask.shape[1]
    if mask_length == encoder_length:
        return attention_mask.to(device=device)

    if mask_length == encoder_length * group_size:
        return attention_mask.reshape(
            batch_size, encoder_length, group_size
        ).bool().any(dim=-1).to(dtype=attention_mask.dtype, device=device)

    raise ValueError(
        f"attention_mask length must be {encoder_length} (compressed) or "
        f"{encoder_length * group_size} (token-level); got {mask_length}."
    )


class PianoT5GemmaEncoder(T5GemmaPreTrainedModel):
    _can_record_outputs = {
        "attentions": T5GemmaSelfAttention,
        "hidden_states": T5GemmaEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = PianoEncoderEmbeddings(config)
        self.norm = T5GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [
                T5GemmaEncoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        # Build the default mask before input_ids is discarded by embedding.
        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError("input_ids must have shape (batch, sequence).")
            batch_size, token_length = input_ids.shape
            if token_length % 8:
                raise ValueError(
                    f"Encoder input length must be divisible by 8; got {token_length}."
                )
            if attention_mask is None:
                attention_mask = make_default_2d_attention_mask(
                    input_ids, None, self.config.pad_token_id
                )
            inputs_embeds = self.embeddings(input_ids=input_ids)
        else:
            if inputs_embeds.ndim != 3:
                raise ValueError(
                    "inputs_embeds must have shape (batch, sequence, hidden_size)."
                )
            batch_size, token_length, _ = inputs_embeds.shape
            if token_length % 8:
                raise ValueError(
                    f"Encoder input length must be divisible by 8; got {token_length}."
                )
            inputs_embeds = self.embeddings(inputs_embeds=inputs_embeds)

        encoder_length = inputs_embeds.shape[1]
        attention_mask = _compress_encoder_attention_mask(
            attention_mask,
            batch_size=batch_size,
            encoder_length=encoder_length,
            device=inputs_embeds.device,
        )

        if position_ids is None:
            position_ids = torch.arange(
                encoder_length, device=inputs_embeds.device
            ).unsqueeze(0)
        elif position_ids.shape[-1] == token_length:
            # If callers supply token-level positions, use the first position
            # in each eight-token group as that frame's position.
            if position_ids.ndim == 1:
                position_ids = position_ids.reshape(-1, 8)[:, 0].unsqueeze(0)
            else:
                position_ids = position_ids.reshape(
                    position_ids.shape[0], encoder_length, 8
                )[:, :, 0]
        elif position_ids.shape[-1] != encoder_length:
            raise ValueError(
                f"position_ids length must be {encoder_length} or {token_length}."
            )

        cache_position = torch.arange(
            encoder_length, device=inputs_embeds.device
        )

        # Do not forward generation-only/model-output kwargs into encoder layers.
        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("return_dict", None)
        layer_kwargs.pop("output_hidden_states", None)
        layer_kwargs.pop("output_attentions", None)
        layer_kwargs.pop("use_cache", None)
        layer_kwargs.pop("cache_position", None)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            self_attn_mask_mapping = {
                "full_attention": create_causal_mask(
                    **mask_kwargs,
                    or_mask_function=bidirectional_mask_function(attention_mask),
                ),
                "sliding_attention": create_sliding_window_causal_mask(
                    **mask_kwargs,
                    or_mask_function=sliding_window_bidirectional_mask_function(
                        self.config.sliding_window
                    ),
                    and_mask_function=bidirectional_mask_function(attention_mask),
                ),
            }
        else:
            self_attn_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **layer_kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class PianoT5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config: PianoT5GemmaConfig):
        super().__init__(config)

        if not config.is_encoder_decoder:
            raise ValueError(
                "PianoT5Gemma requires an encoder-decoder T5Gemma configuration."
            )

        self.encoder = PianoT5GemmaEncoder(config.encoder)
        self.decoder = T5GemmaDecoder(config.decoder)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.encoder.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0]
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state
        batch_size, encoder_length = encoder_hidden_states.shape[:2]

        # Cross-attention sees the compressed frame-level encoder mask, never
        # the original eight-times-longer token-level mask.
        encoder_attention_mask = _compress_encoder_attention_mask(
            attention_mask,
            batch_size=batch_size,
            encoder_length=encoder_length,
            device=encoder_hidden_states.device,
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class PianoT5Gemma(T5GemmaPreTrainedModel, GenerationMixin):
    """
    Encoder-decoder MIDI model.

    Encoder: groups eight input tokens into one timestep.
    Decoder: predicts ordinary vocabulary tokens autoregressively.
    """

    config_class = PianoT5GemmaConfig
    main_input_name = "input_ids"
    _tied_weights_keys = [
        "model.decoder.embed_tokens.weight",
        "lm_head.out_proj.weight",
    ]
    # The original implementation registered these parameters but never used
    # them in forward(): a second root-level embedding module and the encoder's
    # unused stock embed_tokens table. Ignore those known legacy checkpoint keys.
    _keys_to_ignore_on_load_unexpected = [
        r"^embeddings\.",
        r"^model\.encoder\.embed_tokens\.weight$",
    ]

    def __init__(self, config: PianoT5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)

        self.model = PianoT5GemmaModel(config)
        self.vocab_size = config.decoder.vocab_size
        self.lm_head = T5GemmaLMHead(
            config.decoder.hidden_size, self.vocab_size
        )

        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head.out_proj

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.out_proj = new_embeddings

    def _tie_weights(self):
        # Restore the tying behavior from the original implementation.
        # Some checkpoints store the decoder embedding once and omit a second
        # serialized copy for lm_head.out_proj.
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(
                self.lm_head.out_proj,
                self.get_decoder().get_input_embeddings(),
            )

    def _shift_right(self, labels: torch.Tensor) -> torch.LongTensor:
        """Create decoder inputs by shifting labels one position to the right."""
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, sequence).")

        start_token_id = self.config.decoder_start_token_id
        if start_token_id is None:
            start_token_id = self.config.pad_token_id
        if start_token_id is None:
            raise ValueError(
                "decoder_start_token_id or pad_token_id must be configured."
            )

        pad_token_id = (
            self.config.pad_token_id
            if self.config.pad_token_id is not None
            else 0
        )
        shifted = labels.new_full(labels.shape, pad_token_id)
        if labels.shape[1] > 1:
            shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0] = start_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Seq2SeqLMOutput]:
        if self.training and self.config._attn_implementation != "eager":
            message = (
                "Training T5Gemma with a non-eager attention implementation may "
                "be less reliable. Consider loading/configuring the model with "
                "attn_implementation='eager' if you encounter training issues."
            )
            if is_torchdynamo_compiling():
                raise ValueError(message)
            logger.warning_once(message)

        return_dict = kwargs.pop("return_dict", self.config.use_return_dict)

        if labels is not None:
            if labels.ndim != 2:
                raise ValueError("labels must have shape (batch, sequence).")
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = self._shift_right(labels)

            # Loss requires logits for every target position.
            logits_to_keep = 0

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if isinstance(logits_to_keep, int):
            hidden_states_for_logits = (
                hidden_states[:, -logits_to_keep:, :]
                if logits_to_keep > 0
                else hidden_states
            )
        else:
            hidden_states_for_logits = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states_for_logits)

        softcap = getattr(self.get_decoder().config, "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if labels is not None:
            if logits.shape[:2] != labels.shape:
                raise ValueError(
                    "Logit and label sequence dimensions must match when "
                    "computing loss. "
                    f"Got logits {tuple(logits.shape[:2])}, "
                    f"labels {tuple(labels.shape)}."
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )

        result = Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

        return result if return_dict else result.to_tuple()


if __name__ == "__main__":
    # A lightweight smoke test. The sequence length must be divisible by 8.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = PianoT5GemmaConfig(
        hidden_size=768,
        intermediate_size=3072,
        encoder_layers_num=2,
        decoder_layers_num=2,
    )
    model = PianoT5Gemma(config).to(device).eval()

    toy_ids = torch.tensor(
        [[1, 2, 3, 4, 1, 2, 3, 4] * 2],
        dtype=torch.long,
        device=device,
    )
    toy_mask = torch.ones_like(toy_ids)

    with torch.inference_mode():
        output = model(
            input_ids=toy_ids,
            attention_mask=toy_mask,
            decoder_input_ids=torch.tensor(
                [[config.decoder_start_token_id, 2, 4]],
                dtype=torch.long,
                device=device,
            ),
        )
        print("Logits shape:", tuple(output.logits.shape))

        generated = model.generate(
            input_ids=toy_ids,
            attention_mask=toy_mask,
            max_new_tokens=8,
        )
        print("Generated shape:", tuple(generated.shape))
