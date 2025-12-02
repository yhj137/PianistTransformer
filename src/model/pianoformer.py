from typing import Callable, Optional, Union
from transformers import T5GemmaModel, T5GemmaConfig, T5GemmaModuleConfig, T5GemmaPreTrainedModel, T5GemmaForConditionalGeneration, AutoTokenizer
import torch
from transformers.models.t5gemma.modeling_t5gemma import (
    T5GemmaLMHead, 
    GenerationMixin, 
    logger,
    T5GemmaSelfAttention,
    T5GemmaEncoderLayer,
    T5GemmaRMSNorm,
    T5GemmaRotaryEmbedding,
    make_default_2d_attention_mask,
    create_causal_mask,
    bidirectional_mask_function,
    create_sliding_window_causal_mask,
    sliding_window_bidirectional_mask_function,
    T5GemmaDecoder
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
import torch.nn as nn

class PianoT5GemmaConfig(T5GemmaConfig):

    def __init__(
            self, 
            hidden_size=768, 
            intermediate_size=3072,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            encoder_layers_num=8,
            decoder_layers_num=4,
            **kwargs
        ):
        total_vocab_size = 5389

        self.mask_token_id = 1
        self.bos_token_id = 2
        self.play_token_id = 4
        self.pitch_start = 5
        self.velocity_start = 5 + 128
        self.timing_start = 5 + 128 + 128
        self.pedal_start = 5 + 128 + 128 + 5000
        self.hidden_size = hidden_size

        self.valid_id_range = [
            (5, 133),
            (261, 5252),   
            (133, 261),
            (261, 5261),
            (5261, 5389),
            (5261, 5389),
            (5261, 5389),
            (5261, 5389),
        ]

        encoder_config = T5GemmaModuleConfig(
            vocab_size=total_vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=encoder_layers_num,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )
        decoder_config = T5GemmaModuleConfig(
            vocab_size=total_vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=decoder_layers_num,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )

        super().__init__(
            encoder=encoder_config,
            decoder=decoder_config,
            vocab_size=total_vocab_size,
            **kwargs,
        )

class PianoEncoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if config.hidden_size % 8 != 0:
            raise ValueError("Invalid hidden size: must be a multiple of 8.")
        self.projection_layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size // 8) for i in range(8)])
        self.hidden_size = config.hidden_size

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()

        batch_size = input_shape[0]
        seq_length = input_shape[1]

        inputs_embeds = self.word_embeddings(input_ids)
        grouped_embeds = inputs_embeds.view(batch_size, seq_length // 8, 8, -1)
        projection_list = []
        for i in range(8):
            projection_list.append(self.projection_layers[i](grouped_embeds[:,:,i,:]))
        projection_cat = torch.cat(projection_list, dim=-1)
        inputs_embeds = projection_cat.view(batch_size, -1, self.hidden_size)
        embeddings = inputs_embeds
        return embeddings


class PianoT5GemmaEncoder(T5GemmaPreTrainedModel):
    _can_record_outputs = {
        "attentions": T5GemmaSelfAttention,
        "hidden_states": T5GemmaEncoderLayer,
    }

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embeddings = PianoEncoderEmbeddings(config)
        self.norm = T5GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = T5GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [T5GemmaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
            input_ids = None

        cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None:
            B, L = attention_mask.shape
            block_mask = attention_mask.view(B, L // 8, 8)
            mask2 = block_mask.any(dim=-1).long()
            attention_mask = mask2.view(B, -1)

        if attention_mask is None:
            attention_mask = make_default_2d_attention_mask(input_ids, inputs_embeds, self.config.pad_token_id)
        
        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
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
                    or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                    and_mask_function=bidirectional_mask_function(attention_mask),
                ),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )

class PianoT5GemmaModel(T5GemmaPreTrainedModel):
    def __init__(self, config: T5GemmaConfig):
        super().__init__(config)

        if not config.is_encoder_decoder:
            raise ValueError("T5GemmaModel only support encoder-decoder modeling. Use `T5GemmaEncoderModel` instead.")

        self.encoder = PianoT5GemmaEncoder(config.encoder)
        self.decoder = T5GemmaDecoder(config.decoder)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        encoder_hidden_states = encoder_outputs.last_hidden_state

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states
            if kwargs.get("output_hidden_states", False)
            else (decoder_outputs.last_hidden_state,),
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class PianoT5Gemma(T5GemmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["model.decoder.embed_tokens.weight", "lm_head.out_proj.weight"]
    _tp_plan = {"lm_head.out_proj": "colwise_rep"}
    _pp_plan = {"lm_head.out_proj": (["hidden_states"], ["logits"])}

    def __init__(self, config: PianoT5GemmaConfig):
        config.is_encoder_decoder = True
        super().__init__(config)
        self.embeddings = PianoEncoderEmbeddings(config)
        self.model = PianoT5GemmaModel(config)
        self.vocab_size = config.decoder.vocab_size
        self.lm_head = T5GemmaLMHead(config.decoder.hidden_size, self.vocab_size)
        self.loss_type = "ForMaskedLM"

        self.post_init()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.out_proj = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.out_proj

    def _tie_weights(self):
        # Decoder input and output embeddings are tied.
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.lm_head.out_proj, self.get_decoder().get_input_embeddings())

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
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
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        if self.training and self.config._attn_implementation != "eager":
            msg = (
                "It is strongly recommended to train T5Gemma models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
            if is_torchdynamo_compiling():
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        #if input_ids is not None:
        #    inputs_embeds = self.embeddings(input_ids)
        
        #if attention_mask is not None:
        #    B, L = attention_mask.shape
        #    block_mask = attention_mask.view(B, L // 8, 8)
        #    mask2 = block_mask.any(dim=-1).long()
        #    attention_mask = mask2.view(B, -1)

        #print(attention_mask)

        decoder_outputs: Seq2SeqModelOutput = self.model(
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

        hidden_states = decoder_outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        decoder_config = self.get_decoder().config
        if decoder_config.final_logit_softcapping is not None:
            logits = logits / decoder_config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * decoder_config.final_logit_softcapping

        loss = None
        if labels is not None:
            # Input has right-shifted so we directly perform masked lm loss
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.decoder_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
            encoder_hidden_states=decoder_outputs.encoder_hidden_states,
            encoder_attentions=decoder_outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)



if __name__ == "__main__":
    config = PianoT5GemmaConfig()
    test = PianoEncoderEmbeddings(config)
    model = PianoT5Gemma(config).cuda()
    #encoder_config = T5GemmaModuleConfig(num_hidden_layers=1)
    #decoder_config = T5GemmaModuleConfig(num_hidden_layers=1)
    #config = T5GemmaConfig(encoder_config, decoder_config, attn_implementation='eager')

    #model = T5GemmaForConditionalGeneration(config).cuda()

    toy_ids = torch.tensor([[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]], dtype=torch.long).cuda()
    #tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2")
    #input_text = "Write me a poem about Machine Learning. Answer:"
    #input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    print(model.generate(toy_ids, decoder_input_ids=toy_ids, max_new_tokens=32))

   
    #print(model(input_ids=toy_ids, decoder_input_ids=toy_ids).logits.shape)

