import json
import argparse
import datetime
import os

from datasets import load_from_disk
import torch
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

from src.model.pianoformer import PianoT5Gemma, PianoT5GemmaConfig
from src.utils.func import filter_valid_args

os.environ["WANDB_PROJECT"] = "pianist-transformer"
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Parameters: {(total_params - trainable_params):,}")

    print("--------------------------------------------------")
    print(f"Total Parameters (M):     {total_params / 1_000_000:.2f}M")
    print(f"Trainable Parameters (M): {trainable_params / 1_000_000:.2f}M")
    print("--------------------------------------------------")

class UltimateDataCollator:
    def __init__(self, config, transposition_range=(-5, 6)):
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        
        self.pitch_token_start = config.valid_id_range[0][0]
        self.pitch_token_end = config.valid_id_range[0][1]
        
        self.valid_id_range = config.valid_id_range

        self.transposition_range = transposition_range

    def __call__(self, examples):
        input_tensors = [torch.tensor(f["input_ids"]).long() for f in examples]
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=self.pad_token_id)
        
        attention_mask = (input_ids != self.pad_token_id).long()

        original_padded = pad_sequence(input_tensors, batch_first=True, padding_value=self.pad_token_id)
        
        decoder_input_ids = original_padded.new_zeros(original_padded.shape)
        decoder_input_ids[:, 1:] = original_padded[:, :-1].clone()
        decoder_input_ids[:, 0] = self.bos_token_id
        
        #seq_len = input_ids.shape[1]
        #positional_mask = (torch.arange(seq_len, device=input_ids.device) % 8 == 0)

        #pitch_mask = positional_mask.unsqueeze(0) & (attention_mask == 1)

        #transpose_values = torch.randint(
        #    self.transposition_range[0],
        #    self.transposition_range[1] + 1,
        #    (input_ids.shape[0], 1),
        #    device=input_ids.device
        #)
        
        #transposition_tensor = torch.zeros_like(input_ids)
        #transposition_tensor[pitch_mask] = transpose_values.expand(-1, seq_len)[pitch_mask]

        #input_ids += transposition_tensor
        #input_ids[pitch_mask] = torch.clamp(input_ids[pitch_mask], self.pitch_token_start, self.pitch_token_end - 1)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        mask_probs = torch.full_like(input_ids, 0.3, dtype=torch.float32, device=input_ids.device)
        #expanded_probs = mask_probs.expand_as(input_ids)
        masked_ind = torch.bernoulli(mask_probs).bool() & (attention_mask == 1)

        #rand = torch.rand_like(input_ids, dtype=torch.float)
        #mask_80 = masked_ind & (rand < 0.9)
        #mask_20 = masked_ind & (rand >= 0.9)

        #range_low = torch.zeros_like(input_ids, dtype=torch.long).view(-1, seq_len // 8, 8)
        #range_high = torch.zeros_like(input_ids, dtype=torch.long).view(-1, seq_len // 8, 8)
        #for i in range(8):
        #    range_low[:,:,i] = self.valid_id_range[i][0]
        #    range_high[:,:,i] = self.valid_id_range[i][1] - self.valid_id_range[i][0]
        #range_low = range_low.view(-1, seq_len)
        #range_high = range_high.view(-1, seq_len)
        
        #rand_replace = range_low + torch.randint_like(input_ids, low=0, high=100000000) % range_high

        #input_ids[mask_80] = self.mask_token_id
        #input_ids[mask_20] = rand_replace[mask_20]

        labels[~masked_ind] = -100
        input_ids[masked_ind] = self.mask_token_id

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}

if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    outname = "pretrain_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/pretrain_config.json")
    parser.add_argument('--deepspeed', type=str, help='Path to DeepSpeed config')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from deepspeed')
    parser.add_argument('--encoder_layers_num', type=int, default=None, help='local rank passed from deepspeed')
    parser.add_argument('--decoder_layers_num', type=int, default=None, help='local rank passed from deepspeed')
    parser.add_argument('--hidden_size', type=int, default=None, help='local rank passed from deepspeed')
    parser.add_argument('--intermediate_size', type=int, default=None, help='local rank passed from deepspeed')
    parser.add_argument('--data_size', type=float, default=None, help='local rank passed from deepspeed')
    parser.add_argument('--eval_steps', type=int, default=None, help='local rank passed from deepspeed')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        train_config = json.load(f)
    train_config["output_dir"] = os.path.join(train_config["output_dir"], outname)
    train_config["run_name"] = outname
    train_config["logging_dir"] = os.path.join(train_config["logging_dir"], outname)
    if args.encoder_layers_num:
        train_config["encoder_layers_num"] = args.encoder_layers_num
    if args.decoder_layers_num:
        train_config["decoder_layers_num"] = args.decoder_layers_num
    if args.hidden_size:
        train_config["hidden_size"] = args.hidden_size
    if args.intermediate_size:
        train_config["intermediate_size"] = args.intermediate_size
    if args.data_size:
        train_config["data_size"] = args.data_size
    if args.eval_steps:
        train_config["eval_steps"] = args.eval_steps
    
    if not os.path.exists(train_config["output_dir"]):
        os.makedirs(train_config["output_dir"])
    with open(os.path.join(train_config["output_dir"], "config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    config = PianoT5GemmaConfig(
        hidden_size=train_config["hidden_size"], 
        intermediate_size=train_config["intermediate_size"],
        num_attention_heads=train_config["num_attention_heads"],
        num_key_value_heads=train_config["num_key_value_heads"],
        head_dim=train_config["head_dim"],
        encoder_layers_num=train_config["encoder_layers_num"],
        decoder_layers_num=train_config["decoder_layers_num"],
        torch_dtype=torch.bfloat16   
    )

    dataset = load_from_disk(train_config["data_paths"])
    dataset = dataset.shuffle(seed=42) 
    dataset = dataset.train_test_split(0.02, shuffle=False)
    dataset_valid = dataset["test"]
    if train_config["data_size"] < 1.0:
        dataset_train = dataset["train"].train_test_split(train_config["data_size"], shuffle=False)["test"]
    else:
        dataset_train = dataset["train"]

    data_collator = UltimateDataCollator(config)

    if train_config["resume_path"] is None:
        model = PianoT5Gemma(config)
    else:
        model = PianoT5Gemma.from_pretrained(
            train_config["resume_path"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        print(f"Loaded pretrained model from {train_config['resume_path']}!")

    model.to(device)
    print_model_parameters(model)

    training_args = filter_valid_args(train_config, TrainingArguments)

    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator, 
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
    )

    if not train_config["resume_path"]:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=train_config["resume_path"])
    trainer.save_model()

