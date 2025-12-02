import json
import argparse
import datetime
import os
import random

from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

from src.model.pianoformer import PianoT5GemmaConfig, PianoT5Gemma
from src.utils.func import filter_valid_args

os.environ["WANDB_PROJECT"] = "pianist-transformer"
def group_ids(examples, block_size, overlap_ratio):
    def slide_window(total_len, window_len):
        window_len = window_len // 8 * 8
        out = []
        start = 0
        while start + window_len <= total_len:
            out.append((start, start + window_len))
            start += int(window_len * (1 - overlap_ratio)) // 8 * 8
        if len(out) == 0 or out[-1][1] != total_len:
            out.append((start, total_len))
        return out
    def random_cut(windows):
        out = []
        for start, end in windows:
            origin_len = end - start
            rand_len = random.randint(8, origin_len) // 8 * 8
            rand_start = random.randint(start, end - rand_len) // 8 * 8
            out.append((rand_start, rand_start + rand_len))
        return out
    xs = []
    labels = []
    for i in range(len(examples["x"])):
        label_ = []
        for j in range(len(examples["label"][i])):
            if j % 8 > 3:
                if examples["label"][i][j] >= 5261 + 64:
                    label_.append(5261 + 127)
                else:
                    label_.append(5261)
            else:
                label_.append(examples["label"][i][j])
        windows = slide_window(len(examples["x"][i]), block_size)
        random_windows = random_cut(windows)
        for start, end in windows:
            x = examples["x"][i][start: end]
            label = label_[start: end]
            xs.append(x)
            labels.append(label)
        for start, end in random_windows:
            x = examples["x"][i][start: end]
            label = label_[start: end]
            xs.append(x)
            labels.append(label)
    return {"input_ids": xs, "labels": labels}


class DiffusionSFTDataCollator:
    def __init__(self, config, transposition_range=(-3, 3)):
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        
        self.pitch_token_start = config.valid_id_range[0][0]
        self.pitch_token_end = config.valid_id_range[0][1]
        
        self.valid_id_range = config.valid_id_range

        self.transposition_range = transposition_range


    def __call__(self, examples):
        #len_list = [len(f["input_ids"]) for f in examples]
        #max_length = max(len_list)
        #input_ids = torch.tensor([f["input_ids"] + [self.pad_token_id] * (max_length - len(f["input_ids"])) for f in examples]).long()
        #label_ids = torch.tensor([f["labels"] + [-100] * (max_length - len(f["labels"])) for f in examples]).long()
        #attention_mask = torch.tensor([[1] * len(f["input_ids"]) + [0] * (max_length - len(f["input_ids"])) for f in examples]).long()
        
        input_tensors = [torch.tensor(f["input_ids"]).long() for f in examples]
        label_tensors = [torch.tensor(f["labels"]).long() for f in examples]
        input_ids = pad_sequence(input_tensors, batch_first=True, padding_value=self.pad_token_id)
        label_ids = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        #original_padded = pad_sequence(label_tensors, batch_first=True, padding_value=self.pad_token_id)
        
        # 右移操作
        #decoder_input_ids = original_padded.new_zeros(original_padded.shape)
        #decoder_input_ids[:, 1:] = original_padded[:, :-1].clone()
        #decoder_input_ids[:, 0] = self.bos_token_id
        
        # decoder的attention mask也需要根据原始padding来
        # decoder_attention_mask = (decoder_input_ids != self.pad_token_id).long()

        attention_mask = (input_ids != self.pad_token_id).long()
        
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
        #label_ids += transposition_tensor
        #input_ids[pitch_mask] = torch.clamp(input_ids[pitch_mask], self.pitch_token_start, self.pitch_token_end - 1)
        #label_ids[pitch_mask] = torch.clamp(label_ids[pitch_mask], self.pitch_token_start, self.pitch_token_end - 1)

        #label_ids[pitch_mask] = -100
        #batch_size = input_ids.shape[0]
        #t = 1 - torch.rand((batch_size, 1))
        #mask_p = torch.ones_like(input_ids) * t
        #unmask_ind = torch.tensor([[1] * (len_list[i] // 2 + 4) + \
        #                           [1, 0, 0, 0, 0, 0, 0, 0] * ((len_list[i] // 2 - 4) // 8) + \
        #                            [1] * (max_length - len_list[i]) for i in range(batch_size)]).bool()
        #mask_p[unmask_ind] = 0
        #masked_ind = torch.bernoulli(mask_p).bool()
        #label_ids[~masked_ind] = -100
        #input_ids[masked_ind] = self.mask_token_id
        return {"input_ids": input_ids, "labels": label_ids, "attention_mask": attention_mask}#, "decoder_input_ids": decoder_input_ids}

if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    outname = "sft_" + current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/sft_config.json")
    parser.add_argument('--deepspeed', type=str, help='Path to DeepSpeed config')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from deepspeed')

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
    
    config = PianoT5GemmaConfig(
        encoder_layers_num=10,
        decoder_layers_num=2,
        torch_dtype=torch.bfloat16   
    )
    
    dataset = load_dataset("json", data_files=train_config["data_paths"])
    dataset = dataset.shuffle(seed=42) 
    
    train_dataset = dataset.filter(lambda example: example['split'] == 'train')
    valid_dataset = dataset.filter(lambda example: example['split'] == 'test')

    train_dataset = train_dataset.map(
        group_ids, 
        fn_kwargs={
            "block_size": train_config["block_size"], 
            "overlap_ratio": train_config["overlap_ratio"], 
        }, 
        batched=True, 
        remove_columns=['x', 'label', 'score_source', 'performance_source', 'cut', 'split']
    )
    valid_dataset = valid_dataset.map(
        group_ids, 
        fn_kwargs={
            "block_size": train_config["block_size"], 
            "overlap_ratio": train_config["overlap_ratio"], 
        }, 
        batched=True, 
        remove_columns=['x', 'label', 'score_source', 'performance_source', 'cut', 'split']
    )

    data_collator = DiffusionSFTDataCollator(config)
    if train_config["pretrained_model"] is None:
        model = PianoT5Gemma(config)
    else:
        model = PianoT5Gemma.from_pretrained(
            train_config["pretrained_model"],
            torch_dtype=torch.bfloat16
        )
        print(f"Loaded pretrained model from {train_config['pretrained_model']}")

    model.to(device)
        
    training_args = filter_valid_args(train_config, TrainingArguments)

    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator, 
        train_dataset=train_dataset["train"],
        eval_dataset=valid_dataset["train"],
    )

    trainer.train()
    trainer.save_model()

