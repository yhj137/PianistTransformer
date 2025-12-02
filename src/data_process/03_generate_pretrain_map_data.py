import json
import argparse
import os
import random
from itertools import chain
from copy import deepcopy
from tqdm import tqdm
import math
import glob

from src.model.pianoformer import PianoT5GemmaConfig
from src.utils.midi import enhanced_ids_uniform

def group_ids(examples, block_size, config, enhanced=False):
    """
    数据处理函数保持不变。
    这个函数会被多个进程并行调用。
    """
    ids_list = []
    if enhanced:
        for i in range(len(examples["input_ids"])):
            ids_list.append(enhanced_ids_uniform(config, examples["input_ids"][i]))
    else:
        ids_list = examples["input_ids"]

    total_ids = list(chain.from_iterable(ids_list))
    total_len = len(total_ids)
    
    #total_len = (total_len // block_size) * block_size
    #total_ids = total_ids[:total_len]
    
    input_ids = [total_ids[i : i + block_size] for i in range(0, total_len, block_size)]
    
    if not input_ids:
        return {"input_ids": []}

    short_size = int(len(input_ids) * 0.1)
    index = list(range(len(input_ids)))
    random.shuffle(index)
    for i in index[:short_size]:
        max_len = len(input_ids[i])
        if max_len <= 8:
            continue
        random_len = random.randint(8, max_len) // 8 * 8
        if random_len == 0:
            random_len = 8

        start_ind_max = max_len - random_len
        if start_ind_max < 0:
            continue
        start_ind = random.randint(0, start_ind_max) // 8 * 8
        input_ids[i] = input_ids[i][start_ind: start_ind + random_len]
        
    random.shuffle(input_ids)
    results = {"input_ids": input_ids}
    return results

def main():
    parser = argparse.ArgumentParser(description="Accelerated Preprocessing for PianoBert training.")
    parser.add_argument("--config", type=str, default="configs/pretrain_config.json", help="Path to the training configuration file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory to load the origin .jsonl files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed .jsonl files.")
    parser.add_argument("--lines_per_file", type=int, default=20000, help="Maximum number of lines per output JSONL file.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid"], help="Which data split to process.")
    # 新增的核心参数
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Number of processes for the map function.")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of input files to load and process in one go (per chunk).")
    parser.add_argument("--enhanced", action="store_true", help="Whether to enhance data.")
    
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    print(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        train_config = json.load(f)

    config = PianoT5GemmaConfig()

    # --- 2. 创建输出目录 ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 3. 获取并分块输入文件 ---
    from datasets import load_dataset
    
    path_patterns = [os.path.join(args.input_dir ,"*.jsonl")]
    
    print("Expanding glob patterns...")
    input_files = []
    for pattern in path_patterns:
        # 使用 glob.glob 解析每个模式，并将匹配到的文件列表合并
        matched_files = glob.glob(pattern)
        if not matched_files:
            print(f"Warning: Pattern '{pattern}' did not match any files.")
        input_files.extend(matched_files)
    
    # 对文件进行排序，确保每次运行的顺序一致
    input_files.sort()

    if not input_files:
        print(f"No input files found for split '{args.split}'. Exiting.")
        return

    # 将所有输入文件路径按 chunk_size 分成多个组
    file_chunks = [input_files[i:i + args.chunk_size] for i in range(0, len(input_files), args.chunk_size)]
    print(f"Found {len(input_files)} input files, divided into {len(file_chunks)} chunks of size {args.chunk_size}.")
    print(f"Using {args.num_proc} processes for parallel processing.")

    # --- 4. 核心处理与保存逻辑 ---
    output_file_index = 0
    line_count = 0
    output_path = os.path.join(args.output_dir, f"{args.split}_{output_file_index}.jsonl")
    output_file = open(output_path, "w")
    print(f"Starting processing. Initial output file: {output_path}")

    # 遍历每一个文件块
    for i, chunk in enumerate(tqdm(file_chunks, desc="Processing file chunks")):
        print(f"\nProcessing chunk {i+1}/{len(file_chunks)} with files: {chunk}")
        
        # 1. 加载当前块的数据 (非流式)
        dataset_chunk = load_dataset("json", data_files=chunk, split="train")

        # 2. 使用多进程 map 进行并行处理
        processed_dataset = dataset_chunk.map(
            group_ids,
            fn_kwargs={
                "block_size": train_config["block_size"],
                "config": config,
                "enhanced": args.enhanced,
            },
            batched=True,
            batch_size=500, # 这是传递给 group_ids 的批大小
            num_proc=args.num_proc,
            remove_columns=dataset_chunk.column_names # 移除旧列，只保留 group_ids 返回的列
        )

        # 3. 将处理好的数据写入文件
        for record in tqdm(processed_dataset, desc=f"Writing chunk {i+1} to output"):
            output_file.write(json.dumps(record) + "\n")
            line_count += 1

            # 如果达到单个文件的行数上限，则切换到新文件
            if line_count >= args.lines_per_file:
                output_file.close()
                print(f"\nFile {output_path} finished with {line_count} lines.")
                output_file_index += 1
                line_count = 0
                output_path = os.path.join(args.output_dir, f"{args.split}_{output_file_index}.jsonl")
                output_file = open(output_path, "w")
                print(f"Switched to new file: {output_path}")

    # 收尾工作
    output_file.close()
    print(f"\nFinished processing. Last file {output_path} has {line_count} lines.")
    print(f"All processed data saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
