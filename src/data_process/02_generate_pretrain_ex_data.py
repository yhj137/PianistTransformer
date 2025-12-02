import json
import os
import argparse
from glob import glob
import random

from miditoolkit import MidiFile
from tqdm import tqdm

# 假设这些模块在你自己的项目中是可用的
from src.utils.midi import midi_to_ids
from src.model.pianoformer import PianoT5GemmaConfig

def filter_files_by_size(file_list, threshold_bytes):
    """
    过滤文件列表中文件大小小于给定阈值的文件。

    Args:
        file_list (list): 包含文件路径的列表。
        threshold_bytes (int): 文件大小阈值（字节）。小于此大小的文件将被过滤掉。

    Returns:
        list: 过滤后的文件列表，只包含大小大于或等于阈值的文件。
    """
    filtered_list = []
    for file_path in file_list:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            if file_size >= threshold_bytes:
                filtered_list.append(file_path)
        else:
            print(f"警告：文件或路径不存在，已跳过：{file_path}")
    return filtered_list


def find_midi_files(search_dir):
    """
    递归查找指定目录下的所有MIDI文件。

    Args:
        search_dir (str): 要搜索的根目录。

    Returns:
        list: 所有找到的MIDI文件的完整路径列表。
    """
    midi_files = []
    # 定义不区分大小写的MIDI文件后缀
    extensions = ['*.mid', '*.midi', '*.MID', '*.MIDI']
    for ext in extensions:
        # 使用 recursive=True 来进行递归搜索
        midi_files.extend(glob(os.path.join(search_dir, '**', ext), recursive=True))
    
    # 去除重复的文件（如果存在符号链接等情况）
    return list(set(midi_files))

def create_dataset(input_dir, output_dir, batch_size=1000, filter_size=0):
    """
    从一个包含MIDI文件的目录创建数据集。

    Args:
        input_dir (str): 包含.mid, .midi文件的根目录。
        output_dir (str): 保存处理后的.jsonl文件的目录。
        batch_size (int): 每个.jsonl文件中包含的样本数量。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在从 '{input_dir}' 查找MIDI文件...")
    midi_file_paths = find_midi_files(input_dir)
    
    if not midi_file_paths:
        print("错误：在指定目录下没有找到任何MIDI文件。")
        return
    
    random.shuffle(midi_file_paths)

    print(f"找到了 {len(midi_file_paths)} 个MIDI文件。现在开始处理...")

    midi_file_paths = filter_files_by_size(midi_file_paths, filter_size)

    print(f"过滤后有 {len(midi_file_paths)} 个MIDI文件。现在开始处理...")
    
    config = PianoT5GemmaConfig()
    output_batch = []
    file_counter = 0
    
    for file_path in tqdm(midi_file_paths, desc="处理MIDI文件"):
        try:
            # 加载MIDI文件
            midi_obj = MidiFile(file_path)
            # 将MIDI转换为ID序列
            ids = midi_to_ids(config, midi_obj)
            
            # 如果ID序列不为空，则添加到输出列表
            if ids:
                output_batch.append({
                    "input_ids": ids, 
                    "source": os.path.basename(file_path) # 只保留文件名作为来源标识
                })
            
            # 如果批次大小达到阈值，则写入文件
            if len(output_batch) >= batch_size:
                output_filename = os.path.join(output_dir, f"{file_counter}.jsonl")
                with open(output_filename, "w", encoding="utf-8") as f:
                    for item in output_batch:
                        f.write(json.dumps(item) + "\n")
                file_counter += 1
                output_batch = [] # 重置批次
        
        except Exception as e:
            # 捕获并打印处理单个文件时可能出现的任何错误（例如，损坏的MIDI文件）
            print(f"\n处理文件 '{file_path}' 时出错: {e}")
            pass

    # 处理并写入最后一个不满批次的批次
    if output_batch:
        output_filename = os.path.join(output_dir, f"{file_counter}.jsonl")
        with open(output_filename, "w", encoding="utf-8") as f:
            for item in output_batch:
                f.write(json.dumps(item) + "\n")
    
    print(f"\n数据集创建完成！总共生成了 {file_counter + (1 if output_batch else 0)} 个文件，保存在 '{output_dir}' 目录中。")


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="从MIDI文件目录创建数据集。")
    parser.add_argument("--input_dir", type=str, default="data/midis/scores", help="包含MIDI文件的输入根目录。")
    parser.add_argument("--output_dir", type=str, default="data/processed/pretrain/raw/scores",help="用于保存生成的 .jsonl 文件的输出目录。")
    parser.add_argument("--batch_size", type=int, default=1000, help="每个输出的 .jsonl 文件中包含的样本数量。")
    parser.add_argument("--filter_size", type=int, default=7, help="过滤样本的大小(KB)")

    args = parser.parse_args()

    # 调用主函数
    create_dataset(args.input_dir, args.output_dir, args.batch_size, args.filter_size)
