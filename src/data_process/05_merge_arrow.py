import argparse
import os
from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm

def main():
    """
    高效地合并多个大型Arrow数据集，不进行混洗。
    """
    parser = argparse.ArgumentParser(description="Merge multiple large Arrow datasets efficiently without shuffling.")
    parser.add_argument("--input_dirs", nargs='+', required=True, 
                        help="A list of directories containing the Arrow datasets to merge. Separate paths with spaces.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory where the final merged dataset will be saved.")
    
    args = parser.parse_args()

    print("--- 开始合并数据集任务 ---")
    print(f"输入的数据集目录: {args.input_dirs}")
    print(f"最终输出目录: {args.output_dir}")

    # 检查输出目录是否存在，避免意外覆盖
    if os.path.exists(args.output_dir):
        print(f"错误: 输出目录 '{args.output_dir}' 已存在。请指定一个新的目录或手动删除旧的目录。")
        return

    # 1. 加载所有数据集的引用 (这一步是内存高效的)
    print("\n步骤 1/3: 加载所有数据集引用...")
    try:
        datasets_to_merge = [load_from_disk(path) for path in tqdm(args.input_dirs, desc="加载数据集中")]
        print(f"成功加载 {len(datasets_to_merge)} 个数据集的引用。")
    except Exception as e:
        print(f"错误: 加载数据集时出错。请检查输入路径是否都为有效的 Arrow 数据集目录。")
        print(f"详细错误: {e}")
        return

    # 2. 合并数据集 (concatenate_datasets 对于 Arrow 格式非常高效)
    print("\n步骤 2/3: 正在合并数据集...")
    merged_dataset = concatenate_datasets(datasets_to_merge)
    total_samples = len(merged_dataset)
    print(f"合并完成。合并后的总样本数: {total_samples}")

    # 3. 保存最终合并好的数据集
    print(f"\n步骤 3/3: 正在将最终合并的数据集保存到: {args.output_dir}")
    merged_dataset.save_to_disk(args.output_dir)
    print("最终数据集已成功保存。")
    
    print("\n--- 所有操作成功完成！ ---")
    print(f"您合并后的数据集已准备就绪，位于: '{args.output_dir}'")
    print("该数据集是按输入顺序拼接的，并未混洗。")
    print(f"在您的训练脚本中，可以使用以下代码加载它:")
    print(f"from datasets import load_from_disk")
    print(f"my_dataset = load_from_disk('{args.output_dir}')")

if __name__ == "__main__":
    main()