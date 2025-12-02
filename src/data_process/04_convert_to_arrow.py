import argparse
import os
import glob
import shutil
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Convert a large collection of JSONL files to a single, efficient Arrow dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the preprocessed .jsonl files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the final Arrow dataset will be saved.")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of JSONL files to process in one memory-safe chunk. Adjust based on your RAM and file sizes.")
    
    args = parser.parse_args()

    print(f"Searching for .jsonl files in: {args.input_dir}")
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))

    if not input_files:
        print(f"Error: No .jsonl files found in '{args.input_dir}'. Please check the path.")
        return

    file_chunks = [input_files[i:i + args.chunk_size] for i in range(0, len(input_files), args.chunk_size)]
    print(f"Found {len(input_files)} files, which will be processed in {len(file_chunks)} chunks of up to {args.chunk_size} files each.")

    temp_storage_dir = f"{args.output_dir}_temp_chunks"
    if os.path.exists(temp_storage_dir):
        print(f"Warning: Found existing temporary directory. Removing it: {temp_storage_dir}")
        shutil.rmtree(temp_storage_dir)
    os.makedirs(temp_storage_dir)

    processed_chunk_paths = []
    for i, chunk_of_files in enumerate(tqdm(file_chunks, desc="Processing Chunks")):
        print(f"\nProcessing chunk {i+1}/{len(file_chunks)}...")
        
        dataset_chunk = load_dataset("json", data_files=chunk_of_files, split="train")

        chunk_path = os.path.join(temp_storage_dir, f"chunk_{i}")
        dataset_chunk.save_to_disk(chunk_path)
        processed_chunk_paths.append(chunk_path)
        print(f"Chunk {i+1} converted and saved to: {chunk_path}")

    print("\nAll chunks processed. Now merging into a single dataset...")
    
    all_datasets = [load_from_disk(path) for path in processed_chunk_paths]
    
    final_dataset = concatenate_datasets(all_datasets)
    print(f"Merge complete. Final dataset has {len(final_dataset)} samples.")

    print(f"Saving the final dataset to: {args.output_dir}")
    final_dataset.save_to_disk(args.output_dir)
    
    print(f"Cleaning up temporary directory: {temp_storage_dir}")
    shutil.rmtree(temp_storage_dir)
    
    print("Success! Your dataset is fully converted and ready for high-speed training.")
    print(f"In your training script, load it using: datasets.load_from_disk('{args.output_dir}')")

if __name__ == "__main__":
    main()
