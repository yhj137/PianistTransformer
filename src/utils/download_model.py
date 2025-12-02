import os
import argparse
import sys

def download_hf(repo_id, local_dir):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[-] Please install huggingface_hub by: pip install huggingface_hub")
        sys.exit(1)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[+] Success to Download: {local_dir}")
    except Exception as e:
        print(f"[-] Failed to Download: {e}")

def download_ms(repo_id, local_dir):
    
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("[-] Please install modelscope by: pip install modelscope")
        sys.exit(1)

    try:
        snapshot_download(
            model_id=repo_id,
            local_dir=local_dir,
        )
        print(f"[+] Success to Download: {local_dir}")
    except Exception as e:
        print(f"[-] Failed to Download: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    
    parser.add_argument(
        "--source", 
        type=str, 
        default="huggingface", 
        choices=["huggingface", "modelscope"], 
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="models/sft", 
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    if args.source == "huggingface":
        download_hf("yhj137/pianist-transformer-rendering", args.output)
    elif args.source == "modelscope":
        download_ms("yhj137/pianist-transformer-rendering", args.output)