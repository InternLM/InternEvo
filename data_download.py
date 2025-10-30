import os
import argparse
from datasets import load_dataset, DatasetDict, DownloadConfig

def main():
    # os.environ["HF_DATASETS_OFFLINE"] = "1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="wikimedia/wikipedia", help="ModelScope 数据集ID")
    ap.add_argument("--subset", default="20231101.en", help="子配置名，如 20231101.en")
    ap.add_argument("--split", default="train")
    ap.add_argument("--target_dir", required=True, help="save_to_disk 输出目录，如 /data/wiki_20231101_en_saved")
    ap.add_argument("--cache_dir", default=os.getenv("/data/MS_data"))
    args = ap.parse_args()

    ds = load_dataset(args.dataset_id, name=args.subset, split=args.split,
                        cache_dir=args.cache_dir, trust_remote_code=True,)

    out_dir = os.path.join(args.target_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)
    ds.to_json(os.path.join(out_dir, "c4_data.jsonl"))
    print(f"Saved {args.dataset_id}:{args.subset}/{args.split} to {out_dir}")

if __name__ == "__main__":
    main()