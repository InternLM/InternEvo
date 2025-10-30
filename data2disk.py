import os
import argparse

# 在 import 之前设置环境变量
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_dataset, DatasetDict, DownloadConfig

def resolve_hashed_config(cache_dir: str, base_config: str) -> str | None:
    """
    在本地缓存中查找以 base_config 开头的带 hash 配置名。
    例如 base_config='20231101.en' -> '20231101.en-530e0ee51b14b68f'
    """
    root = os.path.join(cache_dir, "wikimedia___wikipedia")
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        if name.startswith(base_config):
            return name
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, help="HF 缓存根目录，如 /data/HF_data")
    ap.add_argument("--config", required=True, help="wikipedia 配置，例如 20231101.en 或 20230601.zh")
    ap.add_argument("--target_dir", required=True, help="save_to_disk 输出目录，如 /data/wiki_20231101_en_saved")
    args = ap.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    # 仅使用本地缓存
    dcfg = DownloadConfig(cache_dir=args.cache_dir, local_files_only=True)

    ds = None
    mode = ""

    # 方案A：优先使用内置脚本 'wikipedia'（无需从Hub取脚本）
    try:
        ds = load_dataset(
            "wikipedia",
            args.config,
            cache_dir=args.cache_dir,
            download_config=dcfg,
            download_mode="reuse_cache_if_exists",
            ignore_verifications=True,
        )
        mode = "wikipedia(builtin)"
    except Exception as e1:
        # 方案B：回退到 'wikimedia/wikipedia'，自动解析带 hash 的配置名
        hashed = resolve_hashed_config(args.cache_dir, args.config)
        if not hashed:
            raise RuntimeError(
                f"本地缓存未找到配置 {args.config} 对应的带hash目录，"
                f"请检查 {os.path.join(args.cache_dir,'wikimedia__wikipedia')} 是否存在。"
            ) from e1
        ds = load_dataset(
            "wikimedia/wikipedia",
            hashed,
            cache_dir=args.cache_dir,
            download_config=dcfg,
            download_mode="reuse_cache_if_exists",
            ignore_verifications=True,
        )
        mode = f"wikimedia/wikipedia:{hashed}"

    # 保存到磁盘
    if isinstance(ds, DatasetDict):
        for split_name, sub in ds.items():
            sub.save_to_disk(os.path.join(args.target_dir, split_name))
    else:
        ds.save_to_disk(args.target_dir)

    print(f"Saved to {args.target_dir} via {mode}")

if __name__ == "__main__":
    main()