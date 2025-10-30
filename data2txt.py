import os
from datasets import load_from_disk, load_dataset, DatasetDict
from tqdm import tqdm
from itertools import islice
from typing import Optional

def export_local_wikipedia_to_txt(dataset_dir: str, out_path: str, split: str = "train",
                                config: Optional[str] = None, max_docs: int = -1, offline: bool = False):

    os.environ["HF_DATASETS_OFFLINE"] = "1"
        
    try:
        ds = load_from_disk(dataset_dir)
        if isinstance(ds, DatasetDict):
            ds = ds[split]
        mode = "from disk"
    except Exception as e:
        print(f"加载本地 wikipedia 数据集失败，需指定 --config 参数 error:{e}")
        ds = load_dataset('wikimedia/wikipedia', config, split=split, cache_dir=dataset_dir, streaming=True, trust_remote_code=True)
        mode = f"cache_dir:{config}"
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    it = ds if max_docs < 0 else islice(ds, max_docs)

    cnt = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(it, desc=f"Exporting local wikipedia:[{mode}] {split}"):
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            text = " ".join(text.split())
            f.write(text + "\n")
            cnt += 1
    print(f"Wrote {cnt} docs to {out_path}")

if __name__ == "__main__":
    
    dataset_dir = "/data/HF_data"
    out_path = "/data/wikipedia/en/wikipedia_20231101.en.txt"
    export_local_wikipedia_to_txt(dataset_dir=dataset_dir, out_path=out_path, split="train", config="20231101.en", max_docs=-1, offline=True)