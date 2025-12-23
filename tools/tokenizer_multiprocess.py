import argparse
import json
import os
import sys
from transformers import AutoTokenizer
import numpy as np
import tqdm

# --- 1. 设置环境变量，避免多进程警告 ---
# 这一行对于 "Fast" tokenizer 很重要
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 2. (可选) 将 BATCH_SIZE 设为可配置 ---
# 你可以根据内存调整这个值。1000 到 5000 都是合理范围。
DEFAULT_BATCH_SIZE = 3000

# (代码中其余部分)
# ... (sys.path.append, model_path 等) ...

# tokenizer_path = "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/tokenizer/llama2" # Internlm2分词器
tokenizer_path = "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/tokenizer/llama2" # 替换为你本地的 Tokenizer 路径
try:
    print("loading tokenizer------")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    if not tokenizer.is_fast:
        print("警告: 未能加载 'Fast' Tokenizer (Rust 核心)，速度会较慢。")
except Exception as e:
    print(f"fail to load tokenizer, exit. error: {e}")
    exit()


def process_and_write_batch(batch_contexts: list, bin_file) -> list:
    """
    对一个批次的文本进行分词、编码，并写入bin文件。
    
    Args:
        batch_contexts (list): 包含多个原始文本字符串的列表。
        bin_file (file handler): 已打开的二进制文件句柄。

    Returns:
        list: 包含元信息 (token_length, byte_length) 的元组列表。
    """
    batch_meta_info = []

    # --- 关键优化：批量分词 ---
    # 我们一次性处理所有上下文，不使用填充和截断
    outputs = tokenizer(batch_contexts, truncation=False, padding=False)
    
    # outputs["input_ids"] 是一个列表的列表, e.g., [[...], [...], ...]
    list_of_token_lists = outputs["input_ids"]

    for tokens in list_of_token_lists:
        # --- 沿用原始代码的存储逻辑 ---
        data = dict(tokens=tokens)
        saved_bin = str.encode(json.dumps(data) + "\n")
        
        # 写入文件
        bin_file.write(saved_bin)

        # --- 关键优化：立即收集元数据 ---
        token_length = len(tokens)
        byte_length = len(saved_bin) # 这是写入的字节长度
        batch_meta_info.append((token_length, byte_length))
        
    return batch_meta_info


def save_meta(all_meta_info: list, bin_output_path: str):
    """
    将收集到的所有元数据保存到 .meta 文件。
    
    Args:
        all_meta_info (list): (cur_offset, length, index) 元组的完整列表。
        bin_output_path (str): .bin 文件的路径。
    """
    meta_fp = bin_output_path + ".meta"
    # 将列表转换为 NumPy 数组
    meta = np.array(all_meta_info, dtype=np.int64)
    
    with open(meta_fp, "wb") as f:
        np.save(f, meta)
    print(f"Successfully generated {meta_fp}")


def text2bin(text_input_path: str, bin_output_path: str, batch_size: int):
    """
    读取内容，批量分词，并写入 bin 文件，同时生成元数据。
    """
    if not os.path.isfile(text_input_path):
        raise FileNotFoundError(f"{text_input_path} does not exist.")

    file_format = text_input_path.split(".")[-1]
    assert file_format in ["txt", "json", "jsonl"], (
        "Invalid input file type. Currently support `txt`, `json` and `jsonl`."
    )

    # --- 存储元数据 ---
    all_meta_info = []
    current_byte_offset = 0
    new_index = 0
    batch_contexts = []

    with open(text_input_path, "r") as text_file, open(bin_output_path, "wb") as bin_file: # 使用 'wb' 覆盖
        
        # 内部函数，用于处理一个已满的批次
        def process_batch():
            nonlocal current_byte_offset, new_index
            if not batch_contexts:
                return

            # 批量分词并写入
            batch_meta = process_and_write_batch(batch_contexts, bin_file)
            
            # 收集元数据
            for (token_len, byte_len) in batch_meta:
                all_meta_info.append((current_byte_offset, token_len, new_index))
                current_byte_offset += byte_len
                new_index += 1
            
            # 清空批次
            batch_contexts.clear()

        # --- 根据文件格式处理 ---
        
        if file_format == "txt":
            for line in tqdm(text_file, desc="Processing txt"):
                stripped_line = line.strip()
                if stripped_line:
                    batch_contexts.append(stripped_line)
                    if len(batch_contexts) >= batch_size:
                        process_batch()

        elif file_format == "json":
            data = json.load(text_file)
            index = 10 # 遵循原始代码的逻辑，只处理10个
            for record in tqdm(data, desc="Processing json"):
                if index <= 0:
                    break
                
                context = json.dumps(record)
                batch_contexts.append(context)
                if len(batch_contexts) >= batch_size:
                    process_batch()
                
                index -= 1

        elif file_format == "jsonl":
            for i, line in enumerate(tqdm.tqdm(text_file, desc="Processing jsonl")):
                # if i >= 10000: # 遵循原始代码的注释
                #     break
                
                line_data = json.loads(line)
                context = line_data['text'] # 假设总是存在 'text' 键
                batch_contexts.append(context)
                
                if len(batch_contexts) >= batch_size:
                    process_batch()

        # --- 处理最后一批不满的数据 ---
        process_batch()

    return all_meta_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_input_path",
        type=str,
        required=True,
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--bin_output_path", type=str, required=True, help="Path to the output bin file."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE, 
        help=f"Number of lines to process in one batch (default: {DEFAULT_BATCH_SIZE})."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 注意：'wb' 模式会覆盖旧文件。如果你想追加，请使用 'ab'
    # 但如果使用 'ab' (追加)，元数据会出错。
    # 推荐的预处理流程是每次都重新生成。
    
    # text2bin 现在返回元数据
    print(f"Starting conversion, batch size = {args.batch_size}")
    all_meta = text2bin(args.text_input_path, args.bin_output_path, args.batch_size)
    print(f"Successfully converted {args.text_input_path} to {args.bin_output_path}")

    # 不再需要 prepare_meta()，我们直接保存元数据
    save_meta(all_meta, args.bin_output_path)


if __name__ == "__main__":
    main()