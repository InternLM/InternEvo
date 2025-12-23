import argparse
import json
import os
import sys
import numpy as np
import tqdm

# 移除了 transformers 和 InternLMTokenizer 的导入，因为不再需要分词

# 移除了所有加载分词器的代码


def write_bin_pretokenized(tokens: list, bin_file) -> None:
    """
    根据一个预先分好词的 token 列表写入 bin 文件。

    Args:
        tokens (list): token ID 列表。
        bin_file (file handler): 打开的 bin 文件。

    Example:
    >>> write_bin_pretokenized([67577, 69095, 63010], "out.bin")
    >>> out.bin (文件内容)
    >>> {"tokens": [67577, 69095, 63010]}
    """
    # 确保传入的是一个列表
    if not isinstance(tokens, list):
        # 如果不是列表（可能是None或其它类型），写入一个空列表以避免错误
        print(f"Warning: received non-list data, writing empty list.", file=sys.stderr)
        tokens = []
        
    # 将列表转换为字典，键为 'tokens'
    # eg. {"tokens": [67577, 69095, 63010, 61770, 67783, 69301, 74732]}
    data = dict(tokens=tokens)
    
    # 将字典编码为
    saved_bin = str.encode(json.dumps(data) + "\n")

    # 将字节写入 bin_file
    bin_file.write(saved_bin)


def prepare_meta(bin_output_path: str):
    """
    为给定的 bin 文件准备元数据。
    (此函数与原代码完全相同，无需更改)

    Args:
        bin_output_path (str): 输出 bin 文件的路径。
    """
    meta = []
    cur = 0
    new_index = 0
    print('writing meta information------')
    with open(bin_output_path, "rb") as f:
        while True:
            # 读取行
            line = f.readline()
            # 如果行是空的，则跳出
            if line == b"":
                break
            # 获取每行的 token 数量
            try:
                length = len(json.loads(line)["tokens"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line at offset {cur}: {e}. Skipping.", file=sys.stderr)
                length = 0 # 记录为0，但继续处理
                
            # meta 是一个元组列表 (cur, length)
            # cur: 每行的起始索引
            # length: 每行的 token 数量
            meta.append((cur, length, new_index))
            # 更新 cur 以生成下一行的 meta 信息
            cur += len(line)
            new_index += 1

    # 定义生成的 meta 文件的路径
    meta_fp = bin_output_path + ".meta"
    # 保存生成的 meta 信息
    with open(meta_fp, "wb") as f:
        meta = np.array(meta, dtype=np.int64)
        np.save(f, meta)


def text2bin(text_input_path: str, bin_output_path: str):
    """
    从输入文件读取内容并写入 bin 文件。
    此修改版本仅支持 'jsonl' 输入格式，
    并假定 'jsonl' 文件每行包含一个带 "tokens" 键的 JSON 对象。

    Args:
        text_input_path (str): txt 文件路径。
        bin_output_path (str): 输出 bin 文件路径。
    """
    # 检查 jsonl 文件是否存在
    if not os.path.isfile(text_input_path):
        raise FileNotFoundError(f"{text_input_path} does not exist.")

    file_format = text_input_path.split(".")[-1]
    
    # 修改断言，只支持 'jsonl'
    assert file_format in ["jsonl"], print(
        "Invalid input file type. This modified script only supports pre-tokenized `jsonl` files."
    )
    
    index = []
    
    print(f"Processing pre-tokenized file: {text_input_path}")
    
    with open(text_input_path, "r") as text_file, open(bin_output_path, "ab") as bin_file:
        
        # 移除了 'txt' 和 'json' 的处理逻辑

        if file_format == "jsonl":
            for i, line in enumerate(tqdm.tqdm(text_file, desc="Processing lines")):
                try:
                    # 加载 JSON 行
                    line_data = json.loads(line)
                    
                    # *** 关键 ***
                    # 假设预分词数据存储在 'tokens' 键中
                    # 如果你的 .jsonl 文件使用不同的键（例如 'text' 或 'ids'），请在此处更改 'tokens'
                    tokens_list = line_data['input_ids']
                    
                    # 调用新的写入函数，它接受一个列表
                    write_bin_pretokenized(tokens_list, bin_file)
                    
                    # (可选) 如果原始 .jsonl 中有 'id'，也收集它
                    if "id" in line_data:
                        index.append(line_data["id"])

                except json.JSONDecodeError:
                    print(f"Skipping line {i+1}: Invalid JSON format.", file=sys.stderr)
                except KeyError:
                    # 如果 'tokens' 键不存在，则发出警告
                    print(f"Skipping line {i+1}: 'tokens' key not found in JSON object.", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing line {i+1}: {e}", file=sys.stderr)

    return index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_input_path",
        type=str,
        required=True,
        help="Path to the input .jsonl file (must contain a 'tokens' key).",
    )
    parser.add_argument("--bin_output_path", type=str, required=True, help="Path to the output bin file.")

    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()

    # 运行转换
    text2bin(args.text_input_path, args.bin_output_path)
    print(f"Successfully converted {args.text_input_path} to {args.bin_output_path}")

    # 准备元数据
    prepare_meta(args.bin_output_path)
    print(f"Successfully generated {args.text_input_path}.meta")


if __name__ == "__main__":
    main()