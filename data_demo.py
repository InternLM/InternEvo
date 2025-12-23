import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from modelscope.msdatasets import MsDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import csv

def data_visualization_raw():
    outputfile_name = "wikipedia_distribution"
    # 加载数据集
    print("loading dataset ..")

    try:
        ds = MsDataset.load('wikimedia/wikipedia', subset_name='20231101.en', cache_dir="/data/HF_data", split='train')
        subset_ds = ds
        print(f"successfully load dataset, choose {len(subset_ds)} samples to analyse")
    except Exception as e:
        print(f"fail to load dataset, error: {e}")
        exit()

    # 加载分词器
    tokenizer_name = "/mnt/shared-storage-user/ailab-sys/lusitian/workspace/InternEvo/tokenizer/Internlm2" # Internlm2分词器
    print(f"loading tokenizer {tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"fail to load tokenizer, try to use 'bert-base-uncased'. error: {e}")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print("Successfully load tokenizer")


    # 计算序列长度 
    print("caculating the length of sequences")

    #lengths = [len(tokenizer.encode(text)) for text in tqdm(subset_ds['text'])]

    lengths = []
    lengths_count = np.array([])

    try:

        with open(f"./DataDistribution/wikipedia/{outputfile_name}.csv", 'w', newline='', encoding='utf-8') as csvfile:
            # 创建一个CSV写入器
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow(['Sequence_Length', 'Sequence_index'])
            
            # 遍历dataset
            for item in tqdm(subset_ds, desc="Processing and Saving"):
                text = item['text']
                index = item['id']
                
                # 对文本进行分词并计算长度
                length = len(tokenizer.encode(text))
                lengths.append(length)
                # text_sample_prefix = text.replace('\n', ' ') # 截取前100个字符并替换换行符
                writer.writerow([length, index])

        print(f"Successfully save data to {outputfile_name}")

    except IOError as e:
        print(f"fail to write the file, error: {e}")
        exit()

    # 计算数据集中不同长度sequence的个数
    lengths_sr = pd.Series(lengths)
    lengths_count = lengths_sr.value_counts().sort_index()
    df_lengths_count = lengths_count.reset_index()
    df_lengths_count.columns = ['Sequence_Length', 'Count']

    try:
        df_lengths_count.to_csv(f"./DataDistribution/wikipedia/{outputfile_name}_count.csv", index=False)
        print("Successfully save the count list of dataset")
    except IOError as e:
        print(f"fail to write count list csv file caused by IOError: {e}")
        

    print("Successfully caculate the sequence length ")


    # draw
    print("正在生成分布图...")
    # 将长度数据转换为Pandas Series方便处理
    lengths_sr = pd.Series(lengths)

    # 设置绘图风格
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))

    # 绘制直方图和KDE曲线，与论文风格保持一致
    ax = sns.histplot(
        data=lengths_sr, 
        log_scale=True,  # X轴使用对数尺度
        kde=True,        # 绘制核密度估计曲线
        color='#4C9A2A', # 使用类似Wikipedia的绿色
        line_kws={'linewidth': 2.5},
        alpha=0.2 # 填充区域的透明度
    )

    # --- 配置坐标轴以匹配论文插图 ---
    # 设置X轴的刻度值
    ticks = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ax.set_xticks(ticks)

    # 自定义X轴刻度标签，将数字转换为 '1K', '2K' 的格式
    @mticker.FuncFormatter
    def custom_formatter(x, pos):
        if x >= 1024:
            return f'{int(x/1024)}K'
        return str(int(x))

    ax.xaxis.set_major_formatter(custom_formatter)
    plt.xticks(rotation=45, ha="right") # 旋转标签防止重叠

    # 隐藏Y轴刻度和标签，因为我们只关心分布形状
    ax.set_yticks([])
    ax.set_ylabel('')

    # 添加标题和标签
    plt.title('Distribution of Sequence Lengths in Wikipedia (20231101.en)', fontsize=16)
    plt.xlabel('Sequence Lengths', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./DataDistribution/wikipedia/{outputfile_name}.png")

    # 显示图像
    plt.show()

    # --- 5. 输出统计信息以验证观察结果 ---
    print("\n--- 数据分布统计 ---")
    print(f"平均序列长度: {lengths_sr.mean():.2f}")
    print(f"序列长度中位数: {lengths_sr.median():.2f}")
    print(f"最大序列长度: {lengths_sr.max()}")
    print(f"长度小于 8K 的序列占比: {(lengths_sr < 8192).mean() * 100:.2f}%")
    print(f"长度超过 32K 的序列占比: {(lengths_sr > 32768).mean() * 100:.2f}%")
    
def data_visualization_withmeta():
    # 配置参数
    input_file = "/mnt/shared-storage-user/lusitian/data/data_jsonl/github/tokenized_llama2/output.bin.meta"  # 输入文件路径
    output_dir = "./DataDistribution/github/"  # 输出目录
    output_name = "token_distribution_llama2"

    print("Loading tokenized data from meta file...")

    # 读取.meta文件
    try:
        # 假设meta文件是numpy格式
        meta = np.load(input_file, allow_pickle=True)
        # 提取token数 (meta[:, 1]为各document的token数)
        lengths = meta[:, 1].astype(int).tolist()
        print(f"Successfully loaded {len(lengths)} documents")
    except Exception as e:
        print(f"Failed to load meta file, error: {e}")
        exit()

    # # 保存原始数据到CSV
    # print("Saving token lengths to CSV...")
    # try:
    #     with open(f"{output_dir}{output_name}.csv", 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['Sequence_Length', 'Document_Index'])
    #         for idx, length in enumerate(lengths):
    #             writer.writerow([length, idx])
    #     print(f"Successfully saved data to {output_name}.csv")
    # except IOError as e:
    #     print(f"Failed to write CSV file, error: {e}")
    #     exit()

    # 计算不同长度的统计
    lengths_sr = pd.Series(lengths)
    lengths_count = lengths_sr.value_counts().sort_index()
    df_lengths_count = lengths_count.reset_index()
    df_lengths_count.columns = ['Sequence_Length', 'Count']

    try:
        df_lengths_count.to_csv(f"{output_dir}{output_name}_count.csv", index=False)
        print("Successfully saved count statistics")
    except IOError as e:
        print(f"Failed to write count CSV file, error: {e}")

    # 绘制分布图
    print("Generating distribution plot...")

    # 设置绘图风格
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 6))

    # 绘制直方图和KDE曲线
    ax = sns.histplot(
        data=lengths_sr, 
        log_scale=True,  # X轴使用对数尺度
        kde=True,        # 绘制核密度估计曲线
        color='#4C72B0', # 使用蓝色
        line_kws={'linewidth': 2.5},
        alpha=0.2  # 填充区域的透明度
    )

    # 设置X轴的刻度值
    ticks = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ax.set_xticks(ticks)

    # 自定义X轴刻度标签,将数字转换为 '1K', '2K' 的格式
    @mticker.FuncFormatter
    def custom_formatter(x, pos):
        if x >= 1024:
            return f'{int(x/1024)}K'
        return str(int(x))

    ax.xaxis.set_major_formatter(custom_formatter)
    plt.xticks(rotation=45, ha="right")  # 旋转标签防止重叠

    # 隐藏Y轴刻度和标签
    ax.set_yticks([])
    ax.set_ylabel('')

    # 添加标题和标签
    plt.title('Distribution of Token Lengths in Tokenized Dataset', fontsize=16)
    plt.xlabel('Token Lengths', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}{output_name}.png", dpi=300)

    print(f"Successfully saved plot to {output_name}.png")

    # 输出统计信息
    print("\n--- Token Distribution Statistics ---")
    print(f"Total documents: {len(lengths_sr)}")
    print(f"Average token length: {lengths_sr.mean():.2f}")
    print(f"Median token length: {lengths_sr.median():.2f}")
    print(f"Max token length: {lengths_sr.max()}")
    print(f"Min token length: {lengths_sr.min()}")
    print(f"Sequences < 8K tokens: {(lengths_sr < 8192).mean() * 100:.2f}%")
    print(f"Sequences > 32K tokens: {(lengths_sr > 32768).mean() * 100:.2f}%")
    print(f"Sequences in [8K, 32K]: {((lengths_sr >= 8192) & (lengths_sr <= 32768)).mean() * 100:.2f}%")

    # 显示图像
    plt.show()
    
data_visualization_withmeta()
