# InternEvo

<div align="center">

<img src="./doc/imgs/InternEvo_logo.png" width="800"/>

[![使用文档](https://readthedocs.org/projects/internevo/badge/?version=latest)](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)
[![license](./doc/imgs/license.svg)](./LICENSE)

[📘使用教程](./doc/usage.md) |
[🛠️安装指引](./doc/install.md) |
[📊框架性能](./doc/train_performance.md) |
[🤔问题报告](https://github.com/InternLM/InternEvo/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>


### 新闻 🔥

- 2024/08/29: InternEvo支持流式加载huggingface格式的数据集。新增详细数据流程说明的指导文档。

- 2024/04/17: InternEvo支持在NPU-910B集群上训练模型。

- 2024/01/17: 更多关于InternLM系列模型的内容，请查看组织内的 [InternLM](https://github.com/InternLM/InternLM)


## 简介

InternEvo是一个开源的轻量级训练框架，旨在支持无需大量依赖关系的模型预训练。凭借单一代码库，InternEvo支持在具有上千GPU的大规模集群上进行预训练，并在单个GPU上进行微调，同时可实现显著的性能优化。当在1024个GPU上进行训练时，InternEvo可实现近90%的加速效率。

基于InternEvo训练框架，我们累计发布了一系列大语言模型，包括InternLM-7B系列和InternLM-20B系列，这些模型在性能上显著超越了许多知名的开源LLMs，如LLaMA和其他模型。

## 安装

首先，安装指定版本的torch, torchvision, torchaudio, and torch-scatter.
例如:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

安装InternEvo:
```bash
pip install InternEvo
```

安装flash-attention (version v2.2.1):

如果需要使用flash-attention加速训练, 并且环境中支持, 按如下方式安装:
```bash
pip install flash-attn==2.2.1
```

有关安装环境以及源码方式安装的更多详细信息，请参考[安装文档](https://internevo.readthedocs.io/zh-cn/latest/install.html#)

## 快速开始

### 训练脚本

首先，准备训练脚本，参考：[train.py](https://github.com/InternLM/InternEvo/blob/develop/train.py)

有关训练脚本的更多详细解释，请参考[训练文档](https://internevo.readthedocs.io/zh-cn/latest/training.html#)

### 数据准备

其次，准备训练或者微调的数据。

从huggingface下载数据集，以 `roneneldan/TinyStories` 数据集为例:
```bash
huggingface-cli download --repo-type dataset --resume-download "roneneldan/TinyStories" --local-dir "/mnt/petrelfs/hf-TinyStories"
```

获取分词器到本地路径。例如，从 `https://huggingface.co/internlm/internlm2-7b/tree/main` 下载special_tokens_map.json、tokenizer.model、tokenizer_config.json、tokenization_internlm2.py和tokenization_internlm2_fast.py文件，并保存到本地路径： `/mnt/petrelfs/hf-internlm2-tokenizer` 。

然后，修改配置文件：
```bash
TRAIN_FOLDER = "/mnt/petrelfs/hf-TinyStories"
data = dict(
    type="streaming",
    tokenizer_path="/mnt/petrelfs/hf-internlm2-tokenizer",
)
```

对于其他数据集类型的准备方式，请参考：[用户文档](https://internevo.readthedocs.io/zh-cn/latest/usage.html#)

### 配置文件

配置文件的内容，请参考：[7B_sft.py](https://github.com/InternLM/InternEvo/blob/develop/configs/7B_sft.py)

关于配置文件更多详细的说明，请参考：[用户文档](https://internevo.readthedocs.io/zh-cn/latest/usage.html#)

### 开启训练

可以在 slurm 或者 torch 分布式环境中开始训练。

slurm环境，双机16卡，启动训练命令如下：
```bash
$ srun -p internllm -N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python -m internlm.launcher.launch --config ./configs/7B_sft.py
```

torch环境，单机8卡，启动训练命令如下：
```bash
$ torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```

## 系统架构

系统架构细节请参考：[系统架构文档](./doc/structure.md)

## 特性列表

<div align="center">
  <b>InternEvo 特性列表</b>
</div>
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>数据集</b>
      </td>
      <td>
        <b>模型</b>
      </td>
      <td>
        <b>并行模式</b>
      </td>
      <td>
        <b>工具</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>已分词数据集</li>
        <li>流式数据集</li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/7B_isp_sft.py">InternLM</a></li>
        <li><a href="configs/7B_internlm2.py">InternLM2</a></li>
        <li><a href="configs/8B_internlm3.py">InternLM3</a></li>
        <li><a href="configs/7B_llama2.py">Llama2</a></li>
        <li><a href="configs/7B_qwen2.py">Qwen2</a></li>
        <li><a href="configs/7B_baichuan2.py">Baichuan2</a></li>
        <li><a href="configs/7B_gemma.py">gemma</a></li>
        <li><a href="configs/57B_qwen2_MoE.py">Qwen2-MoE</a></li>
        <li><a href="configs/8x7B_mixtral.py">Mixtral</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li>ZeRO 1.5</li>
          <li>1F1B 流水线并行</li>
          <li>PyTorch FSDP 训练</li>
          <li>Megatron-LM 张量并行 (MTP)</li>
          <li>Megatron-LM 序列化并行 (MSP)</li>
          <li>Flash-Attn 序列化并行 (FSP)</li>
          <li>Intern 序列化并行 (ISP)</li>
          <li>内存性能分析</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="huggingface_models/README-zh-Hans.md">将ckpt转为huggingface格式</a></li>
          <li><a href="huggingface_models/README-zh-Hans.md">将ckpt从huggingface格式转为InternEvo格式</a></li>
          <li><a href="tools/tokenizer.py">原始数据分词器</a></li>
          <li><a href="tools/alpaca_tokenizer.py">Alpaca数据分词器</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 常见tips

<div align="center">
</div>
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>现象</b>
      </td>
      <td>
        <b>介绍</b>
      </td>
    </tr>
    <tr valign="bottom">
      <td>
        <b>在Vocab维度并行计算loss</b>
      </td>
      <td>
        <b><a href="doc/parallel_output.md">说明</a></b>
      </td>
    </tr>
  </tbody>
</table>

## 贡献

我们感谢所有的贡献者为改进和提升 InternEvo 所作出的努力。非常欢迎社区用户能参与进项目中来。请参考贡献指南来了解参与项目贡献的相关指引。

## 致谢

InternEvo 代码库是一款由上海人工智能实验室和来自不同高校、企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供新功能支持的贡献者，以及提供宝贵反馈的用户。我们希望这个工具箱和基准测试可以为社区提供灵活高效的代码工具，供用户微调 InternEvo 并开发自己的新模型，从而不断为开源社区提供贡献。特别鸣谢 [flash-attention](https://github.com/HazyResearch/flash-attention) 与 [ColossalAI](https://github.com/hpcaitech/ColossalAI) 两项开源项目。

## 引用

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```
