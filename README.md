# InternEvo

<div align="center">

<img src="./doc/imgs/InternEvo_logo.png" width="800"/>

[![Documentation Status](https://readthedocs.org/projects/internevo/badge/?version=latest)](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)
[![license](./doc/imgs/license.svg)](./LICENSE)

[📘Usage](./doc/en/usage.md) |
[🛠️Installation](./doc/en/install.md) |
[📊Performance](./doc/en/train_performance.md) |
[🤔Reporting Issues](https://github.com/InternLM/InternEvo/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>


### Latest News 🔥

- 2024/08/29: InternEvo supports streaming dataset of huggingface format. Add detailed instructions of data flow.

- 2024/04/17: InternEvo supports training model on NPU-910B cluster.

- 2024/01/17: To delve deeper into the InternLM series of models, please check [InternLM](https://github.com/InternLM/InternLM) in our organization.


## Introduction

InternEvo is an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies. With a single codebase, it supports pre-training on large-scale clusters with thousands of GPUs, and fine-tuning on a single GPU while achieving remarkable performance optimizations. InternEvo achieves nearly 90% acceleration efficiency during training on 1024 GPUs.

Based on the InternEvo training framework, we are continually releasing a variety of large language models, including the InternLM-7B series and InternLM-20B series, which significantly outperform numerous renowned open-source LLMs such as LLaMA and other leading models in the field.

## Installation

First, install the specified versions of torch, torchvision, torchaudio, and torch-scatter.
For example:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

Install InternEvo:
```bash
pip install InternEvo
```

Install flash-attention (version v2.2.1):

If you need to use flash-attention to accelerate training, and it is supported in your environment, install as follows:
```bash
pip install flash-attn==2.2.1
```

For more detailed information about installation environment or source code installation, please refer to [Install Tutorial](https://internevo.readthedocs.io/en/latest/install.html#)

## Quick Start

### Train Script

Firstly, prepare training script as [train.py](https://github.com/InternLM/InternEvo/blob/develop/train.py)

For more detailed explanation, please refer to [Training Tutorial](https://internevo.readthedocs.io/en/latest/training.html#)

### Data Preparation

Secondly, prepare data for training or fine-tuning.

Download dataset from huggingface, take `roneneldan/TinyStories` dataset as example:
```bash
huggingface-cli download --repo-type dataset --resume-download "roneneldan/TinyStories" --local-dir "/mnt/petrelfs/hf-TinyStories"
```

Achieve tokenizer to local path. For example, download special_tokens_map.json、tokenizer.model、tokenizer_config.json、tokenization_internlm2.py and tokenization_internlm2_fast.py from `https://huggingface.co/internlm/internlm2-7b/tree/main` to local `/mnt/petrelfs/hf-internlm2-tokenizer` .

Then modify configuration file as follows:
```bash
TRAIN_FOLDER = "/mnt/petrelfs/hf-TinyStories"
data = dict(
    type="streaming",
    tokenizer_path="/mnt/petrelfs/hf-internlm2-tokenizer",
)
```

For other type dataset preparation, please refer to [Usage Tutorial](https://internevo.readthedocs.io/en/latest/usage.html#)

### Configuration File

The content of configuration file is as [7B_sft.py](https://github.com/InternLM/InternEvo/blob/develop/configs/7B_sft.py)

For more detailed introduction, please refer to [Usage Tutorial](https://internevo.readthedocs.io/en/latest/usage.html#)

### Train Start

Training can be started on slurm or torch distributed environment.

On slurm, using 2 nodes and 16 cards, the command is as follows:
```bash
$ srun -p internllm -N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py
```

On torch, using 1 node and 8 cards, the command is as follows:
```bash
$ torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```

## System Architecture

Please refer to the [System Architecture document](./doc/en/structure.md) for architecture details.

## Feature Zoo

<div align="center">
  <b>InternEvo Feature Zoo</b>
</div>
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Data</b>
      </td>
      <td>
        <b>Model</b>
      </td>
      <td>
        <b>Parallel</b>
      </td>
      <td>
        <b>Tool</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>Tokenized</li>
        <li>Streaming</li>
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
          <li>1F1B Pipeline Parallel</li>
          <li>PyTorch FSDP Training</li>
          <li>Megatron-LM Tensor Parallel (MTP)</li>
          <li>Megatron-LM Sequence Parallel (MSP)</li>
          <li>Flash-Attn Sequence Parallel (FSP)</li>
          <li>Intern Sequence Parallel (ISP)</li>
          <li>Memory Profiling</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="tools/transformers/README.md">Convert ckpt to HF</a></li>
          <li><a href="tools/transformers/README.md">Revert ckpt from HF</a></li>
          <li><a href="tools/tokenizer.py">Raw Data Tokenizer</a></li>
          <li><a href="tools/alpaca_tokenizer.py">Alpaca data Tokenizer</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## Common Tips

<div align="center">
</div>
<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Item</b>
      </td>
      <td>
        <b>Introduction</b>
      </td>
    </tr>
    <tr valign="bottom">
      <td>
        <b>Parallel Computing Loss</b>
      </td>
      <td>
        <b><a href="doc/en/parallel_output.md">link</a></b>
      </td>
    </tr>
  </tbody>
</table>

## Contribution

We appreciate all the contributors for their efforts to improve and enhance InternEvo. Community users are highly encouraged to participate in the project. Please refer to the contribution guidelines for instructions on how to contribute to the project.

## Acknowledgements

InternEvo codebase is an open-source project contributed by Shanghai AI Laboratory and researchers from different universities and companies. We would like to thank all the contributors for their support in adding new features to the project and the users for providing valuable feedback. We hope that this toolkit and benchmark can provide the community with flexible and efficient code tools for fine-tuning InternEvo and developing their own models, thus continuously contributing to the open-source community. Special thanks to the two open-source projects, [flash-attention](https://github.com/HazyResearch/flash-attention) and [ColossalAI](https://github.com/hpcaitech/ColossalAI).

## Citation

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```
