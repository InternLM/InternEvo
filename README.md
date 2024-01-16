# InternEvo

<div align="center">

<img src="./doc/imgs/logo.svg" width="200"/>
  <div> </div>
  <div align="center">
    <b><font size="5">InternEvo</font></b>
    <sup>
      <a href="https://internlm.intern-ai.org.cn/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    <div> </div>
  </div>

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

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>


### Latest News 🔥

- 2024/01/17: To delve deeper into the InternLM series of models, please check [InternLM](https://github.com/InternLM/InternLM) in our organization.


## Introduction

InternEvo is an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies. With a single codebase, it supports pre-training on large-scale clusters with thousands of GPUs, and fine-tuning on a single GPU while achieving remarkable performance optimizations. InternEvo achieves nearly 90% acceleration efficiency during training on 1024 GPUs.

Based on the InternEvo training framework, we are continually releasing a variety of large language models, including the InternLM-7B series and InternLM-20B series, which significantly outperform numerous renowned open-source LLMs such as LLaMA and other leading models in the field.


## Quick Start

Please refer to [Usage Tutorial](./doc/en/usage.md) to start InternEvo installation, data processing, pre-training and fine-tuning.

For more details, please check [internevo.readthedocs.io](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)

## System Architecture

Please refer to the [System Architecture document](./doc/en/structure.md) for architecture details.

## Performance

InternEvo deeply integrates Flash-Attention, Apex and other high-performance model operators to improve training efficiency. By building the Hybrid Zero technique, it achieves efficient overlap of computation and communication, significantly reducing cross-node communication traffic during training. InternEvo supports expanding the 7B model from 8 GPUs to 1024 GPUs, with an acceleration efficiency of up to 90% at the thousand-GPU scale, a training throughput of over 180 TFLOPS, and an average of over 3600 tokens per GPU per second. The following table shows InternEvo's scalability test data at different configurations:

| GPU Number         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |

TGS represents the average number of tokens processed per GPU per second. For more performance test data, please refer to the [Training Performance document](./doc/en/train_performance.md) for further details.

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
