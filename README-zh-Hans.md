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

[![使用文档](https://readthedocs.org/projects/internevo/badge/?version=latest)](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)
[![license](./doc/imgs/license.svg)](./LICENSE)

[📘使用教程](./doc/en/usage.md) |
[🛠️安装指引](./doc/en/install.md) |
[📊框架性能](./doc/en/train_performance.md) |
[🤔问题报告](https://github.com/InternLM/InternEvo/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

<p align="center">
    👋 加入我们的 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> 和 <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">微信群</a>
</p>


### 新闻 🔥

- 2024/01/17: 更多关于InternLM系列模型的内容，请查看组织内的 [InternLM](https://github.com/InternLM/InternLM)


## 简介

InternEvo是一个开源的轻量级训练框架，旨在支持无需大量依赖关系的模型预训练。凭借单一代码库，InternEvo支持在具有上千GPU的大规模集群上进行预训练，并在单个GPU上进行微调，同时可实现显著的性能优化。当在1024个GPU上进行训练时，InternEvo可实现近90%的加速效率。

基于InternEvo训练框架，我们累计发布了一系列大语言模型，包括InternLM-7B系列和InternLM-20B系列，这些模型在性能上显著超越了许多知名的开源LLMs，如LLaMA和其他模型。

## 快速开始

请查看 [Usage Tutorial](./doc/en/usage.md) 来开始InternEvo的安装、数据处理、预训练与微调。

更多细节请查看文档 [internevo.readthedocs.io](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)

## 系统架构

Please refer to the [System Architecture document](./doc/en/structure.md) for architecture details.

## 框架性能

InternEvo深度集成了Flash-Attention、Apex等高性能计算库，以提高训练效率。通过构建Hybrid Zero技术，InternEvo可在训练过程中实现计算和通信的有效重叠，显著降低跨节点通信流量。InternEvo支持将7B模型从8个GPU扩展到1024个GPU，在千卡规模下可实现高达90%的加速效率，超过180 TFLOPS的训练吞吐量，平均每个GPU每秒可处理超过3600个tokens。下表展示了InternEvo在不同配置下的可扩展性测试数据：

| GPU Number         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |

TGS表示每张GPU每秒可处理的平均Tokens数量。更多模型性能测试数据细节请查看 [Training Performance document](./doc/en/train_performance.md)


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
