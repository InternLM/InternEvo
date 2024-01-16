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

[![license](./doc/imgs/license.svg)](./LICENSE)
[![evaluation](./doc/imgs/compass_support.svg)](https://github.com/internLM/OpenCompass/)
[![Documentation Status](https://readthedocs.org/projects/internlm/badge/?version=latest)](https://internlm.readthedocs.io/zh_CN/latest/?badge=latest)

[📘Usage](./doc/en/usage.md) |
[🛠️Installation](./doc/en/install.md) |
[📊Train Performance](./doc/en/train_performance.md) |
[🤗HuggingFace](https://huggingface.co/internlm) |
[🤔Reporting Issues](https://github.com/InternLM/InternEvo/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a>
</p>

## Introduction

InternEvo is an open-sourced lightweight training framework aims to support model pre-training without the need for extensive dependencies. With a single codebase, it supports pre-training on large-scale clusters with thousands of GPUs, and fine-tuning on a single GPU while achieving remarkable performance optimizations. InternEvo achieves nearly 90% acceleration efficiency during training on 1024 GPUs.

Based on the InternEvo training framework, we are continually releasing a variety of large language models, including the InternLM-7B series and InternLM-20B series, which significantly outperform numerous renowned open-source LLMs such as LLAMA and other leading models in the field.

### Dialogue

You can interact with the InternLM Chat 7B model through a frontend interface by running the following code:

```bash
pip install streamlit==1.24.0
pip install transformers==4.30.2
streamlit run web_demo.py
```

The effect is as follows

![demo](https://github.com/InternLM/InternLM/assets/9102141/11b60ee0-47e4-42c0-8278-3051b2f17fe4)

### Deployment

We use [LMDeploy](https://github.com/InternLM/LMDeploy) to complete the one-click deployment of InternEvo.

1. First, install LMDeploy:

```shell
python3 -m pip install lmdeploy
```

2. Use the following command for iteractive communication with `internlm-chat-7b` model on localhost:

```shell
lmdeploy chat turbomind InternLM/internlm-chat-7b --model-name internlm-chat-7b
```

3. Besides chatting via command line, you can start lmdeploy `api_server` as below:

```shell
lmdeploy serve api_server InternLM/internlm-chat-7b --model-name internlm-chat-7b
```
For a comprehensive understanding of the `api_server` RESTful API, kindly consult [this](https://github.com/InternLM/lmdeploy/blob/main/docs/en/restful_api.md) guide. For additional deployment tutorials, feel free to explore [here](https://github.com/InternLM/LMDeploy).

## Training

### Pre-training and Fine-tuning Tutorial

Please refer to [Usage Tutorial](./doc/en/usage.md) to start InternEvo installation, data processing, pre-training and fine-tuning.

### Convert to Transformers Format

The model trained by InternEvo can be easily converted to HuggingFace Transformers format, which is convenient for seamless docking with various open source projects in the community. With the help of `tools/transformers/convert2hf.py`, the weights saved during training can be converted into transformers format with one command:

```bash
python tools/transformers/convert2hf.py --src_folder origin_ckpt/ --tgt_folder hf_ckpt/ --tokenizer ./tools/V7_sft.model
```

After conversion, it can be loaded as transformers by the following code

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

## Training System

### System Architecture

Please refer to the [System Architecture document](./doc/en/structure.md) for further details.

### Training Performance

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

## License

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.

## Citation

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternEvo}},
    year={2023}
}
```
