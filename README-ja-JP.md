# InternEvo

<div align="center">

<img src="./doc/imgs/InternEvo_logo.png" width="800"/>

[![Documentation Status](https://readthedocs.org/projects/internevo/badge/?version=latest)](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)
[![license](./doc/imgs/license.svg)](./LICENSE)

[📘使用方法](./doc/en/usage.md) |
[🛠️インストール](./doc/en/install.md) |
[📊パフォーマンス](./doc/en/train_performance.md) |
[🤔問題報告](https://github.com/InternLM/InternEvo/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>


### 最新ニュース 🔥

- 2024/08/29: InternEvoは、huggingface形式のストリーミングデータセットをサポートしています。データフローの詳細な手順を追加しました。

- 2024/04/17: InternEvoは、NPU-910Bクラスターでモデルのトレーニングをサポートしています。

- 2024/01/17: InternLMシリーズのモデルについてさらに詳しく知りたい方は、当社の組織内の[InternLM](https://github.com/InternLM/InternLM)をご覧ください。


## イントロダクション

InternEvoは、広範な依存関係を必要とせずにモデルの事前トレーニングをサポートすることを目的としたオープンソースの軽量トレーニングフレームワークです。単一のコードベースで、数千のGPUを搭載した大規模クラスターでの事前トレーニングと、単一GPUでのファインチューニングをサポートし、顕著なパフォーマンス最適化を実現しています。InternEvoは、1024個のGPUでのトレーニング中に約90%の加速効率を達成しています。

InternEvoトレーニングフレームワークを基に、当社はInternLM-7BシリーズやInternLM-20Bシリーズを含むさまざまな大規模言語モデルを継続的にリリースしています。これらのモデルは、LLaMAのような数多くの有名なオープンソースの大規模言語モデルや、その他の業界をリードするモデルを大きく上回る性能を発揮しています。

## インストール

まず、指定バージョンのtorch、torchvision、torchaudio、およびtorch-scatterをインストールしてください。
たとえば：
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

InternEvoをインストールします：
```bash
pip install InternEvo
```

flash-attention（バージョンv2.2.1）をインストールします：

もしflash-attentionを使用してトレーニングを加速する必要があり、あなたの環境でサポートされている場合は、以下の通りインストールしてください：
```bash
pip install flash-attn==2.2.1
```

インストール環境やソースコードインストールに関するより詳細な情報については、以下のリンクを参照してください [インストールチュートリアル](https://internevo.readthedocs.io/en/latest/install.html#)

## クイックスタート

### トレーニングスクリプト

まず、トレーニングスクリプトを準備してください [train.py](https://github.com/InternLM/InternEvo/blob/develop/train.py)

より詳細な説明については、以下を参照してください [トレーニングチュートリアル](https://internevo.readthedocs.io/en/latest/training.html#)

### データの準備

次に、トレーニングまたはファインチューニングのためのデータを準備してください。

データセットをHuggingfaceからダウンロードし、roneneldan/TinyStories データセットを例にとって説明します：
```bash
huggingface-cli download --repo-type dataset --resume-download "roneneldan/TinyStories" --local-dir "/mnt/petrelfs/hf-TinyStories"
```

トークナイザーをローカルパスに配置してください。例として、https://huggingface.co/internlm/internlm2-7b/tree/main から special_tokens_map.json、tokenizer.model、tokenizer_config.json、tokenization_internlm2.py、tokenization_internlm2_fast.py をローカルの /mnt/petrelfs/hf-internlm2-tokenizer にダウンロードしてください。

次に、設定ファイルを以下のように変更します：
```bash
TRAIN_FOLDER = "/mnt/petrelfs/hf-TinyStories"
data = dict(
    type="streaming",
    tokenizer_path="/mnt/petrelfs/hf-internlm2-tokenizer",
)
```

他のタイプのデータセットの準備については、以下を参照してください [使用方法のチュートリアル](https://internevo.readthedocs.io/en/latest/usage.html#)

### Configuration File

設定ファイルの内容は以下の通りです：[7B_sft.py](https://github.com/InternLM/InternEvo/blob/develop/configs/7B_sft.py)

より詳細な紹介については、以下を参照してください [使用方法のチュートリアル](https://internevo.readthedocs.io/en/latest/usage.html#)

### トレーニング開始

トレーニングは、slurmまたはtorch distributed環境で開始できます。

Slurm環境で2ノード16カードを使用する場合、コマンドは以下の通りです：
```bash
$ srun -p internllm -N 2 -n 16 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py
```

torchを使用し、1ノード8カードで実行する場合、コマンドは以下の通りです：
```bash
$ torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```

## システムアーキテクチャ

アーキテクチャの詳細については、[システムアーキテクチャドキュメント](./doc/en/structure.md)を参照してください。

## フィーチャーズー

<div align="center">
  <b>InternEvo フィーチャーズー</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>データ</b>
      </td>
      <td>
        <b>モデル</b>
      </td>
      <td>
        <b>並列</b>
      </td>
      <td>
        <b>ツール</b>
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

## コントリビュート

我々は、InternEvo を改善し、向上させるために尽力してくれたすべての貢献者に感謝している。コミュニティ・ユーザーのプロジェクトへの参加が強く推奨されます。プロジェクトへの貢献方法については、貢献ガイドラインを参照してください。

## 謝辞

InternEvo コードベースは、上海 AI 研究所と様々な大学や企業の研究者によって貢献されたオープンソースプロジェクトです。プロジェクトに新機能を追加してくれたすべての貢献者と、貴重なフィードバックを提供してくれたユーザーに感謝したい。私たちは、このツールキットとベンチマークが、InternLM をファインチューニングし、独自のモデルを開発するための柔軟で効率的なコードツールをコミュニティに提供し、オープンソースコミュニティに継続的に貢献できることを願っています。2 つのオープンソースプロジェクト、[flash-attention](https://github.com/HazyResearch/flash-attention) と [ColossalAI](https://github.com/hpcaitech/ColossalAI) に感謝します。

## 引用

```
@misc{2023internlm,
    title={InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities},
    author={InternLM Team},
    howpublished = {\url{https://github.com/InternLM/InternLM}},
    year={2023}
}
```
