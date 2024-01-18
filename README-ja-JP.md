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

[📘使用方法](./doc/en/usage.md) |
[🛠️インストール](./doc/en/install.md) |
[📊パフォーマンス](./doc/en/train_performance.md) |
[🤔問題報告](https://github.com/InternLM/InternEvo/issues/new)

[English](./README.md) |
[简体中文](./README-zh-Hans.md) |
[日本語](./README-ja-JP.md)

</div>

<p align="center">
    👋 <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> と <a href="https://github.com/InternLM/InternLM/assets/25839884/a6aad896-7232-4220-ac84-9e070c2633ce" target="_blank">WeChat</a> で私たちに参加してください
</p>


### 最新ニュース 🔥

- 2024/01/17: InternLMシリーズのモデルについてさらに詳しく知りたい方は、当社の組織内の[InternLM](https://github.com/InternLM/InternLM)をご覧ください。


## イントロダクション

InternEvoは、広範な依存関係を必要とせずにモデルの事前トレーニングをサポートすることを目的としたオープンソースの軽量トレーニングフレームワークです。単一のコードベースで、数千のGPUを搭載した大規模クラスターでの事前トレーニングと、単一GPUでのファインチューニングをサポートし、顕著なパフォーマンス最適化を実現しています。InternEvoは、1024個のGPUでのトレーニング中に約90%の加速効率を達成しています。

InternEvoトレーニングフレームワークを基に、当社はInternLM-7BシリーズやInternLM-20Bシリーズを含むさまざまな大規模言語モデルを継続的にリリースしています。これらのモデルは、LLaMAのような数多くの有名なオープンソースの大規模言語モデルや、その他の業界をリードするモデルを大きく上回る性能を発揮しています。


## クイックスタート

InternEvoのインストール、データ処理、事前トレーニング、およびファインチューニングを開始するためには、[使用チュートリアル](./doc/en/usage.md) を参照してください。

詳細については、以下をご確認ください: [internevo.readthedocs.io](https://internevo.readthedocs.io/zh_CN/latest/?badge=latest)

## システムアーキテクチャ

アーキテクチャの詳細については、[システムアーキテクチャドキュメント](./doc/en/structure.md)を参照してください。

## トレーニングパフォーマンス

InternEvoは、Flash-Attention、Apexなどの高性能モデルオペレーターを深く統合してトレーニング効率を向上させています。Hybrid Zeroテクニックを構築することにより、計算と通信の効率的な重複を実現し、トレーニング中のクロスノード通信トラフィックを大幅に削減します。InternEvoは、7Bモデルを8つのGPUから1024個のGPUに拡張することをサポートし、千のGPUスケールで最大90%の加速効率、180 TFLOPSを超えるトレーニングスループット、そしてGPUあたり秒間3600トークン以上の平均を実現します。以下の表は、異なる構成でのInternEvoのスケーラビリティテストデータを示しています：

| GPU 番号         | 8   | 16  | 32  | 64  | 128  | 256  | 512  | 1024  |
| ---------------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ------ |
| TGS | 4078 | 3939 | 3919 | 3944 | 3928  | 3920  | 3835  | 3625   |
| TFLOPS  | 193 | 191  | 188  | 188  | 187   | 185   | 186   | 184    |

TGSは、GPUごとの秒間平均処理トークン数を表しています。より多くのパフォーマンステストデータについては、[トレーニングパフォーマンスドキュメント](./doc/en/train_performance.md) をご参照ください。

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
