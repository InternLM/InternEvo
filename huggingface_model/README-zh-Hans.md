# InternLM Transformers

[English](./README.md) |
[简体中文](./README-zh-Hans.md)

该文件夹下包含了 transformers 格式的模型及一些辅助脚本。

```bash
├── convert2hf_internlm2.py
├── convert2hf_internlm.py
├── internlm2_model
│   ├── configuration_internlm2.py
│   ├── __init__.py
│   ├── modeling_internlm2.py
│   └── tokenization_internlm2.py
├── internlm_model
│   ├── configuration_internlm.py
│   ├── __init__.py
│   ├── modeling_internlm.py
│   └── tokenization_internlm.py
├── README.md
├── README-zh-Hans.md
├── revert_internlm2.py
└── revert_internlm.py
```

## 加载 HuggingFace 模型

`InternLM` 和 `InternLM2` 的 HuggingFace 模型可以通过指定不同参数来满足不同场景下的应用或部署。下面是常用的几种参数：

- `trust_remote_code=True`：该参数必须指定，这样 HuggingFace 会加载模型路径下的模型文件或 tokenizer 文件。
- `torch_dtype`(*可选*): 指定加载的参数类型：
    - `None`: 当不指定参数或为 `None` 时，加载的模型为 `float32` 类型。
    - `"auto"`： 根据模型路径下 `config.json` 中的 `torch_dtype` 字段设定模型的类型。
    - 具体的类型，如 `torch.float16`、`torch.bfloat16` 等：加载模型为指定的类型。
- `attn_implementation`：该参数可以指定模型是否使用 Flash Attention：
    - `"eager"`：不指定该参数或者指定为 `eager` 时，会使用基础的注意力计算方式。
    - `"flash_attention_2"`: 使用 Flash Attention 2 计算注意力。此时请确保您的环境中有的 [flash_attn](https://github.com/Dao-AILab/flash-attention) 库，并且将 `torch_dtype` 字段设置为 `torch.float16` 或 `torch.bfloat16`，否则程序会报错。
- `device_map`：指定该参数可以将 HuggingFace 模型在多张显卡上运行，一般设置为 `"auto"` 即可。请确保您的环境中安装了 `accelerate` 库。更加详细的设置方法可以参考 [HuggingFace 的文档](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference)

下面是可以参考的例子：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> import torch
# 单卡，使用 float32 加载
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
# 单卡加载，数据类型根据 config.json 的内容指定
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype="auto").cuda()
# 单卡加载，数据类型为 torch.float16
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16).cuda()
# 单卡加载，数据类型为 torch.float16，并且使用 flash attention
# 需要安装 flash_attn 库，注意 flash attention 只能在 float16 和 bfloat16 的情况下使用
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
# 多卡加载，并且指定 dtype（需要安装 accelerate 库：pip install accelerate）
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
# 多卡加载，并且使用 flash attention
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2")
```

## InternEvo适配训练huggingface格式模型

对于huggingface上面发布的模型，使用InternEvo框架训练，需要适配的步骤如下：

### 步骤一 下载模型文件
从huggingface上下载指定模型的config及modeling文件，放入huggingface_model路径下，新建模型路径。
如：
```bash
huggingface_model
├── internlm_model
│   ├── configuration_internlm.py
│   ├── __init__.py
│   ├── modeling_internlm.py
```

在模型路径中创建__init__.py文件，并导出相关接口，如：
```bash
from .configuration_internlm import InternLMConfig
from .modeling_internlm import InternLMForCausalLM
```

### 步骤二 注册模型
在internlm/model/registry.py文件中，需要在register_model_initializer函数里面注册huggingface模型的入口函数，如：
```bash
from huggingface_model.internlm_model.modeling_internlm import InternLMForCausalLM

def register_model_initializer() -> None:
    model_initializer.register_module("INTERNLM_FROM_HF", InternLMForCausalLM)
```

### 步骤三 加载模型配置参数
在internlm/model/builder.py文件中，create_model函数中，在hf_model_conf_map里面新增模型配置接口，如：
```bash
hf_model_conf_map = {
    "INTERNLM_FROM_HF":("huggingface_model.internlm_model.configuration_internlm", "InternLMConfig"),
}
```
注意：这里要求新增的模型model_type定义以_FROM_HF字段结尾。

### 步骤四 修改配置文件，加载huggingface格式数据集及模型权重
我们提供了configs/7B_hf.py配置文件，用来训练huggingface上的模型。其中，需要更改的配置项及说明如下：
```bash
model_type = "INTERNLM_FROM_HF"
MODEL_ONLY_FOLDER = "internlm/internlm-7b"
ckpt = dict(
    load_ckpt_info=dict(path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="hf_model"),
    auto_resume=False,
)
TRAIN_FOLDER = "roneneldan/TinyStories"
data = dict(
    type="hf",
    tokenizer_path="internlm/internlm-7b",
)
```
model_type：设置模型类型，需要以“_FROM_HF”结尾，这里设置的值需要与上述步骤二及步骤三中的值保持一致。
MODEL_ONLY_FOLDER：设置从huggingface上加载的模型路径。
load_ckpt_info：模型加载路径信息，其中，ckpt_type需要设置为"hf_model"。
auto_resume：需要设置为False。
TRAIN_FOLDER：设置从huggingface上加载的数据集路径。
type：数据集类型需要设置为"hf"。
tokenizer_path：设置从huggingface加载的模型tokenizer路径。

## 权重转换 - InternLM

`convert2hf_internlm.py` 可以将训练保存的权重一键转换为 transformers 格式。所需要的参数为：

- `--src`: 需要转换的权重路径。
- `--tgt`: 转换后 HuggingFace 权重的保存路径。
- `--tokenizer`: tokenizer 路径。
- `--dtype`(*可选*): 转换后权重保存的 dtype；默认为 `bfloat16`。
- `--max_shard`(*可选*): 权重切分的最大大小，等同于 `save_pretrained` 函数的 `max_shard_size` 参数。默认为 `10GB`。
- `--max_pos`(*可选*): 模型的最大上下文大小，一般为训练时的最大序列长度。默认为 `4096`。
- `--rotary_type`(*可选*): 位置编码的种类，支持两种：`origin` 为旋转位置编码；`dynamic` 为动态 NTK 旋转编码。默认为 `origin`。
- `--scaling_factor`(*可选*): 动态 NTK 旋转编码的缩放参数，该参数仅当 `--rotary_type=origin` 时有意义。默认为 `1.0`。

在仓库根目录运行以下命令：

```bash
python huggingface_model/convert2hf_internlm.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 4096 --rotary_type origin
```

```bash
# dynamic NTK
python huggingface_model/convert2hf_internlm.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 4096 --rotary_type dynamic --scaling_factor 2.0
```

然后可以使用 `from_pretrained` 接口加载：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`revert_internlm.py` 可以将 HuggingFace 的模型权重转换为 InternLM 原生训练格式的权重。所需要的参数为：

- `--src`: 需要转换的 HuggingFace 权重路径。
- `--tgt`: 转换后权重的保存路径。
- `--tp_size`: 转换后权重的张量并行大小。
- `--version`: MLP 层中 `down_proj` `up_proj` 与 `w2` `w3` 的对应关系。为 `1` 时，HuggingFace 模型中的 `down_proj` 对应 `InternLM` 的 `w3`，`up_proj` 对应 `InternLM` 的 `w2`，为 `2` 时则相反。
- `--embed_split`: 即 `InternEvo` 框架的 `embed_split_hidden` 参数，如果指定了该参数则会对嵌入层在 hidden states 维度上进行切分，反之则在另一个维度进行切分。
- `--use_flash`: 即 `InternEvo` 的 `use_flash_attn` 参数，如果指定了该参数，则在加载后使用 Flash Attention。
- `--safetensors`: 需要转换的模型是否是以 `safetensors` 保存的。如果指定了该参数则表明是 `safetensors` 格式。

在仓库根目录运行以下命令：

```bash
python huggingface_model/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1
```

如果模型是用 `safetensors` 格式保存的，则需要添加 `--safetensors` 参数：

```bash
python huggingface_model/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1 --safetensors
```

## 权重转换 - InternLM2

`convert2hf_internlm2.py` 可以将训练保存的权重一键转换为 transformers 格式。所需要的参数为：

- `--src`: 需要转换的权重路径。
- `--tgt`: 转换后 HuggingFace 权重的保存路径。
- `--tokenizer`: tokenizer 路径。
- `--dtype`(*可选*): 转换后权重保存的 dtype；默认为 `bfloat16`。
- `--max_shard`(*可选*): 权重切分的最大大小，等同于 `save_pretrained` 函数的 `max_shard_size` 参数。默认为 `10GB`。
- `--max_pos`(*可选*): 模型的最大上下文大小，一般为训练时的最大序列长度。默认为 `4096`。
- `--rotary_type`(*可选*): 位置编码的种类，支持两种：`origin` 为旋转位置编码；`dynamic` 为动态 NTK 旋转编码。默认为 `origin`。
- `--scaling_factor`(*可选*): 动态 NTK 旋转编码的缩放参数，该参数仅当 `--rotary_type=origin` 时有意义。默认为 `1.0`。

在仓库根目录运行以下命令：

```bash
python huggingface_model/convert2hf_internlm2.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 32768 --rotary_type origin
```

```bash
# dynamic NTK
python huggingface_model/convert2hf_internlm2.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 32768 --rotary_type dynamic --scaling_factor 2.0
```

然后可以使用 `from_pretrained` 接口加载：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`revert_internlm2.py` 可以将 HuggingFace 的模型权重转换为 InternLM2 原生训练格式的权重，所需要的参数为：

- `--src`: 需要转换的 HuggingFace 权重路径。
- `--tgt`: 转换后权重的保存路径。
- `--tp_size`: 转换后权重的张量并行大小。
- `--embed_split`: 即 `InternEvo` 框架的 `embed_split_hidden` 参数，如果指定了该参数则会对嵌入层在 hidden states 维度上进行切分，反之则在另一个维度进行切分。
- `--use_flash`: 即 `InternEvo` 的 `use_flash_attn` 参数，如果指定了该参数，则在加载后使用 Flash Attention。
- `--safetensors`: 需要转换的模型是否是以 `safetensors` 保存的。如果指定了该参数则表明是 `safetensors` 格式。

在仓库根目录运行以下命令：

```bash
python huggingface_model/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash
```

如果模型是用 `safetensors` 格式保存的，则需要添加 `--safetensors` 参数：

```bash
python huggingface_model/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --safetensors
```
