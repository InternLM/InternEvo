# InternLM Transformers

[English](./README.md) |
[简体中文](./README-zh-Hans.md)

该文件夹下包含了 transformers 格式的 `InternLM2` 模型及一些辅助脚本。

```bash
├── convert2hf_internlm2.py
├── convert2hf_internlm.py
├── internlm2_model
│   ├── configuration_internlm.py
│   ├── __init__.py
│   ├── modeling_internlm2.py
│   ├── __pycache__
│   │   ├── configuration_internlm.cpython-310.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   ├── modeling_internlm2.cpython-310.pyc
│   │   └── tokenization_internlm.cpython-310.pyc
│   └── tokenization_internlm.py
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

## 权重转换 - InternLM

`convert2hf_internlm.py` 可以将训练保存的权重一键转换为 transformers 格式。在仓库根目录运行以下命令：

```bash
python transformers/convert2hf_internlm.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/v13.model --max_pos 4096 --rotary_type origin
```

```bash
# dynamic NTK
python transformers/convert2hf_internlm.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/v13.model --max_pos 4096 --rotary_type dynamic --scaling_factor 2.0
```

然后可以使用 `from_pretrained` 接口加载：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`revert_internlm.py` 可以将 HuggingFace 的模型权重转换为 InternLM 原生训练格式的权重：

```bash
python transformers/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1
```

如果模型是用 `safetensors` 格式保存的，则需要添加 `--safetensors` 参数：

```bash
python transformers/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1 --safetensors
```

## 权重转换 - InternLM2

`convert2hf_internlm2.py` 可以将训练保存的权重一键转换为 transformers 格式。在仓库根目录运行以下命令：

```bash
python transformers/convert2hf_internlm2.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/v13.model --max_pos 32768 --rotary_type origin
```

```bash
# dynamic NTK
python transformers/convert2hf_internlm2.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/v13.model --max_pos 32768 --rotary_type dynamic --scaling_factor 2.0
```

然后可以使用 `from_pretrained` 接口加载：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`revert_internlm2.py` 可以将 HuggingFace 的模型权重转换为 InternLM2 原生训练格式的权重：

```bash
python transformers/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash
```

如果模型是用 `safetensors` 格式保存的，则需要添加 `--safetensors` 参数：

```bash
python transformers/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --safetensors
```
