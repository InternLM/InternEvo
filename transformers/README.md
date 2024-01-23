# InternLM Transformers

[English](./README.md) |
[简体中文](./README-zh-Hans.md)

This folder contains the `InternLM2` model in transformers format and some scripts.

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

## Weight Conversion - InternLM

`convert2hf_internlm.py` can convert saved training InternLM weights into the transformers format with a single command. Execute the command in the root directory of repository:

```bash
python transformers/convert2hf_internlm.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 4096 --rotary_type origin
```

```bash
# dynamic NTK
python transformers/convert2hf_internlm.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 4096 --rotary_type dynamic --scaling_factor 2.0
```

Then, you can load it using the `from_pretrained` interface:

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`revert_internlm.py` can convert huggingface-format checkpoint to training InternLM weights. Execute the command below:

```bash
python transformers/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1
```

If the model is saved with `safetensors`, please add `--safetensors` to the command:

```bash
python transformers/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1 --safetensors
```

## Weight Conversion - InternLM2

`convert2hf_internlm2.py` can convert saved training InternLM2 weights into the transformers format with a single command. Execute the command in the root directory of repository:

```bash
python transformers/convert2hf_internlm2.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 32768 --rotary_type origin
```

```bash
# dynamic NTK
python transformers/convert2hf_internlm2.py --src origin_ckpt/ --tgt hf_ckpt/ --tokenizer ./tools/tokenizer_internlm2.model --max_pos 32768 --rotary_type dynamic --scaling_factor 2.0
```

Then, you can load it using the `from_pretrained` interface:

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
```

`revert_internlm2.py` can convert huggingface-format checkpoint to training InternLM2 weights. Execute the command below:

```bash
python transformers/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash
```

If the model is saved with `safetensors`, please add `--safetensors` to the command:

```bash
python transformers/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --safetensors
```
