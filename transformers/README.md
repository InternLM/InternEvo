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

## Load InternLM and InternLM2 with HuggingFace

`InternLM` and `InternLM2` HuggingFace models can be adapted to different scenarios or deployments by specifying different parameters. Here are some commonly used parameters:

- `trust_remote_code=True`: This parameter must be specified so that HuggingFace can load the model file or tokenizer file located in the model path.
- `torch_dtype`(*Optional*): Specify the data type of the loaded parameters:
    - `None`: When this parameter is not specified or set to None, the loaded model will be of type float32.
    - `"auto"`: The model type will be determined by the torch_dtype field in the config.json file located in the model path.
    - Specific types, such as `torch.float16`, `torch.bfloat16`, etc.: Load the model with the specified data type.
- `attn_implementation`(*Optional*): This parameter can be used to specify whether the model uses Flash Attention:
    - `"eager"`: If this parameter is not specified or set to `"eager"`, the basic attention calculation method will be used.
    - `"flash_attention_2"`: Use Flash Attention 2 to calculate attention. Make sure you have the [flash_attn](https://github.com/Dao-AILab/flash-attention) library in your environment, and set the `torch_dtype` field to `torch.float16` or `torch.bfloat16`. Otherwise, an error will occur.
- `device_map`(*Optional*): Specifying this parameter can run the HuggingFace model on multiple GPUs. Generally, it can be set to `"auto"`. Make sure you have the accelerate library installed in your environment. For more detailed settings, you can refer to the [HuggingFace documentation](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference).

Here are some examples that you can refer to:

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> import torch
# Single GPU, load with float32
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True).cuda()
# Single GPU, load with data type determined by config.json
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype="auto").cuda()
# Single GPU, load with data type torch.float16
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16).cuda()
# Single GPU, load with data type torch.float16 and use flash attention
# flash_attn library needs to be installed, and flash attention can only be used with float16 and bfloat16
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda()
# Multi-GPU load and specify dtype (accelerate library needs to be installed: pip install accelerate)
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
# Multi-GPU load and use flash attention
>>> model = AutoModel.from_pretrained("hf_ckpt/", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2")
```

## Weight Conversion - InternLM

`convert2hf_internlm.py` can convert saved training InternLM weights into the transformers format with a single command. Below are the parameters needed:

- `--src`: Path to the weights to be converted.
- `--tgt`: Path to save the converted HuggingFace weights.
- `--tokenizer`: Path to the tokenizer.
- `--dtype` (*Optional*): The dtype to save the converted weights; defaults to `bfloat16`.
- `--max_shard` (*Optional*): The maximum size of the sharded weights, equivalent to the `max_shard_size` parameter of the `save_pretrained` function. Defaults to `10GB`.
- `--max_pos` (*Optional*): The maximum context size of the model, generally equal to the maximum sequence length during training. Defaults to `4096`.
- `--rotary_type` (*Optional*): The type of positional encoding; supports two options: `origin` for rotary positional encoding, and `dynamic` for dynamic NTK rotary encoding. Defaults to `origin`.
- `--scaling_factor` (*Optional*): The scaling factor for dynamic NTK rotary encoding; this parameter is only relevant when `--rotary_type=origin`. Defaults to `1.0`.

Execute the command in the root directory of repository:

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

`revert_internlm.py` can convert huggingface-format checkpoint to training InternLM weights. Below are the parameters needed"

- `--src`: Path to the HuggingFace weights to be converted.
- `--tgt`: Path to save the converted weights.
- `--tp_size`: Tensor parallel size for the converted weights.
- `--version`: The correspondence between `down_proj` `up_proj` in MLP layers and `w2` `w3` in `InternLM`. Set to `1` if HuggingFace's `down_proj` corresponds to `InternLM`'s `w3` and `up_proj` corresponds to `InternLM`'s `w2`, and set to `2` for the opposite.
- `--embed_split`: The `embed_split_hidden` parameter of the `InternEvo` framework. If specified, the embedding layer will be split along the hidden states dimension; otherwise, it will be split along another dimension.
- `--use_flash`: The `use_flash_attn` parameter of `InternEvo`. If specified, Flash Attention will be used after loading.
- `--safetensors`: Indicates whether the HuggingFace model to is saved with `safetensors`. If specified, it means the model is saved with `safetensors`.

Execute the command below:

```bash
python transformers/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1
```

If the model is saved with `safetensors`, please add `--safetensors` to the command:

```bash
python transformers/revert_internlm.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --version 1 --safetensors
```

## Weight Conversion - InternLM2

`convert2hf_internlm2.py` can convert saved training InternLM2 weights into the transformers format with a single command. Below are the parameters needed:

- `--src`: Path to the weights to be converted.
- `--tgt`: Path to save the converted HuggingFace weights.
- `--tokenizer`: Path to the tokenizer.
- `--dtype` (*Optional*): The dtype to save the converted weights; defaults to `bfloat16`.
- `--max_shard` (*Optional*): The maximum size of the sharded weights, equivalent to the `max_shard_size` parameter of the `save_pretrained` function. Defaults to `10GB`.
- `--max_pos` (*Optional*): The maximum context size of the model, generally equal to the maximum sequence length during training. Defaults to `4096`.
- `--rotary_type` (*Optional*): The type of positional encoding; supports two options: `origin` for rotary positional encoding, and `dynamic` for dynamic NTK rotary encoding. Defaults to `origin`.
- `--scaling_factor` (*Optional*): The scaling factor for dynamic NTK rotary encoding; this parameter is only relevant when `--rotary_type=origin`. Defaults to `2.0`.

Execute the command in the root directory of repository:

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

`revert_internlm2.py` can convert huggingface-format checkpoint to training InternLM2 weights. Below are the parameters needed:

- `--src`: Path to the HuggingFace weights to be converted.
- `--tgt`: Path to save the converted weights.
- `--tp_size`: Tensor parallel size for the converted weights.
- `--embed_split`: The `embed_split_hidden` parameter of the `InternEvo` framework. If specified, the embedding layer will be split along the hidden states dimension; otherwise, it will be split along another dimension.
- `--use_flash`: The `use_flash_attn` parameter of `InternEvo`. If specified, Flash Attention will be used after loading.
- `--safetensors`: Indicates whether the HuggingFace model to is saved with `safetensors`. If specified, it means the model is saved with `safetensors`.

Execute the command below:

```bash
python transformers/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash
```

If the model is saved with `safetensors`, please add `--safetensors` to the command:

```bash
python transformers/revert_internlm2.py --src /path/to/src --tgt /path/to/tgt --tp_size 2 --embed_split --use_flash --safetensors
```
