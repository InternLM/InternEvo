import os

import pytest
import torch
from sentencepiece import SentencePieceProcessor

from internlm.apis.inference import SequenceGenerator, batch_tokenize
from internlm.initialize import initialize_distributed_env  # noqa: E402
from internlm.train import initialize_model, initialize_parallel_communicator


def set_seed(seed: int = 1024):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_generate(path, model_type="INTERNLM2", tokenizer_path=""):
    model_cfg = os.path.join(path, "model_config.pt")
    model_wt = os.path.join(path, "model_tp0_pp0.pt")
    model_config = torch.load(model_cfg)
    model_config["apply_post_layer_norm"] = False
    if model_config.get("adapt_hf") is not None:
        model_config.pop("adapt_hf")
    evo_cfg = dict(
        model_type=model_type,
        model=model_config,
        parallel=dict(
            zero1=dict(size=1, fsdp=False),
            pipeline=dict(size=1, interleaved_overlap=True),
            tensor=dict(size=1, mode="mtp"),
            sequence_parallel=0,
        ),
    )
    initialize_distributed_env(evo_cfg, master_port=23574, args_check=False)

    tokenizer = SentencePieceProcessor(tokenizer_path)  # pylint: disable=E1121

    def convert_to_str(output_ids):
        output_tokens = output_ids.tolist()
        all_output_str = []
        for b in range(len(output_tokens)):
            for sent_idx in range(len(output_tokens[b])):
                cur_output_tokens = output_tokens[b][sent_idx]
                cur_sent = tokenizer.decode(cur_output_tokens)
                all_output_str.append(cur_sent)
        return all_output_str

    model = initialize_model()
    _ = initialize_parallel_communicator(model)
    # Directly get the origin model without NativeAMP wrapper.
    model = model.model

    state_dict = torch.load(model_wt)
    load_info = model.load_state_dict(state_dict, strict=False)
    print(load_info)

    sequenece_generator = SequenceGenerator(
        decoder=model,
        eos_token_id=tokenizer.eos_id(),
        pad_token_id=tokenizer.bos_id(),
        bos_token_id=tokenizer.bos_id(),
        additional_eos_token_list=None,
    )

    test_prompt_0 = "Gold is considered to be a precious metal."
    test_prompt_1 = "what is love? someone think it is a feeling, someone think it is a chemical reaction."
    test_prompt_2 = "kobe bryant is a basketball player."

    prompt_3 = [
        test_prompt_0,
        test_prompt_1,
        test_prompt_2,
    ]
    prompt_2 = [
        test_prompt_0,
        test_prompt_1,
    ]

    prompt_1 = [test_prompt_0]

    def generate(prompt):
        input_ids = batch_tokenize(prompt, tokenizer, pad_token_id=tokenizer.bos_id()).cuda()
        generate_kwargs = {}
        set_seed()
        output_ids = sequenece_generator.generate(
            input_ids,
            num_return_sequences=generate_kwargs.get("num_return_sequences", 1),
            max_length=generate_kwargs.get("max_length", input_ids.shape[1] + 80),
            num_beams=generate_kwargs.get("num_beams", 1),
            do_sample=generate_kwargs.get("do_sample", False),
            temperature=generate_kwargs.get("temperature", 1.0),
            top_k=generate_kwargs.get("top_k", 50),
            top_p=generate_kwargs.get("top_p", 1.0),
            repetition_penalty=generate_kwargs.get("repetition_penalty", 1),
            length_penalty=generate_kwargs.get("repetition_penalty", 1.0),
        )

        all_output_str = convert_to_str(output_ids)
        return all_output_str

    output_3 = generate(prompt_3)
    output_2 = generate(prompt_2)
    output_1 = generate(prompt_1)

    assert output_3[0] == output_2[0]
    assert output_3[1] == output_2[1]
    assert (
        output_1[0]
        == "Gold is considered to be a precious metal. It is a metal that is highly valued for its \
rarity and beauty. Gold is often used in jewelry, coins, and other decorative items. It is also used in \
the production of electronics and other high-tech products. Gold is a highly sought-after metal because \
of its ability to resist corrosion and tarnish. It is also highly resistant to fire and is a good conductor \
of heat and electricity.\n"
    )
    print("test generate done!")


def test_internlm2_1_8B_generate():
    base_model_dir = os.environ.get("qa_data")
    if base_model_dir is not None:
        model_dir = os.path.join(base_model_dir, "internlm2_1_8B")
        tokenizer_path = os.path.join(base_model_dir, "InternLM_CI_assets/v13.model")
        if os.path.exists(model_dir) and os.path.exists(tokenizer_path):
            load_and_generate(model_dir, tokenizer_path=tokenizer_path)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "-v", "test_generate.py"])
