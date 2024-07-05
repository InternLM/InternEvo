import os

import pytest
from sentencepiece import SentencePieceProcessor

import internlm  # noqa: E402
from internlm.apis.inference import SequenceGenerator, batch_tokenize
from internlm.checkpoint import CheckpointManager  # noqa: E402
from internlm.core.context import global_context as gpc  # noqa: E402
from internlm.core.trainer import TrainState, Trainer  # noqa: E402
from internlm.data import build_train_loader_with_data_type  # noqa: E402
from internlm.initialize import initialize_distributed_env  # noqa: E402
from internlm.model.losses import FlashGPTLMLoss  # noqa: E402
from internlm.train import (  # noqa: E402
    get_scheduler_hooks,
    initialize_model,
    initialize_optimizer,
    initialize_parallel_communicator,
)


def setup_generator(config, tokenizer):
    initialize_distributed_env(config=config)

    model = initialize_model()
    isp_communicator = initialize_parallel_communicator(model)

    criterion = FlashGPTLMLoss()

    # initialize the train data loader
    train_dl, _ = build_train_loader_with_data_type()

    # initialize and resume train state
    train_state = TrainState(gpc.config, train_dl.batch_sampler)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

    ckpt_manager = CheckpointManager(
        ckpt_config=gpc.config.ckpt,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dl=train_dl,
        model_config=gpc.config.model,
        feishu_address=gpc.config.monitor.alert.feishu_alert_address,
    )
    ckpt_manager.try_resume_training(train_state)

    # initialize trainer
    engine, scheduler = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=get_scheduler_hooks(None, optimizer, isp_communicator),
    )
    trainer = Trainer(engine, scheduler)

    trainer.schedule.data_process_func = None

    if isinstance(tokenizer, SentencePieceProcessor):
        eos_token_id = tokenizer.eos_id()
        pad_token_id = tokenizer.eos_id()
        bos_token_id = tokenizer.bos_id()
    else:
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        bos_token_id = tokenizer.bos_token_id

    sequenece_generator = SequenceGenerator(
        decoder=trainer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        additional_eos_token_list=None,
    )

    return sequenece_generator


def do_generate(config, tokenizer_path, prompt):
    tokenizer = SentencePieceProcessor(tokenizer_path)  # pylint: disable=E1121

    sequenece_generator = setup_generator(config, tokenizer)
    input_ids = batch_tokenize(prompt, tokenizer, pad_token_id=tokenizer.bos_id()).cuda()

    generate_kwargs = {}
    output_ids = sequenece_generator.generate(
        input_ids,
        num_return_sequences=generate_kwargs.get("num_return_sequences", 1),
        max_length=generate_kwargs.get("max_length", 100),
        num_beams=generate_kwargs.get("num_beams", 1),
        do_sample=generate_kwargs.get("do_sample", True),
        temperature=generate_kwargs.get("temperature", 1.0),
        top_k=generate_kwargs.get("top_k", 50),
        top_p=generate_kwargs.get("top_p", 1.0),
        repetition_penalty=generate_kwargs.get("repetition_penalty", 1),
        length_penalty=generate_kwargs.get("repetition_penalty", 1.0),
    )
    output_tokens = output_ids.tolist()
    all_output_str = []
    for b in range(len(output_tokens)):
        for sent_idx in range(len(output_tokens[b])):
            cur_output_tokens = output_tokens[b][sent_idx]
            cur_sent = tokenizer.decode(cur_output_tokens)
            all_output_str.append(cur_sent)
    return all_output_str


def test_luyou_2B_generate():
    prompt = [
        "user\nHow can I keep flys away from my house\nassistant\n",
        "user\nHow can I keep flys away from my house\nassistant\nThe best way is to keep your house clean, "
        "and sweep away from where your meals are prepared, since flys tend to seek out food particles.\n"
        "user\nAny other advice?\nassistant\n",
    ]

    base_model_dir = os.environ.get("qa_data")
    if base_model_dir is not None:
        config = os.path.join(base_model_dir, "model_configs/Luyou_1B_merged.py")

        tokenizer_path = os.path.join(base_model_dir, "InternLM_CI_assets/v13.model")
        if os.path.exists(config) and os.path.exists(tokenizer_path):
            all_output_str = do_generate(config, tokenizer_path, prompt)
            print("out_str:\n", all_output_str)
            assert (
                all_output_str[0][len(prompt[0]) :]
                == "There are several things you can do to keep flies away from your house:\n\n\
1. Keep your home clean: Flies are attracted to food and dirty surfaces. Make sure that your home \
is well-maintained and"
            )
            assert (
                all_output_str[1][len(prompt[1]) :]
                == "You can also use plastic baggies to keep any food that is dropped on your porch, \
patio, or windowsill from attracting flies.\n[UNUSED_TOKEN_145]\nNo[UNUSED_TOKEN_145]\nYou could also \
use scented candles or diffusers"
            )


@pytest.mark.skip("requires 2 gpu")
def test_internlm2_pp2_generate():
    prompt = [
        "user\nHow can I keep flys away from my house\nassistant\n",
        "user\nHow can I keep flys away from my house\nassistant\nThe best way is to keep your house clean, "
        "and sweep away from where your meals are prepared, since flys tend to seek out food particles.\n"
        "user\nAny other advice?\nassistant\n",
    ]

    base_model_dir = os.environ.get("qa_data")
    if base_model_dir is not None:
        config = os.path.join(base_model_dir, "model_configs/Luyou_1B_PP2.py")
        tokenizer_path = os.path.join(base_model_dir, "InternLM_CI_assets/v13.model")
        if os.path.exists(config) and os.path.exists(tokenizer_path):
            all_output_str = do_generate(config, tokenizer_path, prompt)
            print("out_str:\n", all_output_str)
            assert (
                all_output_str[0][len(prompt[0]) :]
                == "There are several things you can do to keep flies away \
from your house:\n\n1. Keep your home clean: Flies are attracted to food and dirty surfaces. Make sure that your \
home is well-maintained and"
            )
            assert (
                all_output_str[1][len(prompt[1]) :]
                == "You can also use plastic baggies to keep any food that is dropped on your porch, patio, or \
windowsill from attracting flies.\n[UNUSED_TOKEN_145]\nNo[UNUSED_TOKEN_145]\nYou could also use scented candles \
or diffusers"
            )


@pytest.mark.skip("reduce timecost")
def test_internlm2_7B_tp2():
    prompt = [
        "user\nHow can I keep flys away from my house\nassistant\n",
        "user\nHow can I keep flys away from my house\nassistant\nThe best way is to keep your house clean, "
        "and sweep away from where your meals are prepared, since flys tend to seek out food particles.\n"
        "user\nAny other advice?\nassistant\n",
    ]

    base_model_dir = os.environ.get("qa_data")
    if base_model_dir is not None:
        config = os.path.join(base_model_dir, "model_configs/7B_internlm2.py")

        tokenizer_path = os.path.join(base_model_dir, "InternLM_CI_assets/v13.model")
        if os.path.exists(config) and os.path.exists(tokenizer_path):
            all_output_str = do_generate(config, tokenizer_path, prompt)
            print("out_str:\n", all_output_str)
            assert (
                all_output_str[0][len(prompt[0]) :]
                == "You can use natural repellants like lavender, vanilla or lemongrass essential oils. \
Or you can spray essential oil in a spray bottle around doors and windows. Also, using a white vinegar and"
            )
            assert (
                all_output_str[1][len(prompt[1]) :]
                == "You may want to consider using fly trapped to keep or get rid of the flys if need be. \
Also wearing indoor protective clothing may be advised as well since they can be dangerous"
            )


if __name__ == "__main__":
    pytest.main(["-s", "-q", "-v", "test_trainer_generate.py"])
