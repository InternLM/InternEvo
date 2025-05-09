from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-32B"
cache_dir = "/mnt/afs/huangting3/hf/qwen3-32B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    # torch_dtype="auto",
    # device_map="auto"
)

print(f"{model=}", flush=True)
