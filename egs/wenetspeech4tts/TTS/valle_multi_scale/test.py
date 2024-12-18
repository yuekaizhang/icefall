from transformers import AutoConfig

config = AutoConfig.from_pretrained("/workspace/Qwen2.5-0.5B-Instruct")


print(config)
