from datasets import load_dataset

path = "/workspace/VoiceAssistant-400"
dataset = load_dataset(path)

snac_codec = dataset["train"][0]["answer_snac"]
codec_list = snac_codec.split("#")

for i in range(len(codec_list)):
    print(codec_list[i], i, len(codec_list[i]))
