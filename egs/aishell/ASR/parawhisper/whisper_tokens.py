
import random
import re
import whisper
from pathlib import Path

def load_chinese_chars(filename):
    chinese_chars = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            char = line.split()[0]
            chinese_chars.append(char)
    return chinese_chars


def save_new_tokens(chinese_chars, tokenizer, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for i in range(len(chinese_chars)):
            tokens = tokenizer.encode(chinese_chars[i])
            tokens_str = " ".join([str(token) for token in tokens])
            file.write(f"{chinese_chars[i]} {tokens_str}\n")


def load_new_tokens_dict_list(filename, tokenizer):
    chinese_chars = {}
    index_set = set([tokenizer.eot, tokenizer.sot])
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            char = line.split()[0]
            # every value is a list of int
            tokens = [int(token) for token in line.split()[1:]]
            index_set.update(tokens)
            chinese_chars[char] = tokens
    # from 0 - 50364
    suppress_index_list = []
    for i in range(50365):
        if i not in index_set:
            suppress_index_list.append(i)
    index_list = list(index_set)
    index_list.sort()
    return chinese_chars, index_list, suppress_index_list

def encode_with_dict(text_str, dict):
    encoded = []
    for char in text_str:
        if char in dict:
            encoded += dict[char]
    return encoded
    


if __name__ == "__main__":
    # token_path = "./aishell_tokens_wenet.txt"
    # chinese_chars = load_chinese_chars(token_path)
    # print(chinese_chars)
    tokenizer = whisper.tokenizer.get_tokenizer(
        True,
        num_languages=100,
        language="zh",
        task="transcribe",
    )
    new_token_path = "./aishell_tokens_whisper.txt"
    # save_new_tokens(chinese_chars, tokenizer, new_token_path)


    aishell_dict, aishell_index_set, supress_index_list = load_new_tokens_dict_list(new_token_path, tokenizer)
    print(aishell_dict, aishell_index_set, supress_index_list)
    print(len(aishell_dict), len(aishell_index_set), len(supress_index_list))

    test_str = "我是中国人"
    test_str = "今天天气不错,我们一起去爬山吧"
    encoded = encode_with_dict(test_str, aishell_dict)
    print(encoded)
    # decoded = tokenizer.decode(encoded)
    # print(decoded)