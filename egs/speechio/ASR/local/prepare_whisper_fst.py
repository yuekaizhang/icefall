
import whisper
import hanzidentifier
from typing import Dict, List, Union
import k2
import torch

# k2.shortest_path(lattice, use_double_scores=use_double_scores)

def write_mapping(filename: str, sym2id: Dict[str|int, List[int]]) -> None:
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, ids in sym2id.items():
            ids2str = " ".join([str(i) for i in ids])
            f.write(f"{sym} {ids2str}\n")

def load_words_dict(filename, whisper_tokenizer):
    word2wordid, wordid2word, word2tokenid, wordid2tokenid = {}, {}, {}, {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, word_id = line.split()
            word_id = int(word_id)
            word2wordid[word] = word_id
            wordid2word[word_id] = word
            token_ids = whisper_tokenizer.encode(word)
            word2tokenid[word] = token_ids
            wordid2tokenid[word_id] = token_ids
            print(word, word_id, token_ids)
    return word2wordid, wordid2word, word2tokenid, wordid2tokenid

def load_lexicon(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        wordid2word = {}
        for line in file:
            # line's first word is the word, the rest are the tokens
            # 6 32 8176
            word_id, token_ids = line.split()[0], line.split()[1:]
            token_ids = [int(token_id) for token_id in token_ids]
            wordid2word[word_id] = token_ids
            print(word_id, token_ids)
            
    return wordid2word


def lexicon_to_fst_no_sil(
    wordid2tokenid: Dict[int, List[int]],
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    # word_id = 0 is reserved for <eps>
    eps = 0

    for word_id, token_ids in wordid2tokenid.items():
        assert len(token_ids) > 0, f"{word_id} is not in the wordid2tokenid dict"
        cur_state = loop_state

        for i in range(len(token_ids) - 1):
            w = word_id if i == 0 else eps
            arcs.append([cur_state, next_state, token_ids[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(token_ids) - 1
        w = word_id if i == 0 else eps
        arcs.append([cur_state, loop_state, token_ids[i], w, 0])


    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa

def get_texts(
    best_paths: k2.Fsa, return_ragged: bool = False
) -> Union[List[List[int]], k2.RaggedTensor]:
    """Extract the texts (as word IDs) from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2
    if return_ragged:
        return aux_labels
    else:
        return aux_labels.tolist()

if __name__ == "__main__":
    whisper_tokenizer = whisper.tokenizer.get_tokenizer(
        True,
        num_languages=100,
        language="zh",
        task="transcribe",
    )

    # mytest_str = "<|startofprev|> Nvidia<|startoftranscript|><|en|><|transcribe|><|endoftranscript|><|unk|>"
    # mytest_str = "Nvidia"
    # encoding = enc.encode(mytest_str)
    # mystr = enc.decode([50361, 45, 43021, 50258, 50259, 50359])
    # mystr2 = enc.decode([50361, 46284, 50258, 50259, 50359])
    # #print(encoding, mystr, mystr2)
    # # print(
    # #     enc.encode("<|startoftranscript|>",
    # #                allowed_special=enc.special_tokens_set)[0])
    # # print(
    # #     enc.encode("<|endoftext|>",
    # #                allowed_special=enc.special_tokens_set)[0])
    # my_zh_str = "好好学习"
    # encoding = enc.encode(my_zh_str)
    # decoding = enc.decode(encoding)
    # print(encoding, decoding)

    # for i in range(50399):
    #     #print(i, enc.decode([i]))
    #     symbol = enc.decode([i])
    #     # check if the symbol in the range of Unicode Chinese wordacters
    #     if '\u4e00' <= symbol <= '\u9fff':

    #         if hanzidentifier.identify(symbol) is hanzidentifier.SIMPLIFIED or hanzidentifier.identify(symbol) is hanzidentifier.BOTH:
    #             print(i, symbol)
    #         else:
    #             print("================", symbol)
    #     else:
    #         pass

    # word2wordid, wordid2word, word2tokenid, wordid2tokenid = load_words_dict("./words.txt", whisper_tokenizer)
    # # _, wordid2word, _, wordid2tokenid = load_words_dict("./words.txt", whisper_tokenizer)
    # write_mapping("./lexicon.txt", word2tokenid)
    # write_mapping("./lexicon_ids.txt", wordid2tokenid)

    wordid2tokenid = load_lexicon("./lexicon_ids.txt")

    L = lexicon_to_fst_no_sil(wordid2tokenid)
    # L.write(f"{lang_dir}/HL.fst")
    # L.write("./L.fst")
    torch.save(L.as_dict(), "L.pt")
