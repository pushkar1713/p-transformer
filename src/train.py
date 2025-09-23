import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_trainer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else :
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("opus_books", f"{config["src_lang"]}-{config["tgt_lang"]}", split="train")

    tokenizer_src = get_or_build_trainer(config, ds_raw, config["src_lang"])
    tokenizer_tgt = get_or_build_trainer(config, ds_raw, config["tgt_lang"])

    train_ds_size = int(0.9 * len(ds_raw))
    validation_ds_size = len(ds_raw) - train_ds_size
    
    train_ds_raw, val_ds_raw = random_split(ds_raw, lengths=[train_ds_size, validation_ds_size])


