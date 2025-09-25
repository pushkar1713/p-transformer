import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from pathlib import Path
from config import get_config, get_weights_file_path
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset, casual_mask

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

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config["src_lang"], config["tgt_lang"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len, src_seq_len=config["seq_len"], tgt_seq_len=config["seq_len"], d_model=config["d_model"])
    return model


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device :", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    Path(f"{config["datasource"]}_{config["model_folder"]}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, vocab_src_len=tokenizer_src.get_vocab_size(), vocab_tgt_len=tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optim = torch.optim.Adam(model.parameters(), lr = config["lr"], eps=10e-9)

    initial_epoch = 0
    global_step = 0
    preload = config["preload"]

    if(preload):
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"preloading model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optim.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"processing_epoch : {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"]
            decoder_input = batch["decoder_input"]
            decoder_mask = batch["decoder_mask"]
            encoder_mask = batch["encoder_mask"]

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_input, encoder_mask, decoder_input, decoder_mask)

            proj_output = model.project(decoder_output)

            label = batch["label"].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optim.step()
            optim.zero_grad(set_to_none=True)

            global_step += 1

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)


            

