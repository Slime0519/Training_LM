import torch
import argparse, os

import numpy as np

from tqdm import tqdm

from KoGPT.KoDialog.Dataset import KoDialogueDataset
from KoGPT.KoDialogGPT2 import KoDialogGPT2
from transformers import PreTrainedTokenizerFast
from torch.utils.data import SequentialSampler, DataLoader
from torchnlp.samplers import BucketBatchSampler
from KoGPT.data_preprocessing.utils import collate_fn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=5, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--save_step", default=100, type=int)
parser.add_argument("--lr", default=5e-5)
parser.add_argument("--save_path", type=str)
parser.add_argument("--validation", default=False, type=bool)
parser.add_argument("--train_dataset", type=str)
parser.add_argument("--valid_dataset", type=str)
parser.add_argument("--savename", type=str)
train_data_path = "korean_dialog_summary/Training/label_kodialog_summary_train/personal_relationship.json"
valid_data_path = "korean_dialog_summary/Validation/label_kodialog_summary_valid/personal_relationship.json"
save_ckpt_path = "KoDialog_general.pth"

if __name__ =="__main__":
    args = parser.parse_args()
    n_epoch = args.epoch
    batch_size = args.batch_size
    save_step = args.save_step
    learning_rate = args.lr
    save_path = args.save_path
    do_valid = args.validation
    savename = args.savename
    train_data_path = args.train_dataset
    valid_data_path = args.valid_dataset

    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')

    train_dataset = KoDialogueDataset(tokenizer=tokenizer, datapath= train_data_path)
    train_sampler = SequentialSampler(train_dataset)
    train_batch_sampler = BucketBatchSampler(train_sampler, batch_size, True, sort_key=lambda r: len(train_dataset[r]))
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)

    if do_valid:
        valid_dataset = KoDialogueDataset(tokenizer=tokenizer, datapath=valid_data_path)
        valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = KoDialogGPT2()
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(n_epoch):
        count = 0
        losses = []
        valid_losses = []
        with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
            model.train()
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(device)


                outputs = model(data, labels = data)
                _, logits = outputs[:2]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels =data[...,1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
                    save_ckpt_path = os.path.join(save_path, savename+ f"_{epoch}_{count}_{np.mean(losses):.3f}.pth")
                    torch.save({
                        'epoch': epoch,
                        'train_no': count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, save_ckpt_path)
                count += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss.item():.3f} ({np.mean(losses):.3f})")

            if do_valid:
                model.eval()
                for i, data in enumerate(valid_loader):
                    data = data.to(device)
                    outputs = model(data, labels = data)
                    _, logits = outputs[:2]

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = data[..., 1:].contiguous()

                    valid_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    valid_losses.append(valid_loss.item())
                print(f"Validation loss : {np.mean(valid_losses):.3f}")

