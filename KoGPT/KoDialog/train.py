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

data_path = "korean_dialog_summary/Training/label_kodialog_summary_train/personal_relationship.json"
save_ckpt_path = "KoDialog_general.pth"

if __name__ =="__main__":
    args = parser.parse_args()
    n_epoch = args.epoch
    batch_size = args.batch_size
    save_step = args.save_step
    learning_rate = args.lr
    save_path = args.save_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')

    dataset = KoDialogueDataset(tokenizer=tokenizer, datapath= data_path)
    train_sampler = SequentialSampler(dataset)
    train_batch_sampler = BucketBatchSampler(train_sampler, batch_size, True, sort_key=lambda r: len(dataset[r]))
    train_loader = DataLoader(dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)

    model = KoDialogGPT2()
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(n_epoch):
        count = 0
        with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
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
                    save_ckpt_path = os.path.join(save_path, f"KoDialog_general_{epoch}_{count}_{np.mean(losses):.3f}.pth")
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
