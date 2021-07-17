import torch
import argparse

import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
from KoGPT.wellness_data import WellnessDialogDataset
from KoGPT.model import GPTmodel
from transformers import PreTrainedTokenizerFast

from torchnlp.samplers import BucketBatchSampler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=5, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--save_step", default=100, type=int)
parser.add_argument("--lr", default=5e-5)

data_path = "chatbotData .csv"
save_ckpt_path = "./kogpt2-wellnesee-auto-regressive.pth"
if __name__ =="__main__":
    args = parser.parse_args()
    n_epoch = args.epoch
    batch_size = args.batch_size
    save_step = args.save_step
    learning_rate = args.lr

    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')

    dataset = WellnessDialogDataset(tokenizer=tokenizer, filepath= data_path)
    train_loader = DataLoader(dataset, batch_size= batch_size, shuffle=True)

    model = GPTmodel()
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(n_epoch):
        count = 0
        with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                #print(data.shape)

                #data = torch.stack(data)
                #data = data.transpose(1,0)
                #data = data.transpose(1,0)
                data = data.to(device)

                outputs = model(data, labels = data)
                _, logits = outputs[:2]
               # print(logits.shape)
               # print(data.shape)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels =data[...,1:].contiguous()
               # print(shift_logits.shape)
               #print(shift_labels.shape)

                #print(shift_labels)
                #print(shift_logits)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if (count > 0 and count % save_step == 0) or (len(data) < batch_size):
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
