import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from KoGPT.model import GPTmodel
from transformers import PreTrainedTokenizerFast
from KoGPT.wellness_data import WellnessDialogDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
datapath = "kogpt2-wellnesee-auto-regressive1.pth"

data_path = "chatbotData .csv"

def get_len(str):
    for i, index in enumerate(str[0]):
        if index == 3:
            return i

if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')

    checkpoint = torch.load(datapath, map_location=device)
    model = GPTmodel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    dataset = WellnessDialogDataset(tokenizer=tokenizer, filepath=data_path)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()

    count = 0
    output_size = 200

    for i, data in enumerate(train_loader):
        # print(data.shape)

        # data = torch.stack(data)
        # data = data.transpose(1,0)
        # data = data.transpose(1,0)
        data = data.to(device)

        outputs = model(data, labels=data)
        _, logits = outputs[:2]
        # print(logits.shape)
        # print(data.shape)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()

        shift_logits = logits[..., :, :].contiguous()
        shift_labels = data[..., :].contiguous()
        # print(shift_logits.shape)
        # print(shift_labels.shape)
        print(shift_logits.shape)
        converted_logits = torch.softmax(shift_logits, dim=2)
        converted_indicies = torch.argmax(converted_logits, dim=2)

        origin_length = get_len(shift_labels)

       # print(tokenizer.decode([9038]))
        print(converted_indicies[0, : origin_length])
        print(tokenizer.decode(converted_indicies[0, : origin_length]))
        print(shift_labels[0, :origin_length])
        print(tokenizer.decode(shift_labels[0, :origin_length]))
        break
        # print(shift_labels)
        # print(shift_logits)