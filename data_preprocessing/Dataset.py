import torch
import json
from data_preprocessing.utils import *
from transformers import PreTrainedTokenizerFast

from torch.utils.data import Dataset
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                            bos_token='</s>',
                                                            eos_token='</s>',
                                                            unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
DEFAULT_PADDING_INDEX = tokenizer.pad_token_id

class KoDialogueDataset(Dataset):
    def __init__(self, datapath = "korean_dialog_summary/Training/label_kodialog_summary_train/personal_relationship.json"):
        super(KoDialogueDataset, self).__init__()
        with open(datapath, "r", encoding='utf-8') as jsonfile:
            json_temp = json.load(jsonfile)
        dialogues = get_dialog(json_temp, num_participants=2, merge=True)

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                            bos_token='</s>',
                                                            eos_token='</s>',
                                                            unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
        bos_token_id = [self.tokenizer.bos_token_id]
        eos_token_id = [self.tokenizer.eos_token_id]
        pad_token_id = [self.tokenizer.pad_token_id]

        clips = get_twoturn_dialogues(dialogues)
        self.dialogdata = []

        for clip in clips:
            index_of_words = bos_token_id + tokenizer.encode(clip[0]) + eos_token_id + \
                             bos_token_id + tokenizer.encode(clip[1]) + eos_token_id
            if (len(index_of_words) > 100): continue
            self.dialogdata.append(index_of_words)

    def __getitem__(self, index):
        return  torch.Tensor(self.dialogdata[index])

    def __len__(self):
        return len(self.dialogdata)


def collate_fn(batch, train=True):
    """ list of tensors to a batch tensors """
    premise_batch = stack_and_pad_tensors([row for row in batch], padding_index=DEFAULT_PADDING_INDEX)

    return premise_batch

if __name__ == "__main__":
    dataset = KoDialogueDataset()
    from torch.utils.data import SequentialSampler, DataLoader
    from torchnlp.samplers import BucketBatchSampler

    batch_size = 2
    train_sampler = SequentialSampler(dataset)
    train_batch_sampler = BucketBatchSampler(train_sampler, batch_size, True, sort_key=lambda r: len(dataset[r]))
    train_dataloader = DataLoader(dataset, batch_sampler=train_batch_sampler,collate_fn=collate_fn)

    for batch_idx, batch in enumerate(train_dataloader):
        #print(batch)
        print("first : {}   second : {}".format(batch[0].shape, batch[1].shape))
        print([tokenizer.decode(np.array(element).astype(dtype=int)) for element in batch])
        if(batch_idx > 3):
            break


