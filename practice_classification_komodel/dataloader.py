from practice_classification_komodel.tokenization_kobert import KoBertTokenizer
#from tokenization_kobert import KoBertTokenizer

from torch.utils.data import Dataset, DataLoader
import torch

def parsing_data(path):
    dataset = {'id': [], 'body': [], 'label': []}
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        line = f.readline()
        while line:
            id, body, label = line.split('\t')
            dataset['id'].append(id)
            dataset['body'].append(body)
            dataset['label'].append(int(label[0]))
            line = f.readline()

    print("load {} sequences".format(len(dataset['id'])))
    return dataset


class CustomDataset(Dataset):
    def __init__(self, path, tokenizer : KoBertTokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = parsing_data(path)
        self.body = self.data['body']
        self.targets = self.data['label']
        self.max_len = max_len

    def __getitem__(self, index):
        body = self.body[index]
        inputs = self.tokenizer.encode_plus(
            body, None, add_special_tokens=True,
            max_length=self.max_len, padding='max_length',
            return_token_type_ids=True
        )
        target = [0,0]
        target[self.targets[index]] = 1
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
       # print("ids : {} / mask : {} / token_type_ids : {}".format(len(ids), len(mask), len(token_type_ids)))
        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
            'targets' : torch.tensor(target, dtype=torch.float)
        }

    def __len__(self):
        return len(self.targets)


if __name__ == "__main__":

    data_dict = parsing_data('../nsmc-master/ratings_train.txt')
    keys = list(data_dict.keys())
    """
    for i in range(len(data_dict[keys[0]])):
        print(data_dict[keys[0]][i], data_dict[keys[1]][i], data_dict[keys[2]][i])
    """



    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    print(tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. [SEP]"))
    print(tokenizer.encode_plus("[CLS] 한국어 모델을 공유합니다. [SEP]"))

    inputs = tokenizer.encode_plus(
        data_dict['body'][10], None, add_special_tokens=True,
        max_length=200, padding=True,
        return_token_type_ids=True
    )

    print(inputs)
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    print(tokenizer.convert_ids_to_tokens(ids))
    #for ele in ids:

