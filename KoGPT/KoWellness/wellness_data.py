import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

def parse_Chatbotdata(filepath):
    questions =[]
    answers = []
    labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()
        line = f.readline()
        while line:
            parsed_line = line.split(',')
            print(parsed_line)
            question = parsed_line[0]
            answer = parsed_line[1]
            label = parsed_line[2]
            questions.append(question)
            answers.append(answer)
            labels.append(label)
            line = f.readline().split('\n')[0]

    return {'question' : questions, 'answer' : answers, 'label' : labels}

class WellnessDialogDataset(Dataset):
    def __init__(self, tokenizer : PreTrainedTokenizerFast, filepath="ChatbotData .csv", n_ctx=1024):
        super(WellnessDialogDataset, self).__init__()
        self.tokenizer = tokenizer

        bos_token_id = [self.tokenizer.bos_token_id]
        eos_token_id = [self.tokenizer.eos_token_id]
        pad_token_id = [self.tokenizer.pad_token_id]

        data = parse_Chatbotdata(filepath)

        self.question = []
        self.answer = []
        self.data = []
        for i in range(len(data["question"])):
            index_of_words = bos_token_id + self.tokenizer.encode(data["question"][i]) + eos_token_id \
            + bos_token_id + self.tokenizer.encode(data["answer"][i]) + eos_token_id

            pad_token_len = n_ctx -len(index_of_words)
            index_of_words +=pad_token_id*pad_token_len
            self.data.append(index_of_words)

    def __getitem__(self, index):
        #Q_token = self.tokenizer.encode(self.question[index])
        #A_token = self.tokenizer.encode(self.answer[index])
        return torch.tensor(self.data[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    parseddict = parse_Chatbotdata("ChatbotData .csv")
    print(len(parseddict['question']))

    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                       bos_token='</s>',
                                                       eos_token='</s>',
                                                       unk_token='<unk>',
                                                       pad_token='<pad>', mask_token='<mask>')

    print(tokenizer.encode("안녕하세요. 반가워요", add_special_tokens=True))
    print(tokenizer.tokenize("안녕하세요. 반가워요", add_special_tokens=True))

