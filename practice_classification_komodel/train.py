import numpy as np
import pandas as pd
import transformers, torch, os

#from practice_classification_komodel.dataloader import CustomDataset
from dataloader import CustomDataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertModel
#from practice_classification_komodel.tokenization_kobert import KoBertTokenizer
from tokenization_kobert import KoBertTokenizer
import torch.nn as nn

TRAINSET_PATH = "../nsmc-master/ratings_train.txt"
TESTSET_PATH = "../nsmc-master/ratings_test.txt"

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-05

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class DistillBERTClass(nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('monologg/distilkobert')
        self.l2 = nn.Dropout(0.3)
        self.l3 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask = mask)
        output = output[0][:,0]
        output2 = self.l2(output)
        output =self.l3(output2)
        return output


if __name__ == "__main__":

    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    print("--------load trainset--------")
    training_set = CustomDataset(path = TRAINSET_PATH, tokenizer = tokenizer, max_len = MAX_LEN)
    print("--------load trainset--------")
    test_set = CustomDataset(path = TESTSET_PATH, tokenizer = tokenizer, max_len = MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(test_set, **test_params)

    model = DistillBERTClass().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr = LEARNING_RATE, eps=1e-06)


    def train(epoch):
        model.train()
        for i, data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            #outputs = torch.squeeze(outputs)
            loss = criterion(outputs, targets)
            if i % 5000 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                torch.save(model.state_dict(), "trained_classifier_{}_{}_{:.4f}.pth".format(epoch, i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if "trained_classifier_8_15000_0.0299.pth" in os.listdir("./"):
        model.load_state_dict(torch.load("trained_classifier_8_15000_0.0299.pth"))
    else:
        for epoch in range(EPOCHS):
            train(epoch)


    def validation(epoch):
        model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    from sklearn import metrics
    for epoch in range(EPOCHS):
        outputs, targets = validation(epoch)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
      #  f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
       # f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
      #  print(f"F1 Score (Micro) = {f1_score_micro}")
      #  print(f"F1 Score (Macro) = {f1_score_macro}")
