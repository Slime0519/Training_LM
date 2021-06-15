from time import time

import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer

torch.set_grad_enabled(False)

MODEL_NAME = 'bert-base-cased'

model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokens_pt = tokenizer("This is an input example", return_tensors = "pt")
for key, value in tokens_pt.items():
    print("{}:\n\t{}".format(key, value))


outputs = model(**tokens_pt)
last_hidden_state = outputs.last_hidden_state
pooler_output = outputs.pooler_output

print("Token-wise output : {}, Pooled output : {}".format(last_hidden_state.shape, pooler_output.shape))

single_seg_input = tokenizer("This is a sample input")

multi_seg_input = tokenizer("This is segment A", "This is segment B")

print("Single segment token (str): {}".format(tokenizer.convert_ids_to_tokens(single_seg_input['input_ids'])))
print("Single segment token (int): {}".format(single_seg_input['input_ids']))
print("Single segment type       : {}".format(single_seg_input['token_type_ids']))

print()
print("Multi segment token (str): {}".format(tokenizer.convert_ids_to_tokens(multi_seg_input['input_ids'])))
print("Multi segment token (int): {}".format(multi_seg_input['input_ids']))
print("Multi segment type       : {}".format(multi_seg_input['token_type_ids']))

tokens = tokenizer(
    ["This is a sample", "This is another longer sample text"],
    padding =True
)

print()
for i in range(2):
    print("Tokens (int)      : {}".format(tokens['input_ids'][i]))
    print("Tokens (str)      : {}".format([tokenizer.convert_ids_to_tokens(s) for s in tokens['input_ids'][i]]))
    print("Tokens (attn_mask): {}".format(tokens['attention_mask'][i]))
    print()

from transformers import TFBertModel, BertModel

model_tf = TFBertModel.from_pretrained(MODEL_NAME)
model_pt = BertModel.from_pretrained(MODEL_NAME)

input_tf = tokenizer("This is a sample input", return_tensors = 'tf')
input_pt = tokenizer("This is a sample input", return_tensors = 'pt')

output_tf, output_pt = model_tf(input_tf), model_pt(**input_pt)

for name in ["last_hidden_state", "pooler_output"]:
    print("{} differences: {:.5}".format(name, (output_tf[name].numpy() - output_pt[name].numpy()).sum()))

from transformers import DistilBertModel

bert_distil = DistilBertModel.from_pretrained('distilbert-base-cased')
input_pt =tokenizer("This is a sample input to demonstrate performance of distiled models especially inference time",
                    return_tensors='pt')
from time import time

s =time()
bert_distil(input_pt['input_ids'])
distil_delta = time()-s

s =time()
model_pt(input_pt['input_ids'])
origin_delta =time()-s
print("Times for distilbert : {}".format(distil_delta))
print("Times for original bert : {}".format(origin_delta))