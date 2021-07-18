import json
import numpy as np
import re
import torch
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                            bos_token='</s>',
                                                            eos_token='</s>',
                                                            unk_token='<unk>',
                                                            pad_token='<pad>', mask_token='<mask>')
DEFAULT_PADDING_INDEX = tokenizer.pad_token_id


def replace_mask(string : str):
    mask_re = re.compile('#@.{2,5}#')
    patterns = mask_re.finditer(string)
    #print(list(patterns))
    for pattern in patterns:
        #strindex = pattern.start(0)
        category = pattern.group()[2:-1]
        #print(category)
        if category == "이름":
            replace_target = "봇준명"
        elif category == "계정":
            replace_target = "내 계정"
        elif category == "신원":
            replace_target = "2018****"
        elif category == "전번":
            replace_target = "010-2518-****"
        elif category == "금융":
            replace_target = "우리은행 준명봇"
        elif category == "번호":
            replace_target = "번호"
        elif category == "주소":
            replace_target = "광주과기원 학생기숙사 B동"
        elif category == "소속":
            replace_target = "광주과학기술원"
        elif category == "이모티콘":
            replace_target = ""
        else:
            return -1

        string = string.replace(pattern.group(), replace_target)

    return string

def merge_utterance(first, second, mode = "attach"):
    if(first[-1] in "?!.,^~" or first[-1] in "이데서고로에"):
        string = first + " " + second
    elif(first[-1] in "든지네야듯"):
        string = first + ". " + second
    else:
        string = first + ", " + second

    return string

def get_dialog(jsondata, num_participants = None, merge = True, mode ="attach"):
    length = jsondata["numberOfItems"]
    dialogues = []

    for i in range(length):
        if(num_participants):
            if(jsondata['data'][i]['header']["dialogueInfo"]["numberOfParticipants"] > num_participants):
                continue
        utterance = []

        for order, utterance_obj in enumerate(jsondata['data'][i]['body']['dialogue']):

            if(merge and order is not 0 and utterance_obj["participantID"] == \
                    pre_utterance_obj["participantID"] and utterance_obj["turnID"] == pre_utterance_obj["turnID"]):
                utterance[-1] = merge_utterance(pre_utterance_obj['utterance'], utterance_obj['utterance'], mode = mode)
            else:
                utterance.append(utterance_obj['utterance'])
            pre_utterance_obj = utterance_obj
        dialogues.append(utterance)

    for i, element in enumerate(dialogues):
        for j,string in enumerate(element):
            dialogues[i][j] = replace_mask(string)

    return dialogues

def get_stat(utterances):
    total_dialog_num = 0
    total_uttr_len = 0
    len_utterance = []

    for element in utterances:
        total_dialog_num += len(element)
        for string in element:
            if string == -1:
                continue
            #total_uttr_len += len(string)
            len_utterance.append(len(string))
    #print("average len of dialogues : {}".format(total_dialog_num/len(utterances)))
    #print("average len of utterance in each dialogue : {}".format(total_uttr_len/total_dialog_num))

    len_utterance = np.array(len_utterance)
    first_quantile = np.percentile(len_utterance, 25)
    third_quantile = np.percentile(len_utterance, 75)
    avg = np.average(len_utterance)
    median = np.median(len_utterance)
    print("first quantile : {}, third quantile : {}".format(first_quantile, third_quantile))
    print("average : {} , median : {}".format(avg, median))

    return avg, median, first_quantile, third_quantile

    #return total_dialog_num/len(utterances), total_uttr_len/total_dialog_num

def get_twoturn_dialogues(dialogues):
    clips = []
    _, _, first_quantile, third_quantile = get_stat(dialogues)

    for dialogue in dialogues:
        index = 0
        while(index<len(dialogue)-1):
            clip = []

            clip.append(dialogue[index])
            index+=1
            clip.append(dialogue[index])
            index+=1

            if -1 in clip:
                continue
            if len(clip[0]) < first_quantile or len(clip[1]) < first_quantile \
                or len(clip[0]) >third_quantile or len(clip[1]) > third_quantile:
                continue
            clips.append(clip)
    return clips

#for batchsampler
def pad_tensor(tensor, length, padding_index):
    n_padding = length - tensor.shape[0]
    assert n_padding >= 0
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)

def stack_and_pad_tensors(batch, padding_index, dim=0):
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = torch.stack(padded, dim=dim).contiguous()
    for _ in range(dim):
        lengths = lengths.unsqueeze(0)

    return torch.Tensor(padded)

def collate_fn(batch, train=True):
    """ list of tensors to a batch tensors """
    premise_batch = stack_and_pad_tensors([row for row in batch], padding_index=DEFAULT_PADDING_INDEX)

    return premise_batch