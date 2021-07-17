import json
from data_preprocessing.utils import *
from transformers import PreTrainedTokenizerFast

if __name__ ==  "__main__":
    with open("korean_dialog_summary/Training/label_kodialog_summary_train/personal_relationship.json", "r", encoding='utf-8') as jsonfile:
        json_temp = json.load(jsonfile)

    utterances = get_dialog(json_temp, num_participants=2, merge=True)
    avg_len, avg_char = get_stat(utterances)
    for i in range(1,len(utterances), 10000):
        print(utterances[i])
    print("total number of dialogues : {}".format(len(utterances)))
    print("average length of dialogues : {}".format(avg_len))
    print("average length of utterance in each dialogue : {}".format(avg_char))

    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')
    bos_token_id = [tokenizer.bos_token_id]
    eos_token_id = [tokenizer.eos_token_id]
    pad_token_id = [tokenizer.pad_token_id]

    data = []

    clips = get_twoturn_dialogues(utterances)
    print(clips[0])
    """
    for i in range(len(data["question"])):
        index_of_words = bos_token_id + self.tokenizer.encode(data["question"][i]) + eos_token_id \
                         + bos_token_id + self.tokenizer.encode(data["answer"][i]) + eos_token_id

        pad_token_len = n_ctx - len(index_of_words)
        index_of_words += pad_token_id * pad_token_len
        self.data.append(index_of_words)
    """
    dialogdata = []
    tot_twostage = 0
    num_twostage = 0
    len_clips = []
    for clip in clips:
        index_of_words = bos_token_id + tokenizer.encode(clip[0]) + eos_token_id + \
            bos_token_id + tokenizer.encode(clip[1]) + eos_token_id
        if(len(index_of_words)>100): continue

        dialogdata.append(index_of_words)
        #if(len(index_of_words) > 100):
        len_clips.append(len(index_of_words))
        tot_twostage+=len(index_of_words)
        num_twostage+=1
        #print(index_of_words)
        #print(tokenizer.decode(index_of_words))
    print(tot_twostage / num_twostage)
    print("total numbers of clips : {}".format(num_twostage))
    print("average length of encoded seq : {}".format(tot_twostage / num_twostage))

    import matplotlib.pyplot as plt
    plt.hist(len_clips, bins=50)
    plt.show()
