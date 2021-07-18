import pandas as pd
import math

def read_xlsx(datapath):
    xlsx = pd.read_excel(datapath)
    #print(xlsx.head())
    #print(xlsx.shape)
    return xlsx

def excel_to_list(exceldata):
    data = []
    num = exceldata.shape[0]
    for ind in range(num):
        tempdata =[]
        for ele in exceldata.iloc[ind,:]:
            if(isinstance(ele, float) and math.isnan(ele)):
                break
            tempdata.append(ele)
        data.append(tempdata)
    return data

def make_clips(data):
    clips = []
    for dialogue in data:
        dia_len = len(dialogue)
        ind =0
        while ind<dia_len-1:
            clip = []
            clip.append(dialogue[ind])
            ind+=1
            clip.append(dialogue[ind])
            clips.append(clip)
    return clips

if __name__ == "__main__":
    exceldata = read_xlsx("./twitter_dialogue_scenario.xlsx")
#print(exceldata.iloc[0,:])
    data = excel_to_list(exceldata)
    #print(data[0])
    clips = make_clips(data)
    print(clips[0])
    print(clips[10])
    print(len(clips))