#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from pathlib import Path
import re
import jieba
import random


# In[2]:


def choice_rate(i):
    if random.randint(1,100) < i:
        return True
    return False


# In[ ]:





# ## 准备词典

# In[3]:


# !python tools/prep_dict.py lexicon.txt
# !cd gmm && rm -rf data/dict/* && bash local/prepare_dict.sh ../lexicon.txt data/dict


# In[4]:


# !rm gmm/data/dict/*
# !cp lexicon.txt silence_phones.txt nonsilence_phones.txt extra_questions.txt optional_silence.txt gmm/data/dict
# !rm silence_phones.txt nonsilence_phones.txt extra_questions.txt optional_silence.txt


# ## SLR85生成wav.scp

# In[5]:


SLR85_dir = "/home1/meichaoyang/dataset/SLR85/dev/SPEECHDATA/wav"
SLR85_wav_scp = '/home1/meichaoyang/dataset/data_aishell2/wake_up/SLR_wav.scp'
SLR85_trans = '/home1/meichaoyang/dataset/data_aishell2/wake_up/SLR_trans.scp'
SLR_wav_list=list(Path(SLR85_dir).rglob("*.wav"))


# In[6]:


SLR_wav_list.sort(key=lambda Path: os.path.basename(os.path.splitext((str(Path)))[0]))


# In[7]:


len(SLR_wav_list)


# In[8]:


with open(SLR85_wav_scp, 'w') as f:
    for path_item in SLR_wav_list:
        wav_path = str(path_item)
        f.write(os.path.basename(os.path.splitext((wav_path))[0])+'\t'+wav_path+'\n')


# In[9]:


with open(SLR85_trans, 'w') as f:
    for path_item in SLR_wav_list:
        wav_path = str(path_item)
        f.write(os.path.basename(os.path.splitext((wav_path))[0])+'\t'+"你好米雅"+'\n')


# ### 并筛选一部分放入SLR_wav.scp.shuff文件中

# In[10]:


# !bash tools/random_line.sh SLR_wav.scp 5 SLR_wav.scp.shuff


# > 并筛选一部分放入SLR_wav.scp.shuff文件中

# ## Aishell2数据处理

# ### 生成Aishell2的原始wav.scp

# In[11]:


aishell_dir = "/home1/meichaoyang/dataset/data_aishell2/data_aishell/wav"
scp_path="/home1/meichaoyang/dataset/data_aishell2/wake_up/Aishell_wav_raw.scp"


# In[12]:


aishell_wav_list=list(Path(aishell_dir).rglob("*.wav"))
aishell_wav_list.sort(key=lambda Path: os.path.basename(os.path.splitext((str(Path)))[0]))
aishell_scp_raw = {}
with open(scp_path, 'w') as f:
    for path_item in aishell_wav_list:
        wav_path = str(path_item)
        aishell_scp_raw[os.path.basename(os.path.splitext((wav_path))[0])] = wav_path
        f.write(os.path.basename(os.path.splitext((wav_path))[0])+'\t'+wav_path+'\n')


# ### 对train满足条件的标注进行筛选

# In[13]:


cidian="/home1/meichaoyang/kaldi/egs/wakeup_words1/corpus/res/lexicon.txt"
lexicon_map={}


# In[14]:


with open(cidian, "r") as f:
    for line in f:
        data = line.split()
        lexicon_map[data[0]] = " ".join(data[1:])


# In[15]:


# url = '/C0001/IC0001W0001.wav' # 需要拆分的字符串
# result = re.split(r'[/|.]' , url) # 以pattern的值 分割字符串
# print(result)


# In[ ]:





# In[16]:


aishell_2_corp_raw = "/home1/meichaoyang/dataset/data_aishell2/data_aishell/wav/trans.txt"
aishell_scp = "/home1/meichaoyang/dataset/data_aishell2/wake_up/aishell_wav.scp"
aishell_corp = "/home1/meichaoyang/dataset/data_aishell2/wake_up/aishell_trans.txt"


# In[17]:


pattern_utt = re.compile(r'/.*\.')
pattern_Eng = re.compile(u'[a-zA-Z\n]')
corp_map_train = {}
corp_map_test = {}
corp_map_dev = {}
with open(aishell_2_corp_raw, "r") as f:
    for line in f:
        choiced = False
        data = line.split()
        if len(data[1]) > 30 or pattern_Eng.search(data[1]) != None: ##删除小于10和非英文标注
            continue
        for word in lexicon_map.keys():
            if "高雅" in data[1]:
                print(data)
            if word in data[1]:
                corp_map_train[re.split(r'[/|.]' , data[0])[-2]] = data[1]
                choiced = True
                break

        if choiced:
            continue
        elif choice_rate(10):
            corp_map_train[re.split(r'[/|.]' , data[0])[-2]] = data[1]
        elif choice_rate(80):
            corp_map_test[re.split(r'[/|.]' , data[0])[-2]] = data[1]
        elif choice_rate(75):
            corp_map_dev[re.split(r'[/|.]' , data[0])[-2]] = data[1]


# In[18]:


corp_map_train["IC0124W0027"]


# ### 抽取SLR中的部分数据并入aishell2的train、test、dev

# In[19]:


SLR_wav_scp = {}
SLR_wav_corp = {}


# In[20]:


with open(SLR85_wav_scp, "r") as f:
    for line in f:
        data = line.split()
        SLR_wav_scp[data[0]] = data[1]
        SLR_wav_corp[data[0]] = "你好米雅"


# In[21]:


SLR_train_key = random.sample(list(SLR_wav_scp.keys()), 10000)
SLR_test_key = random.sample(list(SLR_wav_scp.keys()), 1000)
SLR_dev_key = random.sample(list(SLR_wav_scp.keys()), 1000)


# In[22]:


SLR_wav_scp_tran=dict([(key, SLR_wav_scp[key]) for key in SLR_train_key])
SLR_wav_corp_tran=dict([(key, SLR_wav_corp[key]) for key in SLR_train_key])


# In[23]:


SLR_wav_scp_test=dict([(key, SLR_wav_scp[key]) for key in SLR_test_key])
SLR_wav_corp_test=dict([(key, SLR_wav_corp[key]) for key in SLR_test_key])


# In[24]:


SLR_wav_scp_dev=dict([(key, SLR_wav_scp[key]) for key in SLR_dev_key])
SLR_wav_corp_dev=dict([(key, SLR_wav_corp[key]) for key in SLR_dev_key])


# #### 训练集

# In[25]:


aishell_scp_train = {}
for i in sorted(aishell_scp_raw):
    if i not in corp_map_train.keys():
        continue
    aishell_scp_train[i] = aishell_scp_raw[i]


# In[26]:


wav_scp_train = {**SLR_wav_scp_tran,**aishell_scp_train}
corpus_train = {**SLR_wav_corp_tran, **corp_map_train}


# In[ ]:





# In[27]:


with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/data/wav.scp', 'w') as f:
    for i in sorted(wav_scp_train):
        f.write(i+'\t'+wav_scp_train[i]+'\n')
        
with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/data/trans.txt', 'w') as f:
    for i in sorted(corpus_train):
        f.write(i+'\t'+corpus_train[i]+'\n')


# #### 测试集

# In[28]:


aishell_scp_test = dict([(key,aishell_scp_raw[key])for key in sorted(corp_map_test.keys())])


# In[29]:


wav_scp_test = {**SLR_wav_scp_test,**aishell_scp_test}
corpus_test = {**SLR_wav_corp_test, **corp_map_test}


# In[30]:


with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/test/wav.scp', 'w') as f:
    for i in sorted(wav_scp_test):
        f.write(i+'\t'+wav_scp_test[i]+'\n')
        
with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/test/trans.txt', 'w') as f:
    for i in sorted(corpus_test):
        f.write(i+'\t'+corpus_test[i]+'\n')


# #### 开发集

# In[31]:


aishell_scp_dev = dict([(key,aishell_scp_raw[key])for key in sorted(corp_map_test.keys())])


# In[32]:


wav_scp_dev = {**SLR_wav_scp_dev,**aishell_scp_dev}
corpus_dev = {**SLR_wav_corp_dev, **corp_map_dev}


# In[33]:


with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/dev/wav.scp', 'w') as f:
    for i in sorted(wav_scp_dev):
        f.write(i+'\t'+wav_scp_dev[i]+'\n')
        
with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/dev/trans.txt', 'w') as f:
    for i in sorted(corpus_dev):
        f.write(i+'\t'+corpus_dev[i]+'\n')


# In[ ]:





# In[ ]:




