#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from pathlib import Path
import re
import jieba
import random


# ## 准备词典

# In[2]:


# !python tools/prep_dict.py lexicon.txt
# !cd gmm && rm -rf data/dict/* && bash local/prepare_dict.sh ../lexicon.txt data/dict


# In[3]:


# !rm gmm/data/dict/*
# !cp lexicon.txt silence_phones.txt nonsilence_phones.txt extra_questions.txt optional_silence.txt gmm/data/dict
# !rm silence_phones.txt nonsilence_phones.txt extra_questions.txt optional_silence.txt


# ## SLR85生成wav.scp

# In[4]:


SLR85_dir = "/home1/meichaoyang/dataset/SLR85/dev/SPEECHDATA/wav"
SLR85_wav_scp = '/home1/meichaoyang/dataset/data_aishell2/wake_up/SLR_wav.scp'
SLR85_trans = '/home1/meichaoyang/dataset/data_aishell2/wake_up/SLR_trans.scp'
SLR_wav_list=list(Path(SLR85_dir).rglob("*.wav"))


# In[5]:


SLR_wav_list.sort(key=lambda Path: os.path.basename(os.path.splitext((str(Path)))[0]))


# In[6]:


len(SLR_wav_list)


# In[7]:


with open(SLR85_wav_scp, 'w') as f:
    for path_item in SLR_wav_list:
        wav_path = str(path_item)
        f.write(os.path.basename(os.path.splitext((wav_path))[0])+'\t'+wav_path+'\n')


# In[8]:


with open(SLR85_trans, 'w') as f:
    for path_item in SLR_wav_list:
        wav_path = str(path_item)
        f.write(os.path.basename(os.path.splitext((wav_path))[0])+'\t'+"你好米雅"+'\n')


# ### 并筛选一部分放入SLR_wav.scp.shuff文件中

# In[9]:


# !bash tools/random_line.sh SLR_wav.scp 5 SLR_wav.scp.shuff


# > 并筛选一部分放入SLR_wav.scp.shuff文件中

# ## Aishell2数据处理

# ### 生成Aishell2的原始wav.scp

# In[10]:


aishell_dir = "/home1/meichaoyang/dataset/data_aishell2/wav"
scp_path="/home1/meichaoyang/dataset/data_aishell2/wake_up/Aishell_wav_raw.scp"


# In[11]:


aishell_wav_list=list(Path(aishell_dir).rglob("*.wav"))
aishell_wav_list.sort(key=lambda Path: os.path.basename(os.path.splitext((str(Path)))[0]))
aishell_scp_raw = {}
with open(scp_path, 'w') as f:
    for path_item in aishell_wav_list:
        wav_path = str(path_item)
        aishell_scp_raw[os.path.basename(os.path.splitext((wav_path))[0])] = wav_path
        f.write(os.path.basename(os.path.splitext((wav_path))[0])+'\t'+wav_path+'\n')


# ### 对train满足条件的标注进行筛选

# In[12]:


aishell_2_corp_raw = "/home1/meichaoyang/dataset/data_aishell2/wav/trans.txt"
aishell_scp = "/home1/meichaoyang/dataset/data_aishell2/wake_up/aishell_wav.scp"
aishell_corp = "/home1/meichaoyang/dataset/data_aishell2/wake_up/aishell_trans.txt"


# In[13]:


pattern_utt = re.compile(r'/.*\.')
pattern_Eng = re.compile(u'[a-zA-Z\n]')
corp_map = {}
with open(aishell_2_corp_raw, "r") as f:
    for line in f:
        data = line.split()
        if len(data[1]) > 10 or pattern_Eng.search(data[1]) != None: ##删除小于10和非英文标注
            continue
        corp_map[pattern_utt.search(data[0]).group()[1:-1]] = data[1]


# In[14]:


aishell_scp_dic = {}
for i in sorted(aishell_scp_raw):
    if i not in corp_map.keys():
        continue
    aishell_scp_dic[i] = aishell_scp_raw[i]


# In[15]:


with open( aishell_scp, 'w') as f:
    for i in aishell_scp_dic:
        f.write(i+'\t'+aishell_scp_dic[i]+'\n')


# In[16]:


with open( aishell_corp, 'w') as f:
    for i in corp_map:
        f.write(i+'\t'+corp_map[i]+'\n')


# In[17]:


len(corp_map)


# ### 筛选一部分到aishell_2.scp.shuff

# In[21]:


# !bash tools/random_line.sh aishell_2.scp 5 aishell_2.scp.shuff


# ## 拆分数据

# In[22]:


SLR_wav_scp = {}
SLR_wav_corp = {}
aishell_wav_scp = {}
aishell_wav_corp = {} 


# In[23]:



with open(SLR85_wav_scp, "r") as f:
    for line in f:
        data = line.split()
        SLR_wav_scp[data[0]] = data[1]
        SLR_wav_corp[data[0]] = "你好米雅"


# In[24]:



with open(aishell_scp, "r") as f:
    for line in f:
        data = line.split()
        aishell_wav_scp[data[0]] = data[1]
        
        
with open(aishell_corp, "r") as f:
    for line in f:
        data = line.split()
        if data[0] in aishell_wav_scp:
            aishell_wav_corp[data[0]] = data[1]


# In[25]:


a = list(SLR_wav_scp.keys())
b = list(aishell_wav_scp.keys())


# ### 分出训练集

# In[26]:


SLR_train_key = random.sample(a, 10000)
aishell_train_key = random.sample(b, 100000)


# In[52]:


SLR_train_key=[] ####----


# In[27]:


SLR_wav_scp_tran=dict([(key, SLR_wav_scp[key]) for key in SLR_train_key])
SLR_wav_corp_tran=dict([(key, SLR_wav_corp[key]) for key in SLR_train_key])


# In[28]:


aishell_wav_scp_tran=dict([(key, aishell_wav_scp[key]) for key in aishell_train_key])
aishell_wav_corp_tran=dict([(key, aishell_wav_corp[key]) for key in aishell_train_key])


# In[29]:


wav_scp_train = {**SLR_wav_scp_tran,**aishell_wav_scp_tran}
corpus_train = {**SLR_wav_corp_tran, **aishell_wav_corp_tran}


# In[31]:


with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/data/wav.scp', 'w') as f:
    for i in sorted(wav_scp_train):
        f.write(i+'\t'+wav_scp_train[i]+'\n')
        
with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/data/trans.txt', 'w') as f:
    for i in sorted(corpus_train):
        f.write(i+'\t'+corpus_train[i]+'\n')


# ### 分出测试集

# In[32]:


SLR_test_key = random.sample(a, 1000)
aishell_test_key = random.sample(b, 10000)


# In[58]:


SLR_test_key=[] ####----


# In[33]:


SLR_wav_scp_test=dict([(key, SLR_wav_scp[key]) for key in SLR_test_key])
SLR_wav_corp_test=dict([(key, SLR_wav_corp[key]) for key in SLR_test_key])


# In[34]:


aishell_wav_scp_test=dict([(key, aishell_wav_scp[key]) for key in aishell_test_key])
aishell_wav_corp_test=dict([(key, aishell_wav_corp[key]) for key in aishell_test_key])


# In[35]:


wav_scp_test = {**SLR_wav_scp_test,**aishell_wav_scp_test}
corpus_test = {**SLR_wav_corp_test, **aishell_wav_corp_test}


# In[36]:


with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/test/wav.scp', 'w') as f:
    for i in sorted(wav_scp_test):
        f.write(i+'\t'+wav_scp_test[i]+'\n')
        
with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/test/trans.txt', 'w') as f:
    for i in sorted(corpus_test):
        f.write(i+'\t'+corpus_test[i]+'\n')


# ### 分出开发集

# In[37]:


SLR_dev_key = random.sample(a, 1000)
aishell_dev_key = random.sample(b, 10000)


# In[64]:


SLR_test_key=[] ####----


# In[38]:


SLR_wav_scp_dev=dict([(key, SLR_wav_scp[key]) for key in SLR_dev_key])
SLR_wav_corp_dev=dict([(key, SLR_wav_corp[key]) for key in SLR_dev_key])


# In[39]:


aishell_wav_scp_dev=dict([(key, aishell_wav_scp[key]) for key in aishell_dev_key])
aishell_wav_corp_dev=dict([(key, aishell_wav_corp[key]) for key in aishell_dev_key])


# In[40]:


wav_scp_dev = {**SLR_wav_scp_dev,**aishell_wav_scp_dev}
corpus_dev = {**SLR_wav_corp_dev, **aishell_wav_corp_dev}


# In[41]:


with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/dev/wav.scp', 'w') as f:
    for i in sorted(wav_scp_dev):
        f.write(i+'\t'+wav_scp_dev[i]+'\n')
        
with open('/home1/meichaoyang/dataset/data_aishell2/wake_up/dev/trans.txt', 'w') as f:
    for i in sorted(corpus_dev):
        f.write(i+'\t'+corpus_dev[i]+'\n')


# In[85]:


len(b)


# In[ ]:




