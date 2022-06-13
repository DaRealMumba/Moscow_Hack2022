#!/usr/bin/env python
# coding: utf-8

# ## Генерируем фейк ньюз с помощью **GPT** модели для дальнейшего обучения модели классификации

# ## Импортируем библиотеки

# In[2]:


# загружаем трансформер
get_ipython().system('pip install transformers')


# In[3]:


import numpy as np
import pandas as pd
import re
import random
import gc
import textwrap
import random

import torch
from tqdm.notebook import tqdm
import transformers
import tensorflow as tf

from transformers import GPT2LMHeadModel, AdamW

if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[4]:


from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')


# ## Открываем спарсенные с сайта mos.ru данные и подготавливаем их для подачи в модель

# In[5]:


path = 'df_mosru.csv'


# In[6]:


df = pd.read_csv(path)
df


# In[7]:


#обрабатываем данные
all_txt = df['Текст'].tolist()
all_txt = [re.sub('\]', '', i) for i in all_txt]
all_txt = [re.sub('\[', '', i) for i in all_txt]
all_txt = [i.strip() for i in all_txt]
all_txt = [re.sub('\n', "", i) for i in all_txt]
all_txt = [re.sub('\xa0', '', i) for i in all_txt]


# In[10]:


# токенизируем наш текст
train = []
test = []
max_length = 0
index_max_length = 0

for idx, value in enumerate(all_txt):
    tokens = tokenizer.encode(value, add_special_tokens=True)
    tokens = np.array(tokens)

    curr_len = len(tokens)
    if curr_len >= max_length:
    max_length = curr_len
    index_max_length = idx

    if idx <= (len(all_txt) * .90):
    train.append(tokens)

    else:
    test.append(tokens)

train = np.array(train)
test = np.array(test)

print('len(train), len(test): ', len(train), len(test))
print('max_length, index_max_length: ', max_length, index_max_length)


# In[15]:


# делаем паддинг для нашей трейновой выборки
train_2 = tf.keras.preprocessing.sequence.pad_sequences(
    train,
    maxlen=500,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0.0)


# In[16]:


# делаем паддинг для нашей тестовой выборки
test_2 = tf.keras.preprocessing.sequence.pad_sequences(
    test,
    maxlen=500,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0.0)


# ## Загружаем модель и обучаем ее

# In[17]:


model = GPT2LMHeadModel.from_pretrained(
    'sberbank-ai/rugpt3small_based_on_gpt2',
    output_attentions = False,
    output_hidden_states = False,
)

model.to(device);


# In[18]:


batch_size = 4
epochs = 3

n_train = len(train_2)//(batch_size+1)
n_test = len(test_2)//(batch_size+1)
print(n_train, n_test)

optimizer = AdamW(model.parameters(), lr = 1e-5, eps = 1e-8)

total_steps = n_train * epochs
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)


def accuracy(y_true, logits):
    return torch.mean((y_true[1:] == torch.argmax(logits, dim=2)[:-1]).float()).detach().cpu().numpy()


# In[20]:


def prep_tensors(x, i, batch_size=batch_size):
    start_idx = i*batch_size
    end_idx = start_idx + batch_size
    batch_ids = x[start_idx: end_idx]
    batch_ids = torch.LongTensor(batch_ids).to(device)
    return torch.tensor(batch_ids).to(device)

preped = prep_tensors(train_2, 17)
print('preped shape: ', preped.shape)

for epoch in range(1, epochs+1):
    print(f'epoch {epoch}/{epochs} : training')

    train_loss = []
    train_acc = []
    model.train()
    pbar = tqdm(range(n_train))
    for i in pbar:
        batch_ids = prep_tensors(train_2, i)

        model.zero_grad()
        torch.cuda.empty_cache()
        loss, logits, _ = model(batch_ids,
                             token_type_ids=None, 
                             labels=batch_ids
                             ).values()

        gc.collect()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss.append(loss.item())
        train_acc.append(accuracy(batch_ids, logits))
        pbar.set_description(f'acc {np.mean(train_acc):.4f} loss {np.mean(train_loss):.4f}', refresh=True)

    
    print('epoch {epoch}/{epochs} : validation')
    model.eval()
    val_acc = []
    val_loss = []
    pbar = tqdm(range(n_test))
    for i in pbar:
        batch_ids = prep_tensors(test_2, i)
        with torch.no_grad():        
            loss, logits, _ = model(batch_ids, 
                                token_type_ids=None, 
                                labels=batch_ids
                                 ).values()
        
        val_loss.append(loss.item())
        val_acc.append(accuracy(batch_ids, logits))
        pbar.set_description(f'acc {np.mean(val_acc):.4f} loss {np.mean(val_loss):.4f}', refresh=True)


# In[30]:


# сохраняем нашу модель
torch.save(model, 'model_v2.pt')


# ## Генерируем фейк новости

# In[159]:


# создаем список из первых 3 слов наших новостей для последующей подачи их на генерацию
gen_list = [' '.join(i.split(' ')[:3]) for i in all_txt[:4000]]


# In[151]:


# датафрейм для сбора сгенерированных фейк новостей
fake_df = pd.DataFrame(columns = ["Текст"])


# In[ ]:


# генерим новости
for i in gen_list:
    prompt = i
    prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = model.generate(
    input_ids=prompt,
    max_length=random.randint(200, 400), #устанавливаем диапазон длины генерируемых новостей
    num_beams=5,
    do_sample=True,
    temperature=3.,
    top_k=80,
    top_p=0.8,
    no_repeat_ngram_size=5,
    num_return_sequences=1,
    ).cpu().numpy()
    sequence = tokenizer.decode(out[0])
    fake_df=fake_df.append({'Текст': textwrap.fill(sequence[:sequence.rfind('.')+1])}, ignore_index=True)
    if fake_df.shape[0] % 5 == 0:
    fake_df.to_csv('df_fake2.csv', index=False) # сохраняем модель
    print(fake_df.shape)

