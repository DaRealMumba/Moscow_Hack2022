#!/usr/bin/env python
# coding: utf-8

# In[39]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# ## Парсинг сайта panorama.pub

# In[18]:


url_list=[]


# In[19]:


# собираем ссылки на новости
for i in range(100):
    url = f'https://panorama.pub/society?page={i}'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")
    for a in soup.find_all('a', href=True):
        if 'news' in a['href']:
            url_list.append('https://panorama.pub' + a['href'])


# In[33]:


# создаем датафрейм для сбора данных
df = pd.DataFrame(columns = ['Заголовок', 'Текст'])


# In[35]:


# парсим новости/заголовки к ним, и добавляем в наш датафрейм 
for i in url_list:
    r = requests.get(i)
    soup = BeautifulSoup(r.text, "lxml")
    df=df.append({'Текст': soup.find('div', class_='entry-contents pr-0 md:pr-8').text,                  'Заголовок':soup.find('h1',                     class_='font-bold text-2xl md:text-3xl lg:text-4xl pl-1 pr-2 self-center').text},                     ignore_index=True)


# In[37]:


# сохраняем датафрейм
df.to_csv('df_panorama.csv', index=False)

