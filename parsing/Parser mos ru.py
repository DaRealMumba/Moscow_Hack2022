#!/usr/bin/env python
# coding: utf-8

# In[517]:


from selenium.webdriver import Chrome 
from bs4 import BeautifulSoup
import requests
from time import sleep
import pandas as pd


# ## Парсинг сайта mos.ru

# In[416]:


# окно для selenium
browser = Chrome("/Users/masha/chromedriver")


# In[417]:


# ссылка за последние 9 месяцев
url = "https://www.mos.ru/news/all/all/from/01.09.2021/to/11.06.2022/"


# In[418]:


browser.get(url)


# In[421]:


# кнопка для дополнения информации 'показать еще'
button=browser.find_element_by_xpath('//*[@id="all-news"]/div/div[2]/div/div[2]/mos-button/a')


# In[476]:


x = 1
y = 10
a=1
b=1000


# In[423]:


links=[]


# In[477]:


# собираем ссылки на новости за последние 9 месяцев и добавляем в наш список
for j in range(a,b):
    for i in range(x,y):
        links.append(browser.find_element_by_xpath        (f'//*[@id="all-news"]/div/div[2]/div/div[1]/events-list/div/div[{j}]/div[{i}]/article-card/div/div/a').        get_attribute('href'))
        if i == 9: 
            print(f'9 новостей{j}')
            button.click()
            sleep(8)


# In[480]:


# сохраняем ссылки в текстовый файл для дальнейшего использования
with open("urls_mosru.txt", "w") as output:
    output.write(str(set(links2)))


# In[501]:


# открываем файл и редактируем ссылки
with open('urls_mosru.txt') as f:
    text = f.read().replace('{', '').replace('}', '').replace("'", '').split(', ')

text


# In[546]:


# создаем датафрейм для сбора данных
df = pd.DataFrame(columns = ['Заголовок', 'Дата', 'Текст'])


# In[548]:


# парсим новости/заголовки к ним/дату и добавляем в наш датафрейм
for i in text:
    r = requests.get(i)
    soup = BeautifulSoup(r.text, "lxml")
    df=df.append({'Заголовок': soup.find('h1', class_='news-article-title-container__title').text,               'Дата':soup.find('time', class_='news-article__date').text,               'Текст':soup.find('div', class_='content-text').text}, ignore_index=True)


# In[549]:


df


# In[559]:


# сохраняем датафрейм
df.to_csv('df_mosru.csv', index=False)

