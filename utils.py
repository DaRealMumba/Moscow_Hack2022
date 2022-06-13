#TODO: написать функцию в колабе


#Это все нужно установить в виртуальное окружение или просто блять как нибудь
#!pip install transformers[sentencepiece]
#!pip install transformers
#!pip install -U sentence-transformers
#!pip install pymorphy2
#!pip install ufal.udpipe
#!pip install corpy
#!pip install wget
#!pip install faiss-cpu --no-cache



#ПРОПИШИТЕ ПУТИ КО ВСЕМ ФАЙЛАМ ОТСЮДА САМИ БРАТЬЯ И СЕСТРА

from transformers import pipeline
import pandas as pd
import numpy as np
import transformers
from transformers import AutoModel, BertTokenizerFast
from sentence_transformers import SentenceTransformer
import faiss
import glob
import gensim
import os
import shutil
import ufal.udpipe as udp
import corpy.udpipe as crp
import wget
from datetime import datetime as dt
import re


def get_results(sequence_to_classify, date_news=None):

  #переделываем даты
  df_news = pd.read_csv('/Users/Muminsho/Desktop/DS/Hakaton/MoscowHack/bot_try/df_news.csv', parse_dates=['Дата']) #вставьте свой путь
#   df_news['Заголовок_и_текст'] = df_news['Заголовок'] + '. \n' + df_news['Текст'] 
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: x[:-7])
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' января ', '/01/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' февраля ', '/02/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' марта ', '/03/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' апреля ', '/04/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' мая ', '/05/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' июня ', '/06/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' июля ', '/07/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' августа ', '/08/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' сентября ', '/09/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' октября ', '/10/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' ноября ', '/11/', x))
#   df_news['Дата'] = df_news['Дата'].apply(lambda x: re.sub(' декабря ', '/12/', x))
#   df_news['Дата'] = pd.to_datetime(df_news['Дата'])

  #загружаем модель для эмбедингов
  model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
  #эмбединги новостей
  sentence_embeddings = np.load("/Users/Muminsho/Desktop/DS/Hakaton/MoscowHack/bot_try/embeddings/embeddings.npy")
  #подрубаем фаисс
  d = sentence_embeddings.shape[1]
  index = faiss.IndexFlatL2(d)
  index.add(sentence_embeddings)
  index.ntotal


  #ищем кандидатов, пришлось переделать конструкцию if/else в цикле
  k = 10

  news_ebd = model.encode(sequence_to_classify, normalize_embeddings=True, show_progress_bar=True)
  query_vector = news_ebd.reshape((1, news_ebd.shape[0]))
  distances, indices = index.search(query_vector, k)


  candidate_indexs = []
  print(f"Top {k} elements in the dataset for max inner product search:")
  for i, (dist, indice) in enumerate(zip(distances[0], indices[0])):
      print(f"{i+1}: Vector number {indice:4} with distance {dist}")
      diff_between_dates_news = 0
      # отбираем только наиболее близких кандидатов и более релевантные новости по дате
      if not date_news is None:
          diff_between_dates_news = (pd.to_datetime(date_news) - df_news['Дата'].iloc[indice]).days
    
      if dist < 0.2 and diff_between_dates_news < 15:
          candidate_indexs.append(indice)

  #подгружаем зеро-шот модель
  classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")
  

  #дальше пиздец какой-то, но не я писал
  candidate_labels = df_news['Заголовок_и_текст'].iloc[candidate_indexs].to_list()
  url = df_news['Ссылки'].iloc[candidate_indexs]



  result_l = []
  result_s = []
  result_sum = []
  for candidat in range(len(candidate_labels)):
      result = classifier(candidate_labels[candidat], [sequence_to_classify], multi_label = True)
      result_l.append(result['labels'][0])
      result_s.append(result['scores'][0])
    
      # Лейблирование
    #   if result['scores'][0] > 0.9:
    #       result_sum.append('Подтверждено')
    #   elif result['scores'][0] < 0.2:
    #       result_sum.append('Фейк')
    #   else:
    #       result_sum.append('Не найдено похожих новостей')

      if result['scores'][0] > 0.8:
          result_sum.append('Подтверждено')
          return [result_sum, result_s, url]
      elif result['scores'][0] < 0.2:
          result_sum.append('Фейк')
          return [result_sum, result_s, url]
      else:
          result_sum.append('Не найдено похожих новостей')
          return [result_sum, result_s, '-']



#   df_result = pd.DataFrame([result_sum, result_s, result_l, candidate_labels, url]).T
#   df_result.columns = ['Фейк_или_нет', 'Скор', 'Проверяемая_новость', 'Новость_из_белого_списка', 'Ссылка']


# return df_result