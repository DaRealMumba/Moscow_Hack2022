"""
This is a echo bot.
It echoes any incoming text messages.
"""

import logging
from datetime import datetime as dt
import pandas as pd
import traceback

 
from aiogram import Bot, Dispatcher, executor, types
from utils import get_results
 
API_TOKEN = '' # вставьте свой токен



dc = {'news': None, 
        'date': None}
# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Привет!\nЭто бот для распознавания фейк-ньюз.\n\nCперва с помощью команды '\\news <текст новости>' введите текст новости\n\nЗатем с помощью команды '\\date <дата>' введите дату новости в формате  YYYY-MM-DD\n\nЕсли даты нет - то просто отправьте пустую команду '\\date'")


@dp.message_handler(commands=['news'])
async def news_reader(message: types.Message):
    dc['news'] = message.text[len('/news')+1:]
    await message.answer("А теперь введите дату:")


@dp.message_handler(commands=['date'])
async def news_reader(message: types.Message):
    dc['date'] = message.text[len('/date')+1:]
    if dc['news'] is None:
        print("Сначала введите текст новости с помощью соотвествующей команды")
        dc['date'] = None
        return

    try:
        result = get_results(dc['news'], pd.to_datetime(dc['date']))
        #print(news_text)
        out = 'Статус новости: '
        if result[0][0] == 'Подтверждено':
            out += "Не фейк, найден первоисточник в белом списке\n\n"
        elif result[0][0] == 'Фейк':
            out += "Фейк, найден первоисточник в белом списке и данная новость ему противоречит\n\n"
        else:
            out += 'Первоисточник не найден\n\n'

        out += 'Вероятность:' + str(result[1]) + '\n\n'

        # out += 'Проверяемая новость: ' + str(result['Проверяемая_новость']) + '\n\n'

        # out += 'Новость из белого списка: ' + str(result['Новость_из_белого_списка']) + '\n\n'

        out += 'Ссылка: ' + str(result[2]) 

        await message.answer(out)
        news_text = None #после того как результат получен - обнуляем значения переменных, что сразу можно было вводить следующие данные
        date = None 

    except Exception as e:
        traceback.print_exc()
        await message.answer('Введите дату в соотвествующем формате')
    

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)