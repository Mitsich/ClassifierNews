#Библиотеки
import pandas as pd
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem

#Предобработка Текста#
#Убираем пунктуацию
def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])
#Убираем цифры
def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])
#Убираем множественные пробелы
def remove_multiple_spaces(text):
	return re.sub(r'\s+', ' ', text, flags=re.I)
#Убираем стоп-слова
def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
    return " ".join(tokens)
#лемматизация текста
def lemmatize_text(text,mystem):
    text_lem = mystem.lemmatize(text)
    tokens = [token for token in text_lem if token != ' ' and token not in russian_stopwords]
    return " ".join(tokens)
#стемминг текста
def stemming_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
    return " ".join(stemmed_tokens)


if __name__ == '__main__':
    data = pd.read_csv('data/data.csv')
    mystem = Mystem()
    category = ['Москва', 'Культура', 'Спорт', 'Экономика', 'В мире'] #Категории нашего классификтора
    stemmer = SnowballStemmer("russian") #Инициализация стеммера
    stemmed_texts_list = []
    lemm_texts_list = []
    russian_stopwords = stopwords.words("russian") #даталист русских стоп-слов
    russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д'])
    df_res = pd.DataFrame()
    for topic in (category):
        df_topic = data[data['category'] == topic]
        df_res = df_res.append(df_topic, ignore_index=True)
        try:
            prep_text = [remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))) for text in df_res['text']]
        except Exception as e:
            print(e)
    print("1. Предоброботка выполнена успешно!")
    df_res['text_prep'] = prep_text
    df_res.to_csv('data/data_prep.csv') #Текст после стандартной обработки
    print("Предобработка загружена в файл 'data_prep.csv'!")
    for text in df_res['text_prep']:
        try:
            text = stemming_text(text)
            stemmed_texts_list.append(text)
        except Exception as e:
            print(e)
    df_res['text_stem'] = stemmed_texts_list
    remove_stop_words(text)
    print("2. Стемминг выполнен успешно!")
    df_res.to_csv('data/data_stemmed.csv') #Текст после стемминга
    print("Стемминг загружен в файл 'data_stemmed.csv'!")
    for text in df_res['text_stem']:
        try:
            text = lemmatize_text(text,mystem)
            lemm_texts_list.append(text)
        except Exception as e:
            print(e)
    df_res['text_lemm'] = lemm_texts_list
    print("3. Лемматизация выполнена успешно!")
    df_res.to_csv('data/data_lemmed.csv')#Текст после лемматизации, стеминга и постобработки
    print("Лемматизация загружена в файл 'data_lemmed.csv'!")






