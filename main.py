import pandas as pd
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem

#Предобработка Текста
def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])

def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

def remove_multiple_spaces(text):
	return re.sub(r'\s+', ' ', text, flags=re.I)

#Убираем стоп-слова
def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
    return " ".join(tokens)

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    category = ['Москва', 'Культура', 'Спорт', 'Экономика', 'В Росии', 'В мире'] #Категории нашего классификтора
    stemmer = SnowballStemmer("russian") #Инициализация стеммера
    stemmed_texts_list = []
    russian_stopwords = stopwords.words("russian") #даталист русских стоп-слов
    russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д'])
    df_res = pd.DataFrame()
    for topic in (category):
        df_topic = data[data['category'] == category]
        df_res = df_res.append(df_topic, ignore_index=True)
    prep_text = [remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))) for text in df_res['text']]
    df_res['text_prep'] = prep_text
    df_res.to_csv('data_prep.csv') #Текст после стандартной обработки
    for text in df_res['text_prep']:
        tokens = word_tokenize(text)
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
        text = " ".join(stemmed_tokens)
        stemmed_texts_list.append(text)
    df_res['text_stem'] = stemmed_texts_list
    remove_stop_words(text)
    df_res.to_csv('data_stemmed.csv') #Текст после стемминга
