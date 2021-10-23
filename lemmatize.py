from nltk.corpus import stopwords
from pymystem3 import Mystem
import pandas as pd
import datetime


def lemmatize_text(text, mystem):
    text_lem = mystem.lemmatize(text)
    tokens = [token for token in text_lem if token != ' ' and token not in russian_stopwords]
    return " ".join(tokens)


if __name__ == '__main__':
    mystem = Mystem()
    # nltk.download('stopwords') #установка один раз
    russian_stopwords = stopwords.words("russian")  # даталист русских стоп-слов
    russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д'])
    lemm_texts_list = [] #список для лемма_текстов

    data = pd.read_csv("data/new_data_count_lemmatime.csv") #[:] - обрез, если нужно
    #print(data)

    starttime = datetime.datetime.now()
    for text in data['text_stem']:
        try:
            text = lemmatize_text(text, mystem)
            lemm_texts_list.append(text)
        except Exception as e:
            print(e)
    final_time = datetime.datetime.now()
    print("Лемматизация прошла за %s" %(final_time - starttime))
    data['text_lemm'] = lemm_texts_list
    data.to_csv("data/lemma.csv")
