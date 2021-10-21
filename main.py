import pandas as pd
import string
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem
from tqdm.auto import tqdm
#Для нейросетки
#from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


# Предобработка Текста#
# Убираем пунктуацию
def remove_punctuation(text):
    return "".join([ch if ch not in string.punctuation else ' ' for ch in text])

# Убираем цифры
def remove_numbers(text):
    return ''.join([i if not i.isdigit() else ' ' for i in text])

# Убираем множественные пробелы
def remove_multiple_spaces(text):
    return re.sub(r'\s+', ' ', text, flags=re.I)

# Убираем стоп-слова
def remove_stop_words(text):
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in russian_stopwords and token != ' ']
    return " ".join(tokens)

#Обработка текста после предобработки
# лемматизация текста
def lemmatize_text(text,mystem):
    text_lem = mystem.lemmatize(text)
    tokens = [token for token in text_lem if token != ' ' and token not in russian_stopwords]
    return " ".join(tokens)

# стемминг текста
def stemming_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in russian_stopwords]
    return " ".join(stemmed_tokens)

#Подготовка дата сета
def preparing_text(mystem,stemmer,category,russian_stopwords):
    data = pd.read_csv('data/news.csv')
    data = data[:100000]
    stemmed_texts_list = []
    lemm_texts_list = []
    df_res = pd.DataFrame()
    news_in_cat_count = 2000
    for topic in tqdm(category):
        df_topic = data[data['topic'] == topic][:news_in_cat_count]
        df_res = df_res.append(df_topic, ignore_index=True)
        try:
            prep_text = [remove_multiple_spaces(remove_numbers(remove_punctuation(text.lower()))) for text in
                         df_res['text']]
        except Exception as e:
            print(e)
    print("1. Предоброботка выполнена успешно!")
    df_res['text_prep'] = prep_text
    df_res.to_csv('data/news_prepared.csv')  # Текст после стандартной обработки
    print("Предобработка загружена в файл 'data_prep.csv'!")
    for text in tqdm(df_res['text_prep']):
        try:
            text = stemming_text(text)
            stemmed_texts_list.append(text)
        except Exception as e:
            print(e)
    df_res['text_stem'] = stemmed_texts_list
    remove_stop_words(text)
    print("2. Стемминг выполнен успешно!")
    df_res.to_csv('data/news_stemmed.csv')  # Текст после стемминга
    print("Стемминг загружен в файл 'data_stemmed.csv'!")
    for text in df_res['text_stem']:
        try:
            text = lemmatize_text(text,mystem)
            lemm_texts_list.append(text)
        except Exception as e:
            print(e)
    df_res['text_lemm'] = lemm_texts_list
    print("3. Лемматизация выполнена успешно!")
    df_res.to_csv('data/news_lemmatized.csv')#Текст после лемматизации, стеминга и постобработки
    print("Лемматизация загружена в файл 'data_lemmed.csv'!")

if __name__ == '__main__':
    mystem = Mystem()
    stemmer = SnowballStemmer("russian")  # Инициализация стеммера
    russian_stopwords = stopwords.words("russian")  # даталист русских стоп-слов
    russian_stopwords.extend(['…', '«', '»', '...', 'т.д.', 'т', 'д'])
    ###Классы нашего классификатора###
    category = ['Спорт', 'Культура', 'Интернет и СМИ', 'Наука и техника', 'Экономика']

    # preparing_text(mystem,stemmer,category,russian_stopwords)

    #Подргружаем файл датасета
    data = pd.read_csv('data/data.csv')
    #Читаем текст и категории из файла
    X = data['text_stem']
    y = data['topic']
    #Разделение датасета
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #Запускаем конвеер
    #nb = Pipeline([('vect', CountVectorizer()),
     #              ('tfidf', TfidfTransformer()),
      #             ('clf', MultinomialNB()),
       #            ])

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])

    #Запускаем обучение
    #nb.fit(X_train, y_train)
    sgd.fit(X_train,y_train)
    #Запускаем предсказание на тестовой выборке
    y_pred = sgd.predict(X_test)

    ##############Классификация текста
    #Ввод данных
    try:
        ex_text = input("Введите текст для классификации:\n")
        if len(ex_text) < 100 or ex_text.isdigit():
            print("ОШИБКА. Введите текст с большим содержанием.")
        else:
    #Предобрабыватываем текст
            ex_text = remove_multiple_spaces(remove_numbers(remove_punctuation(ex_text.lower())))
    #Убираем стоп-слова
            ex_text = remove_stop_words(ex_text)
    #Лемматизация текста
            # ex_text = lemmatize_text(ex_text, mystem)
    #Стемминг текста
            ex_text = stemming_text(ex_text)
    #Предсказываем категорию обработанного текста
            text_pred = sgd.predict([ex_text])
    #Выводим текст
            print("\nКатегория текста:" + text_pred[0])
    except Exception as e:
        print(e)
    ###############Точность
    print("___Точность работы классификатора___")
    accuracy = accuracy_score(y_pred, y_test)
    report = classification_report(y_test, y_pred, target_names=category)
    print('Точность: %s' % accuracy)
    print(report)