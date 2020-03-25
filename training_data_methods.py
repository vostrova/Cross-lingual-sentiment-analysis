import csv
import pymorphy2
import spacy
from nltk import word_tokenize
from spacy.lang.ru import Russian


def get_train_test(file):
    rows = []
    with open(file, mode='r') as csvfile:
        sreader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in sreader:
            rows.append(row)
    return rows


def split_tt(data_even, n):
    train_set = data_even[: int(0.8 * n)]
    test_set = data_even[int(0.8 * n): n]
    return train_set, test_set


def count_tags(l):
    neg = 0
    pos = 0
    for i in range(len(l)):
        if l[i][1] == 'neg':
            neg += 1
        else:
            pos += 1
    return pos, neg


def lemmatize(tokens, lang):
    lemmas_ = []
    if lang == "rus":
        morph = pymorphy2.MorphAnalyzer()
        for token in tokens:
            p = morph.parse(token)
            lemmas_.append(p[0][2])
    if lang == "ger":
        nlp = spacy.load('de_core_news_sm')
        for token in tokens:
            doc = nlp(token)
            lemmas_.append(doc[0].lemma_)
    return lemmas_


def tokenize_set(s, lemmas, lang):
    if lemmas:
        return [lemmatize(word_tokenize(instance[0]), lang) for instance in s]
    if not lemmas:
        return [word_tokenize(instance[0]) for instance in s]


def get_y(s):
    return [instance[1] for instance in s]


ru_nlp = Russian()
de_nlp = spacy.load('de_core_news_sm')


def delete_stop_words(lang, tweet):
    doc = ''
    allowed_words = []
    if lang == "rus":
        doc = ru_nlp(tweet)
        allowed_words = ["не"]
    elif lang == "ger":
        doc = de_nlp(tweet)
        allowed_words = ["gut", "gute", "guter", "gutes", "kaum", "kein", "keine", "keinem", "keinen",
                         "keiner", "nicht", "nichts", "nie", "niemand", "niemandem", "niemanden", "schlecht"]
    else:
        raise Exception("Unknown language " + lang)

    non_stop_words = filter(lambda x: not x.is_stop or str(x) in allowed_words, doc)

    return " ".join(map(lambda x: str(x), non_stop_words))
