from nltk.sentiment import SentimentAnalyzer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment.util import *
from random import shuffle
from training_data_methods import *
from sklearn.model_selection import cross_val_score


def train_classifier(classifier, num_of_tweets, gram, lang, lemmas):
    sentim_analyzer = SentimentAnalyzer()

    print("num_of_tweets, gram, lang, lemmas_bool:")
    print(num_of_tweets, gram, lang, lemmas)

    training = []
    testing = []
    if lang == "rus":
        training = get_train_test("train.csv")
        testing = get_train_test("test.csv")
    if lang == "ger":
        training = get_train_test("train_de.csv")
        testing = get_train_test("test_de.csv")

    data = training + testing

    def removeStopWords(item):
        item[0] = delete_stop_words(lang, item[0])
        return item

    data_neg = []
    data_pos = []
    for i in data:
        if i[1] == 'neg':
            data_neg.append(i)
        if i[1] == 'pos':
            data_pos.append(i)

    data_even = []
    for i in range(len(data_neg)):
        data_even.append(data_neg[i])
        data_even.append(data_pos[i])

    training_data = data_even[:num_of_tweets]

    dict_1 = {}
    dict_1["Accuracy"] = 0
    dict_1["Precision [pos]"] = 0
    dict_1["Recall [pos]"] = 0
    dict_1["F-measure [pos]"] = 0
    dict_1["Precision [neg]"] = 0
    dict_1["Recall [neg]"] = 0
    dict_1["F-measure [neg]"] = 0
    vocab = 0
    unigram = 0
    bigram = 0

    for i in range(5):

        test = training_data[int(len(training_data) / 5) * i:int((len(training_data) / 5)) * (i + 1)]
        train = training_data[:int(len(training_data) / 5) * i] + training_data[
                                                                  int((len(training_data) / 5)) * (i + 1):len(
                                                                      training_data)]

        train = list(map(removeStopWords, train))
        test = list(map(removeStopWords, test))

        # print(train)

        shuffle(train)
        shuffle(test)

        print("len(train+test):")
        print(len(train) + len(test))

        print("train: pos, neg:")
        print(count_tags(train))

        print("test: pos, neg:")
        print(count_tags(test))

        vocabulary = sentim_analyzer.all_words(tokenize_set(train, lemmas, lang))
        print("vocab len:")
        print(len(vocabulary))
        vocab += len(vocabulary)
        # print("vocabulary[0]:")
        # print(vocabulary[0])

        if gram == "unigram":
            unigram_features = sentim_analyzer.unigram_word_feats(vocabulary)
            print("unigram feats len:")
            print(len(unigram_features))
            unigram += len(unigram_features)
            # print("unigram_features[0]:")
            # print(unigram_features[0])

            sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)
        if gram == "bigram":
            bigram_features = sentim_analyzer.bigram_collocation_feats(tokenize_set(train, lemmas, lang))

            print("bigram feats len:")
            print(len(bigram_features))
            bigram += len(bigram_features)
            # print("bigram_features[0]:")
            # print(bigram_features[0])
            # print("bigram_features[5]:")
            # print(bigram_features[5])
            sentim_analyzer.add_feat_extractor(extract_bigram_feats, bigrams=bigram_features)

        _train_X = sentim_analyzer.apply_features(tokenize_set(train, lemmas, lang), labeled=False)
        _train_Y = get_y(train)

        _test_X = sentim_analyzer.apply_features(tokenize_set(test, lemmas, lang), labeled=False)
        _test_Y = get_y(test)
        sentim_analyzer.train(classifier.train, list(zip(_train_X, _train_Y)))
        dict = sentim_analyzer.evaluate(list(zip(_test_X, _test_Y)))
        print(dict)
        dict_1["Accuracy"] += dict.get('Accuracy')
        dict_1["Precision [pos]"] += dict.get('Precision [pos]')
        dict_1["Recall [pos]"] += dict.get('Recall [pos]')
        dict_1["F-measure [pos]"] += dict.get('F-measure [pos]')
        dict_1["Precision [neg]"] += dict.get('Precision [neg]')
        dict_1["Recall [neg]"] += dict.get('Recall [neg]')
        dict_1["F-measure [neg]"] += dict.get('F-measure [neg]')

    print("Accuracy:")
    print(dict_1.get('Accuracy') / 5)
    print("Precision [pos]:")
    print(dict_1.get('Precision [pos]') / 5)
    print("Precision [neg]:")
    print(dict_1.get('Precision [neg]') / 5)
    print("F-measure [pos]:")
    print(dict_1.get('F-measure [pos]') / 5)
    print("F-measure [neg]:")
    print(dict_1.get('F-measure [neg]') / 5)
    print("Recall [pos]:")
    print(dict_1.get('Recall [pos]') / 5)
    print("Recall [neg]:")
    print(dict_1.get('Recall [neg]') / 5)
    print("vocab length: ")
    print(vocab / 5)
    if gram == "bigram":
        print("bigram features:")
        print(bigram / 5)
    if gram == "unigram":
        print("unigram features:")
        print(unigram / 5)


svm = SklearnClassifier(LinearSVC())

grams = ["unigram", "bigram"]
langs = ["rus", "ger"]
classifiers = [svm, NaiveBayesClassifier]
# NaiveBayesClassifier
for classifier in classifiers:
    for gram in grams:
        for lang in langs:
            print("------------------------------------------------------")
            train_classifier(classifier, 500, gram, lang, False)
            print("------------------------------------------------------")
            train_classifier(classifier, 1500, gram, lang, False)
