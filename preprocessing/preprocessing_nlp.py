import spacy
spacy.cli.download("fr_core_news_md")
nlp = spacy.load("fr_core_news_md")

import numpy as np
import pandas as pd

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

STOPWORDS = list(fr_stop)

####################
# ngrams functions
####################

def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


def count_ngrams(data, y, n_gram):
    ngrams_0 = defaultdict(int)
    ngrams_1 = defaultdict(int)

    for text in data[y.isin([y.unique()[0]])]:
        for word in generate_ngrams(text, n_gram):
            ngrams_0[word] += 1

    for text in data[y.isin([y.unique()[1]])]:
        for word in generate_ngrams(text, n_gram):
            ngrams_1[word] += 1

    df_ngrams_0 = pd.DataFrame(sorted(ngrams_0.items(), key=lambda x: x[1])[::-1])
    df_ngrams_1 = pd.DataFrame(sorted(ngrams_1.items(), key=lambda x: x[1])[::-1])

    return df_ngrams_0, df_ngrams_1


def show_best_ngrams_by_label(data, y, n_gram, N):
    fig, axes = plt.subplots(ncols=2, figsize=(9, 10), dpi=100)
    plt.tight_layout()

    df_ngrams_0, df_ngrams_1 = count_ngrams(data, y, n_gram)

    sns.barplot(y=df_ngrams_0[0].values[:N], x=df_ngrams_0[1].values[:N], ax=axes[0], color='red')
    sns.barplot(y=df_ngrams_1[0].values[:N], x=df_ngrams_1[1].values[:N], ax=axes[1], color='green')

    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].tick_params(axis='x', labelsize=13)
        axes[i].tick_params(axis='y', labelsize=13)

    axes[0].set_title('Top ' + str(N) + ' of ' + str(y.unique()[0]) + ' target', fontsize=15)
    axes[1].set_title('Top ' + str(N) + ' of ' + str(y.unique()[1]) + ' target', fontsize=15)

    plt.show()


# %% [code]
# n_gram = 2
# N = 20
# show_best_ngrams_by_label(data.text, data.sentiment, n_gram, N)

####################
# clean and preprocess text
####################

def small_clean_text(text):
    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('’', ' ', text)

    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)
    return text


def clean_text(text):
    text = str(text).lower()

    text = re.sub('\n', ' ', text)
    text = re.sub('\r', ' ', text)

    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    text = re.sub('’', ' ', text)

    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(' +', ' ', text)

    return text


def nlp_preprocessing_spacy(data):
    """ nlp.pipe preprocessing from spacy """
    list_content = list(data)
    doc_spacy_data = [doc for doc in nlp.pipe(list_content, disable=["ner", "tok2vec", "parser"])]
    return doc_spacy_data


# doc_spacy_data = nlp_preprocessing_spacy(data.text)

def reduce_text_data(doc_spacy_data, keep_pos_tag, lemmatize):
    data = []
    for text in doc_spacy_data:
        if keep_pos_tag == 'all':
            if lemmatize:
                new_text = [token.lemma_ for token in text]
            else:
                new_text = [token.text for token in text]
        else:
            if lemmatize:
                new_text = [token.lemma_ for token in text if token.pos_ in keep_pos_tag]
            else:
                new_text = [token.text for token in text if token.pos_ in keep_pos_tag]
        data.append(clean_text(" ".join(new_text)))
    return data

def clean_data_2(data):
    """ remove words shorter than 3 and remove double spaces """
    # data = data.apply(lambda text: text.lower())
    data = data.apply(lambda text: ' '.join([w for w in text.split() if len(w) >= 3]))
    data = data.apply(lambda text: re.sub(' +', ' ', text))
    return data


def top_k_frequent_n_gram(data, n_gram, k):
    """ Show frequent k n-grams """
    freq_word = defaultdict(int)

    for doc in data:
        for word in generate_ngrams(doc, n_gram):
            freq_word[word] += 1

    sort_freq_word = sorted(freq_word.items(), key=lambda x: x[1], reverse=True)

    return sort_freq_word[:k]

#############################
#############################
#############################

class Preprocessing_NLP:

    def __init__(self, data, column_text=None, target=None):
        self.data = data
        self.column_text = column_text

        self.target = target
        if isinstance(self.target, list):
            self.target = self.target
        else:
            self.target = [self.target]

        self.Y = self.data[[col for col in self.target if col in self.data.columns]]
        self.data = self.data[[self.column_text]]

    def fit_transform(self, apply_small_clean=False):
        self.apply_small_clean = apply_small_clean
        if self.apply_small_clean:
            self.data[self.column_text] = self.data[self.column_text].apply(lambda text: small_clean_text(text))

        self.doc_spacy_data = nlp_preprocessing_spacy(self.data[self.column_text])

        return self.data

    def transform(self, data_test):
        has_target = False
        if self.target is not None:
            Y_test = data_test[[col for col in self.target if col in data_test.columns]]
            has_target = True

        if self.apply_small_clean:
            data_test[self.column_text] = data_test[self.column_text].apply(lambda text: small_clean_text(text))

        self.doc_spacy_data_test = nlp_preprocessing_spacy(data_test[self.column_text])

        if has_target:
            data_test[self.target] = Y_test[self.target]

        return data_test