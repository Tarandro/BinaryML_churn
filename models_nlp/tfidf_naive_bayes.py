from class_models import Model
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from scipy.stats import uniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class NaiveBayes_skl(Model):

    def __init__(self, objective, seed = 15, column_text = None, class_weight = None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'tf-idf+Naive_Bayes'

    def hyper_params(self, size_params = 'small'):
        if size_params == 'small':
            self.parameters = dict(
                vect__tfidf__lowercase = [True, False],
                vect__tfidf__binary=[False, True],
                vect__tfidf__ngram_range = [(1, 1), (1, 2), (1 ,3)],
                vect__tfidf__stop_words = [None, list(fr_stop)],
                clf__alpha = uniform(0, 1)
            )
        else:
            self.parameters = dict(
                vect__tfidf__lowercase = [True, False],
                vect__tfidf__binary=[False, True],
                vect__tfidf__ngram_range = [(1, 1), (1, 2), (1 ,3)],
                vect__tfidf__stop_words = [None, list(fr_stop)],
                clf__alpha = uniform(0, 1)
            )
        return self.parameters

    def model(self, hyper_params_clf = {}, hyper_params_vect = {}):
        tfidf_ngrams = TfidfVectorizer(**hyper_params_vect)
        clf = MultinomialNB(
            **hyper_params_clf
        )

        vect = ColumnTransformer(transformers=[
            ('tfidf', tfidf_ngrams, self.column_text)
        ])

        pipeline = Pipeline(steps=[('vect', vect), ('clf', clf)])

        return pipeline