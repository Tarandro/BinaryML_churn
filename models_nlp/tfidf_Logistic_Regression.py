from class_models import Model
from sklearn.decomposition import TruncatedSVD
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from scipy.stats import uniform, loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class TfidfLogisticRegression_skl(Model):

    def __init__(self, objective, seed=15, column_text=None, class_weight=None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'tf-idf+Logistic_Regression'

    def hyper_params(self, size_params='small'):
        if size_params == 'small':
            self.parameters = dict(
                vect__text__tfidf__lowercase=[True, False],
                vect__text__tfidf__binary=[False, True],
                vect__text__tfidf__ngram_range=[(1, 1), (1, 2), (1, 3)],
                vect__text__tfidf__stop_words=[None, list(fr_stop)],
                vect__text__reduce_dim=[None, TruncatedSVD(2), TruncatedSVD(5), TruncatedSVD(10), TruncatedSVD(20),
                                        TruncatedSVD(50), TruncatedSVD(100)],
                clf__C=loguniform(1e-2, 1e2),
                clf__penalty=['l2', 'l1']
            )
        else:
            self.parameters = dict(
                vect__text__tfidf__lowercase=[True, False],
                vect__text__tfidf__binary=[False, True],
                vect__text__tfidf__ngram_range=[(1, 1), (1, 2), (1, 3)],
                vect__text__tfidf__stop_words=[None, list(fr_stop)],
                vect__text__reduce_dim=[None, TruncatedSVD(2), TruncatedSVD(5), TruncatedSVD(10), TruncatedSVD(20),
                                        TruncatedSVD(50), TruncatedSVD(100)],
                clf__C=loguniform(1e-2, 1e2),
                clf__penalty=['l2', 'l1', 'elasticnet', 'None'],
                clf__max__iter=randint(50, 150)
            )
        return self.parameters

    def model(self, hyper_params_clf={}, hyper_params_vect={}):
        tfidf_ngrams = TfidfVectorizer(**hyper_params_vect)
        clf = LogisticRegression(
            random_state=self.seed,
            class_weight=self.class_weight,
            solver="saga",
            **hyper_params_clf
        )

        nlp_vect = Pipeline(steps=[
            ('tfidf', tfidf_ngrams),
            ('reduce_dim', 'passthrough')])

        vect = ColumnTransformer(transformers=[
            ('text', nlp_vect, self.column_text)
        ])

        pipeline = Pipeline(steps=[('vect', vect), ('clf', clf)])

        return pipeline
