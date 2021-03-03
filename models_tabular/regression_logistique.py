from class_models import Model
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, loguniform, randint


class LogisticRegression_skl(Model):

    def __init__(self, objective, seed=15, column_text=None, class_weight=None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'Logistic_Regression'

    def hyper_params(self, size_params='small'):
        if size_params == 'small':
            self.parameters = dict(C=loguniform(1e-2, 1e2),
                                   penalty=['l2', 'l1'])
        else:
            self.parameters = dict(C=loguniform(1e-2, 1e2),
                                   penalty=['l2', 'l1', 'elasticnet', 'None'],
                                   max_iter=randint(50, 150))
        return self.parameters

    def model(self, hyper_params={}):
        m = LogisticRegression(
            random_state=self.seed,
            class_weight=self.class_weight,
            solver="saga",
            **hyper_params
        )
        return m
