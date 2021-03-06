from class_models import Model
import numpy as np
import catboost as cat
from scipy.stats import uniform, loguniform, randint
from sklearn.utils.class_weight import compute_class_weight


class CatBoost(Model):

    def __init__(self, objective, seed = 15, column_text = None, class_weight = None, y_train=None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'CatBoost'
        self.y_train = y_train

    def hyper_params(self, size_params='small'):
        if size_params == 'small':
            self.parameters = dict(iterations=randint(20, 200),
                                   depth=randint(2, 9),
                                   learning_rate=uniform(0.04, 0.3),
                                   subsample=uniform(0.5, 0.5))
        else:
            self.parameters = dict(iterations=randint(20, 200),
                                   depth=randint(2, 9),
                                   learning_rate=uniform(0.04, 0.3),
                                   l2_leaf_reg=randint(1, 15),
                                   subsample=uniform(0.5, 0.5)
                                   )
        return self.parameters

    def model(self, hyper_params={}):
        if self.class_weight == 'balanced':
            self.class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train.values.reshape(-1)),
                                                     y=self.y_train.values.reshape(-1))
            # self.class_weight = dict(zip(np.unique(self.y_train[target]), weights))
        else:
            self.class_weight = None

        m = cat.CatBoostClassifier(
            random_state=self.seed,
            verbose=False,
            class_weights=self.class_weight,
            bootstrap_type='Bernoulli',
            **hyper_params
        )
        return m