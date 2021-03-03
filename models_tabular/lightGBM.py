from class_models import Model
import lightgbm as lgb
from scipy.stats import uniform, loguniform, randint


class LightGBM(Model):

    def __init__(self, objective, seed = 15, column_text = None, class_weight = None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'LightGBM'

    def hyper_params(self, size_params='small'):
        if size_params == 'small':
            self.parameters = dict(n_estimators=randint(20, 200),
                                   num_leaves=randint(50, 150),
                                   learning_rate=uniform(0.04, 0.3))
        else:
            self.parameters = dict(n_estimators=randint(20, 200),
                                   num_leaves=randint(50, 150),
                                   learning_rate=uniform(0.04, 0.3),
                                   min_child_weight=loguniform(1e-3, 0.1),
                                   feature_fraction=uniform(0.5, 0.5),
                                   reg_alpha=loguniform(0.01, 0.5),
                                   reg_lambda=loguniform(0.01, 0.5))
        return self.parameters

    def model(self, hyper_params={}):
        m = lgb.LGBMClassifier(
            random_state=self.seed,
            class_weight=self.class_weight,
            verbosity=0,
            force_col_wise=True,
            **hyper_params
        )
        return m