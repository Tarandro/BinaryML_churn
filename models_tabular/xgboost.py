from class_models import Model
import xgboost as xgb
from scipy.stats import uniform, loguniform, randint


class XGBoost(Model):

    def __init__(self, objective, seed = 15, column_text = None, class_weight = None):
        Model.__init__(self, objective, seed, column_text, class_weight)
        self.name_model = 'XGBoost'

    def hyper_params(self, size_params='small'):
        if size_params == 'small':
            self.parameters = dict(n_estimators=randint(20, 200),
                                   max_depth=randint(3, 10),
                                   learning_rate=uniform(0.04, 0.3),
                                   subsample=uniform(0.5, 0.5))
        else:
            self.parameters = dict(n_estimators=randint(20, 200),
                                   max_depth=randint(3, 10),
                                   learning_rate=uniform(0.04, 0.3),
                                   eta=uniform(0, 1),
                                   gamma=loguniform(1e-2, 3),
                                   reg_alpha=loguniform(0.01, 0.5),
                                   reg_lambda=loguniform(0.01, 1),
                                   min_child_weight=loguniform(1e-1, 4),
                                   subsample=uniform(0.5, 0.5))
        return self.parameters

    def model(self, hyper_params={}):
        m = xgb.XGBClassifier(
            random_state=self.seed,
            verbosity = 0,
            **hyper_params  # ,
            # scale_pos_weight = count(negative examples)/count(Positive examples)
        )
        return m