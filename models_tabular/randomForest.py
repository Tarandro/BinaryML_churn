from class_models import Model
from scipy.stats import uniform, loguniform, randint
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifier_skl(Model):

    def __init__(self, objective, seed=15, class_weight=None):
        Model.__init__(self, objective, seed, class_weight)
        self.name_model = 'Random_Forest'

    def hyper_params(self, size_params='small'):
        if size_params == 'small':
            self.parameters = dict(n_estimators=randint(20, 300),  # randint(20,100)
                                   max_depth=randint(5, 50),  # randint(5,75)
                                   min_samples_split=randint(2, 6),  # randint(5,15)
                                   max_samples=uniform(0.4, 0.6))  # uniform(0.4,0.6)
        else:
            self.parameters = dict(n_estimators=randint(20, 500),  # [75, 100, 125, 150],
                                   max_depth=randint(5, 75),  # [7, 10, 15, 20, 30, 40, 50, 75]
                                   criterion=['gini', 'entropy'],
                                   min_samples_split=randint(2, 6),
                                   max_samples=uniform(0.4, 0.6))
        return self.parameters

    def model(self, hyper_params={}):
        return RandomForestClassifier(
            random_state=self.seed,
            class_weight=self.class_weight,
            **hyper_params
        )