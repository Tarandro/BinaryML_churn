from class_models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier


class Stacking(Model):

    def __init__(self, objective, seed = 15, class_weight = None, models = {}, nfolds = 5):
        Model.__init__(self, objective, seed, class_weight)
        self.name_model = 'Stacking'
        self.models = models
        self.nfolds = nfolds

    def hyper_params(self, y):
        self.shape_y = y.shape[1]

    def model(self):
        estimators = [ (name_model, self.models[name_model].best_model) for name_model in self.models.keys()]

        model_for_stacking = RandomForestClassifier(
            random_state = self.seed,
            class_weight = self.class_weight
        )

        self.best_model = StackingClassifier(
            estimators = estimators, final_estimator = model_for_stacking, cv = self.nfolds
        )
        return self.best_model