import numpy as np
from utils import *


class Prediction:

    def __init__(self, objective, name=None, is_NN=False, class_weight=None):
        self.objective = objective
        self.name = name
        self.is_NN = is_NN
        self.class_weight = class_weight

    def fit(self, model, x, y=None, print_result=False):

        if self.is_NN:
            # validation for neural network models :
            if 'pandas' in str(type(x)):
                self.prediction = model.predict(x.values)
            else:
                self.prediction = model.predict(x)
        else:
            # validation for sklearn models, catboost, xgboost and lightgbm :
            if 'binary_proba' in self.objective:
                if 'pandas' in str(type(x)):
                    self.prediction = model.predict_proba(x.values)[:, 1]
                else:
                    self.prediction = model.predict_proba(x)[:, 1]
            else:
                if 'pandas' in str(type(x)):
                    self.prediction = model.predict(x.values)
                else:
                    self.prediction = model.predict(x)

        if y is not None:
            # calculate metrics if y_true is provided
            if 'binary' in self.objective and self.is_NN:
                self.prediction = np.argmax(self.prediction, axis=1).reshape(-1)
            else:
                self.prediction = self.prediction.reshape(-1)
            self.acc_test, self.f1_test, self.recall_test, self.pre_test, self.roc_auc_test = calcul_metric_binary(y,
                                                                                                                   self.prediction,
                                                                                                                   print_result)

    def get_prediction(self):
        return self.prediction

    def get_scores(self):
        return self.acc_test, self.f1_test, self.recall_test, self.pre_test, self.roc_auc_test