from utils import *


class BlendModel:
    """ Average of model predictions """
    def __init__(self, objective, seed = 15):
        self.objective = objective
        self.seed = seed
        self.name_model = 'BlendModel'
        self.info_scores = {}

    def validation(self, models, x_train, y_train, print_result = False, thr_1 = 0.5):
        """ Average validation predictions of all models (models[name_model].info_scores['oof_val'])  """

        oof_val = None
        self.y_shape1 = y_train.shape[1]

        for name_model in models.keys():
            if name_model not in ['Stacking', 'BlendModel']:
                if oof_val is None:
                    oof_val = models[name_model].info_scores['oof_val'].reshape(-1 ,1)
                else:
                    oof_val = np.concatenate([oof_val, models[name_model].info_scores['oof_val'].reshape(-1 ,1)], axis=1)

        oof_val = np.mean(oof_val, axis = 1)

        if self.objective == 'binary':
            oof_val = np.where(oof_val > 0.5, 1, 0)

        self.info_scores['accuracy_val'], self.info_scores['f1_val'], self.info_scores['recall_val'], self.info_scores['precision_val'], self.info_scores['roc_auc_val'] = calcul_metric_binary(y_train, oof_val, print_result, thr_1)
        self.info_scores['fpr'], self.info_scores['tpr'] = roc(y_train.values, oof_val)

        self.info_scores['fold_id'], self.info_scores['oof_val'] = models[name_model].info_scores['fold_id'], oof_val

    def prediction(self, models, x_test, y_test = None, print_result = False, thr_1 = 0.5):
        """ Average test predictions of all models (models[name_model].info_scores['prediction'])  """

        prediction = None

        for name_model in models.keys():
            if name_model not in ['Stacking', 'BlendModel']:
                if prediction is None:
                    prediction = models[name_model].info_scores['prediction'].reshape(-1 ,1)
                else:
                    prediction = np.concatenate([prediction, models[name_model].info_scores['prediction'].reshape(-1 ,1)], axis=1)

        prediction = np.mean(prediction, axis = 1)

        if self.objective == 'binary':
            prediction = np.where(prediction > 0.5, 1, 0)

        if y_test is not None:
            self.info_scores['accuracy_test'], self.info_scores['f1_test'], self.info_scores['recall_test'], self.info_scores['precision_test'], self.info_scores['roc_auc_test'] = calcul_metric_binary(y_test, prediction, print_result, thr_1)

        self.info_scores['prediction'] = prediction