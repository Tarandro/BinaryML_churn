import numpy as np
from sklearn.model_selection import KFold

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import *


class Validation:

    def __init__(self, objective, seed=15, is_NN=False, name=None, class_weight=None):
        self.seed = seed
        self.objective = objective
        self.is_NN = is_NN
        self.name = name
        self.class_weight = class_weight

    def fit(self, model, x, y, nfolds=5, scoring='accuracy', print_result=False):

        self.oof_val = np.zeros((len(y),))
        self.fold_id = np.zeros((len(y),))

        if self.is_NN:
            # validation for neural network models :
            total_epochs = 0

            kf = KFold(n_splits=nfolds, shuffle=True, random_state=self.seed)

            for num_fold, (train_index, val_index) in enumerate(kf.split(y)):
                if isinstance(x, dict):
                    x_train, x_val = {}, {}
                    for col in x.keys():
                        x_train[col], x_val[col] = x[col][train_index], x[col][val_index]
                    y_train, y_val = y.values[train_index], y.values[val_index]
                elif isinstance(x, list):
                    x_train, x_val = [], []
                    for col in range(len(x)):
                        x_train.append(x[col][train_index])
                        x_val.append(x[col][val_index])
                    y_train, y_val = y.values[train_index], y.values[val_index]
                else:
                    try:
                        x_train, x_val = x.values[train_index], x.values[val_index]
                    except:
                        x_train, x_val = x[train_index], x[val_index]
                    try:
                        y_train, y_val = y.values[train_index], y.values[val_index]
                    except:
                        y_train, y_val = y[train_index], y[val_index]

                K.clear_session()

                model_nn = model()

                if scoring == 'accuracy':
                    monitor = 'accuracy'
                else:
                    monitor = 'loss'
                rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=3,
                                        verbose=1, epsilon=1e-4, mode='auto', min_lr=1e-4)  ########## !!!!!!!!!!!!

                # ckp = ModelCheckpoint(f'model_{n}.hdf5', monitor = 'val_loss', verbose = 0,
                #                      save_best_only = True, save_weights_only = True, mode = 'min')

                es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=4, mode='auto',
                                   baseline=None, restore_best_weights=True, verbose=0)

                train_history = model_nn.fit(x_train, y_train,
                                             validation_data=(x_val, y_val),
                                             class_weight=compute_dict_class_weight(y_train, self.class_weight,
                                                                                    self.objective),
                                             epochs=50, batch_size=16, verbose=1, callbacks=[rlr, es])

                print('Kfold #', num_fold, ' : train', monitor, '=', train_history.history[monitor][-5], ' and val',
                      monitor, '=', train_history.history['val_' + monitor][-5])
                total_epochs += len(train_history.history[monitor][:-5])

                if 'binary_proba' in self.objective:
                    self.oof_val[val_index] = model_nn.predict(x_val).reshape(-1)
                else:
                    self.oof_val[val_index] = np.argmax(model_nn.predict(x_val), axis=1).reshape(-1)
                self.fold_id[val_index] = num_fold

        else:
            # validation for sklearn models, catboost, xgboost and lightgbm :
            kf = KFold(n_splits=nfolds, shuffle=True, random_state=self.seed)

            for num_fold, (train_index, val_index) in enumerate(kf.split(x)):
                try:
                    x_train, x_val = x.values[train_index], x.values[val_index]
                except:
                    x_train, x_val = x[train_index], x[val_index]
                try:
                    y_train, y_val = y.values[train_index], y.values[val_index]
                except:
                    y_train, y_val = y[train_index], y[val_index]

                model.fit(x_train, y_train)

                if 'binary_proba' in self.objective:
                    self.oof_val[val_index] = model.predict_proba(x_val)[:, 1].reshape(x_val.shape[0], )
                else:
                    self.oof_val[val_index] = model.predict(x_val).reshape(x_val.shape[0], )
                self.fold_id[val_index] = num_fold

        self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val = calcul_metric_binary(y,
                                                                                                          self.oof_val,
                                                                                                          print_result, 0.5)
        self.fpr, self.tpr = roc(y.values, self.oof_val)

        del x_train, x_val, y_train, y_val, model

    def get_cv_prediction(self):
        return self.fold_id, self.oof_val

    def get_scores(self):
        return self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val

    def get_roc(self):
        return self.fpr, self.tpr