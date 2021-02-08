import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
import catboost as cat

import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
from hyperopt import hp, fmin, tpe, Trials

from utils import compute_dict_class_weight



def optimization_gridsearch(x, y, model, distributions, time_limit_per_model, nfolds, scoring, objective):
    """ gridsearch function for sklearn model with time integrated """

    print_details = False

    n_iter = 1

    start, last_call = time.perf_counter(), time.perf_counter()

    df_results = pd.DataFrame()
    df_all_results = pd.DataFrame()
    best_n_iter = 0

    while time.perf_counter() - start < time_limit_per_model:

        clf = RandomizedSearchCV(model, distributions, random_state=None, n_iter=n_iter, cv=nfolds, scoring=scoring)
        search = clf.fit(x, y)

        approx_time = (time.perf_counter() - last_call) / n_iter
        if print_details:
            print('time :', approx_time * n_iter, 'and n_iter :', n_iter)
            print(search.best_params_)
            print(search.best_score_)
            print()
        if n_iter > best_n_iter:
            best_approx_time = approx_time
            best_n_iter = n_iter

        n_iter = int((time_limit_per_model - (time.perf_counter() - start)) / best_approx_time / 3)
        last_call = time.perf_counter()

        df_results = pd.concat([df_results, pd.DataFrame(clf.cv_results_)], axis=0).reset_index(drop=True)

        if n_iter < 3 and time.perf_counter() - start > time_limit_per_model * 0.8:
            break

        if n_iter < 1:
            break

    print('  Total time :', np.round(time.perf_counter() - start, 3), 'and n_iter :', len(df_results))

    df_all_results = pd.concat([df_all_results, df_results])

    best_mean = np.mean(df_results.mean_test_score)
    best_score = np.max(df_results.mean_test_score)
    if print_details:
        plt.hist(df_results.mean_test_score)
        plt.show()

    return df_all_results


class GridSearch:

    def __init__(self, model, hyper_params):
        self.model = model
        self.model_2 = model
        self.hyper_params = hyper_params

    def train(self, x, y, nfolds=5, scoring='accuracy', verbose=0, time_limit_per_model=60, objective='binary'):
        self.df_all_results = optimization_gridsearch(x, y, self.model, self.hyper_params, time_limit_per_model, nfolds,
                                                      scoring, objective)
        self.index_best_score = self.df_all_results.mean_test_score.argmax()

    def show_distribution_score(self):
        plt.hist(self.df_all_results.mean_test_score)
        plt.show()

    def best_params(self, print_result=False):
        params = self.df_all_results.loc[self.index_best_score, 'params']
        print_params = params.copy()
        if print_result:
            if 'vect__text__tfidf__stop_words' in params.keys() and params['vect__text__tfidf__stop_words'] is not None:
                print_params['vect__text__tfidf__stop_words'] = True
            if 'vect__tfidf__stop_words' in params.keys() and params['vect__tfidf__stop_words'] is not None:
                print_params['vect__tfidf__stop_words'] = True
            print('Best parameters: ', print_params)
        return params

    def best_score(self, print_result=False):
        score = self.df_all_results.loc[self.index_best_score, 'mean_test_score']
        if print_result:
            print('Mean cross-validated score of the best_estimator: ', np.round(score, 4))
        return score

    def best_estimator(self, objective):
        if 'catboost' in str(type(self.model_2)):
            return cat.CatBoostClassifier(
                random_state=self.model_2.get_param('random_state'),
                class_weights=self.model_2.get_param('class_weights'),
                verbose=False,
                bootstrap_type='Bernoulli',
                **self.best_params()
            )
        else:
            return self.model_2.set_params(**self.best_params())

    def get_grid(self, sort_by='mean_test_score'):
        return self.df_all_results[['mean_fit_time', 'params', 'mean_test_score', 'std_test_score']].sort_values(
            by=sort_by, ascending=False).reset_index(drop=True)


class GridSearch_NN:

    def __init__(self, Model_NN, hyper_params):
        self.Model_NN = Model_NN
        self.hyper_params = hyper_params

    def optimise(self, params):

        self.Model_NN.initialize_params(self.x, self.y, params)

        print(self.Model_NN.p)

        oof_val = np.zeros((self.y.shape[0], self.y.shape[1]))
        start = time.perf_counter()

        for n, (tr, te) in enumerate(KFold(n_splits=self.nfolds,
                                           random_state=self.Model_NN.seed,
                                           shuffle=True).split(self.y)):

            if isinstance(self.x, dict):
                x_tr, x_val = {}, {}
                for col in self.x.keys():
                    x_tr[col], x_val[col] = self.x[col][tr], self.x[col][te]
                y_tr, y_val = self.y.values[tr], self.y.values[te]
            else:
                x_tr, x_val = self.x.values[tr], self.x.values[te]
                y_tr, y_val = self.y.values[tr], self.y.values[te]

            model = self.Model_NN.model()

            if self.scoring == 'accuracy':
                monitor = 'accuracy'
            else:
                monitor = 'loss'

            rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=3,
                                    verbose=0, epsilon=1e-4, mode='auto', min_lr=1e-4)

            # ckp = ModelCheckpoint(f'model_{n}.hdf5', monitor = 'val_loss', verbose = 0,
            #                      save_best_only = True, save_weights_only = True, mode = 'min')

            es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=4, mode='auto',
                               baseline=None, restore_best_weights=True, verbose=0)

            history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                epochs=60, batch_size=16,
                                class_weight=compute_dict_class_weight(y_tr, self.Model_NN.class_weight,
                                                                       self.Model_NN.objective),
                                callbacks=[rlr, es], verbose=0)

            hist = pd.DataFrame(history.history)

            if 'binary_proba' in self.Model_NN.objective:
                oof_val[te, :] = model.predict(x_val)
            else:
                oof_val[te, :] = np.argmax(model.predict(x_val), axis=1).reshape(-1, 1)

            self.total_epochs += len(history.history['val_loss'][:-5])

            K.clear_session()
            del model, history, hist
            d = gc.collect()

        metrics = []
        oof_val = np.where(oof_val > 0.5, 1, 0).reshape(-1)
        if 'f1' in self.scoring:
            metrics.append(-f1_score(self.y.values.reshape(-1), oof_val))
        elif 'recall' in self.scoring:
            metrics.append(-recall_score(self.y.values.reshape(-1), oof_val))
        elif 'precision' in self.scoring:
            metrics.append(-precision_score(self.y.values.reshape(-1), oof_val))
        elif 'roc' in self.scoring or 'auc' in self.scoring:
            metrics.append(-roc_auc_score(self.y.values.reshape(-1), oof_val))
        else:
            metrics.append(-accuracy_score(self.y.values.reshape(-1), oof_val))

        score = -np.mean(metrics)
        print('oof_val score', self.scoring, 'Metric', score)

        if 'hidden_units' in self.Model_NN.p.keys():
            self.list_hist[len(self.Model_NN.p['hidden_units']) - 1].append(score)
        else:
            self.list_hist[0].append(score)
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        return np.mean(metrics)

    def train(self, x_, y_, nfolds=5, scoring='accuracy', verbose=0, time_limit_per_model=60,
              name_model='SimpleNeuralNetwork'):
        self.x = x_  # .copy().reset_index(drop=True)
        self.y = y_  # .copy().reset_index(drop=True)
        self.nfolds = nfolds
        self.scoring = scoring
        self.df_all_results = {'mean_fit_time': [], 'params': [], 'mean_test_score': [], 'std_test_score': []}
        self.list_hist = [[] for name in self.hyper_params.keys() if 'hidden_unit' in name]
        if len(self.list_hist) == 0:
            self.list_hist = [[]]
        self.total_epochs = 0
        trials = Trials()

        self.hopt = fmin(fn=self.optimise,
                         space=self.hyper_params,
                         algo=tpe.suggest,
                         max_evals=100,
                         timeout=time_limit_per_model,
                         trials=trials,
                         )

        self.df_all_results = pd.DataFrame(self.df_all_results)
        self.df_all_results['model'] = name_model
        self.index_best_score = self.df_all_results.mean_test_score.argmax()
        self.mean_epochs = int(self.total_epochs / self.nfolds) + 1

    def show_distribution_score(self):
        rows, cols = 1, 3
        fig, ax = plt.subplots(rows, cols, figsize=(50, 20))

        for row in range(rows):
            for col in range(cols):
                if row * cols + col + 1 <= len(self.list_hist) and len(self.list_hist[row * cols + col]) > 0:
                    ax[col].hist(self.list_hist[row * cols + col])
                    for tick in ax[col].xaxis.get_major_ticks():
                        tick.label.set_fontsize(30)
        plt.show()

    def best_params(self, print_result=False):
        params = self.df_all_results.loc[self.index_best_score, 'params']
        if print_result:
            print('Best parameters: ', params)
        return params

    def best_score(self, print_result=False):
        score = self.df_all_results.loc[self.index_best_score, 'mean_test_score']
        if print_result:
            print('Mean cross-validated score of the best_estimator: ', np.round(score, 4))
        return score

    def best_estimator(self, objective):
        self.Model_NN.initialize_params(self.x, self.y, self.best_params())

        model = self.Model_NN.model()

        if self.scoring == 'accuracy':
            monitor = 'accuracy'
        else:
            monitor = 'loss'
        rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=3,
                                verbose=1, epsilon=1e-4, mode='auto', min_lr=1e-4)

        # ckp = ModelCheckpoint(f'model_{n}.hdf5', monitor = 'val_loss', verbose = 0,
        #                      save_best_only = True, save_weights_only = True, mode = 'min')

        es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=5, mode='auto',
                           baseline=None, restore_best_weights=True, verbose=0)

        for n, (tr, te) in enumerate(KFold(n_splits=10,
                                           random_state=self.Model_NN.seed,
                                           shuffle=True).split(self.y)):

            if isinstance(self.x, dict):
                x_tr, x_val = {}, {}
                for col in self.x.keys():
                    x_tr[col], x_val[col] = self.x[col][tr], self.x[col][te]
                y_tr, y_val = self.y.values[tr], self.y.values[te]
            else:
                x_tr, x_val = self.x.values[tr], self.x.values[te]
                y_tr, y_val = self.y.values[tr], self.y.values[te]

            history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                epochs=60, batch_size=16,
                                class_weight=compute_dict_class_weight(y_tr, self.Model_NN.class_weight,
                                                                       self.Model_NN.objective),
                                callbacks=[rlr, es], verbose=0)
            break  # best_model train on only one validation
        return model

    def get_grid(self, sort_by='mean_test_score'):
        return self.df_all_results[['mean_fit_time', 'params', 'mean_test_score', 'std_test_score']].sort_values(
            by=sort_by, ascending=False).reset_index(drop=True)