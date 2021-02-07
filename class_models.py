from class_gridsearch import *
from validation_tabular import *
from prediction_tabular import *

class Model:  # class parent

    def __init__(self, objective, seed, class_weight):
        self.objective = objective
        self.seed = seed
        self.class_weight = class_weight
        self.is_NN = False
        self.info_scores = {}

    def fit_greadsearch(self, x, y, nfolds = 5, scoring = 'accuracy', size_params = 'small', verbose = 0,
                        time_limit_per_model = 60, print_result = False):
        if self.is_NN:
            self.gridsearch = GridSearch_NN(self, self.hyper_params(size_params))
            self.gridsearch.train(x, y, nfolds, scoring, verbose, time_limit_per_model, self.name_model)
        else:
            self.gridsearch = GridSearch(self.model(), self.hyper_params(size_params))
            self.gridsearch.train(x, y, nfolds, scoring, verbose, time_limit_per_model, self.objective)
        self.best_params = self.gridsearch.best_params(print_result)
        self.best_cv_score = self.gridsearch.best_score(print_result)
        self.best_model = self.gridsearch.best_estimator(self.objective)
        self.df_all_results = self.gridsearch.get_grid()
        if print_result:
            self.gridsearch.show_distribution_score()

    def validation(self, model, x_train, y_train, nfolds = 5, scoring = 'accuracy', print_result = False):

        val = Validation(self.objective, self.seed, self.is_NN, self.name_model, self.class_weight)
        val.fit(model, x_train, y_train, nfolds, scoring, print_result)

        self.info_scores['fold_id'], self.info_scores['oof_val'] = val.get_cv_prediction()

        self.info_scores['accuracy_val'], self.info_scores['f1_val'], self.info_scores['recall_val'], self.info_scores['precision_val'], self.info_scores['roc_auc_val'] = val.get_scores()
        self.info_scores['fpr'], self.info_scores['tpr'] = val.get_roc()


    def prediction(self, model, x_test = None, y_test = None, print_result = False):
        pred = Prediction(self.objective, self.name_model, self.is_NN, self.class_weight)

        pred.fit(model, x_test, y_test, print_result)
        self.info_scores['prediction'] = pred.get_prediction()

        if y_test is not None:
            self.info_scores['accuracy_test'], self.info_scores['f1_test'], self.info_scores['recall_test'], self.info_scores['precision_test'], self.info_scores['roc_auc_test'] = pred.get_scores()


    def binaryml(self, x_train, y_train = None, nfolds = 5, scoring = 'accuracy', size_params = 'small', verbose = 0,
               time_limit_per_model = 60, print_result = False):

        ### GridSearch
        start = time.perf_counter()
        self.fit_greadsearch(x_train, y_train, nfolds, scoring, size_params, verbose, time_limit_per_model, print_result)
        print('Time search :', time.perf_counter() - start)

        ### Validation
        start = time.perf_counter()
        if not self.is_NN:
            self.validation(self.best_model, x_train, y_train, nfolds, scoring, print_result)
        else:
            self.initialize_params(x_train, y_train, self.best_params)
            self.validation(self.model, x_train, y_train, nfolds, scoring, print_result)
        print('Time validation :', time.perf_counter() - start)

