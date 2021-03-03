from class_gridsearch import *
from validation import *
from prediction import *
from preprocessing_data.preprocessing_nlp import reduce_text_data


class Model:
    """ Parent class of models """

    def __init__(self, objective, seed, column_text, class_weight):
        """
        Args:
            objective (str) : 'binary' or 'binary_proba' or 'text_binary' or 'text_binary_proba'
            seed (int)
            column_text (str) : name of the column with texts
            class_weight (None or 'balanced') : apply a weight for each class
        """
        self.objective = objective
        self.seed = seed
        self.column_text = column_text
        self.class_weight = class_weight
        self.is_NN = False    # (Boolean) is the model a Neural Network ?
        self.info_scores = {}

    def fit_greadsearch(self, x, y, nfolds=5, scoring='accuracy', size_params='small', verbose=0,
                        time_limit_per_model=60, print_result=False):
        """ Apply gridsearch for the model by optimizing 'scoring' with a time less than time_limit_per_model
            (use class_gridsearch.py) """
        if self.is_NN:
            self.gridsearch = GridSearch_NN(self, self.hyper_params(size_params))
            self.gridsearch.train(x, y, nfolds, scoring, verbose, time_limit_per_model, self.name_model)
        else:
            self.gridsearch = GridSearch(self.model(), self.hyper_params(size_params))
            self.gridsearch.train(x, y, nfolds, scoring, verbose, time_limit_per_model)
        self.best_params = self.gridsearch.best_params(print_result)
        self.best_cv_score = self.gridsearch.best_score(print_result)
        self.best_model = self.gridsearch.best_estimator(self.objective)
        self.df_all_results = self.gridsearch.get_grid()
        if print_result:
            self.gridsearch.show_distribution_score()

    def validation(self, model, x_train, y_train, nfolds=5, scoring='accuracy', print_result=False):
        """ Apply validation for the model with cross-validation on (x_train,y_train)
            (use validation.py) """

        val = Validation(self.objective, self.seed, self.is_NN, self.name_model, self.class_weight)
        val.fit(model, x_train, y_train, nfolds, scoring, print_result)

        self.info_scores['fold_id'], self.info_scores['oof_val'] = val.get_cv_prediction()

        self.info_scores['accuracy_val'], self.info_scores['f1_val'], self.info_scores['recall_val'], self.info_scores[
            'precision_val'], self.info_scores['roc_auc_val'] = val.get_scores()
        self.info_scores['fpr'], self.info_scores['tpr'] = val.get_roc()

    def prediction(self, model, x_test=None, y_test=None, doc_spacy_data_test=[], print_result=False):
        """ Apply prediction for the model on (x_test,y_test)
            (use prediction.py) """
        pred = Prediction(self.objective, self.name_model, self.is_NN, self.class_weight)

        if 'text' in self.objective:
            if self.is_NN:
                x_test = self.preprocessing_transform(x_test)
            else:
                x_test[self.column_text] = reduce_text_data(doc_spacy_data_test,
                                                            self.method_embedding[self.name_model_bml][0],
                                                            self.method_embedding[self.name_model_bml][1])

        pred.fit(model, x_test, y_test, print_result)
        self.info_scores['prediction'] = pred.get_prediction()

        if y_test is not None:
            self.info_scores['accuracy_test'], self.info_scores['f1_test'], self.info_scores['recall_test'], \
            self.info_scores['precision_test'], self.info_scores['roc_auc_test'] = pred.get_scores()

    def binaryml(self, x_train, y_train=None, nfolds=5, scoring='accuracy', size_params='small', verbose=0,
                 doc_spacy_data_train=[], method_embedding={},
                 name_model_bml=None, time_limit_per_model=60, print_result=False):
        """ Apply fit_gridsearch and validation on the best model from gridsearch """

        self.method_embedding = method_embedding
        self.name_model_bml = name_model_bml

        if 'text' in self.objective:
            # preprocess text on x_train :
            if self.is_NN:
                # tokenization for neural network :
                x_train = self.preprocessing_fit_transform(x_train, size_params, method_embedding[self.name_model])
            else:
                # reduction by spacy : (pos_tag + lemmatization)
                x_train[self.column_text] = reduce_text_data(doc_spacy_data_train, method_embedding[name_model_bml][0],
                                                             method_embedding[name_model_bml][1])  # split train/test

        ### GridSearch
        start = time.perf_counter()
        self.fit_greadsearch(x_train, y_train, nfolds, scoring, size_params, verbose, time_limit_per_model,
                             print_result)
        print('Time search :', time.perf_counter() - start)

        ### Validation
        start = time.perf_counter()
        if not self.is_NN:
            self.validation(self.best_model, x_train, y_train, nfolds, scoring, print_result)
        else:
            self.initialize_params(x_train, y_train, self.best_params)
            self.validation(self.model, x_train, y_train, nfolds, scoring, print_result)
        print('Time validation :', time.perf_counter() - start)


