from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from preprocessing_data.preprocessing_tabular import *
from preprocessing_data.preprocessing_nlp import *
from models_tabular.regression_logistique import LogisticRegression_skl
from models_tabular.randomForest import RandomForestClassifier_skl
from models_tabular.catBoost import CatBoost
from models_tabular.lightGBM import LightGBM
from models_tabular.xgboost import XGBoost
from models_tabular.simpleNeuralNetwork import SimpleNeuralNetwork
from models_tabular.stacking import Stacking
from models_tabular.blend_models import BlendModel

from models_nlp.tfidf_naive_bayes import NaiveBayes_skl
from models_nlp.tfidf_SGDclassifier import SGDClassifier_skl
from models_nlp.tfidf_Logistic_Regression import TfidfLogisticRegression_skl
from models_nlp.fastText import Fasttext_Attention
from models_nlp.camembert import BERT
from utils import calcul_metric_binary


class BinaryML:

    def __init__(self, exclude_model=[], max_nb_model=None, max_run_time=None, max_run_time_per_model=60,
                 objective='binary', nfolds=5, seed=15,
                 class_weight=False, print_result=False, scoring="accuracy", verbose=0,
                 size_params='small', apply_stacking=False, apply_blend_model=False, method_scaling="MinMaxScaler",
                 method_embedding={}):
        """
        Args:
            exclude_model (list) : name of models to exclude
            max_nb_model (int) : (not used) maximum of gridsearch iteration to try for each model
            max_run_time (int) : (not used) maximum time in seconds for gridsearch
            max_run_time_per_model (int) : maximum time in seconds for gridsearch for each model
            nfolds (int) : number of folds during gridsearch and validation
            seed (int)
            class_weight (Boolean) : apply a weight for each class
            print_result (Boolean) : show intermediate result
            scoring (str) : metric optimized during gridsearch
            verbose (int) : verbose for gridsearch
            objective (str) : 'binary' or 'binary_proba' or 'text_binary' or 'text_binary_proba'
            size_params (str) : 'small' or 'big', size of parameters range for gridsearch
            apply_stacking (Boolean)
            apply_blend_model (Boolean)
            method_scaling (str) : 'MinMaxScaler' or 'RobustScaler' or 'StandardScaler' sklearn method to scale data
            method_embedding (dict) : information about embedding method
                e.g : {'Fasttext_Attention': '/kaggle/input/fasttext-french-2b-300d/cc.fr.300.vec', #path of pre-train model
                       'BERT': 'CamemBERT',     # or 'Roberta'
                       'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False), (['ADJ', 'NOUN'], True)]} #(tag_to_keep, lemmatize)
        """
        self.exclude_model = exclude_model
        self.max_nb_model = max_nb_model
        self.max_run_time = max_run_time
        self.max_run_time_per_model = max_run_time_per_model
        self.nfolds = nfolds
        self.seed = seed
        if not class_weight:
            self.class_weight = None
        else:
            self.class_weight = "balanced"
        self.print_result = print_result
        self.scoring = scoring
        self.verbose = verbose
        self.objective = objective
        self.size_params = size_params
        self.apply_stacking = apply_stacking
        self.apply_blend_model = apply_blend_model
        self.method_scaling = method_scaling
        self.method_embedding = method_embedding
        self.info_scores = {}

    def split_data(self, frac=0.8):
        """ split data and build X_train, X_test, Y_train, Y_test
            if text : build doc_spacy_data_train and doc_spacy_data_test, documents preprocessed by spacy"""
        train_data = self.data.sample(frac=frac, random_state=self.seed)
        self.X_train = train_data.copy()
        self.Y_train = self.Y.loc[train_data.index, :]
        if self.doc_spacy_data is not None:
            self.doc_spacy_data_train = np.array(self.doc_spacy_data)[list(train_data.index)]
        else:
            self.doc_spacy_data_train = None

        if frac < 1:
            test_data = self.data.drop(train_data.index)
            self.X_test = test_data.copy()
            self.Y_test = self.Y.drop(train_data.index)
            if self.doc_spacy_data is not None:
                self.doc_spacy_data_test = np.array(self.doc_spacy_data)[
                    [i for i in range(len(self.doc_spacy_data)) if i not in list(train_data.index)]]
            else:
                self.doc_spacy_data_test = None

    def normalize_data(self):
        """ Normalize X_train and X_test, fit apply only on X_train"""
        self.features = self.X_train.columns.values
        if self.method_scaling == 'MinMaxScaler':
            self.scaler = MinMaxScaler(feature_range=(0, 1))  # or (-1,1)
        elif self.method_scaling == 'RobustScaler':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.features = list(self.X_train.columns)
        self.scaler.fit(self.X_train)

        self.X_train[self.features] = self.scaler.transform(self.X_train[self.features])
        try:
            self.X_test[self.features] = self.scaler.transform(self.X_test[self.features])
        except:
            pass

    def data_preprocessing(self, data, target=None, column_text=None,
                           frac=0.8, normalize=True, remove_multicollinearity=False, feature_selection=False, subsample=1,
                           info_pca={}, info_tsne={}, info_stats={}, remove_low_variance=False,
                           remove_percentage=0.8, multicollinearity_threshold=0.9, feature_selection_threshold=0.8,
                           method_nan_categorical='constant', method_nan_numeric='mean',
                           apply_small_clean=False):
        """ Apply Preprocessing_tabular from preprocessing_tabular.py
               or Preprocessing_NLP from preprocessing_nlp.py  """

        if not 'text' in self.objective:
            self.pre = Preprocessing_tabular(data, target, None)
            self.data = self.pre.fit_transform(remove_multicollinearity=remove_multicollinearity,
                                               feature_selection=feature_selection,
                                               class_weight=self.class_weight, subsample=subsample,
                                               info_stats=info_stats, info_pca=info_pca, info_tsne=info_tsne,
                                               remove_low_variance=remove_low_variance,
                                               remove_percentage=remove_percentage,
                                               multicollinearity_threshold=multicollinearity_threshold,
                                               feature_selection_threshold=feature_selection_threshold,
                                               method_nan_categorical=method_nan_categorical,
                                               method_nan_numeric=method_nan_numeric)
            self.doc_spacy_data = None
            self.column_text = None
        else:
            self.pre = Preprocessing_NLP(data, column_text, target)
            self.data = self.pre.fit_transform(apply_small_clean=apply_small_clean)
            self.doc_spacy_data = self.pre.doc_spacy_data
            self.column_text = list(self.data.columns).index(column_text)

        self.target = self.pre.target
        self.Y = self.pre.Y

        self.split_data(frac)

        self.normalize = normalize
        if self.normalize and 'text' not in self.objective:
            self.normalize_data()

    def preprocess_test_data(self, data_test):
        """ apply same transformation as in the fit_transform for data_test
        Args:
            data_test (dataframe)
        """
        self.data_test = self.pre.transform(data_test)
        if self.normalize and 'text' not in self.objective:
            self.data_test = pd.DataFrame(self.scaler.transform(self.data_test))
            self.data_test.columns = self.features
        if 'text' in self.objective:
            self.doc_spacy_data_test = self.pre.doc_spacy_data_test
        else:
            self.doc_spacy_data_test = None
        return self.data_test

    def train(self, x=None, y=None):
        """ Apply gridsearch, save the best model and apply validation for each model """
        # if x and y are None use X_train and Y_train else use x and y :
        if x is not None:
            self.x_train = x
        else:
            self.x_train = self.X_train
        if y is not None:
            self.y_train = y
        else:
            self.y_train = self.Y_train

        if 'text' not in self.objective:
            self.name_models = ['Logistic_Regression', 'Random_Forest', 'LightGBM', 'XGBoost', 'CatBoost',
                                'SimpleNeuralNetwork']
            class_models = [LogisticRegression_skl, RandomForestClassifier_skl, LightGBM, XGBoost, CatBoost,
                            SimpleNeuralNetwork]
        else:
            if 'spacy' not in self.method_embedding.keys() or self.method_embedding['spacy'] == []:
                self.name_models = ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression']
                self.name_models += [name for name in [ 'Fasttext_Attention', 'BERT'] if name in self.method_embedding.keys()]
                class_models = [NaiveBayes_skl, SGDClassifier_skl, TfidfLogisticRegression_skl]
                class_models += [[Fasttext_Attention, BERT][i] for i in range(2) if ['Fasttext_Attention', 'BERT'][i] in self.method_embedding.keys()]
            else:
                self.name_models = []
                class_models = []
                for (keep_pos_tag, lemmatize) in self.method_embedding['spacy']:
                    if keep_pos_tag == 'all':
                        if lemmatize == True:
                            for name in ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression']:
                                self.name_models.append(name + '_lem')
                                self.method_embedding[name + '_lem'] = (keep_pos_tag, lemmatize)
                                if name in self.exclude_model:
                                    self.exclude_model.append(name + '_lem')
                        else:
                            for name in ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression']:
                                self.name_models.append(name)
                                self.method_embedding[name] = (keep_pos_tag, lemmatize)
                    else:
                        if lemmatize == True:
                            for name in ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression']:
                                self.name_models.append(name + '_' + "_".join(keep_pos_tag) + '_lem')
                                self.method_embedding[name + '_' + "_".join(keep_pos_tag) + '_lem'] = (
                                    keep_pos_tag, lemmatize)
                                if name in self.exclude_model:
                                    self.exclude_model.append(name + '_' + "_".join(keep_pos_tag) + '_lem')
                        else:
                            for name in ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression']:
                                self.name_models.append(name + '_' + "_".join(keep_pos_tag))
                                self.method_embedding[name + '_' + "_".join(keep_pos_tag)] = (keep_pos_tag, lemmatize)
                                if name in self.exclude_model:
                                    self.exclude_model.append(name + '_' + "_".join(keep_pos_tag))
                    class_models += [NaiveBayes_skl, SGDClassifier_skl, TfidfLogisticRegression_skl]
                self.name_models += [name for name in ['Fasttext_Attention', 'BERT'] if
                                     name in self.method_embedding.keys()]
                class_models += [[Fasttext_Attention, BERT][i] for i in range(2) if
                                 ['Fasttext_Attention', 'BERT'][i] in self.method_embedding.keys()]

        self.models = {}

        for i, name_model in enumerate(self.name_models):
            if name_model not in self.exclude_model:
                print('\n\033[4m' + name_model + ' Model\033[0m', ':' if self.print_result else '...', '\n')
                if name_model == 'CatBoost' or name_model == 'XGBoost':
                    self.models[name_model] = class_models[i](self.objective, self.seed, self.column_text,
                                                              self.class_weight, self.y_train)
                else:
                    self.models[name_model] = class_models[i](self.objective, self.seed, self.column_text,
                                                              self.class_weight)

                self.models[name_model].binaryml(self.x_train, self.y_train, self.nfolds, self.scoring,
                                                 self.size_params, self.verbose,
                                                 self.doc_spacy_data_train, self.method_embedding, name_model,
                                                 self.max_run_time_per_model, self.print_result)

    def ensemble(self):
        """ Apply ensemble model : Stacking and BlendModel """
        # stacking :
        if self.apply_stacking:
            allow_models = {name: model for name, model in self.models.items() if not self.models[name].is_NN}
            if len(allow_models.keys()) >= 2 and 'text' not in self.objective:
                print('\n\033[4m' + 'Stacking' + ' Model\033[0m', ':' if self.print_result else '...', '\n')
                model_stacking = Stacking(self.objective, self.seed, self.column_text, self.class_weight, allow_models,
                                          self.nfolds)
                model_stacking.hyper_params(self.y_train)
                model_stacking.validation(model_stacking.model(), self.x_train, self.y_train, self.nfolds, self.scoring,
                                          self.print_result)
            else:
                self.apply_stacking = False

        # blend model average:
        if self.apply_blend_model:
            if len(self.models.keys()) >= 2:
                print('\n\033[4m' + 'Blend' + ' Model\033[0m', ':' if self.print_result else '...', '\n')
                model_blend = BlendModel(self.objective, self.seed)
                model_blend.validation(self.models, self.x_train, self.y_train, self.print_result)
            else:
                self.apply_blend_model = False

        if self.apply_stacking:
            self.models['Stacking'] = model_stacking
        if self.apply_blend_model:
            self.models['BlendModel'] = model_blend

    def get_leaderboard(self, dataset='val', sort_by='accuracy', ascending=False):
        """ Metric scores for each best model
        Args:
            dataset (str) : 'val' or 'test', which prediction to use
        Return:
             self.leaderboard (dataframe)
        """
        self.metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
        self.leaderboard = {"name": list(self.models.keys())}
        for metric in self.metrics:
            self.info_scores[metric + '_' + dataset] = [self.models[name_model].info_scores[metric + '_' + dataset] for
                                                        name_model in self.models.keys()]
            self.leaderboard[metric + '_' + dataset] = np.round(self.info_scores[metric + '_' + dataset], 4)
        self.leaderboard = pd.DataFrame(self.leaderboard).sort_values(by=sort_by + '_' + dataset, ascending=ascending)
        return self.leaderboard

    def get_leaderboard_threshold(self, list_threshold_1 = [0.5], sort_by='accuracy', ascending=False):
        """ Metric scores for each best model with different threshold (only on validation prediction)
        Args:
            list_threshold_1 (list) : threshold to try (value > threshold -> value = 1)
        Return:
             self.leaderboard (dataframe)
        """
        if list_threshold_1 == None or list_threshold_1 == []:
            return None

        dataset = 'val'
        self.metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
        self.leaderboard_thr = {"name": [], "thr_1": []}
        for metric in self.metrics:
            self.leaderboard_thr[metric + '_' + dataset] = []

        for thr in list_threshold_1:
            for name_model in self.models.keys():
                y_pred = self.models[name_model].info_scores['oof_val']
                y_true = self.y_train
                acc_val, f1_val, recall_val, pre_val, roc_auc_val = calcul_metric_binary(y_true, y_pred, False, thr_1=thr)
                self.leaderboard_thr["name"].append(name_model)
                self.leaderboard_thr["thr_1"].append(thr)
                self.leaderboard_thr["accuracy_val"].append(np.round(acc_val, 4))
                self.leaderboard_thr["recall_val"].append(np.round(recall_val, 4))
                self.leaderboard_thr["precision_val"].append(np.round(pre_val, 4))
                self.leaderboard_thr["f1_val"].append(np.round(f1_val, 4))
                self.leaderboard_thr["roc_auc_val"].append(np.round(roc_auc_val, 4))

        self.leaderboard_thr = pd.DataFrame(self.leaderboard_thr).sort_values(by=sort_by + '_' + dataset, ascending=ascending)
        return self.leaderboard_thr

    def get_df_all_results(self):
        """ Information gridsearch for each model
        Return:
            df_all_results (dataframe)
        """
        df_all_results = pd.DataFrame()
        for name_model in self.models.keys():
            if name_model not in ['Stacking', 'BlendModel']:
                df_all_results_model = self.models[name_model].df_all_results
                df_all_results_model['model'] = name_model
                df_all_results = pd.concat([df_all_results, df_all_results_model], axis=0).reset_index(drop=True)
        return df_all_results

    def show_distribution_scores(self):
        df_all_results = self.get_df_all_results()
        list_name_models = list(df_all_results.model.unique())
        rows, cols = 2, 3
        fig, ax = plt.subplots(rows, cols, figsize=(50, 20))

        for row in range(rows):
            for col in range(cols):
                if row * cols + col + 1 <= len(list_name_models):
                    name_model = list_name_models[row * cols + col]
                    values = df_all_results[df_all_results.model.isin([name_model])].mean_test_score
                    if np.std(values) < 1e-4:
                        ax[row, col].hist(values, range=(values.min() - 1e-3, values.max() + 1e-3))
                    else:
                        ax[row, col].hist(values)
                    ax[row, col].set_xlabel(name_model + ' (' + str(len(values)) + ' models)', size=30)
                    ax[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    for tick in ax[row, col].xaxis.get_major_ticks():
                        tick.label.set_fontsize(30)
        plt.show()

    def get_roc_curves(self):
        plt.figure(figsize=(15, 15), linewidth=1)
        for name_model, model in self.models.items():
            plt.plot(model.info_scores['fpr'], model.info_scores['tpr'], label=name_model)
        plt.plot([0, 1], [0, 1], 'k--', label='Random: 0.5')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        if 'text' in self.objective:
            plt.savefig('roc_curves_text.png')
        else:
            plt.savefig('roc_curves.png')
        plt.show()

    def correlation_models(self):
        """ correlation between cross-validation prediction of best models """
        result_val = pd.DataFrame(
            {name_model: self.models[name_model].info_scores['oof_val'].reshape(-1) for name_model in
             self.models.keys()})
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.heatmap(result_val.corr(), annot=True, cmap=sns.cm.rocket_r)
        plt.show()

    def leader_predict(self, on_test_data=True, x=None, y=None, thr_1 = 0.5):
        """ Prediction on x or X_test (if on_test_data=True or x == None) for each best models """
        if on_test_data:  # predict on self.X_test
            for name_model in self.models.keys():
                if name_model == 'BlendModel':
                    self.models[name_model].prediction(self.models, self.X_test, self.Y_test, False, thr_1=thr_1)
                else:
                    self.models[name_model].prediction(self.models[name_model].best_model, self.X_test, self.Y_test,
                                                       self.doc_spacy_data_test, print_result=False, thr_1=thr_1)

        else:  # predict on x
            for name_model in self.models.keys():
                if name_model == 'BlendModel':
                    self.models[name_model].prediction(self.models, x, y, False, thr_1=thr_1)
                else:
                    self.models[name_model].prediction(self.models[name_model].best_model, x, y,
                                                       self.doc_spacy_data_test, print_result=False, thr_1=thr_1)

        # Create a dataframe with predictions of each model + y_true
        dict_prediction = {}
        if on_test_data:
            if self.Y_test is not None:
                dict_prediction['y_true'] = np.array(self.Y_test).reshape(-1)
        else:
            if y is not None:
                dict_prediction['y_true'] = np.array(y).reshape(-1)

        for name_model in self.models.keys():
            dict_prediction[name_model] = np.round(self.models[name_model].info_scores['prediction'].reshape(-1), 3)
        self.dataframe_predictions = pd.DataFrame(dict_prediction)
