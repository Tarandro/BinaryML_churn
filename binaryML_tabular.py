from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from preprocessing_tabular import *
from models_tabular.regression_logistique import LogisticRegression_skl
from models_tabular.randomForest import RandomForestClassifier_skl
from models_tabular.catBoost import CatBoost
from models_tabular.lightGBM import LightGBM
from models_tabular.xgboost import XGBoost
from models_tabular.simpleNeuralNetwork import SimpleNeuralNetwork
from models_tabular.stacking import Stacking
from models_tabular.blend_models import BlendModel

class BinaryML_tabular:

    def __init__(self, exclude_model = [], max_nb_model = None, max_run_time = None, max_run_time_per_modele = 60, early_stopping = False, objective = 'binary',
                 nfolds = 5, keep_cv_pred =True, seed = 15, class_weight = False, print_result = False, scoring = "accuracy", verbose = 0,
                 size_params = 'small', apply_stacking = False, apply_blend_model = False, method_scaling = "MinMaxScaler"):
        self.exclude_model = exclude_model
        self.max_nb_model = max_nb_model
        self.max_run_time = max_run_time
        self.max_run_time_per_modele = max_run_time_per_modele
        self.early_stopping = early_stopping
        self.nfolds = nfolds
        self.keep_cv_pred = keep_cv_pred
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
        self.info_scores = {}

    def split_data(self, frac = 0.8):
        train_data = self.data.sample(frac=frac, random_state=self.seed)
        self.X_train = train_data.copy()
        self.Y_train = self.Y.loc[train_data.index ,:]

        if frac < 1:
            test_data = self.data.drop(train_data.index)
            self.X_test = test_data.copy()
            self.Y_test = self.Y.drop(train_data.index)

    def normalize_data(self):
        self.features = self.X_train.columns.values
        if self.method_scaling == 'MinMaxScaler':
            self.scaler = MinMaxScaler(feature_range = (0 ,1)) # or (-1,1)
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


    def data_preprocessing(self, data, target = None,
                           frac = 0.8, normalize = True, remove_multicollinearity = False, feature_selection = False, subsample = 1,
                           info_pca = {}, info_tsne = {}, info_stats = {}, remove_low_variance = False,
                           remove_percentage = 0.8, multicollinearity_threshold = 0.9, feature_selection_threshold = 0.8,
                           method_nan_categorical = 'constant', method_nan_numeric = 'mean'):

        self.pre = Preprocessing(data, target, None)
        self.data = self.pre.fit_transform(remove_multicollinearity = remove_multicollinearity, feature_selection = feature_selection,
                                           class_weight = self.class_weight, subsample = subsample ,info_stats = info_stats, remove_low_variance = remove_low_variance,
                                           remove_percentage = remove_percentage, multicollinearity_threshold = multicollinearity_threshold,
                                           feature_selection_threshold = feature_selection_threshold,
                                           method_nan_categorical = method_nan_categorical, method_nan_numeric = method_nan_numeric)

        self.target = self.pre.target
        self.Y = self.pre.Y

        self.split_data(frac)

        self.normalize = normalize
        if self.normalize:
            self.normalize_data()

    def preprocess_test_data(self, data_test):
        self.data_test = self.pre.transform(data_test)
        if self.normalize:
            self.data_test = pd.DataFrame(self.scaler.transform(self.data_test))
            self.data_test.columns = self.features
        return self.data_test

    def train(self, x = None, y = None, timesteps = 2):
        if x is not None:
            self.x_train = x
        else:
            self.x_train = self.X_train
        if y is not None:
            self.y_train = y
        else:
            self.y_train = self.Y_train


        self.name_models = ['Logistic_Regression', 'Random_Forest', 'LightGBM', 'XGBoost', 'CatBoost', 'SimpleNeuralNetwork']
        class_models = [LogisticRegression_skl, RandomForestClassifier_skl, LightGBM, XGBoost, CatBoost, SimpleNeuralNetwork]

        self.models = {}

        for i, name_model in enumerate(self.name_models):
            if name_model not in self.exclude_model:
                print('\n\033[4m' + name_model + ' Model\033[0m', ':' if self.print_result else '...' ,'\n')
                if name_model == 'CatBoost':
                    self.models[name_model] = class_models[i](self.objective, self.seed, self.class_weight, self.y_train)
                else:
                    self.models[name_model] = class_models[i](self.objective, self.seed, self.class_weight)

                self.models[name_model].binaryml(self.x_train, self.y_train, self.nfolds, self.scoring, self.size_params, self.verbose,
                                               self.max_run_time_per_modele, self.print_result)


    def ensemble(self):
        # stacking :
        if self.apply_stacking:
            allow_models = {name :model for name, model in self.models.items() if not self.models[name].is_NN}
            if len(allow_models.keys()) >= 2:
                print('\n\033[4m' + 'Stacking' + ' Model\033[0m', ':' if self.print_result else '...' ,'\n')
                model_stacking = Stacking(self.objective, self.seed, self.class_weight, allow_models, self.nfolds)
                model_stacking.hyper_params(self.y_train)
                model_stacking.validation(model_stacking.model(), self.x_train, self.y_train, self.nfolds, self.scoring, self.print_result)
            else:
                self.apply_stacking = False

        # blend model average or vote classifier :
        if self.apply_blend_model:
            if len(self.models.keys()) >= 2:
                print('\n\033[4m' + 'Blend' + ' Model\033[0m', ':' if self.print_result else '...' ,'\n')
                model_blend = BlendModel(self.objective, self.seed)
                model_blend.validation(self.models, self.x_train, self.y_train, self.print_result)
            else:
                self.apply_blend_model = False

        if self.apply_stacking:
            self.models['Stacking'] = model_stacking
        if self.apply_blend_model:
            self.models['BlendModel'] = model_blend

    def get_leaderboard(self, dataset = 'val', sort_by = 'accuracy', ascending = False):
        self.metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
        self.leaderboard = {"name" :list(self.models.keys())}
        for metric in self.metrics:
            self.info_scores[metric + '_' + dataset] = [self.models[name_model].info_scores[metric + '_' + dataset] for name_model in self.models.keys()]
            self.leaderboard[metric + '_' + dataset] = np.round(self.info_scores[metric + '_' + dataset], 4)
        self.leaderboard = pd.DataFrame(self.leaderboard).sort_values(by = sort_by + '_' + dataset, ascending = ascending)
        return self.leaderboard

    def get_df_all_results(self):
        df_all_results = pd.DataFrame()
        for name_model in self.models.keys():
            if name_model not in ['Stacking', 'BlendModel']:
                df_all_results_model = self.models[name_model].df_all_results
                df_all_results_model['model'] = name_model
                df_all_results = pd.concat([df_all_results, df_all_results_model], axis = 0).reset_index(drop=True)
        return df_all_results

    def show_distribution_scores(self):
        df_all_results = self.get_df_all_results()
        list_name_models = list(df_all_results.model.unique())
        rows, cols = 2, 3
        fig, ax = plt.subplots(rows, cols, figsize=(50 ,20))

        for row in range(rows):
            for col in range(cols):
                if row * cols + col + 1 <= len(list_name_models):
                    name_model = list_name_models[row * cols + col]
                    values = df_all_results[df_all_results.model.isin([name_model])].mean_test_score
                    if np.std(values) < 1e-4:
                        ax[row, col].hist(values, range = (values.min( ) -1e-3, values.max( ) +1e-3))
                    else:
                        ax[row, col].hist(values)
                    ax[row, col].set_xlabel(name_model + ' ( ' +str(len(values) ) +' models)', size = 30)
                    ax[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    for tick in ax[row, col].xaxis.get_major_ticks():
                        tick.label.set_fontsize(30)
        plt.show()

    def get_roc_curves(self):
        plt.figure(figsize = (15 ,15), linewidth= 1)
        for name_model, model in self.models.items():
            plt.plot(model.info_scores['fpr'], model.info_scores['tpr'], label = name_model)
        plt.plot([0 ,1], [0 ,1], 'k--', label = 'Random: 0.5')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.savefig('roc_curves.png')
        plt.show()

    def correlation_models(self):
        result_val = pd.DataFrame \
            ({name_model: self.models[name_model].info_scores['oof_val'].reshape(-1) for name_model in self.models.keys()})
        sns.set(rc={'figure.figsize' :(12 ,12)})
        sns.heatmap(result_val.corr(), annot=True, cmap=sns.cm.rocket_r)
        plt.show()

    def leader_predict(self, on_test_data = True, x = None, y = None):
        if on_test_data:  # predict on self.X_test
            for name_model in self.models.keys():
                if name_model == 'BlendModel':
                    self.models[name_model].prediction(self.models, self.X_test, self.Y_test, False)
                else:
                    self.models[name_model].prediction(self.models[name_model].best_model, self.X_test, self.Y_test)

        else:  # predict on x
            for name_model in self.models.keys():
                if name_model == 'BlendModel':
                    self.models[name_model].prediction(self.models, x, y, False)
                else:
                    self.models[name_model].prediction(self.models[name_model].best_model, x, y)

        # Create a dataframe with predictions of each model + y_true
        dict_prediction = {}
        if on_test_data:
            if self.Y_test is not None:
                dict_prediction['y_true'] = np.array(self.Y_test).reshape(-1)
        else:
            if y is not None:
                dict_prediction['y_true'] = np.array(y).reshape(-1)

        for name_model in self.models.keys():
            dict_prediction[name_model] = np.round(self.models[name_model].info_scores['prediction'].reshape(-1) ,3)
        self.dataframe_predictions = pd.DataFrame(dict_prediction)