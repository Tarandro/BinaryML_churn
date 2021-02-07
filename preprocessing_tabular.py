import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random as rd

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


#####################
# Reduce memory usage
#####################

def reduce_mem_usage(df, verbose=True):
    """ Technique to reduce memory usage of dataframe """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


####################
# data preparation
####################

def interpolate_missing_data_categorical(data, method):
    """Interpolate missing data and return as pandas data frame."""

    if method == "constant":
        return data.fillna('not_available')
    elif method == 'ffill':
        return data.fillna(method='ffill').fillna(method='bfill')
    else:
        return data.fillna(data.mode())


def interpolate_missing_data_numeric(data, method):
    """Interpolate missing data and return as pandas data frame."""

    def is_number(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    if method == "mean":
        return data.fillna(data.mean())
    elif method == 'ffill':
        return data.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        return data.interpolate()
    elif is_number(str(method)):
        return data.fillna(float(method))
    else:
        return data.fillna(data.median())


def one_hot_encode(data):
    """ Perform a one-hot encoding and return only n-1 columns (avoid multicorrelation) """
    return pd.get_dummies(data).iloc[:, :-1]


def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        if s == 'nan':
            return False
        float(s)
        return True
    except ValueError:
        return False


###########################################
# Feature Importance
###########################################

def get_features_importance(data, Y, subsample, class_weight):
    """ Apply RandomForest and get features importances """
    index_sample = rd.sample(list(data.index), int(len(data) * subsample))
    clf = RandomForestClassifier(random_state=15, class_weight=class_weight, n_estimators=50, max_samples=0.8)
    clf.fit(data.loc[index_sample, :], Y.loc[index_sample, :])
    return clf.feature_importances_


def find_optimal_number(X):
    """ Try different k for KMEANS and choose best k according to silhouette_score """
    sil_score_max = 0
    from sklearn.metrics import silhouette_score
    for n_clusters in range(3, 15):
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1, random_state=0)
        labels = model.fit_predict(X)
        sil_score = silhouette_score(X, labels)
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    return best_n_clusters


#############################
#############################
#############################

class Preprocessing:

    def __init__(self, data, target=None, type_columns=None):
        self.data = data

        self.target = target
        if isinstance(self.target, list):
            self.target = self.target
        else:
            self.target = [self.target]

        self.Y = self.data[[col for col in self.target if col in self.data.columns]]
        for col in self.target:
            if col in self.data.columns:
                self.data = self.data.drop([col], axis=1)

        if type_columns == None:
            self.type_columns = {
                'numeric': list(self.data.loc[:, self.data.dtypes.astype(str).isin(
                    ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float32',
                     'float64'])].columns),
                'categorical': list(
                    self.data.loc[:, self.data.dtypes.astype(str).isin(['O', 'object', 'category', 'bool'])].columns),
                'date': list(self.data.loc[:,
                             self.data.dtypes.astype(str).isin(['datetime64', 'datetime64[ns]', 'datetime'])].columns)
            }
        else:
            self.type_columns = type_columns

        self.base_features = list(self.data.columns)  # useful for pca / tsne in case of categorical features

    def preprocessing_mandatory(self):
        """ 3 steps :  1. change format according to numeric or categorical type
                       2. fillna in  each column according to numeric or categorical type
                       3. one hot encode column if categorical column """
        # 1st step
        for col in self.type_columns['numeric']:
            if self.data[col].dtypes not in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                             'float32', 'float64']:
                self.data[col] = self.data[col].astype('float32')

        for col in self.type_columns['categorical']:
            if self.data[col].dtypes in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                         'float32', 'float64']:
                self.data[col] = self.data[col].astype(str)

        # 2nd step
        for col in self.data.columns:
            if col in self.type_columns['categorical']:
                if self.data[col].isnull().sum() > 0:
                    self.data[col] = interpolate_missing_data_categorical(self.data[col], self.method_nan_categorical)
            if col in self.type_columns['numeric']:
                if self.data[col].isnull().sum() > 0:
                    self.data[col] = interpolate_missing_data_numeric(self.data[col], self.method_nan_numeric)

                    # 3rd step
        for col_categorical in self.type_columns['categorical']:
            if col_categorical in self.data.columns:
                self.apply_dummies = True
                break

        if self.apply_dummies:
            self.start_data = self.data.copy()
            self.info_feat_categorical = {}

            for col_categorical in self.type_columns['categorical']:
                # Perform a one-hot encoding and return only n-1 columns (avoid multicorrelation) :
                dummies = one_hot_encode(self.data[col_categorical])
                if dummies.shape[1] == 1:
                    dummies.columns = [col_categorical + '_' + dummies.columns[0]]
                index_column = list(self.data.columns).index(col_categorical)
                order_columns = list(self.data.columns)[:index_column] + list(dummies.columns) + list(
                    self.data.columns)[(index_column + 1):]
                self.data = pd.concat([dummies, self.data.drop([col_categorical], axis=1)], axis=1)
                self.data = self.data[order_columns]
                self.info_feat_categorical[col_categorical] = list(dummies.columns)

    def preprocessing_mandatory_transform(self, data_test):
        """ Previous function but for transforming data_test """

        for col in self.type_columns['numeric']:
            if data_test[col].dtypes not in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                             'float32', 'float64']:
                data_test[col] = data_test[col].astype('float32')

        for col in self.type_columns['categorical']:
            if data_test[col].dtypes in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
                                         'float32', 'float64']:
                data_test[col] = data_test[col].astype(str)

        for col in data_test.columns:
            if col in self.type_columns['categorical']:
                if data_test[col].isnull().sum() > 0:
                    data_test[col] = interpolate_missing_data_categorical(data_test[col], self.method_nan_categorical)
            if col in self.type_columns['numeric']:
                if data_test[col].isnull().sum() > 0:
                    data_test[col] = interpolate_missing_data_numeric(data_test[col], self.method_nan_numeric)

        if self.apply_dummies:
            concat_data = pd.concat([self.start_data, data_test], axis=0, ignore_index=True)

            for col_categorical in self.type_columns['categorical']:
                dummies = one_hot_encode(concat_data[col_categorical])
                dummies_parallel = one_hot_encode(self.start_data[col_categorical])
                if dummies.shape[1] == 1:
                    dummies.columns = [col_categorical + '_' + dummies.columns[0]]
                if dummies_parallel.shape[1] == 1:
                    dummies_parallel.columns = [col_categorical + '_' + dummies_parallel.columns[0]]

                for col in dummies.columns:
                    if col not in dummies_parallel.columns:
                        dummies = dummies.drop([col], axis=1)

                if len(dummies.columns) != len(dummies_parallel.columns):
                    print('error in one_hot encoding categorical')

                index_column = list(concat_data.columns).index(col_categorical)
                order_columns = list(concat_data.columns)[:index_column] + list(dummies.columns) + list(
                    concat_data.columns)[(index_column + 1):]
                concat_data = pd.concat([dummies, concat_data.drop([col_categorical], axis=1)], axis=1)
                concat_data = concat_data[order_columns]

            data_test = concat_data[len(self.start_data):]
        return data_test

    def build_fe_pca(self):

        def create_pca(data_, kind, features, n_components):
            """ Function to extract pca features from data_[features] """
            if features == 'all':
                features = self.base_features
            true_features = []
            for col in features:
                if col in self.info_feat_categorical.keys():
                    true_features = true_features + self.info_feat_categorical[col]
                else:
                    true_features.append(col)
            features = [col for col in true_features if col in data_.columns]
            data = data_[features].copy()
            if data.shape[1] > n_components:
                pca = PCA(n_components=n_components, random_state=15)
                data = pca.fit_transform(data)
                columns = [f'pca_{kind}{i + 1}' for i in range(data.shape[1])]
                data = pd.DataFrame(data, columns=columns)
                data = pd.concat([data_, data], axis=1)
                del data_
                return data, features, pca, columns
            else:
                del data
                return data_, features, 0, []

        self.info_pca_for_transform = {}
        for name in self.info_pca.keys():
            self.data, features, pca, columns = create_pca(self.data, name, self.info_pca[name][0],
                                                           self.info_pca[name][1])
            if len(columns) > 0:
                self.info_pca_for_transform[name] = (features, pca, columns)

    def build_fe_pca_transform(self, data_test):
        for name in self.info_pca_for_transform.keys():
            data = data_test[self.info_pca_for_transform[name][0]].copy()
            pca = self.info_pca_for_transform[name][1]
            data = pca.transform(data)
            data = pd.DataFrame(data, columns=self.info_pca_for_transform[name][2])
            data_test = pd.concat([data_test, data], axis=1)
        return data_test

    def build_fe_tsne(self):

        def create_tsne(data_, kind, features, n_components):
            """ Function to extract tsne features from data_[features] """
            if features == 'all':
                features = self.base_features
            true_features = []
            for col in features:
                if col in self.info_feat_categorical.keys():
                    true_features = true_features + self.info_feat_categorical[col]
                else:
                    true_features.append(col)
            features = [col for col in true_features if col in data_.columns]
            data = data_[features].copy()
            if data.shape[1] > n_components:
                tsne = TSNE(n_components=n_components, random_state=15, verbose=0)
                data = tsne.fit_transform(data)
                columns = [f'tsne_{kind}{i + 1}' for i in range(data.shape[1])]
                data = pd.DataFrame(data, columns=columns)
                data = pd.concat([data_, data], axis=1)
                del data_
                return data, features, tsne, columns
            else:
                del data
                return data_, features, 0, []

        self.info_tsne_for_transform = {}
        for name in self.info_tsne.keys():
            self.data, features, tsne, columns = create_tsne(self.data, name, self.info_tsne[name][0],
                                                             self.info_tsne[name][1])
            if len(columns) > 0:
                self.info_tsne_for_transform[name] = (features, tsne, columns)

    def build_fe_tsne_transform(self, data_test):
        for name in self.info_tsne_for_transform.keys():
            data = data_test[self.info_tsne_for_transform[name][0]].copy()
            tsne = self.info_tsne_for_transform[name][1]
            data = tsne.transform(data)
            data = pd.DataFrame(data, columns=self.info_tsne_for_transform[name][2])
            data_test = pd.concat([data_test, data], axis=1)
        return data_test

    def build_fe_stats(self):
        """ build features stats : sum / mean / std / kurtosis / skew / multi / div / power """

        self.info_stats_for_transform = {}
        for name in self.info_stats.keys():

            features = self.info_stats[name][1]
            if features == 'all':
                features = self.base_features
            true_features = []
            for col in features:
                if col in self.info_feat_categorical.keys():
                    true_features = true_features + self.info_feat_categorical[col]
                else:
                    true_features.append(col)
            features = [col for col in true_features if col in self.data.columns]

            if type(self.info_stats[name][0]) is list:
                method = self.info_stats[name][0]
            else:
                method = [self.info_stats[name][0]]
            if len(features) >= 2:
                if 'sum' in method:
                    self.data['sum_' + name] = self.data[features].sum(axis=1)
                    self.info_stats_for_transform[name] = ('sum', features, 'sum_' + name)
                if 'mean' in method:
                    self.data['mean_' + name] = self.data[features].mean(axis=1)
                    self.info_stats_for_transform[name] = ('mean', features, 'mean_' + name)
                if 'std' in method:
                    self.data['std_' + name] = self.data[features].std(axis=1)
                    self.info_stats_for_transform[name] = ('std', features, 'std_' + name)
                if 'kurtosis' in method:
                    self.data['kurtosis_' + name] = self.data[features].kurtosis(axis=1)
                    self.info_stats_for_transform[name] = ('kurtosis', features, 'kurtosis_' + name)
                if 'skew' in method:
                    self.data['skew_' + name] = self.data[features].skew(axis=1)
                    self.info_stats_for_transform[name] = ('skew', features, 'skew_' + name)

                if 'multi' in method:
                    data_multi = self.data[features[0]]
                    for col in features[1:]:
                        data_multi = data_multi * self.data[features[1]]
                    name_column = 'multi_' + '_'.join(features)
                    self.data[name_column[0:60]] = data_multi
                    self.info_stats_for_transform[name] = ('multi', features, name_column[0:60])

                if 'div' in method:
                    if len(features) == 2:
                        self.data[features[0] + '_ratio_' + features[1]] = np.round(
                            self.data[features[0]] / (self.data[features[1]] + 0.001), 3)
                        self.info_stats_for_transform[name] = ('div', features, features[0] + '_ratio_' + features[1])

                if 'power' in method:
                    for col in features:
                        self.data[col + '_power2'] = self.data[col] ** 2
                        self.data = self.data.drop([col], axis=1)
                    self.info_stats_for_transform[name] = ('power', features, col + '_power2')

    def build_fe_stats_transform(self, data_test):
        for name in self.info_stats_for_transform.keys():
            method, features, col_name = self.info_stats_for_transform[name]
            if 'sum' in method:
                data_test[col_name] = data_test[features].sum(axis=1)
            if 'mean' in method:
                data_test[col_name] = data_test[features].mean(axis=1)
            if 'std' in method:
                data_test[col_name] = data_test[features].std(axis=1)
            if 'kurtosis' in method:
                data_test[col_name] = data_test[features].kurtosis(axis=1)
            if 'skew' in method:
                data_test[col_name] = data_test[features].skew(axis=1)

            if 'multi' in method:
                data_multi = data_test[features[0]]
                for col in features[1:]:
                    data_multi = data_multi * data_test[features[1]]
                data_test[col_name] = data_multi

            if 'div' in method:
                if len(features) == 2:
                    data_test[col_name] = np.round(data_test[features[0]] / (data_test[features[1]] + 0.001), 3)

            if 'power' in method:
                for col in features:
                    data_test[col_name] = data_test[col] ** 2
                    data_test = data_test.drop([col], axis=1)

        return data_test

    def feature_selection_VarianceThreshold(self, remove_percentage=0.8):
        """ For numeric features, remove features with variance > percentile(remove_percentage) """

        Numeric_features = list(self.data.loc[:, self.data.dtypes.astype(str).isin(
            ['uint8', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64'])].columns)
        Numeric_features = [col for col in Numeric_features if col not in list(self.info_feat_categorical.values())]

        thresh = np.percentile([np.var(self.data[col]) for col in Numeric_features], remove_percentage)

        self.columns_to_drop_variance = []
        for col in Numeric_features:
            if np.var(self.data[col]) <= thresh:
                self.columns_to_drop_variance.append(col)
        print('columns remove due to low variance :\n')
        print(self.columns_to_drop_variance)
        self.data = self.data.drop(self.columns_to_drop_variance, axis=1)

    def feature_selection_VarianceThreshold_transform(self, data_test):
        data_test = data_test.drop(self.columns_to_drop_variance, axis=1)
        return data_test

    def fct_remove_multicollinearity(self, multicollinearity_threshold=0.9):
        """ For feature pairs with correlation > multicollinearity_threshold, remove the one with the lowest importance in RandomForest classification """
        name_columns = self.data.columns
        matrix_corr = np.array(self.data.corr())

        features_importance = get_features_importance(self.data, self.Y, self.subsample, self.class_weight)

        self.columns_to_drop_multicollinearity = []
        for i in range(len(name_columns)):
            for j in range(i + 1, len(name_columns)):
                if name_columns[i] not in self.columns_to_drop_multicollinearity and name_columns[
                    j] not in self.columns_to_drop_multicollinearity:
                    cor_ij = matrix_corr[i][j]
                    if np.abs(cor_ij) > multicollinearity_threshold:
                        importance_i = features_importance[list(name_columns).index(name_columns[i])]
                        importance_j = features_importance[list(name_columns).index(name_columns[j])]

                        if importance_i >= importance_j:
                            self.columns_to_drop_multicollinearity.append(name_columns[j])
                        else:
                            self.columns_to_drop_multicollinearity.append(name_columns[i])
        print('columns remove due to high multicollinearity :\n')
        print(self.columns_to_drop_multicollinearity)
        self.data = self.data.drop(self.columns_to_drop_multicollinearity, axis=1)

    def fct_remove_multicollinearity_transform(self, data_test):
        data_test = data_test.drop(self.columns_to_drop_multicollinearity, axis=1)
        return data_test

    def select_feature_by_importance(self, feature_selection_threshold=0.8):
        """ Keep only feature_selection_threshold*100 % features with highest importances """
        name_columns = self.data.columns

        features_importance = get_features_importance(self.data, self.Y, self.subsample, self.class_weight)

        sorted_idx = features_importance.argsort()
        name_columns_sorted = name_columns[sorted_idx][::-1]

        nb_column_to_keep = int(feature_selection_threshold * len(name_columns))
        print('\ncolumns remove due to low importance value :\n')
        self.columns_to_drop_importance = list(name_columns_sorted)[nb_column_to_keep:]
        print(self.columns_to_drop_importance)
        self.data = self.data.drop(self.columns_to_drop_importance, axis=1)

    def select_feature_by_importance_transform(self, data_test):
        data_test = data_test.drop(self.columns_to_drop_importance, axis=1)
        return data_test

    def print_feature_importances(self):
        features_importance = get_features_importance(self.data, self.Y, self.subsample, self.class_weight)
        sorted_idx = features_importance.argsort()
        plt.barh(self.data.columns[sorted_idx], features_importance[sorted_idx])
        plt.show()

    def print_feature_correlation(self):
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.heatmap(self.data.corr(), annot=True, cmap=sns.cm.rocket_r)
        plt.show()

    def fit_transform(self, remove_multicollinearity=False, multicollinearity_threshold=0.9, feature_selection=False,
                      feature_selection_threshold=0.8,
                      class_weight=None, method_nan_categorical='constant', method_nan_numeric='mean',
                      subsample=1, info_pca={}, info_tsne={}, info_stats={}, remove_low_variance=False,
                      remove_percentage=0.8):

        self.method_nan_categorical = method_nan_categorical
        self.method_nan_numeric = method_nan_numeric
        self.class_weight = class_weight
        self.subsample = subsample  ### % of train data for training RandomForest and calculate features importances

        self.remove_multicollinearity = remove_multicollinearity
        self.feature_selection = feature_selection
        self.remove_low_variance = remove_low_variance
        self.info_pca = info_pca
        self.info_tsne = info_tsne
        self.info_stats = info_stats

        self.preprocessing_mandatory()

        # Create features :
        if len(self.info_tsne.keys()) > 0:
            self.build_fe_tsne()
        if len(self.info_pca.keys()) > 0:
            self.build_fe_pca()
        if len(self.info_stats.keys()) > 0:
            self.build_fe_stats()

        # Remove features :
        if self.remove_low_variance:
            self.feature_selection_VarianceThreshold(remove_percentage)
        if self.remove_multicollinearity:
            self.fct_remove_multicollinearity(multicollinearity_threshold)
        if self.feature_selection:
            self.select_feature_by_importance(feature_selection_threshold)

        return self.data

    def transform(self, data_test):
        has_target = False
        if self.target is not None:
            Y_test = data_test[[col for col in self.target if col in data_test.columns]]
            has_target = True

        data_test = self.preprocessing_mandatory_transform(data_test)

        if len(self.info_tsne.keys()) > 0:
            data_test = self.build_fe_tsne_transform()
        if len(self.info_pca.keys()) > 0:
            data_test = self.build_fe_pca_transform()
        if len(self.info_stats.keys()) > 0:
            data_test = self.build_fe_stats_transform()

        if self.remove_low_variance:
            data_test = self.feature_selection_VarianceThreshold_transform(data_test)
        if self.remove_multicollinearity:
            data_test = self.fct_remove_multicollinearity_transform(data_test)
        if self.feature_selection:
            data_test = self.select_feature_by_importance_transform(data_test)

        for col in data_test.columns:
            if col not in self.data.columns:
                print(col, 'is not in original data')

        for col in self.data.columns:
            if col not in data_test.columns:
                print(col, 'is not in test data')

        if has_target:
            data_test[self.target] = Y_test[self.target]

        return data_test


