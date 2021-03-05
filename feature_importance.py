from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import shap
import eli5
from eli5.sklearn import PermutationImportance
import streamlit.components.v1 as components
from binaryML import BinaryML
shap.initjs()
"""
data = pd.read_csv('./results/results_tabular/data_preprocessed.csv')
X_train = pd.read_csv('./results/results_tabular/X_train.csv')
X_test = pd.read_csv('./results/results_tabular/X_test.csv')
Y_train = pd.read_csv('./results/results_tabular/Y_train.csv')
Y_test = pd.read_csv('./results/results_tabular/Y_test.csv')
"""
data = pd.read_csv('./data/Churn_Modelling.csv')

objective = 'binary_proba' # or 'binary'                ('binary_proba' : predict proba and 'binary' : predict 0 or 1)
target = 'Exited'

frac = 0.8  # train_test_split fraction                 (data is split in train/test with frac = % for training dataset)

max_run_time_per_model = 20                            # (limit gridsearch time for each model)

scoring = 'f1'  # ['accuracy','f1','recall','precision','roc_auc'] # metric to optimize during gridsearch
sort_leaderboard = 'f1'   # ['accuracy','f1','recall','precision','roc_auc']  # sort dataframe leaderboard by a metric

# number of folds during gridsearch and validation :
nfolds = 5

class_weight = True
method_scaling = 'MinMaxScaler'  # MinMaxScaler, RobustScaler, StandardScaler
method_nan_categorical = None  # 'constant', 'ffill', 'mode'                       (fill na for categorical features)
method_nan_numeric = None      # 'mean', 'ffill', int, 'median'                     (fill na for numeric features)

### Create new features (TSNE, PCA, STATS):
# format for tsne and pca (dict): {name_of_new_feature : ([list of features], n_dimension)}
# if you want to apply on all features : replace [list of features] by 'all'

# format for stats (dict): {name_of_new_feature : (name_method,[list of features])} or ([list of name_methods],[list of features])}
# choice name_method : 'sum' / 'mean' / 'std' / 'kurtosis' / 'skew' / 'multi' / 'div' / 'power'

info_tsne = {}  # {'all':('all',2), 'g':(['Gender','Balance','IsActiveMember','EstimatedSalary'],2)}
info_pca = {}  # {'all':('all',2), 'g':(['Gender','Balance','IsActiveMember','EstimatedSalary'],2)}
info_stats = {}
# {'stat_1':('sum',['CreditScore','Balance','EstimatedSalary']), 'stat_2':(['sum','mean'],(['CreditScore','Gender','Balance','EstimatedSalary']))}
# {'multi_1':('multi',['NumOfProducts','HasCrCard','IsActiveMember']), 'div_1': ('div',['EstimatedSalary','CreditScore'])}
# {'power_1': ('power',['Age','Balance'])}

# For numeric features, remove features with variance > percentile(remove_percentage) :
remove_low_variance = False
remove_percentage = 0.8
# For feature pairs with correlation > multicollinearity_threshold, remove the one with the lowest importance in RandomForest classification :
remove_multicollinearity = False
multicollinearity_threshold = 0.9
# Keep only feature_selection_threshold*100 % features with highest importances :
feature_selection = False
feature_selection_threshold = 0.8

apply_stacking = True
apply_blend_model = True

list_threshold_1 = [0.45, 0.5, 0.55] # threshold for probability of 1, use for validation leaderboard
thr_1_test = 0.5 # threshold for probability of 1

# show result:
print_result = False
# list of models to exclude :
exclude_model = ['SimpleNeuralNetwork']  # ['Logistic_Regression', 'Random_Forest', 'LightGBM', 'XGBoost', 'CatBoost', 'SimpleNeuralNetwork']


bml = BinaryML(scoring=scoring, objective=objective, nfolds=nfolds, class_weight=class_weight,
               print_result=print_result, max_run_time_per_model=max_run_time_per_model,
               apply_stacking=apply_stacking, apply_blend_model=apply_blend_model,
               exclude_model=exclude_model,
               method_scaling=method_scaling)

bml.data_preprocessing(data, target=target, frac=frac,
                       method_nan_categorical=method_nan_categorical, method_nan_numeric=method_nan_numeric,
                       info_pca=info_pca, info_tsne=info_tsne, info_stats=info_stats,
                       remove_low_variance=remove_low_variance, remove_percentage=remove_percentage,
                       remove_multicollinearity=remove_multicollinearity,
                       multicollinearity_threshold=multicollinearity_threshold,
                       feature_selection=feature_selection, feature_selection_threshold=feature_selection_threshold)

data.to_csv('./results/results_tabular/data.csv', index=False)
bml.data.to_csv('./results/results_tabular/data_preprocessed.csv', index=False)
bml.X_test.to_csv('./results/results_tabular/X_test.csv', index=False)
bml.Y_train.to_csv('./results/results_tabular/Y_train.csv', index=False)
bml.Y_test.to_csv('./results/results_tabular/Y_test.csv', index=False)



if print_result:
    bml.pre.print_feature_importances()
    bml.pre.print_feature_correlation()

#####################
# Training
#####################

# bml.train() uncomment pour le final


models = ['Logistic_Regression', 'Random_Forest', 'LightGBM', 'XGBoost', 'CatBoost']

# model_fi = RandomForestClassifier(max_depth=7, n_estimators=150, class_weight='balanced')
# model_fi.fit(X_train, Y_train.values.ravel())


""" Feature importance """

""" Shap """


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


""" Specific row """
"""
row_to_show = 0

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
# Create object that can calculate shap values
model_fi.predict_proba(data_for_prediction_array)

# Calculate Shap values
shap_values_row = explainer.shap_values(data_for_prediction_array)
"""