import pandas as pd
from binaryML_tabular import BinaryML_tabular

#####################
# Parameters
#####################

objective = 'binary_proba' # or 'binary'                ('binary_proba' : predict proba and 'binary' : predict 0 or 1)
target = 'Exited'

frac = 0.8  # train_test_split fraction                 (data is split in train/test with frac = % for training dataset)

max_run_time_per_modele = 20                            # (limit gridsearch time for each model)

scoring = 'f1'  # ['accuracy','f1','recall','precision','roc_auc']
sort_leaderboard = 'f1'   # ['accuracy','f1','recall','precision','roc_auc']

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

# show result:
print_result = True
# list of models to exclude :
exclude_model = []  # ['Logistic_Regression', 'Random_Forest', 'LightGBM', 'XGBoost', 'CatBoost', 'SimpleNeuralNetwork']

if __name__ == '__main__':
    #####################
    # data
    #####################

    data = pd.read_csv('./data/Churn_Modelling.csv')
    data = data.iloc[:, 3:]

    bml = BinaryML_tabular(scoring=scoring, objective=objective, nfolds=nfolds, class_weight=class_weight,
                           print_result=print_result, max_run_time_per_modele=max_run_time_per_modele,
                           apply_stacking=apply_stacking, apply_blend_model=apply_blend_model,
                           exclude_model=exclude_model,
                           method_scaling=method_scaling)

    #####################
    # Preprocessing
    #####################

    bml.data_preprocessing(data, target=target, frac=frac,
                           method_nan_categorical=method_nan_categorical, method_nan_numeric=method_nan_numeric,
                           info_pca=info_pca, info_tsne=info_tsne, info_stats=info_stats,
                           remove_low_variance=remove_low_variance, remove_percentage=remove_percentage,
                           remove_multicollinearity=remove_multicollinearity,
                           multicollinearity_threshold=multicollinearity_threshold,
                           feature_selection=feature_selection, feature_selection_threshold=feature_selection_threshold)
    if print_result:
        bml.pre.print_feature_importances()
        bml.pre.print_feature_correlation()

    #####################
    # Training
    #####################

    bml.train()

    #####################
    # Ensemble
    #####################

    bml.ensemble()

    #####################
    # Leaderboard (Validation score)
    #####################

    leaderboard_val = bml.get_leaderboard(sort_by = sort_leaderboard, dataset = 'val')
    print(leaderboard_val)

    bml.correlation_models()
    df_all_results = bml.get_df_all_results()
    bml.show_distribution_scores()

    df_oof_val = bml.Y_train
    for name in bml.models.keys():
        df_oof_val[name] = bml.models[name].info_scores['oof_val']

    if bml.objective == 'binary_proba':
        bml.get_roc_curves()

    #####################
    # Testing
    #####################

    on_test_data = True
    bml.leader_predict(on_test_data)  # or bml.leader_predict(aml.X_test, aml.Y_test)

    df_prediction = bml.dataframe_predictions

    leaderboard_test = bml.get_leaderboard(sort_by=sort_leaderboard, dataset='test')
    print(leaderboard_test)
