import pandas as pd
from binaryML_tabular import BinaryML_tabular

#####################
# Parameters
#####################

objective = 'binary_proba' # or 'binary'
target = 'Exited'

frac = 0.8  # train_test_split fraction

max_run_time_per_modele = 20

scoring = 'f1'  # ['accuracy','f1','recall','precision','roc_auc']
sort_leaderboard = 'f1'   # ['accuracy','f1','recall','precision','roc_auc']

nfolds = 5

class_weight = True
method_scaling = 'MinMaxScaler'  # MinMaxScaler, RobustScaler, StandardScaler
method_nan_categorical = None  # 'constant', 'ffill', 'mode'
method_nan_numeric = None      # 'mean', 'ffill', int, 'median'

info_tsne = {}  # {'all':('all',2), 'g':(['Gender','Balance','IsActiveMember','EstimatedSalary'],2)}
info_pca = {}  # {'all':('all',2), 'g':(['Gender','Balance','IsActiveMember','EstimatedSalary'],2)}
info_stats = {}
# {'stat_1':('sum',['CreditScore','Balance','EstimatedSalary']), 'stat_2':(['sum','mean'],(['CreditScore','Gender','Balance','EstimatedSalary']))}
# {'multi_1':('multi',['NumOfProducts','HasCrCard','IsActiveMember']), 'div_1': ('div',['EstimatedSalary','CreditScore'])}
# {'power_1': ('power',['Age','Balance'])}

remove_low_variance = False  # remove_percentage
remove_multicollinearity = False  # multicollinearity_threshold
feature_selection = False  # feature_selection_threshold
subsample = 1

apply_stacking = True
apply_blend_model = True

print_result = False
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
                           remove_multicollinearity=remove_multicollinearity, feature_selection=feature_selection,
                           subsample=subsample,
                           info_pca=info_pca, info_tsne=info_tsne, info_stats=info_stats,
                           remove_low_variance=remove_low_variance,
                           method_nan_categorical=method_nan_categorical, method_nan_numeric=method_nan_numeric)
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
