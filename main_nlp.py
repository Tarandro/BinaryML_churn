import pandas as pd
from binaryML import BinaryML

#####################
# Parameters
#####################

objective = 'text_binary_proba'  # or 'text_binary'            ('binary_proba' : predict proba and 'binary' : predict 0 or 1)
target = 'sentiment'

column_text = 'text'  # (column with texts)

frac = 0.8  # train_test_split fraction                 (data is split in train/test with frac = % for training dataset)

max_run_time_per_modele = 60  # (limit gridsearch time for each model)

scoring = 'f1'  # ['accuracy','f1','recall','precision','roc_auc']
sort_leaderboard = 'f1'  # ['accuracy','f1','recall','precision','roc_auc']

# number of folds during gridsearch and validation :
nfolds = 5

class_weight = True

apply_stacking = False
apply_blend_model = True

# show result:
print_result = True
# list of models to exclude :
exclude_model = ['Fasttext_Attention', 'BERT']   # FastText work only with pre-training dataset on kaggle (see url method_embedding)
                                                      # Need GPU for BERT
# NLP : ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression', 'Fasttext_Attention', 'BERT']

method_embedding = {'Fasttext_Attention': '/kaggle/input/fasttext-french-2b-300d/cc.fr.300.vec',
                    'BERT': 'CamemBERT',
                    'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False),
                              (['ADJ', 'NOUN'], True)]}

if __name__ == '__main__':
    #####################
    # data
    #####################

    data = pd.read_csv('./data/TrustPilot_data.csv')
    data = data[~data.text.isnull()].reset_index(drop=True)
    data['sentiment'] = data.star.map({5: 1, 4: 1, 3: 0, 2: 0, 1: 0})

    bml = BinaryML(scoring=scoring, objective=objective, nfolds=nfolds, class_weight=class_weight,
                   print_result=print_result, max_run_time_per_modele=max_run_time_per_modele,
                   apply_stacking=apply_stacking, apply_blend_model=apply_blend_model, exclude_model=exclude_model,
                   method_embedding=method_embedding)

    #####################
    # Preprocessing
    #####################

    bml.data_preprocessing(data, target=target, column_text=column_text, frac=frac)

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

    if 'binary_proba' in bml.objective:
        bml.get_roc_curves()

    #####################
    # Testing
    #####################

    on_test_data = True
    bml.leader_predict(on_test_data)  # or bml.leader_predict(aml.X_test, aml.Y_test)

    df_prediction = bml.dataframe_predictions

    leaderboard_test = bml.get_leaderboard(sort_by=sort_leaderboard, dataset='test')
    print(leaderboard_test)

