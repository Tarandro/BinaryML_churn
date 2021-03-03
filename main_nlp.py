import pandas as pd
from binaryML import BinaryML
from extraction_words import *

#####################
# Parameters
#####################

objective = 'text_binary_proba'  # or 'text_binary'       ('binary_proba' : predict proba and 'binary' : predict 0 or 1)
target = 'sentiment'

column_text = 'text'  # (column with texts)

frac = 0.8  # train_test_split fraction                 (data is split in train/test with frac = % for training dataset)

max_run_time_per_model = 30  # (limit gridsearch time for each model)

scoring = 'f1'  # ['accuracy','f1','recall','precision','roc_auc']  # metric to optimize during gridsearch
sort_leaderboard = 'f1'  # ['accuracy','f1','recall','precision','roc_auc']  # sort dataframe leaderboard by a metric

# number of folds during gridsearch and validation :
nfolds = 5

class_weight = True

apply_stacking = False
apply_blend_model = True

thr_1_test = 0.5 # threshold for probability of 1

# show result:
print_result = True
# list of models to exclude :
exclude_model = ['Fasttext_Attention', 'BERT']   # FastText work only with pre-training dataset on kaggle (see url method_embedding)
                                                      # Need GPU for BERT
# NLP models: ['tf-idf+Naive_Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic_Regression', 'Fasttext_Attention', 'BERT']

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
                   print_result=print_result, max_run_time_per_model=max_run_time_per_model,
                   apply_stacking=apply_stacking, apply_blend_model=apply_blend_model, exclude_model=exclude_model,
                   method_embedding=method_embedding)

    #####################
    # Preprocessing
    #####################

    bml.data_preprocessing(data, target=target, column_text=column_text, frac=frac)

    data.to_csv('./results/results_nlp/data.csv', index=False)
    bml.data.to_csv('./results/results_nlp/data_preprocessed.csv', index=False)
    bml.X_train.to_csv('./results/results_nlp/X_train.csv', index=False)
    bml.X_test.to_csv('./results/results_nlp/X_test.csv', index=False)
    bml.Y_train.to_csv('./results/results_nlp/Y_train.csv', index=False)
    bml.Y_test.to_csv('./results/results_nlp/Y_test.csv', index=False)

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
    print('\nValidation Leaderboard (threshold = 0.5)')
    print(leaderboard_val)
    leaderboard_val.to_csv('./results/results_nlp/leaderboard_val.csv', index=False)

    bml.correlation_models()
    df_all_results = bml.get_df_all_results()
    df_all_results.to_csv('./results/results_nlp/df_all_results.csv', index=False)

    df_all_results_mean = df_all_results.groupby('model').mean().sort_values('mean_test_score', ascending=False)
    print('\nGridSearch information Leaderboard')
    print(df_all_results_mean)
    df_all_results.to_csv('./results/results_nlp/df_all_results_mean.csv', index=False)
    bml.show_distribution_scores()

    df_oof_val = bml.Y_train.copy()
    for name in bml.models.keys():
        df_oof_val[name] = bml.models[name].info_scores['oof_val']
    df_oof_val.to_csv('./results/results_nlp/df_oof_val.csv', index=False)

    if 'binary_proba' in bml.objective:
        bml.get_roc_curves()

    #####################
    # Testing
    #####################

    on_test_data = True
    bml.leader_predict(on_test_data, thr_1 = thr_1_test)  # or bml.leader_predict(aml.X_test, aml.Y_test)

    df_prediction = bml.dataframe_predictions
    df_prediction.to_csv('./results/results_nlp/df_prediction.csv', index=False)

    leaderboard_test = bml.get_leaderboard(sort_by=sort_leaderboard, dataset='test')
    print('\nTest Leaderboard (threshold = ' + str(thr_1_test) + ')')
    print(leaderboard_test)
    leaderboard_test.to_csv('./results/results_nlp/leaderboard_test.csv', index=False)

    ##################
    # Extraction words
    ##################
    pr = {0: 'NEGATIVE', 1: 'POSITIVE'}
    n_influent_word = 10
    type_data = 'train'  # 'test'

    if 'Fasttext_Attention' in bml.models.keys() or 'BERT' in bml.models.keys():
        html = extract_influent_word(bml, type_data, n_influent_word, pr)

        Html_file = open("./results/results_nlp/extract_word.html", "w")
        Html_file.write(html)
        Html_file.close()

