from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import shap
import eli5
from eli5.sklearn import PermutationImportance
import streamlit.components.v1 as components

shap.initjs()

data = pd.read_csv('./results/results_tabular/data_preprocessed.csv')
X_train = pd.read_csv('./results/results_tabular/X_train.csv')
X_test = pd.read_csv('./results/results_tabular/X_test.csv')
Y_train = pd.read_csv('./results/results_tabular/Y_train.csv')
Y_test = pd.read_csv('./results/results_tabular/Y_test.csv')
model_fi = RandomForestClassifier(max_depth=7, n_estimators=150, class_weight='balanced')
model_fi.fit(X_train, Y_train.values.ravel())


""" Feature importance """


perm = PermutationImportance(model_fi, random_state=15, scoring = 'f1').fit(X_test, Y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


""" Shap """

X_shap = data.sample(frac=0.01, random_state=15)
explainer = shap.TreeExplainer(model_fi)
shap_values = explainer.shap_values(X_shap)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


""" Specific row """

row_to_show = 0

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
# Create object that can calculate shap values
model_fi.predict_proba(data_for_prediction_array)

# Calculate Shap values
shap_values_row = explainer.shap_values(data_for_prediction_array)
