import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from PIL import Image
from eda_utils import *
from feature_importance import *
from pdpbox import pdp

st.set_option('deprecation.showPyplotGlobalUse', False)
### run application : streamlit run streamlit_tabular.py

################
# Load data
################


st.header('BinaryML Tabular Visualization')

data = pd.read_csv('./results/results_tabular/data.csv')
data_preprocessed = pd.read_csv('./results/results_tabular/data_preprocessed.csv')
Y_train = pd.read_csv('./results/results_tabular/Y_train.csv')

df_all_results = pd.read_csv('./results/results_tabular/df_all_results.csv')

leaderboard_val = pd.read_csv('./results/results_tabular/leaderboard_val.csv')
leaderboard_test = pd.read_csv('./results/results_tabular/leaderboard_test.csv')

oof_val = pd.read_csv('./results/results_tabular/df_oof_val.csv')

try:
    roc_curves = Image.open('./roc_curves.png')
    SP = Image.open('./SP.PNG')
    FI = Image.open('./PI.PNG')
except:
    pass

Section = st.sidebar.radio(
    'Section :', ['Score', 'Data', 'Machine Learning explainability'])

if Section == 'Score':

    """ Validation score """
    leaderboard_val

    """ Test score """
    leaderboard_test

    """ Distribution validation score """
    list_name_models = list(df_all_results.model.unique())
    rows, cols = 2, 3
    fig, ax = plt.subplots(rows, cols, figsize=(50,20))

    for row in range(rows):
        for col in range(cols):
            if row * cols + col + 1 <= len(list_name_models):
                name_model = list_name_models[row * cols + col]
                values = df_all_results[df_all_results.model.isin([name_model])].mean_test_score
                if np.std(values) < 1e-4:
                    ax[row, col].hist(values, range = (values.min()-1e-3, values.max()+1e-3))
                else:
                    ax[row, col].hist(values)
                ax[row, col].set_xlabel(name_model + ' ('+str(len(values))+' models)', size = 30)
                ax[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                for tick in ax[row, col].xaxis.get_major_ticks():
                    tick.label.set_fontsize(30)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(oof_val.drop(['BlendModel', 'Stacking'], axis=1).corr(), annot=True, cmap=sns.cm.rocket_r)
    st.write(fig)

    """ Roc Curves"""
    st.image(roc_curves, use_column_width = True)

elif Section =="Data":

    """ Data provided """
    data

    """ Data preprocessed """
    data_preprocessed

    st.write('Predicted value :', ", ".join([col for col in Y_train.columns]))


    """Corrélation entre variables"""
    sns.set(rc={'figure.figsize': (12, 12)})
    heatmap = sns.heatmap(data.corr(), annot=True)
    fig = heatmap.get_figure()
    st.pyplot(fig)


    """Plots"""
    """Chart pour chaque feature"""
    showChart1 = st.checkbox("Afficher feature chart")
    if showChart1:
        st.plotly_chart(subplot_hist(data))

    """Comparaison churn VS non-churn"""
    # data_sample_names = ['base', 'kept', 'churn']
    # graph1 = st.selectbox(
    #     'Choisir le dataset de gauche :',
    #     data_sample_names
    # )
    # graph2 = st.selectbox(
    #     'Choisir le dataset de droite :',
    #     data_sample_names
    # )

    graph1 = 'kept'
    graph2 = 'churn'
    lost_clients = data[data['Exited'] == 1]
    kept_clients = data[data['Exited'] == 0]

    list_of_cat_var = ['Gender', 'Geography', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']
    list_of_cont_var = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    var_selec = list_of_cat_var + list_of_cont_var

    var_choice1 = st.selectbox(
        'Choisir une variable à comparer',
        var_selec
    )

    showChart2 = st.checkbox("Afficher comparison chart")
    if showChart2:
        #comp_data1 = choose_datase(graph1, data_sample_names, [data, kept_clients, lost_clients])
        #comp_data2 = choose_datase(graph2, data_sample_names, [data, kept_clients, lost_clients])

        if var_choice1 in ['Age', 'CreditScore', 'EstimatedSalary', 'Tenure'] :
            st.plotly_chart(hist_2distrib(kept_clients, lost_clients, var_choice1))

        elif var_choice1 == 'Balance':
            st.plotly_chart(hist_2distrib(kept_clients[kept_clients['Balance']>0], lost_clients[lost_clients['Balance']>0], 'Balance'))

        elif var_choice1 in ['NumOfProducts', 'IsActiveMember', 'HasCrCard', 'Gender','Geography']:
            new_title = var_choice1 + ' by group'
            st.plotly_chart(barplot_comp(data, var_choice1, title=new_title))

    """Part de churn en fonction de la valeur d'une variable"""
    var_choice2 = st.selectbox(
        'Choisir une variable à étudier',
        var_selec
    )
    if var_choice2 in list_of_cat_var:
        countdata = barplot_countdata(data, var_choice2)
        fig = px.bar(countdata, x=var_choice2, y="percentage", color='Exited', title="Exited by {}".format(var_choice2))
        st.plotly_chart(fig)

    elif var_choice2 in list_of_cont_var:
        data_cut = cutting(data, ['CreditScore', 'Balance', 'EstimatedSalary'], 10)
        data_cut = cutting_bins(data_cut, 'Age')
        countdata = barplot_countdata(data_cut, var_choice2)
        fig = px.bar(countdata, x=var_choice2, y="percentage", color='Exited', title="Exited by {}".format(var_choice2))
        st.plotly_chart(fig)

    else:
        raise ValueError('Choisir une variable existante')

elif Section == "Machine Learning explainability":


    """ Sélection du modèle"""

    model_name = st.selectbox(
        "Choisir le modèle à étudier",
        models)


    #selected_model = bml.models[str(model_name)].best_model


    selected_model = RandomForestClassifier()
    selected_model.fit(bml.X_train, bml.Y_train)
    """ Permutation importance """
    perm = PermutationImportance(selected_model, random_state=15, scoring='f1').fit(bml.X_test, bml.Y_test)
    eli5.show_weights(perm, feature_names=bml.X_test.columns.tolist())

    st.image(FI)

    """ Summary plot"""
    data_SP = data_preprocessed
    X_shap = data_SP.sample(frac=0.01, random_state=15)
    print(X_shap)
    explainer = shap.TreeExplainer(selected_model)
    shap_values = explainer.shap_values(X_shap)
    shap.summary_plot(shap_values[1], X_shap)
    st.pyplot()

    """ Partial plots """
    var_selec = ['Tenure', 'NumOfProducts', 'Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    selected_var = st.selectbox(
        "Choisir une variable à étudier",
        var_selec
    )
    pdp_dist = pdp.pdp_isolate(model=selected_model, dataset=bml.X_test,
                               model_features=bml.X_test.columns.tolist(), feature=selected_var)
    pdp.pdp_plot(pdp_dist, selected_var)
    st.pyplot()

    """ Specific row """

    row_to_show = st.selectbox(
        "Choisir le client que vous voulez étudier",
        np.arange(len(bml.Y_train))
    )

    data_for_prediction = bml.X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    # Create object that can calculate shap values
    selected_model.predict_proba(data_for_prediction_array)

    # Calculate Shap values
    shap_values_row = explainer.shap_values(data_for_prediction_array)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_row[1], data_for_prediction_array))
