import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from PIL import Image

### run application : streamlit run streamlit_nlp.py

################
# Load data
################


st.header('BinaryML NLP')

data = pd.read_csv('./results/results_nlp/data.csv')
data_preprocessed = pd.read_csv('./results/results_nlp/data_preprocessed.csv')
Y_train = pd.read_csv('./results/results_nlp/Y_train.csv')

df_all_results = pd.read_csv('./results/results_nlp/df_all_results.csv')

leaderboard_val = pd.read_csv('./results/results_nlp/leaderboard_val.csv')
leaderboard_test = pd.read_csv('./results/results_nlp/leaderboard_test.csv')

oof_val = pd.read_csv('./results/results_nlp/df_oof_val.csv')

try:
    roc_curves = Image.open('./roc_curves_text.png')
except:
    pass

Section = st.sidebar.radio(
    'Section :', ['Data', 'Score', 'Extraction'])

if Section == 'Data':
    """ Dataset provided : """
    data

    """ Data preprocessed : """
    data_preprocessed

    st.write('Predicted value :', ", ".join([col for col in Y_train.columns]))
    st.write('Percentage training set :', str(int(len(Y_train)/len(data)*100)), "%")

elif Section == 'Score':
    """ Cross-Validation score """
    leaderboard_val

    """ Test score """
    leaderboard_test

    #""" Distribution validation score """
    #list_name_models = list(df_all_results.model.unique())
    #rows, cols = 2, 3
    #fig, ax = plt.subplots(rows, cols, figsize=(50,20))

    #for row in range(rows):
    #    for col in range(cols):
    #        if row * cols + col + 1 <= len(list_name_models):
    #            name_model = list_name_models[row * cols + col]
    #            values = df_all_results[df_all_results.model.isin([name_model])].mean_test_score
    #            if np.std(values) < 1e-4:
    #                ax[row, col].hist(values, range = (values.min()-1e-3, values.max()+1e-3))
    #            else:
    #                ax[row, col].hist(values)
    #            ax[row, col].set_xlabel(name_model + ' ('+str(len(values))+' models)', size = 30)
    #            ax[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    #            for tick in ax[row, col].xaxis.get_major_ticks():
    #                tick.label.set_fontsize(30)
    #st.pyplot(fig)

    """ Correlation between cross-validation predictions and target : """
    fig, ax = plt.subplots()
    sns.heatmap(oof_val.corr(), annot=True, cmap=sns.cm.rocket_r)
    st.write(fig)

    """ Roc Curves """
    st.image(roc_curves, use_column_width = True)

else:
    Html_file = open("./results/results_nlp/extract_word.html", "r")
    html_string = Html_file.read()
    html_string_split = html_string.split('<br><br><br><br>')
    for html_text in html_string_split:
        st.markdown('<br><br>'+html_text, unsafe_allow_html=True)