import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np
import pandas as pd

# fonctions d'histogrammes
def hist_or_circ(colname):  # donne la nature du plot pour une variable
    L_hist = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    L_circ = ['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']
    if colname in L_hist:
        return ('histogram')
    elif colname in L_circ:
        return ('pie')
    else:
        raise ValueError('Donnez un nom de variable valide.')


def subplot_hist(df):  # fais un subplot pour un seul DF
    plot_type = [hist_or_circ(col) for col in df.columns]
    titles = []
    for col in df.columns:
        if hist_or_circ(col) == 'histogram':
            titles.append(f"Histogram of {col}")
        elif hist_or_circ(col) == 'pie':
            titles.append(f"Pie chart of {col}")

    specs_list = []
    for ind in range(0, len(plot_type) - 1, 2):
        specs_list.append([{"type": plot_type[ind]}, {"type": plot_type[ind + 1]}])
    if len(plot_type) % 2 != 0:
        specs_list.append([{"type": plot_type[len(plot_type) - 1]}, {"type": 'histogram'}])

    nrow = len(df.columns) // 2 + len(df.columns) % 2
    fig = make_subplots(rows=nrow,
                        cols=2,
                        specs=specs_list,
                        subplot_titles=titles)

    for i, colname in enumerate(df.columns):
        if hist_or_circ(colname) == 'histogram':
            fig.add_histogram(x=df[colname],
                              row=(i // 2) + 1,
                              col=(i % 2) + 1)


        elif hist_or_circ(colname) == 'pie':
            labels_list = list(np.unique(df[colname]))
            count_list = [sum(df[colname] == label) for label in labels_list]
            fig.add_pie(values=count_list, labels=labels_list,
                        row=(i // 2) + 1, col=(i % 2) + 1)

    fig.update_layout(height=2000, showlegend=False)
    return(fig)


def subplot_hist_comp(df1, df2, sel_col=[], fig_height=4000):  # mettre celui avec le plus de colonnes en df1
    if len(sel_col) > 0:
        df1 = df1[sel_col]
        df2 = df2[sel_col]
    plot_type = [hist_or_circ(col) for col in df1.columns]
    titles = []
    for col in df1.columns:
        if hist_or_circ(col) == 'histogram':
            titles.append(f"Histogram of {col}")
            titles.append(f"Histogram of {col}")
        elif hist_or_circ(col) == 'pie':
            titles.append(f"Pie chart of {col}")
            titles.append(f"Pie chart of {col}")
    # titles = [[f"Histogram of {col}", f"Histogram of {col}"] for col in df1.columns]
    # titles = [item for sublist in titles for item in sublist]

    specs_list = []
    for tp in plot_type:
        specs_list.append([{"type": tp}, {"type": tp}])

    nrow = max(len(df1.columns), len(df2.columns))
    fig = make_subplots(rows=nrow,
                        cols=2,
                        specs=specs_list,
                        subplot_titles=titles)

    for i, col in enumerate(df1.columns):
        if plot_type[i] == "histogram":
            fig.add_histogram(x=df1[col],
                              row=i + 1,
                              col=1, histnorm='probability')
            fig.add_histogram(x=df2[col],
                              row=i + 1,
                              col=2, histnorm='probability')
        elif plot_type[i] == "pie":
            labels_list = list(np.unique(df1[col]))
            count_list1 = [sum(df1[col] == label) for label in labels_list]
            count_list2 = [sum(df2[col] == label) for label in labels_list]
            fig.add_pie(values=count_list1, labels=labels_list,
                        row=i + 1, col=1, textinfo='value+percent+label')
            fig.add_pie(values=count_list2, labels=labels_list,
                        row=i + 1, col=2, textinfo='value+percent+label')
    fig.update_layout(height=fig_height, showlegend=False, font={'size': 18})
    return(fig)

def choose_datase(choice, list_of_choices, list_of_datasets):
    if choice in list_of_choices:
        for i,choice_from_list in enumerate(list_of_choices):
            if choice_from_list == choice:
                return(list_of_datasets[i])
                break
    else:
        raise ValueError('Enter a valid choice')


def cutting(data, col_to_cut, nbre_cut):
    for col in col_to_cut:
        data[col] = pd.qcut(data[col].rank(method='first'), nbre_cut, labels=[i for i in range(nbre_cut)]) #ajout de .rank(method='first') pour débuguer
        #data[col] = data[col].astype(str)
    return data

def cutting_bins(data, one_col_to_cut, bins=[0, 18, 25, 45, 65, 75]):
    bins.append(max(data[one_col_to_cut].values))
    labels = [f'{i}-{j}' for i, j in zip(bins[:-1], bins[1:])]
    data[one_col_to_cut] = pd.cut(data[one_col_to_cut], bins=bins, labels=labels)
    return(data)

def barplot_countdata(data, groupbyname): #dataframe for barplot based on grouping
    countdata = pd.DataFrame(data.groupby([groupbyname, 'Exited']).count()).reset_index().iloc[:,0:3]
    freqname = countdata.columns[2]
    countdata = countdata.rename(columns={freqname:'count'})
    tot_by_group=data.groupby([groupbyname]).count()[freqname].reset_index()
    countdata = countdata.merge(tot_by_group).rename(columns={freqname:'tot_by_group'})
    countdata['percentage'] = countdata['count']/countdata['tot_by_group']
    return(countdata)