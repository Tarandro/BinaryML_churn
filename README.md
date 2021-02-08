# BinaryML_churn

utiliser de préférence les notebooks:

2 types de notebooks:
 - full : toutes les fonctions sont contenus dans le notebook
 - main : les fonctions sont récupérés par un git clone (préférable car update des fonctions)

Pour lancer l'application streamlit :
streamlit run streamlit.py

A faire:

 - developper feature importances dans le script preprocessing tabular
 - option threshold 0.5 pour binary_proba
 - ajouter Scraping script
 - ajouter xgboost class_weight
 - afficher some logs duing epoch training (seulement lors de la validation)
 - random-state in gridsearch_sklearn = le fixer et avoir le même que dans validation?

 - continuer développement streamlit
 - ajouter eda et feature importances dans streamlit
 - ajouter extraction des mots influents
