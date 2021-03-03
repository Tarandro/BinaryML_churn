# BinaryML_churn

Structure code :

binaryML est la classe principale / initialisation, preprocessing, train_test_split, normalisation

class_model est la classe parent des modèles

Pour chaque modèle :
	- on lance un gridsearch (class_gridsearch)
	- puis validation (validation.py)

Possibilité de predictions avec prediction.py


code sur les notebooks:

2 types de notebooks (aussi partagé sur kaggle):
 - full : toutes les fonctions sont contenus dans le notebook
 - main : les fonctions sont récupérés par un git clone (préférable car update des fonctions)

sinon lancer code avec:
 - python3 main_tabular.py
 - python3 main_nlp.py

Pour lancer l'application streamlit :
streamlit run streamlit.py

A faire:

 - test PCA results + viz en 2D
 - option threshold 0.5 pour binary_proba --
 - ajouter Scraping script + scrap more comments --
 - viz en 2D des embeddings documents et mots
 - add class_weight pour xgboost
 - afficher some logs during epoch training (seulement lors de la validation) --
 - random-state in gridsearch_sklearn = le fixer et avoir le même que dans validation?
 - 5-10 experiences feature engineering + feature importance
 - presentation des modèles
 - écrire un readme --

 - continuer développement streamlit
 - ajouter extraction des mots influents --
 - résoudre erreur BERT --
