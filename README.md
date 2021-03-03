# BinaryML churn analysis

L'analyse de churn est un domaine qui a pour objectif d’évaluer la perte de clients d’une entreprise, d’en comprendre les causes et éventuellement d’effectuer des prédictions, afin notamment de prendre des décisions stratégiques, d’adapter son offre et d’améliorer ses résultats. 

Dans le cadre de ce projet, l'analyse de démission est étudié de façon binaire, soit avec des données numériques sur une base de données clientèles soit avec des données textuelles issues de commentaires des clients.

Datasets:
	- **Churn_Modelling** (données numériques):
		*donnée clientèle d'une banque, 10000 observations
		*variables : 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
		*target : 'Exited'
	- **TrustPilot_data** (données textuelles):
		*données de commentaires clients du site TustPilot, 440 observations
		*variable : 'text'
		*target : 'sentiment' (1 si nombre_étoiles >= 4 sinon 0)

Le code est construit dans un but d'automatisation de la classification binaire. Il est adapté à tous types de données numériques et données textuelles.

### Classification binaire pour donnéees numériques:

> `python main_tabular.py`

visualisation streamlit :
> `streamlit run streamlit_tabular.py`

### Classification binaire pour donnéees textuelles:

> `python main_nlp.py`

visualisation streamlit :
> `streamlit run streamlit_nlp.py`

### Architecture du code

**binaryML** est la classe principale : initialisation, preprocessing, train/test split, normalisation

**class_model** est la classe parent des modèles

Pour chaque modèle :
	* on lance un **gridsearch** (class_gridsearch.py)
	* puis **validation** (validation.py)

Possibilité de **predictions** avec les meilleurs modèles choisis par gridsearch (prediction.py)

Visualisation sur app **streamlit** : EDA, importance des variables, scores des modèles, ...

### Listes des modèles:

Données numériques: 'Logistic Regression', 'Random Forest', 'LightGBM', 'XGBoost', 'CatBoost', 'SimpleNeuralNetwork'

Données textuelles: 'tf-idf+Naive Bayes', 'tf-idf+SGDClassifier', 'tf-idf+Logistic Regression', 'Fasttext Attention head', 'BERT'

### Résultats sur données test (20%) (classé par f1 score)

Churn_Modelling dataset :

|name|accuracy|recall|precision|f1|roc_auc|
|---------------|-------|-------|-------|-------|-------|
|Random Forest|0.8650|0.5332|0.7306|0.6165|0.8515|
|BlendModel|0.8610|0.5209|0.7186|0.6040|0.8531|
|XGBoost|0.8620|0.5061|0.7331|0.5988|0.8529|
|LightGBM|0.8275|0.6265|0.5692|0.5965|0.8429|
|CatBoost|0.7880|0.7248|0.4860|0.5819|0.8527|
|Stacking|0.8550|0.4840|0.7112|0.5760|0.8396|
|Logistic Regression|0.6985|0.6904|0.3707|0.4824|0.7570|
|SimpleNeuralNetwork|0.7965|0.0000|0.0000|0.0000|0.5000|


TODO:

 - test PCA results + viz en 2D
 - ajouter Scraping script + scrap more comments --
 - viz en 2D des embeddings documents et mots
 - add class_weight pour xgboost
 - random-state in gridsearch_sklearn = le fixer et avoir le même que dans validation?
 - 5-10 experiences feature engineering + feature importance
 - presentation des modèles
 - continuer développement streamlit
