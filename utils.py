import numpy as np
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_class_weight

#####################
# Metrics
#####################

def calcul_metric_binary(y_true_, y_pred, print_results):
    try: # try to pass y_true_ to type numpy
        y_true = y_true_.values.copy()
    except:
        y_true = y_true_.copy()

    report = classification_report(y_true.reshape(-1), np.where(y_pred <0.5 ,0 ,1).reshape(-1), digits = 4, output_dict = True)
    acc = np.round(report['accuracy'] ,4)
    f1 = np.round(report['1']['f1-score'] ,4)
    recall = np.round(report['1']['recall'] ,4)
    precision = np.round(report['1']['precision'] ,4)
    # roc_auc = np.round(roc_auc_score(y_true.values, np.where(y_pred<0.5,0,1)),4)
    fp_rate, tp_rate, thresholds = roc_curve(y_true.reshape(-1), y_pred.reshape(-1))
    roc_auc = np.round(auc(fp_rate, tp_rate) ,4)

    if print_results:
        print('\nCross validation score :')
        print()
        print('roc_auc =', roc_auc)
        print('precision 1 =', precision)
        print('recall 1 =', recall)
        print('f1 score 1 =' ,f1)
        print()
        print(classification_report(y_true.reshape(-1), np.where(y_pred <0.5 ,0 ,1).reshape(-1), digits = 3))

    return acc, f1, recall, precision, roc_auc

def roc(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr

#####################
# Class weight for Neural Network
#####################

def compute_dict_class_weight(y, class_weight, objective):
    if class_weight == "balanced":
        if ('binary' in objective) or (y.shape[1] == 1 and 'classification' in objective):
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y.reshape(-1)), y=y.reshape(-1))
            return dict(zip(np.unique(y.reshape(-1)), weights))
        else:
            return None
    else:
        return None