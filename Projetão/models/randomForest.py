import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

import scikitplot as skplt
from matplotlib import pyplot as plt

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):

    mse = mean_squared_error(y, y_pred_class)
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (mse, accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)

    return performance_metrics

def print_metrics_summary(mse, accuracy, recall, precision, f1, auroc=None, aupr=None):

    print()
    print("{metric:<18}{value:.4f}".format(metric="MSE:", value=mse))
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))

    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))

<<<<<<< HEAD
def get_data():
=======
## Lê datasest, usando a primeira coluna como índice
df = pd.read_csv('../readyData/readyDataTrain.csv', index_col=0)
## Divide Label para y e atributos para X
X, y = df.drop(['PROPHET_LABEL'], axis=1), df['PROPHET_LABEL']
>>>>>>> 731a556fa6c37ede370abbbcdacc7a38abf70b4a

    ## Lê dados de traino usando a primeira coluna como índice
    df = pd.read_csv('../readyData/readyDataTrain.csv', index_col=0)
    ## Divide Label para y e atributos para X
    X_train, y_train = df.drop(['ALVO'], axis=1), df['ALVO']

    ## Lê dados de validação usando a primeira coluna como índice
    df = pd.read_csv('../readyData/readyDataValid.csv', index_col=0)

    ## Concatena dados de validação e treino já que usaremos cross-validation
    X_train = pd.concat([X_train, df.drop(['ALVO'], axis=1)], ignore_index=True)
    y_train = pd.concat([y_train, df['ALVO']], ignore_index=True)

    ## Lê dados de teste usando a primeira coluna como índice
    df = pd.read_csv('../readyData/readyDataTest.csv', index_col=0)
    ## Divide Label para y e atributos para X
    X_test, y_test = df.drop(['ALVO'], axis=1), df['ALVO']
    
    return X_train, y_train, X_test, y_test

## Lê dados
X_train, y_train, X_test, y_test = get_data()

## inicializar um classificador (daqui pra baixo é padrão pra todos classificadores do scikitlearn)
rfc = RandomForestClassifier(n_estimators=10)
## quantos folds no cross-validation e o tamanho do fold de test
cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0)
## faz o cross-validation e guarda cada modelo (5 folds, 5 treinos, 5 modelos)
## o y.round() é feito pois o python reclama dos números quebrados, i.e., 1.0 ou 0.0; mas como temos um problema de classificação, o .round() não prejudica em nada a solução
scores = cross_validate(rfc, X_train, y_train.round(), cv=cv, return_estimator=True)

## pega o modelo que obteve o melhor score (o método de score default é accuracy)
bestRfc = scores['estimator'][np.argmax(scores['test_score'])]
## faz predição
y_pred = bestRfc.predict(X_test)
y_pred_scores = bestRfc.predict_proba(X_test)
## as funções abaixo foram copiadas do github que o proferssor indica no site, exceto pela parte de mse e matriz de confusão
mse, accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test.round(), y_pred, y_pred_scores)
print('Performance no conjunto de teste:')
print_metrics_summary(mse, accuracy, recall, precision, f1, auroc, aupr)
plot_confusion_matrix(bestRfc, X_test, y_test.round())
plt.show()
