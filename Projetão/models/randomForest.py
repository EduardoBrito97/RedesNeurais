import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt
from matplotlib import pyplot as plt
import seaborn as sns

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None, estimator = None, depth = None):

    mse = mean_squared_error(y, y_pred_class)
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (mse, accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.savefig('../results/randomForest/plots/Estimator-' + str(estimator) + '_depth-' + str(depth) + '_KS.png')
        plt.clf()
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

def get_data(path_atributes, path_labels):

    y = pd.read_csv(path_labels, index_col=0).values.squeeze()
    X = StandardScaler().fit_transform(pd.read_csv(path_atributes, index_col=0))
    
    return X, y

## Lê dados
X_train, y_train = get_data('../data/X_train_over.csv', '../data/y_train_over.csv')
X_test, y_test = get_data('../data/X_test.csv', '../data/y_test.csv')

max_depths = [5, 10, 20]
n_estimators = [10, 50, 100]

## quantos folds no cross-validation e o tamanho do fold de test
cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0)

results = {'EST': [], 'DEPTH': [], 'MSE': [], 'ACC': [], 'REC': [], 'PREC': [], 'F1': [], 'AUROC': [], 'AUPR': []}
for estimator in n_estimators:
    for depth in max_depths:
        ## inicializar um classificador (daqui pra baixo é padrão pra todos classificadores do scikitlearn)
        rfc = RandomForestClassifier(n_estimators=estimator, verbose=True, max_depth=depth)
        
        ## faz o cross-validation e guarda cada modelo (5 folds, 5 treinos, 5 modelos)
        ## o y.round() é feito pois o python reclama dos números quebrados, i.e., 1.0 ou 0.0; mas como temos um problema de classificação, o .round() não prejudica em nada a solução
        scores = cross_validate(rfc, X_train, y_train.round(), cv=cv, return_estimator=True)

        ## pega o modelo que obteve o melhor score (o método de score default é accuracy)
        bestRfc = scores['estimator'][np.argmax(scores['test_score'])]
        ## faz predição
        y_pred = bestRfc.predict(X_test)
        y_pred_scores = bestRfc.predict_proba(X_test)
        ## as funções abaixo foram copiadas do github que o proferssor indica no site, exceto pela parte de mse e matriz de confusão
        mse, accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test.round(), y_pred, y_pred_scores, estimator = estimator, depth = depth)

        results['EST'].append(estimator)
        results['DEPTH'].append(depth)
        results['MSE'].append(mse)
        results['ACC'].append(accuracy)
        results['REC'].append(recall)
        results['PREC'].append(precision)
        results['F1'].append(f1)
        results['AUROC'].append(auroc)
        results['AUPR'].append(aupr)

        print('Performance no conjunto de teste para est. = ' + str(estimator) + ' e depth = ' + str(depth) + ':')
        print_metrics_summary(mse, accuracy, recall, precision, f1, auroc, aupr)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
        plt.savefig('../results/randomForest/plots/Estimator-' + str(estimator) + '_depth-' + str(depth) + '_matrix.png')
        plt.clf()

data = pd.DataFrame(results)
data.to_csv('../results/randomForest/results.csv')