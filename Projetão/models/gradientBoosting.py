import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt
from matplotlib import pyplot as plt

import seaborn as sns

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None, base_name = None):
    mse = mean_squared_error(y, y_pred_class)
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (mse, accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.savefig('../results/gradient_boosting/plots/' + base_name + '_KS.png')
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

estimators = [10, 50, 100]
learning_rates = [0.1, 0.001, 0.00001, 0.0000001]
subsamples = [0.1, 0.5, 1.0]

## quantos folds no cross-validation e o tamanho do fold de test
cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0)

results = {'EST': [], 'LR': [], 'SUBSAMP':[], 'MSE': [], 'ACC': [], 'REC': [], 'PREC': [], 'F1': [], 'AUROC': [], 'AUPR': []}
for estimator in estimators:
    for lr in learning_rates:
        for subsample in subsamples:
            ## inicializar um classificador (daqui pra baixo é padrão pra todos classificadores do scikitlearn)
            gbc = GradientBoostingClassifier(n_estimators=estimator, learning_rate=lr, subsample=subsample)
            
            ## faz o cross-validation e guarda cada modelo (5 folds, 5 treinos, 5 modelos)
            ## o y.round() é feito pois o python reclama dos números quebrados, i.e., 1.0 ou 0.0; mas como temos um problema de classificação, o .round() não prejudica em nada a solução
            scores = cross_validate(gbc, X_train, y_train.round(), cv=cv, return_estimator=True)

            ## pega o modelo que obteve o melhor score (o método de score default é accuracy)
            bestGbc = scores['estimator'][np.argmax(scores['test_score'])]
            ## faz predição
            y_pred = bestGbc.predict(X_test)
            y_pred_scores = bestGbc.predict_proba(X_test)
            
            base_name = "est-" + str(estimator) + "_lr-" + str(lr*1000000) + "_subsample-" + str(subsample)
            
            ## as funções abaixo foram copiadas do github que o proferssor indica no site, exceto pela parte de mse e matriz de confusão
            mse, accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test.round(), y_pred, y_pred_scores, base_name)
            results['EST'].append(estimator)
            results['LR'].append(lr)
            results['SUBSAMP'].append(subsample)
            results['MSE'].append(mse)
            results['ACC'].append(accuracy)
            results['REC'].append(recall)
            results['PREC'].append(precision)
            results['F1'].append(f1)
            results['AUROC'].append(auroc)
            results['AUPR'].append(aupr)

            print('Performance no conjunto de teste ' + base_name + ': ')        
            print_metrics_summary(mse, accuracy, recall, precision, f1, auroc, aupr)
            plot_confusion_matrix(bestGbc, X_test, y_test.round())
            plt.savefig('../results/gradient_boosting/plots/' + base_name + '_matrix.png')
            plt.clf()

data = pd.DataFrame(results)
data.to_csv('../results/gradient_boosting/results.csv', index=False, float_format='%.4f')