import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score

import scikitplot as skplt
from matplotlib import pyplot as plt
import seaborn as sns

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None, lr_mul = None):

    mse = mean_squared_error(y, y_pred_class)
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (mse, accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.savefig('../results/ensemble_mlp/plots/lr_mul-' + str(lr_mul) + '_KS.png')
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

# Cria o dicionário com diferentes mlps para o ensemble

hidden_layers = [1, 2, 3]
hidden_neurons = [10, 20]
learning_rates = [0.001, 0.01, 0.1]

results = {'LR_MULTIPLIER': [], 'MSE': [], 'ACC': [], 'REC': [], 'PREC': [], 'F1': [], 'AUROC': [], 'AUPR': []}
for lr_mul in [1, 2, 3]:
    mlps_clf = []
    for hl in hidden_layers:
        for hn in hidden_neurons:
            for lr in learning_rates:
                hidden_layers_size = (hn)
                if hl == 2:
                    hidden_layers_size = (hn, hn)
                elif hl == 3:
                    hidden_layers_size = (hn, hn, hn)

                # Para cada valor de learning rate, hidden neurons e hidden layers, cria um modelo e atribui à legenda
                mlp_ens_clf = MLPClassifier(hidden_layer_sizes=hidden_layers_size,
                                            activation='relu',
                                            learning_rate_init=lr*lr_mul,
                                            learning_rate='constant',
                                            max_iter=100,
                                            )
                mlp_alias = 'mlp_hl-' + str(hl) + '_hn-' + str(hn) + '_lr-' + str(lr)
                mlps_clf.append( (mlp_alias, mlp_ens_clf) )

    # Cria o Ensemble
    mlp_ens = VotingClassifier(mlps_clf, voting='soft')

    # Treina o Ensemble
    mlp_ens.fit(X_train, y_train)

    # Prediz os próximos valores
    y_pred = mlp_ens.predict(X_test)
    y_pred_scores = mlp_ens.predict_proba(X_test)

    mse, accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test.round(), y_pred, y_pred_scores, lr_mul=lr_mul)

    results['LR_MULTIPLIER'].append(lr_mul)
    results['MSE'].append(mse)
    results['ACC'].append(accuracy)
    results['REC'].append(recall)
    results['PREC'].append(precision)
    results['F1'].append(f1)
    results['AUROC'].append(auroc)
    results['AUPR'].append(aupr)

    print('Performance no conjunto de teste ensemble com lr_mul = ' + str(lr_mul) +':')
    print_metrics_summary(mse, accuracy, recall, precision, f1, auroc, aupr)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.savefig('../results/ensemble_mlp/plots/lr_mul-' + str(lr_mul) + '_matrix.png')
    plt.clf()

data = pd.DataFrame(results)
data.to_csv('../results/ensemble_mlp/results.csv')