import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
import seaborn as sns

def extract_final_losses(history):
    """Função para extrair o melhor loss de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado 
    no menor loss de validação.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history):
    """Função para plotar as curvas de erro do treinamento da rede neural.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    mse = mean_squared_error(y, y_pred_class)
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (mse, accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.savefig('../results/mlp/plots/BS' + str(bs) + 'HL' + str(hl) + 'HN' + str(hn) + 'LR' + str(int(lr * 10000)) + optimizer + '_KS.png')
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

    y = pd.read_csv(path_labels, index_col=0).values
    X = pd.read_csv(path_atributes, index_col=0).values
    
    return X, y 

X_train, y_train = get_data('../data/X_train_over.csv', '../data/y_train_over.csv')
X_val, y_val = get_data('../data/X_valid.csv', '../data/y_valid.csv')
X_test, y_test = get_data('../data/X_test.csv', '../data/y_test.csv')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]

batch_size = [64, 256, 512]
hidden_layers = [1, 2, 3]
hidden_neurons = [20, 40]
learning_rates = [0.0001, 0.001, 0.01, 0.1]
optimizers = ['adam', 'SGD']

results = {'BS': [], 'HL': [], 'HN': [], 'LR': [], 'OPTIM': [], 'MSE': [], 'ACC': [], 'REC': [], 'PREC': [], 'F1': [], 'AUROC': [], 'AUPR': []}
for bs in batch_size:
    for hl in hidden_layers:
        for hn in hidden_neurons:
            for lr in learning_rates:
                for optimizer in optimizers:
                
                    classifier = Sequential()

                    classifier.add(Dense(hn, activation='tanh', input_dim=input_dim))
                    for i in range(hl):
                        classifier.add(Dense(hn, activation='tanh'))
                    classifier.add(Dense(1, activation='sigmoid'))

                    classifier.compile(optimizer=optimizer, loss='mean_squared_error')

                    mlp_clf = KerasClassifier(build_fn=lambda: classifier)
                    
                    mlp_clf.fit(X_train, y_train, batch_size=bs, verbose=0, epochs=100000, callbacks=[EarlyStopping(patience=3)], validation_data=(X_val, y_val))

                    y_pred = mlp_clf.predict(X_test)
                    y_pred_scores = mlp_clf.predict_proba(X_test)
                    mse, accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test.squeeze(), y_pred.squeeze(), y_pred_scores)
                    results['BS'].append(bs)
                    results['HL'].append(hl)
                    results['HN'].append(hn)
                    results['LR'].append(lr)
                    results['OPTIM'].append(optimizer)
                    results['MSE'].append(mse)
                    results['ACC'].append(accuracy)
                    results['REC'].append(recall)
                    results['PREC'].append(precision)
                    results['F1'].append(f1)
                    results['AUROC'].append(auroc)
                    results['AUPR'].append(aupr)

                    print('Performance BS' + str(bs) + 'HL' + str(hl) + 'HN' + str(hn) + 'LR' + str(lr) + optimizer)
                    print_metrics_summary(mse, accuracy, recall, precision, f1, auroc, aupr)
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
                    plt.savefig('../results/mlp/plots/BS' + str(bs) + 'HL' + str(hl) + 'HN' + str(hn) + 'LR' + str(int(lr * 10000)) + optimizer + '_confusion_matrix.png')
                    plt.clf()

data = pd.DataFrame(results)
data.to_csv('../results/mlp/results.csv')