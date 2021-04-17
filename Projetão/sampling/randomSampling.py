import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def get_data():

    ## Lê dados de traino
    originalTrain = pd.read_csv('../data/BASE-PREPROCESSED(TRAIN)', sep='\t', engine='python')
    ## Dropa colunas PROPHET_LABEL e PROPHET_NORM_FEATURES
    originalTrain = originalTrain.drop(['PROPHET_LABEL', 'PROPHET_NORM_FEATURES', 'NEURO_LABEL'], axis=1)
    ## Lê dados de validação
    originalValid = pd.read_csv('../data/BASE-PREPROCESSED(VALIDACAO)', sep='\t', engine='python')
    ## Dropa colunas PROPHET_LABEL e PROPHET_NORM_FEATURES
    originalValid = originalValid.drop(['PROPHET_LABEL', 'PROPHET_NORM_FEATURES', 'NEURO_LABEL'], axis=1)
    ## Lê dados de teste
    originalTest = pd.read_csv('../data/BASE-PREPROCESSED(TESTE)', sep='\t', engine='python')
    ## Dropa colunas PROPHET_LABEL e PROPHET_NORM_FEATURES
    originalTest = originalTest.drop(['PROPHET_LABEL', 'PROPHET_NORM_FEATURES', 'NEURO_LABEL'], axis=1)

    ## Só usamos as colunas(atributos) comuns entre datasets de treino e validação
    ## Também eliminamos linhas duplicadas
    usefulTrain = originalTrain[originalTrain.columns.intersection(originalValid.columns)].drop_duplicates()
    usefulValid = originalValid[originalTrain.columns.intersection(originalValid.columns)].drop_duplicates()
    usefulTest = originalTest[originalTrain.columns.intersection(originalValid.columns)].drop_duplicates()

    return usefulTrain, usefulValid, usefulTest

## importa dados
usefulTrain, usefulValid, usefulTest = get_data()

oversample = RandomOverSampler(sampling_strategy='minority')
undersample = RandomUnderSampler(sampling_strategy='majority')

X_train, y_train = usefulTrain.drop(['ALVO'], axis=1), usefulTrain[['ALVO']]
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

X_valid, y_valid = usefulValid.drop(['ALVO'], axis=1), usefulValid[['ALVO']]

X_test, y_test = usefulTest.drop(['ALVO'], axis=1), usefulTest[['ALVO']]

X_train_over.to_csv('../data/X_train_over.csv')
y_train_over.to_csv('../data/y_train_over.csv')

X_train_under.to_csv('../data/X_train_under.csv')
y_train_under.to_csv('../data/y_train_under.csv')

X_valid.to_csv('../data/X_valid.csv')
y_valid.to_csv('../data/y_valid.csv')

X_test.to_csv('../data/X_test.csv')
y_test.to_csv('../data/y_test.csv')