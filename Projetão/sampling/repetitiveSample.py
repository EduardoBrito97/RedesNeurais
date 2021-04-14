import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

def get_data():

    ## Lê dados de traino
    originalTrain = pd.read_csv('../originalData/BASE-PREPROCESSED(TRAIN)', sep='\t', engine='python')
    ## Dropa colunas PROPHET_LABEL e PROPHET_NORM_FEATURES
    originalTrain = originalTrain.drop(['PROPHET_LABEL', 'PROPHET_NORM_FEATURES'], axis=1)
    ## Lê dados de validação
    originalValid = pd.read_csv('../originalData/BASE-PREPROCESSED(VALIDACAO)', sep='\t', engine='python')
    ## Dropa colunas PROPHET_LABEL e PROPHET_NORM_FEATURES
    originalValid = originalValid.drop(['PROPHET_LABEL', 'PROPHET_NORM_FEATURES'], axis=1)
    ## Lê dados de teste
    originalTest = pd.read_csv('../originalData/BASE-PREPROCESSED(TESTE)', sep='\t', engine='python')
    ## Dropa colunas PROPHET_LABEL e PROPHET_NORM_FEATURES
    originalTest = originalTest.drop(['PROPHET_LABEL', 'PROPHET_NORM_FEATURES'], axis=1)

    ## Só usamos as colunas(atributos) comuns entre datasets de treino e validação
    usefulTrain = originalTrain[originalTrain.columns.intersection(originalValid.columns)]
    usefulValid = originalValid[originalTrain.columns.intersection(originalValid.columns)]
    usefulTest = originalTest[originalTrain.columns.intersection(originalValid.columns)]

    return usefulTrain, usefulValid, usefulTest

def oversampling(majorClassSize, minorClass):

    ## Quantas vezes devemos aumentar nossa classe minoritária
    limit = majorClassSize - minorClass.shape[0]
    ## Copia nosso dataframe em formato numpy array por questões de eficiência
    minorClassValues = minorClass.values

    for entry in tqdm(range(0, limit)):
        minorClassValues = np.append(minorClassValues, [minorClassValues[entry]], axis=0)

    ## Copia o nosso numpy array de volta para dataframe
    minorClass = pd.DataFrame(minorClassValues, columns=minorClass.columns)

    return minorClass

def separateClasses(data):

    class1 = data.loc[data['ALVO'] == 1]
    class2 = data.loc[data['ALVO'] == 0]

    return class1, class2

def joinAndShuffle(class1, class2):

    class1 = class1.append(class2)
    class1 = shuffle(class1)

    return class1

## importa dados
usefulTrain, usefulValid, usefulTest = get_data()

## separa classes majoritária e minoritária (para os nossos datasets majoritária 1 e minoritária 0)
majorClassTrain, minorClassTrain = separateClasses(usefulTrain)
majorClassValid, minorClassValid = separateClasses(usefulValid)

## faz repetitive oversampling com classes minoritárias
minorClassTrain = oversampling(majorClassTrain.shape[0], minorClassTrain)
minorClassValid = oversampling(majorClassValid.shape[0], minorClassValid)

## junta tuda e faz o shuffle
readyDataTrain = joinAndShuffle(majorClassTrain, minorClassTrain)
readyDataValid = joinAndShuffle(majorClassValid, minorClassValid)

## elimina os índices antigos, que se tornaram inúteis pois estão fora de ordem
readyDataTrain = readyDataTrain.reset_index(drop=True)
readyDataValid = readyDataValid.reset_index(drop=True)

readyDataTrain.to_csv('../readyData/readyDataTrain.csv')
readyDataValid.to_csv('../readyData/readyDataValid.csv')
usefulTest.to_csv('../readyData/readyDataTest.csv')