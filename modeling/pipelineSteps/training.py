""" Training pipeline. Saves result objcets in a specified directory. """

import pandas as pd
import numpy as np
import pickle

from modeling.configs import nXGB, nSVM, nRF, nLR, nMLP, MLPGrid, LRGrid, RFGrid, kSVMGrid, XGBGrid, modelClasses, \
    inputNNeigh, missingVal, nCVs, CVFolds, USnegProp, imputeMaxIter, randomState, resGT6Path, pathGT6, res0To5Path, \
    resFullPath, path0To5
from modeling.utils.pipelineDefinitions import eval_model_pipeline
from modeling.utils.utils import gen_configs


def training_pipeline():
    rng = np.random.default_rng(randomState)

    XGBConfigs = gen_configs(nXGB, XGBGrid, rng)
    kSVMConfigs = gen_configs(nSVM, kSVMGrid, rng)
    RFConfigs = gen_configs(nRF, RFGrid, rng)
    LRConfigs = gen_configs(nLR, LRGrid, rng)
    MLPConfigs = gen_configs(nMLP, MLPGrid, rng)

    modelConfigs = [XGBConfigs, kSVMConfigs, RFConfigs, LRConfigs, MLPConfigs]

    # read pre-processed BUT not imputed full data (not stratified by age)
    fullDataPath = ''
    fullData = pd.read_csv(fullDataPath, index_col=0, dtype=int)

    rng = np.random.default_rng(randomState)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(fullData), p=dataSplits)
    fullDataTrain, fullDataTest = fullData[trainIndexes], fullData[~trainIndexes]

    resValWhole, resTestWhole, bestArchWhole, \
    trModelsWhole, trDataWhole, testDataWhole, fNamesWhole = eval_model_pipeline(modelClasses=modelClasses,
                                                                                 modelConfigs=modelConfigs,
                                                                                 trainingData=fullDataTrain,
                                                                                 testData=fullDataTest,
                                                                                 inputNNeigh=inputNNeigh,
                                                                                 missingVal=missingVal,
                                                                                 nCVs=nCVs,
                                                                                 CVFolds=CVFolds,
                                                                                 USnegProp=USnegProp,
                                                                                 imputeMaxIter=imputeMaxIter,
                                                                                 randomState=randomState)
    with open(resFullPath, 'wb') as handle:
        pickle.dump([resValWhole, resTestWhole, bestArchWhole,
                     trModelsWhole, trDataWhole, testDataWhole, fNamesWhole], handle)

    rng = np.random.default_rng(randomState)

    # read pre-processed BUT not imputed ages 0 to 5 data
    ages0To5 = pd.read_csv(path0To5, index_col=0, dtype=int)
    # ages0To5 = remove_variance(ages0To5, pctThreshold=0.005)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(ages0To5), p=dataSplits)
    ages0To5Train, ages0To5Test = ages0To5[trainIndexes], ages0To5[~trainIndexes]

    resVal0To5, resTest0To5, bestArch0To5, \
    trModels0To5, trData0To5, testData0To5, fNames0To5 = eval_model_pipeline(modelClasses=modelClasses,
                                                                             modelConfigs=modelConfigs,
                                                                             trainingData=ages0To5Train,
                                                                             testData=ages0To5Test,
                                                                             inputNNeigh=inputNNeigh,
                                                                             missingVal=missingVal,
                                                                             nCVs=nCVs,
                                                                             CVFolds=CVFolds,
                                                                             USnegProp=USnegProp,
                                                                             randomState=randomState)
    with open(res0To5Path, 'wb') as handle:
        pickle.dump([resVal0To5, resTest0To5, bestArch0To5, trModels0To5,
                     trData0To5, testData0To5, fNames0To5], handle)

    rng = np.random.default_rng(randomState)

    # read pre-processed BUT not imputed ages > 6 data
    agesGT6 = pd.read_csv(pathGT6, index_col=0, dtype=int)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(agesGT6), p=dataSplits)
    agesGT6Train, agesGT6Test = agesGT6[trainIndexes], agesGT6[~trainIndexes]

    resValGT6, resTestGT6, bestArchGT6, \
    trModelsGT6, trDataGT6, testDataGT6, fNamesGT6 = eval_model_pipeline(modelClasses=modelClasses,
                                                                         modelConfigs=modelConfigs,
                                                                         trainingData=agesGT6Train,
                                                                         testData=agesGT6Test,
                                                                         inputNNeigh=inputNNeigh,
                                                                         missingVal=missingVal,
                                                                         nCVs=nCVs,
                                                                         CVFolds=CVFolds,
                                                                         USnegProp=USnegProp,
                                                                         randomState=randomState)

    with open(resGT6Path, 'wb') as handle:
        pickle.dump([resValGT6, resTestGT6, bestArchGT6, trModelsGT6,
                     trDataGT6, testDataGT6, fNamesGT6],
                    handle)

if __name__ == '__main__':
    training_pipeline()