""" Forward feature selection for checking the contribution of each feature to the models. """

import pickle

import pandas as pd
import numpy as np

from src.modeling.configs import fullDataPath, inputNNeigh, missingVal, nCVs, CVFolds, USnegProp, imputeMaxIter, \
    randomState, resFullPath, fsResWholePath, pathGT6, path0To5, FS0To5ResPath, fg0To5Res, \
    resGT6Path, FSGT6ResPath
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def forward_feature_selection_pipeline():
    with open(resFullPath, 'rb') as handle:
        resValWhole, resTestWhole, bestArchWhole, trModelsWhole, \
        trDataWhole, testDataWhole, fNamesWhole = pickle.load(handle)

    fullData = pd.read_csv(fullDataPath, index_col=0, dtype=int)

    rng = np.random.default_rng(randomState)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(fullData), p=dataSplits)
    fullDataTrain, fullDataTest = fullData[trainIndexes], fullData[~trainIndexes]

    bestArchitecture = resValWhole.iloc[resValWhole.drop('Arch', axis=1).idxmax()['AUC']]['Arch']
    bestConfig = bestArchWhole[bestArchitecture]

    # XGBoost is the best architecture

    modelEvalParams = {'modelClass': XGBClassifier,
                       'configs': [bestConfig],
                       'inputNNeigh': inputNNeigh,
                       'missingVal': missingVal,
                       'nCVs': nCVs,
                       'CVFolds': CVFolds,
                       'USnegProp': USnegProp,
                       'imputeMaxIter': imputeMaxIter,
                       'randomState': randomState}

    FSWholeBestTrainedModel, FSWholeImportantFeatures, FSWholeValMetrics, FSWholeTestMetrics, FSWholeCIs, \
    FSWholeBestConfig, FSWholeTrainData, FSWholeTestData, FSWholeFNames = forward_feature_selection_pipeline()

    with open(fsResWholePath, 'wb') as handle:
        pickle.dump(
            [FSWholeBestTrainedModel, FSWholeImportantFeatures, FSWholeValMetrics, FSWholeTestMetrics, FSWholeCIs, \
             FSWholeBestConfig, FSWholeTrainData, FSWholeTestData, FSWholeFNames], handle)

    # 0-5 DATASET
    with open(fg0To5Res, 'rb') as handle:
        valMetricsFineGrid0To5, testMetricsFineGrid0To5, CIsFineGrid0To5, bestConfigFineGrid0To5, \
        trainedModelFineGrid0To5, trainDataFineGrid0To5, testDataFineGrid0To5, fNamesFineGrid0To5 = pickle.load(handle)

    rng = np.random.default_rng(randomState)

    ages0To5 = pd.read_csv(path0To5, index_col=0, dtype=int)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(ages0To5), p=dataSplits)
    ages0To5Train, ages0To5Test = ages0To5[trainIndexes], ages0To5[~trainIndexes]

    # Random Forest is the best architecture

    modelEvalParams = {'modelClass': RandomForestClassifier,
                       'configs': [bestConfigFineGrid0To5],
                       'inputNNeigh': inputNNeigh,
                       'missingVal': missingVal,
                       'nCVs': nCVs,
                       'CVFolds': CVFolds,
                       'USnegProp': USnegProp,
                       'imputeMaxIter': imputeMaxIter,
                       'randomState': randomState}

    FS05BestTrainedModel, FS05ImportantFeatures, FS05ValMetrics, FS05TestMetrics, FS05CIs, \
    FS05BestConfig, FS05TrainData, FS05TestData, FS05FNames = forward_feature_selection_pipeline()

    with open(FS0To5ResPath, 'wb') as handle:
        pickle.dump([FS05BestTrainedModel, FS05ImportantFeatures, FS05ValMetrics, FS05TestMetrics, FS05CIs, \
                     FS05BestConfig, FS05TrainData, FS05TestData, FS05FNames], handle)

    rng = np.random.default_rng(randomState)

    with open(resGT6Path, 'rb') as handle:
        resValGT6, resTestGT6, bestArchGT6, trModelsGT6, \
        trDataGT6, testDataGT6, fNamesGT6 = pickle.load(handle)

    agesGT6 = pd.read_csv(pathGT6, index_col=0, dtype=int)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(agesGT6), p=dataSplits)
    agesGT6Train, agesGT6Test = agesGT6[trainIndexes], agesGT6[~trainIndexes]

    bestArchitecture = resValGT6.iloc[resValGT6.drop('Arch', axis=1).idxmax()['AUC']]['Arch']
    bestConfig = bestArchGT6[bestArchitecture]

    # SVC is the best architecture

    modelEvalParams = {'modelClass': SVC,
                       'configs': [bestConfig],
                       'inputNNeigh': inputNNeigh,
                       'missingVal': missingVal,
                       'nCVs': nCVs,
                       'CVFolds': CVFolds,
                       'USnegProp': USnegProp,
                       'imputeMaxIter': imputeMaxIter,
                       'randomState': randomState}

    FS617BestTrainedModel, FS617ImportantFeatures, FS617ValMetrics, FS617TestMetrics, FS617CIs, \
    FS617BestConfig, FS617TrainData, FS617TestData, FS617FNames = forward_feature_selection_pipeline()

    with open(FSGT6ResPath, 'wb') as handle:
        pickle.dump([FS617BestTrainedModel, FS617ImportantFeatures, FS617ValMetrics, FS617TestMetrics, FS617CIs, \
                     FS617BestConfig, FS617TrainData, FS617TestData, FS617FNames], handle)


if __name__ == '__main__':
    forward_feature_selection_pipeline()
