import pickle
import numpy as np
import pandas as pd

from src.modeling.configs import randomState, fullDataPath, XGBGrid, fgWholeResPath, path0To5, RFGrid, pathGT6, fg0To5Res, \
    fgResGT6Path
from src.modeling.utils.pipelineDefinitions import eval_model
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from src.modeling.utils.utils import gen_configs


def fine_tuning_pipeline():
    # let us define 70 / 30 train test splits
    rng = np.random.default_rng(randomState)
    fullData = pd.read_csv(fullDataPath, index_col=0, dtype=int)
    trainIndexes = rng.choice([True, False], size=len(fullData), p=[0.7, 0.3])
    fullDataTrain, fullDataTest = fullData[trainIndexes], fullData[~trainIndexes]

    # common parameter values
    inputNNeigh = 10
    missingVal = -1
    nCVs = 5
    CVFolds = 2
    USnegProp = 0.5
    imputeMaxIter = 5

    nXGB = 750
    rng = np.random.default_rng(randomState)
    XGBConfigs = gen_configs(nXGB, XGBGrid, rng)

    # evaluate
    bestAUCsFineGridWhole, valMetricsFineGridWhole, testMetricsFineGridWhole, CIsFineGridWhole, bestConfigFineGridWhole, \
    trainedModelFineGridWhole, trainDataFineGridWhole, testDataFineGridWhole, fNamesFineGridWhole = eval_model(
        modelClass=XGBClassifier,
        trainingData=fullDataTrain,
        testData=fullDataTest,
        configs=XGBConfigs,
        inputNNeigh=inputNNeigh,
        missingVal=missingVal,
        nCVs=nCVs,
        CVFolds=CVFolds,
        USnegProp=USnegProp,
        imputeMaxIter=imputeMaxIter,
        randomState=randomState)

    with open(fgWholeResPath, 'wb') as handle:
        pickle.dump([valMetricsFineGridWhole, testMetricsFineGridWhole, CIsFineGridWhole,
                     bestConfigFineGridWhole, trainedModelFineGridWhole, trainDataFineGridWhole, testDataFineGridWhole,
                     fNamesFineGridWhole], handle)

    rng = np.random.default_rng(randomState)

    ages0To5 = pd.read_csv(path0To5, index_col=0, dtype=int)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(ages0To5), p=dataSplits)
    ages0To5Train, ages0To5Test = ages0To5[trainIndexes], ages0To5[~trainIndexes]

    nRF = 750
    rng = np.random.default_rng(randomState)
    RFConfigs = gen_configs(nRF, RFGrid, rng)

    # evaluate
    bestAUCsFineGrid0To5, valMetricsFineGrid0To5, testMetricsFineGrid0To5, CIsFineGrid0To5, bestConfigFineGrid0To5, \
    trainedModelFineGrid0To5, trainDataFineGrid0To5, testDataFineGrid0To5, fNamesFineGrid0To5 = eval_model(
        modelClass=RandomForestClassifier,
        trainingData=ages0To5Train,
        testData=ages0To5Test,
        configs=RFConfigs,
        inputNNeigh=inputNNeigh,
        missingVal=missingVal,
        nCVs=nCVs,
        CVFolds=CVFolds,
        USnegProp=USnegProp,
        imputeMaxIter=imputeMaxIter,
        randomState=randomState)

    with open(fg0To5Res, 'wb') as handle:
        pickle.dump([valMetricsFineGrid0To5, testMetricsFineGrid0To5, CIsFineGrid0To5, bestConfigFineGrid0To5,
                     trainedModelFineGrid0To5, trainDataFineGrid0To5, testDataFineGrid0To5, fNamesFineGrid0To5], handle)

    # let us define 70 / 30 train test splits
    rng = np.random.default_rng(randomState)

    agesGT6 = pd.read_csv(pathGT6, index_col=0, dtype=int)

    # Define 70 / 30 train test splits
    dataSplits = [0.7, 0.3]
    trainIndexes = rng.choice([True, False], size=len(agesGT6), p=dataSplits)
    agesGT6Train, agesGT6Test = agesGT6[trainIndexes], agesGT6[~trainIndexes]

    nRF = 750
    rng = np.random.default_rng(randomState)
    RFConfigs = gen_configs(nRF, RFGrid, rng)

    # evaluate
    bestAUCsFineGridGT6, valMetricsFineGridGT6, testMetricsFineGridGT6, CIsFineGridGT6, bestConfigFineGridGT6, \
    trainedModelFineGridGT6, trainDataFineGridGT6, testDataFineGridGT6, fNamesFineGridGT6 = eval_model(
        modelClass=RandomForestClassifier,
        trainingData=agesGT6Train,
        testData=agesGT6Test,
        configs=RFConfigs,
        inputNNeigh=inputNNeigh,
        missingVal=missingVal,
        nCVs=nCVs,
        CVFolds=CVFolds,
        USnegProp=USnegProp,
        imputeMaxIter=imputeMaxIter,
        randomState=randomState)

    with open(fgResGT6Path, 'wb') as handle:
        pickle.dump(
            [bestAUCsFineGridGT6, valMetricsFineGridGT6, testMetricsFineGridGT6, CIsFineGridGT6, bestConfigFineGridGT6, \
             trainedModelFineGridGT6, trainDataFineGridGT6, testDataFineGridGT6, fNamesFineGridGT6], handle)


if __name__ == '__main__':
    fine_tuning_pipeline()
