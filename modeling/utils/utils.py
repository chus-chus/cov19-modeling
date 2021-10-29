""" Utility functions """
import itertools

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize

from scipy.stats import ttest_rel

from modeling.utils.pipelineDefinitions import eval_model


def undersample_data(X, y, randomState, negProp=0.5):
    """ Undersample data.
      :param X: (ndarray) data to undersample.
      :param y: (ndarray) binary ground truth labels.
      :param randomState: (int) random state.
      :param negProp: (float) float between 0 and 1 indicating the proportion
                      of negative samples desired. Default: 0.5 """

    rng = np.random.default_rng(randomState)
    nPositive = sum(y)
    nNegative = len(y) - nPositive
    sampleSize = round((negProp * nPositive) / (1 - negProp))
    if sampleSize > nNegative:
        raise Exception('There are not that many negative samples in the data')
    chosenNegatives = rng.choice(list(range(nNegative)), size=sampleSize)
    XNeg, yNeg = X[y == 0][chosenNegatives], y[y == 0][chosenNegatives]
    X = np.concatenate((X[y == 1], XNeg))
    y = np.concatenate((y[y == 1], yNeg))

    # shuffle
    shuffler = rng.permutation(len(y))
    X = X[shuffler]
    y = y[shuffler]

    return X, y


def remove_variance(data: pd.DataFrame, pctThreshold: float) -> pd.DataFrame:
    """ Delete columns from data that comprise less than 'threshold'% of variance. """
    sel = VarianceThreshold()

    dataNan = data.copy()
    dataNan[dataNan == -1] = np.NaN

    # fit to compute threshold variance percentile
    sel.fit(dataNan)
    # l1 normalisation (sum to 1)
    normVars = normalize(sel.variances_.reshape(1, -1), norm='l1').squeeze()

    threshold = np.max(sel.variances_[normVars < pctThreshold])
    sel = VarianceThreshold(threshold=threshold)
    dataThres = sel.fit_transform(dataNan)

    a = np.sort(np.array(list(zip(sel.variances_, data.columns))))
    delCols = {data.columns[i]: round(normVars[i], 5) for i in np.where(sel.variances_ <= threshold)[0]}
    print(f'Deleted {len(delCols)} variables: {delCols}')

    return data.loc[:, set(data.columns) - set(data.columns[np.where(sel.variances_ <= threshold)[0]])]


def gen_configs(n, grid, rng):
    if n is None:
        # all combinations: cartesian product
        prod = list(itertools.product(*[paramValues for paramValues in grid.values()]))
        configs = [{list(grid.keys())[i]: paramValues[i] for i in range(len(prod[0]))} for paramValues in prod]
    else:
        # generate n random combinations checking that they do not repeat
        maxCombinations = 0
        for i, paramKey in enumerate(grid.keys()):
            if i == 0:
                maxCombinations = len(grid[paramKey])
            else:
                maxCombinations *= len(grid[paramKey])
        if n > maxCombinations:
            raise ValueError(
                f'Number of combinations to compute cannot exceed max. number of combinations ({maxCombinations})')
        configs = []
        configSet = set()
        for i in range(n):
            config = {}
            for paramKey in grid.keys():
                config[paramKey] = rng.choice(grid[paramKey], size=1).item()
            if i > 0:
                while tuple(config.values()) in configSet:
                    config = {}
                    for paramKey in grid.keys():
                        config[paramKey] = rng.choice(grid[paramKey], size=1).item()
            configSet.add(tuple(config.values()))
            configs.append(config)
    return configs


def forward_feature_selection(trainData, testData, modelEvalParams, targetName='final_diagnosis_code'):
    """ Greedy forward feature selection.

    **Procedure**:
    1 Compute, for each of the features, the classifier score and sort them in a
    vector vf.
    2 Add the best feature to the vector of features to use f
    3 While vf has features fi
      3.1 If the classifier performs statistically (by mean AUC) better with fi in f:
        3.1.1 add fi to f
      3.2 remove fi from vf

    :param trainData: (pd.DataFrame) paediatric training data
    :param testData: (pd.DataFrame) paediatric testData data
    :param modelEvalParams: (dict) keyword arguments for the eval_model function
    (except 'trainingData' and 'testData')
    :return:
      - model instance trained with the best set of features
      - list of features picked as relevant, in order of importance
      - ... objects returned by the eval_model function """

    features = [[f, []] for f in trainData.columns if f != targetName]

    # 1
    for i, featurePair in enumerate(features):
        feature = featurePair[0]
        trData = trainData.loc[:, [feature, targetName]].copy()
        teData = testData.loc[:, [feature, targetName]].copy()

        AUCs, _, _, _, _, _, _, _ = eval_model(trainingData=trData,
                                               testData=teData,
                                               bootstrapTest=False,
                                               verbose=False,
                                               **modelEvalParams)

        # compare with mean AUC. We keep the array of AUCs themselves to
        # perform statistical tests.
        features[i][1] = AUCs

    bestAUCs = features[np.argmax([np.mean(f[1]) for f in features])][1]
    features = sorted(features, key=lambda t: np.mean(t[1]))
    features = [f[0] for f in features]

    # 2
    bestInitFeature = features.pop()
    importantFeatures = [bestInitFeature]
    print(f'Best Initial Feature: {bestInitFeature} with mean AUC of {round(np.mean(bestAUCs), 3)}')

    # 3
    while features:
        # 3.2
        feature = features.pop()

        trData = trainData.loc[:, importantFeatures + [feature] + [targetName]].copy()
        teData = testData.loc[:, importantFeatures + [feature] + [targetName]].copy()

        AUCs, _, _, _, _, _, _, _ = eval_model(trainingData=trData,
                                               testData=teData,
                                               bootstrapTest=False,
                                               verbose=False,
                                               **modelEvalParams)
        # 3.1 Statistical comparison (90% confidence)
        pVal = ttest_rel(bestAUCs, AUCs)[1]
        if pVal <= 0.1 and np.mean(AUCs) > np.mean(bestAUCs):
            importantFeatures = importantFeatures + [feature]
            print(f'Picked features: {importantFeatures}. Mean AUC from \
      {round(np.mean(bestAUCs), 3)} to {round(np.mean(AUCs), 3)} with p-value of {round(pVal, 5)}')
            bestAUCs = AUCs
        else:
            print(f'{feature} has not been picked')

    # compute metrics with best configuration (now including bootstrap for CIs)
    trData = trainData.loc[:, importantFeatures + [targetName]]
    teData = testData.loc[:, importantFeatures + [targetName]]

    print('Training model with best performing features...')
    AUCs, valMetrics, testMetrics, CIs, bestConfig, \
    trainedModel, trainData, testData, fNames = eval_model(trainingData=trData,
                                                           testData=teData,
                                                           bootstrapTest=True,
                                                           **modelEvalParams)

    return trainedModel, importantFeatures, valMetrics, testMetrics, \
           CIs, bestConfig, trainData, testData, fNames


def change_fnames(names):
    newNames = []
    for name in names:
        if name == 'fever_1':
            newName = 'fever 37.5-38 ºC'
        elif name == 'fever_2':
            newName = 'fever 38-39 ºC'
        elif name == 'fever_3':
            newName = 'fever > 39 ºC'
        elif name == 'total_days_fever_1':
            newName = 'fever for 1-2 days'
        elif name == 'total_days_fever_2':
            newName = 'fever for 3-7 days'
        elif name == 'total_days_fever_3':
            newName = 'fever for > 7 days'
        elif name == 'tos_1':
            newName = 'coughs'
        elif name == 'dysphonia_1':
            newName = 'dysphonia'
        elif name == 'resp_1':
            newName = 'dyspnea'
        elif name == 'tachypnea_1':
            newName = 'tachypnea'
        elif name == 'ausc_resp_1':
            newName = 'normal auscultation'
        elif name == 'ausc_resp_2':
            newName = 'pathological auscultation'
        elif name == 'ausc_type_1':
            newName = 'wheezing'
        elif name == 'ausc_type_2':
            newName = 'crackles'
        elif name == 'ausc_type_3':
            newName = 'wheezing & crackles'
        elif name == 'odynophagia_1':
            newName = 'odynophagia'
        elif name == 'nasal_congestion_1':
            newName = 'nasal congestion'
        elif name == 'fatiga_1':
            newName = 'fatigued'
        elif name == 'headache_1':
            newName = 'headache'
        elif name == 'conjuntivitis_1':
            newName = 'conjuntivitis'
        elif name == 'gi_symptoms_1':
            newName = 'gastrointestinal symptoms'
        elif name == 'abdominal_pain_1':
            newName = 'abdominal pain'
        elif name == 'vomiting_1':
            newName = 'vomits'
        elif name == 'dyarrea_1':
            newName = 'dyarrhea'
        elif name == 'dermatologic_1':
            newName = 'skin signs'
        elif name == 'rash_1':
            newName = 'skin rash'
        elif name == 'adenopathies_1':
            newName = 'lymphadenopathies (>1cm)'
        elif name == 'hemorrhagies_1':
            newName = 'hemorrhagies'
        elif name == 'irritability_1':
            newName = 'irritability (only < 2y-old)'
        elif name == 'neuro_1':
            newName = 'Neurologic manifestations'
        elif name == 'shock_1':
            newName = 'shock signs'
        elif name == 'taste_smell_1':
            newName = 'alteration in taste'
        elif name == 'smell_1':
            newName = 'alteration in smell'
        else:
            # raise ValueError(f'Name not recognized: {name}')
            newName = name
        newNames.append(newName)
    return newNames
