""" We define a function that performs the following procedure for some model type (SVM, NN, XGBoost...) and some
manually defined hyperparam. configurations.

Algorithm 1 (Modeling):

1. Split the data into training (TOTAL) and test.
2. For each Cross Validation Repeat:
  2.1. For each Cross Validation fold:
    - split training (TOTAL) into training and validation.
    - impute training ALONE -> impute validation WITH training
    - one-hot both training and validation
    - undersample the training data
    - For each hyperparam. configuration:
        - train model on training data
        - compute AUC, sensitivity, specificity, precision, F1 on validation data
3. Pick best configurations by performing pairwise t-tests with the obtained metrics
3. Average and save metrics for each config. across folds and CV repeats.
4. Impute training (TOTAL) -> Impute test WITH training (TOTAL)
5. Onehot training (TOTAL) and test
6. Undersample training (TOTAL)
7. Pick the best config. and boostrap to obtain test Confidence Intervals: train model with training (TOTAL) and
   evaluate on test
8. Return best config and evaluation metrics """

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils import resample

from numpy import median
from numpy import percentile
from scipy.stats import ttest_rel

from src.modeling.utils.utils import undersample_data


def eval_model(modelClass, trainingData, testData, configs, inputNNeigh=5,
               missingVal=-1, target='final_diagnosis_code', nCVs=5, CVFolds=2,
               USnegProp=0.5, nBootstraps=100, bootstrapTest=True, imputeMaxIter=5,
               verbose=True, randomState=888):
    """ CV Training and evaluation pipeline that follows the procedure described above
    (not including split into training and validation) with some predefined configurations.

    Prints the best found configuration and performance metrics. The best configuration
    is found with 5 x 2-fold CV and comparing average validation AUCs with paired t-tests as
    recommended in https://ieeexplore.ieee.org/document/6790639

    :param modelClass: (model method) Model class to test configurations on. Note that the parameter
                       should not be the initialized model, but rather the instancer
                       (i.e. 'network' instead of 'network()'). The class must implement
                       a 'fit(X, y)' training method, as well as a 'predict(X)' method for class
                       prediction, which returns the predicted label for a batch of
                       instances X.
    :param trainingData: (pd.DataFrame) training dataset.
    :param testData: (pd.DataFrame) test dataset.
    :param configs: (iterable of dicts) Iterable containing the different model
                    configurations. Each configuration must be a dictionary with
                    the keys being the exact parameter names and the entries
                    being the desired parameter value. Note that the configurations
                    must be able to be done via the initialization of a 'modelClass'
                    instance.
    :param OHEncoder: (sklearn.preprocessing.OneHotEncoder) Fitted one-hot encoder.
    :param inputNNeigh: (int, optional) number of instances considered when imputing with KNN.
    :param missingVal: (int, optional) value that represents a missing in the data.
    :param target: (str, optional) name of the target column.
    :param nCVs: (int, optional) number of times the cross validation is repeated
      (with different data in the folds).
    :param CVFolds: (int, optional) number of cross validation folds.
    :param USnegProp: (float) float between 0 and 1 indicating the proportion of
                      negative samples when undersampling.
    :param nBootstraps: (int) number of bootstrap rounds
    :param bootstrapTest: (bool, default True) should CIs be extracted from test via bootstrapping?
    :param imputeMaxIter: (int) maximum iterations of iterative imputer.
    :param verbose: (bool, default True) Print info. during the execution?
    :param randomState: (int, optional) random state for all purposes.
    :return: bestAUCs (list), valMetrics (dict), testMetrics (dict), CIs (dict of tuples),
             config (dict), trainedModel (callable), trainData (list),
             testData (list), OHFeatureNames (list):
             - bestAUCs: list with the CV validation AUCs of length nCVs * CVFolds
             - valMetrics: average evaluation metrics on the CV validation sets of the best configuration.
             - testMetrics: median bootstrap evaluation (if bootstrapTest=True)
             or single (if bootstrapTest=False) metrics on the test set given all of the training data of the best configuration
             - CIs: confidence intervals @95% for the test metrics (only if bootstrapTest=True)
             - config: best parameter configuration.
             - trainedModel: model instance trained with the best configuration that
                             was used to compute the test scores.
             - trainData: X and y, data used to train the best model (undersampled and imputed).
             - testData: X and y, data used to test the best model (imputed)
             - OHfeatureNames: list of strs. representing the feature names of the OH
                               representation of the data. """

    rng = np.random.default_rng(randomState)

    # fit a one-hot encoder with all the possible features in the data (without the target)
    # because the dataset has missing values (-1), we first replace them with an existing value (0).
    # This deletion is local and only done to initialise the encoder: the missing values will
    # be imputed further down the pipeline.
    # This step does not create information leakage.
    OHEncoder = OneHotEncoder(sparse=False, drop='first')
    OHEncoder.fit(pd.concat((trainingData, testData), axis=0)
                  .drop(target, axis=1)
                  .replace(to_replace=-1, value=0))

    # for each config we will have a dict. containing the specific configuration
    # and its performance metrics across the CV pipeline
    baseMetrics = {'AUC': 0, 'prec': 0, 'sensitivity': 0, 'specificity': 0, 'F1': 0}
    allMetrics = [{**config, **baseMetrics} for config in configs]

    # we will need arrays with the results for the statistical tests.
    for i in range(len(configs)):
        allMetrics[i]['AUC'] = []

    # split target and data and transform to numpy
    yTrainTotal = trainingData.loc[:, target].to_numpy(dtype=int)
    XTrainTotal = trainingData.drop(target, axis=1)

    if verbose:
        print(XTrainTotal.columns)

    XTrainTotal = XTrainTotal.to_numpy()

    yTest = testData.loc[:, target].to_numpy(dtype=int)
    XTest = testData.drop(target, axis=1).to_numpy()

    for nCV in range(nCVs):
        # pre-split training data into folds
        folds = KFold(n_splits=CVFolds, shuffle=True, random_state=randomState + nCV)
        if verbose:
            print(f'\n{nCV + 1}/{nCVs} CV')
        foldIndex = 0
        for trainIndexes, valIndexes in folds.split(XTrainTotal):
            if verbose:
                print(f'\nFold {foldIndex + 1}/{CVFolds}...')
            # get actual CV splits
            XTrain, XVal = XTrainTotal[trainIndexes], XTrainTotal[valIndexes]
            yTrain, yVal = yTrainTotal[trainIndexes], yTrainTotal[valIndexes]
            # impute training first and then validation with training in mind
            # to prevent information leakage
            maxValueTrainInput = np.max(XTrain, axis=0)
            maxValueValInput = np.max(np.concatenate((XTrain, XVal), axis=0), axis=0)

            for i in range(len(maxValueTrainInput)):
                if maxValueTrainInput[i] == 0:
                    # variable does not have multiple levels. Set its max. value to 1 as
                    # a minor hack to the IterativeImputer -> it raises an error if min and
                    # max value is the same. This does not make it impute invalid values.
                    maxValueTrainInput[i] = 1
                if maxValueValInput[i] == 0:
                    maxValueValInput[i] = 1

            imputer = IterativeImputer(estimator=KNeighborsClassifier(n_neighbors=inputNNeigh),
                                       missing_values=missingVal,
                                       initial_strategy='median',
                                       min_value=0,
                                       max_value=maxValueTrainInput,
                                       max_iter=imputeMaxIter,
                                       skip_complete=True,
                                       random_state=randomState)
            XTrain = imputer.fit_transform(XTrain)
            imputer = IterativeImputer(estimator=KNeighborsClassifier(n_neighbors=inputNNeigh),
                                       missing_values=missingVal,
                                       initial_strategy='median',
                                       min_value=0,
                                       max_value=maxValueValInput,
                                       skip_complete=True,
                                       max_iter=imputeMaxIter,
                                       random_state=randomState)
            XVal = imputer.fit(np.concatenate((XTrain, XVal), axis=0)).transform(XVal)

            # one-hot imputed data
            XTrain = OHEncoder.transform(XTrain)
            XVal = OHEncoder.transform(XVal)

            # undersample the training split
            XTrain, yTrain = undersample_data(XTrain, yTrain, rng, negProp=USnegProp)
            # for each config, train and validate
            configIndex = 0
            for config in configs:
                configMetrics = baseMetrics
                # train model
                model = modelClass(**config)
                model.fit(XTrain, yTrain)
                preds = model.predict(XVal)
                # compute validation metrics
                tn, fp, fn, tp = confusion_matrix(yVal, preds).ravel()
                configMetrics['AUC'] = [roc_auc_score(yVal, preds)]
                configMetrics['prec'] = tp / (tp + fp)
                configMetrics['sensitivity'] = tp / (tp + fn)
                configMetrics['specificity'] = tn / (tn + fp)
                configMetrics['F1'] = f1_score(yVal, preds)
                for metric in baseMetrics.keys():
                    allMetrics[configIndex][metric] += configMetrics[metric]
                configIndex += 1
            foldIndex += 1

    # Get best configurations by statistically comparing the resulting AUCs
    bestAUCs = [-1 for _ in range(nCVs * CVFolds)]
    bestConfig = None
    bestPVal = np.inf
    bestConfigIndex = -1

    for i, configMetrics in enumerate(allMetrics):
        for metric in baseMetrics.keys():
            if metric != 'AUC':
                configMetrics[metric] /= nCVs * CVFolds

        # paired t-test
        pVal = ttest_rel(configMetrics['AUC'], bestAUCs)[1]
        if i == 0:
            firstPVal = pVal
        if pVal <= 0.05 and np.mean(configMetrics['AUC']) > np.mean(bestAUCs):
            bestAUCs = configMetrics['AUC']
            bestConfig = configMetrics
            bestPVal = pVal
            bestConfigIndex = i

    if verbose:
        if firstPVal == bestPVal:
            print(
                f'First configuration appears to perform as well as the others (no statistical difference between configurations).')
        else:
            print(f'\nFound best configuration with p-value {round(bestPVal, 5)}')

    bestConfig['AUC'] = np.mean(bestConfig['AUC'])

    if verbose:
        print('Imputing full train dataset...')
    # impute whole train data (without splitting for CV) and test
    maxValueTrainTotalInput = np.max(XTrainTotal, axis=0)
    maxValueTestInput = np.max(np.concatenate((XTrainTotal, XTest), axis=0), axis=0)

    tr, ts = None, None
    for i in range(len(maxValueTrainTotalInput)):
        if maxValueTrainTotalInput[i] == 0:
            # variable does not have multiple levels, so maximum value needs to be bigger
            # than the minimum. The variable will be imputed with the same level, so
            # this does not introduce wrong levels.
            maxValueTrainTotalInput[i] = 1
            tr = i
        if maxValueTestInput[i] == 0:
            maxValueTestInput[i] = 1
            ts = i

    imputer = IterativeImputer(estimator=KNeighborsClassifier(n_neighbors=inputNNeigh),
                               missing_values=missingVal,
                               initial_strategy='median',
                               min_value=0,
                               max_value=maxValueTrainTotalInput,
                               skip_complete=True,
                               max_iter=imputeMaxIter,
                               random_state=randomState)
    XTrainTotal = imputer.fit_transform(XTrainTotal)
    imputer = IterativeImputer(estimator=KNeighborsClassifier(n_neighbors=inputNNeigh),
                               missing_values=missingVal,
                               initial_strategy='median',
                               min_value=0,
                               max_value=maxValueTestInput,
                               skip_complete=True,
                               max_iter=imputeMaxIter,
                               random_state=randomState)
    XTest = imputer.fit(np.concatenate((XTrainTotal, XTest), axis=0)).transform(XTest)

    # one hot whole train and test datasets
    XTrainTotal = OHEncoder.transform(XTrainTotal)
    XTest = OHEncoder.transform(XTest)

    # undersample
    XTrainTotal, yTrainTotal = undersample_data(XTrainTotal, yTrainTotal, rng, negProp=USnegProp)

    if bootstrapTest is True and verbose:
        print('\nBootstrapping test...')
    elif bootstrapTest is False:
        nBootstraps = 1

    metricsTest = {'AUC': [], 'prec': [], 'sensitivity': [], 'specificity': [], 'F1': []}
    for i in range(nBootstraps):
        XBoots, yBoots = resample(XTrainTotal, yTrainTotal, random_state=randomState + i)

        # train best configuration with the bootstrapped training set and
        # get metrics for the test data
        bestModel = modelClass(**configs[bestConfigIndex])
        bestModel.fit(XBoots, yBoots)
        preds = bestModel.predict(XTest)
        tn, fp, fn, tp = confusion_matrix(yTest, preds).ravel()
        metricsTest['AUC'].append(roc_auc_score(yTest, preds))
        metricsTest['prec'].append(tp / (tp + fp))
        metricsTest['sensitivity'].append(tp / (tp + fn))
        metricsTest['specificity'].append(tn / (tn + fp))
        metricsTest['F1'].append(f1_score(yTest, preds))

    if bootstrapTest is True:
        # get 95% confidence intervals
        alpha = 5
        lowerP = alpha / 2
        upperP = (100 - alpha) + (alpha / 2)
        CIs = {}
        for metric, values in metricsTest.items():
            CIs[metric] = (percentile(values, lowerP), percentile(values, upperP))
            metricsTest[metric] = median(values)

        return bestAUCs, {valMetric: bestConfig[valMetric] for valMetric in baseMetrics.keys()}, \
               metricsTest, CIs, configs[bestConfigIndex], bestModel, [XTrainTotal, yTrainTotal], \
               [XTest, yTest], OHEncoder.get_feature_names(trainingData.drop(target, axis=1).columns)

    else:
        # w/o confidence intervals if we do not have bootstrapping
        return bestAUCs, {valMetric: bestConfig[valMetric] for valMetric in baseMetrics.keys()}, \
               metricsTest, configs[bestConfigIndex], bestModel, [XTrainTotal, yTrainTotal], \
               [XTest, yTest], OHEncoder.get_feature_names(trainingData.drop(target, axis=1).columns)


def eval_model_pipeline(modelClasses, modelConfigs, trainingData, testData, **kwargs):
    """ Measures performance of multiple model architectures with different
        configurations following Algorithm 1.

    :param modelClasses: (list-like of model methods) methods to evaluate. Each element
                         of the collection is the model instancer (i.e. 'network'
                         instead of 'network()'). Each class must implement
                         a 'fit(X, y)' training method, as well as a 'predict(X)'
                         method for class prediction, which returns the predicted
                         label for a batch of instances X.
    :param modelConfigs: (2D list-like of dicts) A list containing, for each modelClass,
                         a list of architecture configurations represented as dicts.
                         modelConfigs[i] corresponds to the configurations for modelClasses[i].
    :param trainingData: (pd.DataFrame) Unimputed and unbalanced training data (including the target).
    :param testData: (pd.DataFrame) Unimputed test data (including the target).
    :param kwargs: extra keyword arguments that will be passed onto 'eval_model()'.
    :return: (pd.DataFrame, pd.DataFrame, dict of dicts, dict of callables, list, list)
             - average validation evaluation metrics
             - test evaluation metrics for the best configuration of each model
               class in 'modelClasses' with 95% confidence intervals
             - best performing configurations for each model class in 'modelClasses'
             - the trained models used to compute the test scores, respectively.
             - trainData (X and y) used to train all of the models (imputed and undersampled)
             - testData (X and y) used to test all of the best models configurations. (imputed) """

    resEntries = ['Arch', 'AUC', 'AUC CIs', 'sensitivity', 'sensitivity CIs',
                  'specificity', 'specificity CIs', 'prec', 'prec CIs', 'F1', 'F1 CIs']

    valEntries = ['Arch', 'AUC', 'prec', 'sensitivity', 'specificity', 'F1']

    bestArchitectures = {}
    trainedModels = {}

    resVal = []
    resTest = []
    for i, modelClass in enumerate(modelClasses):
        print(f'\nEvaluating {modelClass}')
        _, valScores, testScores, CIs, bestConfig, \
        trainedModel, finalTrData, finalTestData, OHfNames = eval_model(modelClass=modelClass,
                                                                        configs=modelConfigs[i],
                                                                        testData=testData,
                                                                        trainingData=trainingData,
                                                                        bootstrapTest=True,
                                                                        verbose=True,
                                                                        **kwargs)
        valScores['Arch'] = str(modelClass)
        testScores['Arch'] = str(modelClass)

        resValModel = []
        resTestModel = []

        for metric in valEntries:
            resValModel.append(valScores[metric])
        resVal.append(resValModel)

        for metric in valEntries:
            if metric == 'Arch':
                resTestModel.append(testScores[metric])
            else:
                resTestModel.append(testScores[metric])
                resTestModel.append(CIs[metric])
        resTest.append(resTestModel)

        bestArchitectures[str(modelClass)] = bestConfig
        trainedModels[str(modelClass)] = trainedModel

    resultsVal = pd.DataFrame(data=resVal, columns=valEntries)
    resultsTest = pd.DataFrame(data=resTest, columns=resEntries)

    return resultsVal, resultsTest, bestArchitectures, trainedModels, finalTrData, \
           finalTestData, OHfNames