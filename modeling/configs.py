""" Common parameters and hyperparameter grid configurations """

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from modeling.utils.MLP import MLP

# --------------------------
# common parameters

# data paths MUST BE DEFINED!
fullDataPath = ''
path0To5 = ''
pathGT6 = ''

# training results
resFullPath = ''
resGT6Path = ''
res0To5Path = ''

# fine tuning results
fgWholeResPath = ''
fg0To5Res = ''
fgResGT6Path = ''

# forward feature selection results
fsResWholePath = ''
FS0To5ResPath = ''
FSGT6ResPath = ''

# plot paths
avgWholePltPath = ''
maxWholePltPath = ''
beesWholePltPath = ''

bees0To5PltPath = ''

beesGT6PltPath = ''

# vars
randomState = 888
inputNNeigh = 10
missingVal = -1
nCVs = 5
CVFolds = 2
USnegProp = 0.5
imputeMaxIter = 5

# --------------------------
# model architectures to try

modelClasses = [XGBClassifier, SVC, RandomForestClassifier, LogisticRegression, MLP]

# --------------------------
# defining hyperparam. grids

# XGBoost: 2187 total combinations
XGBGrid = {'learning_rate': [0.1, 0.01, 0.001],
           'n_estimators': [100, 300, 500],
           'max_depth': [4, 6, 8],
           'subsample': [0.25, 0.5, 0.8],
           'gamma': [0, 0.1, 1],  # default is 0
           'reg_alpha': [0, 0.1, 1],  # default is 0
           'reg_lambda': [0, 0.1, 1],  # default is 1
           'random_state': [randomState]}

# kSVM: 120 total combinations
kSVMGrid = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10],
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5, 6],
            'probability': [True],
            'random_state': [randomState]}

# RF: 756 total combinations
RFGrid = {'n_estimators': [25, 50, 100, 200, 300, 400, 500, 750, 1000],
          'max_depth': [3, 4, 5, 6, 7, 8, 9],
          'min_samples_split': [2, 4, 8, 16],  # default is 2
          'max_features': ['auto', 'sqrt', 'log2'],
          'random_state': [randomState]}

# LR: 11 combinations
LRGrid = {'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
          'random_state': [randomState]}

# MLP: 576 combinations
MLPGrid = {'linearLayerSizes': [((1.5, 2), (2, 1.5)),
                                ((1.5, 2), (2, 3), (3, 2), (2, 1), (1, 0.5)),
                                ((1.5, 2), (2, 3), (3, 3.5), (3.5, 3), (3, 2), (2, 1), (1, 0.5)),
                                ((1.5, 2), (2, 3), (3, 4), (4, 3), (3, 2), (2, 1), (1, 0.5))],
           'activation': ['relu', 'rrelu', 'tanh', 'sigmoid'],
           'nDropOutLayers': [0.1, 0.2, 0.3],
           'dropOutP': [0.1, 0.15, 0.2],
           'nEpochs': [25, 50, 75, 100],
           'batchSize': [50],
           'initLR': [1e-3],
           'randomState': [888]}

# --------------------------
# getting n random hyperparam. configurations
# if None, use all combinations

nXGB = 120
nSVM = 120
nRF = 120
nLR = None
nMLP = 120
