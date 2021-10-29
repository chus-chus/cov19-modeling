import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math


class MLP(nn.Module):
    """ Fully connected network (Multi-layer perceptron) semi-skeleton. The size of intermediate layers need to be
    defined. This class is meant to be trained with its `.fit` method. """

    def __init__(self, linearLayerSizes, nDropOutLayers=0.5, dropOutP=0.1,
                 batchSize=15, nEpochs=30, initLR=1e-3, activation='relu', randomState=888):
        """ dropOut layers will be randomly applied to some of the provided Linear Layers.
        ReLU activations will follow all linear layers. The model will return probabilities.

        :param linearLayerSizes: (array-like) array containing the input and output sizes of
        the intermediate nn.Linear layers as multipliers of the number of features in the data,
        in order of execution. Note that the first and last layers are already implemented:
        first layer casts to size round(inputOutputScale*inputSize), last layer casts from size
        round(inputOutputScale*inputSize). inputOutputScale will be induced from the first
        entry of the first linear layer in linearLayerSizes. The following is a correct example of linearLayerSizes,
        equivalent to having intermediate Linear layers of shapes (round(inputScale*inputSize), 2*inputSize)
        and (2*inputSize, round(outputScale*inputSize)):

        >>> linearLayerSizes = [(inputScale, 2), (2, outputScale)]

        :param nDropOutLayers: (int or float) if int, indicates the number of intemediate linear
        layers that will be followed by a dropout <= len(linearLayerSizes). If float,
        it will indicate the percentage of linear layers that will be followed by a dropout
        and ceil(len(linearLayerSizes) * nDropOutLayers) will be picked. In both cases, which
        layers are followed by the dropout will be randomly chosen.
        :param dropOutP: (float) percentage of dropout for the picked layers.
        :param batchSize: (int) batch size for training
        :param nEpochs: (float) number of epochs used for training
        :param initLR: (float) initial learning rate for the adaptive optimizer
        :param activation: (str) Nonlinear activation function to use:
          - 'relu': rectified linear unit
          - 'rrelu': randomized leaky rectified linear unit
          - 'sigmoid': sigmoid
          - 'tanh': hyperbolic tangent"""

        super(MLP, self).__init__()
        torch.manual_seed(randomState)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'rrelu':
            self.activation = F.rrelu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"{activation} activation not valid. Use one in {['relu', 'rrelu', 'sigmoid', 'tanh']}")

        self.rng = np.random.default_rng(randomState)

        self.inputScale = None
        self.outputScale = None

        # it will be induced when calling the .fit method
        self.inputSize = None

        # their input and output sizes will be adjusted when self.inputSize is induced
        self.firstLinLayer = nn.Linear(1, 2)
        self.lastLinLayer = nn.Linear(2, 1)

        # will be filled when self.inputSize is induced in the .fit method
        self.linearLayers = []
        self.linearLayerSizes = linearLayerSizes

        if isinstance(nDropOutLayers, int):
            self.nDropOutLayers = nDropOutLayers
        elif isinstance(nDropOutLayers, float):
            self.nDropOutLayers = math.ceil(nDropOutLayers * len(self.linearLayerSizes))

        if self.nDropOutLayers > len(self.linearLayerSizes) or self.nDropOutLayers < 0:
            raise ValueError(f'Invalid number of dropout layers ({self.nDropOutLayers})')

        # indexes of layers that will be followed by a dropout layer.
        self.dropOutIndexes = set(self.rng.choice(list(range(len(self.linearLayerSizes))),
                                                  size=self.nDropOutLayers,
                                                  replace=False))

        self.dropOut = nn.Dropout(dropOutP)
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.initLR = initLR

        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        """ :param x: (torch.FloatTensor) of shape (nSamples, inputSize) """

        out = self.activation(self.firstLinLayer(x))
        for i, linearLayer in enumerate(self.linearLayers):
            out = self.activation(linearLayer(out))
            if i in self.dropOutIndexes:
                out = self.dropOut(out)

        # return raw scores
        return self.lastLinLayer(out)

    def fit(self, X, y):
        """
        :param X: (np.ndarray) of shape (nSamples, inputSize)
        :param y: (np.ndarray) of shape (nSamples,) """

        # input size is induced when calling this method for similarity to the sklearn
        #  API. The first and last linear layer sizes are adjusted accordingly.

        self.inputSize = X.shape[1]

        # instantiate intermediate layers
        for i, shape in enumerate(self.linearLayerSizes):

            if i == 0:
                self.inputScale = shape[0]
            elif i == len(self.linearLayerSizes) - 1:
                self.outputScale = shape[1]

            inputSize = round(shape[0] * self.inputSize)
            outputSize = round(shape[1] * self.inputSize)

            self.linearLayers.append(nn.Linear(inputSize, outputSize))

        self.firstLinLayer = nn.Linear(self.inputSize, round(self.inputScale * self.inputSize))
        self.lastLinLayer = nn.Linear(round(self.outputScale * self.inputSize), 2)

        # criterion and optimizer are hardcoded (except LR)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.initLR)

        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        self.train()
        for epoch in range(self.nEpochs):
            for batch in range(int(len(X) / self.batchSize)):
                XBatch = X[batch:batch + self.batchSize]
                yBatch = y[batch:batch + self.batchSize]
                self.zero_grad()
                out = self.forward(XBatch)
                loss = self.criterion(out, yBatch)
                loss.backward()
                self.optimizer.step()

        self.eval()

    def predict(self, x):
        """ Predicts class 1 or 0 for an observation or batch of observations.
        :param x: (np.ndarray) single observation of shape (inputSize,) or
        (nSamples, inputSize)
        :return: (np.ndarray) of shape (nSamples, 2) predicted labels """

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # not computing probs. with the class method to reduce calling overhead
        probs = F.softmax(self.forward(torch.FloatTensor(x)), dim=1)

        return torch.argmax(probs, dim=1).detach().numpy()

    def predict_proba(self, x):
        """ Predicts probabilities for class 1 or 0 for an observation or batch of observations.
        :param x: (np.ndarray) single observation of shape (inputSize,) or
        (nSamples, inputSize)
        :return: (np.ndarray) of shape (nSamples, 2) predicted probabilities """

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        out = self.forward(torch.FloatTensor(x))

        return F.softmax(out, dim=1).detach().numpy()