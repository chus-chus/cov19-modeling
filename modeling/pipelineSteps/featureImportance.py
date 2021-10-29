import pickle

import shap
import matplotlib.pyplot as plt

from modeling.configs import fgWholeResPath, avgWholePltPath, maxWholePltPath, beesWholePltPath, fg0To5Res, \
    bees0To5PltPath, fgResGT6Path, beesGT6PltPath
from modeling.utils.utils import change_fnames


def feature_importance_pipeline():
    with open(fgWholeResPath, 'rb') as handle:
        valMetricsFineGridWhole, testMetricsFineGridWhole, CIsFineGridWhole, bestConfigFineGridWhole, \
        trainedModelFineGridWhole, trainDataFineGridWhole, testDataFineGridWhole, fNamesFineGridWhole = pickle.load(
            handle)

    expWhole = shap.Explainer(model=trainedModelFineGridWhole,
                              algorithm='tree',
                              feature_names=change_fnames(fNamesFineGridWhole))
    shapValsTestWhole = expWhole(testDataFineGridWhole[0])

    shap.plots.bar(shapValsTestWhole.abs.mean(0), max_display=11, show=False)
    plot = plt.gcf()
    plot.savefig(avgWholePltPath, dpi=2000, bbox_inches='tight')

    shap.plots.bar(shapValsTestWhole.abs.max(0), max_display=11, show=False)
    plot = plt.gcf()
    plot.savefig(maxWholePltPath, dpi=2000, bbox_inches='tight')

    plot = shap.plots.beeswarm(shapValsTestWhole, max_display=16, show=False)
    plot = plt.gcf()
    plot.savefig(beesWholePltPath, dpi=2000, bbox_inches='tight')

    with open(fg0To5Res, 'rb') as handle:
        valMetricsFineGrid0To5, testMetricsFineGrid0To5, CIsFineGrid0To5, bestConfigFineGrid0To5, \
        trainedModelFineGrid0To5, trainDataFineGrid0To5, testDataFineGrid0To5, fNamesFineGrid0To5 = pickle.load(handle)

    exp0To5 = shap.Explainer(model=trainedModelFineGrid0To5,
                             algorithm='tree',
                             feature_names=change_fnames(fNamesFineGrid0To5))
    shapValsTest0To5 = exp0To5(testDataFineGrid0To5[0])

    shap.plots.beeswarm(shapValsTest0To5[:, :, 1], max_display=11, show=False)
    plot = plt.gcf()
    plot.savefig(bees0To5PltPath, dpi=2000, bbox_inches='tight')

    with open(fgResGT6Path, 'rb') as handle:
        bestAUCsFineGridGT6, valMetricsFineGridGT6, testMetricsFineGridGT6, CIsFineGridGT6, bestConfigFineGridGT6, \
        trainedModelFineGridGT6, trainDataFineGridGT6, testDataFineGridGT6, fNamesFineGridGT6 = pickle.load(handle)

    exp6 = shap.Explainer(model=trainedModelFineGridGT6,
                          algorithm='tree',
                          feature_names=change_fnames(fNamesFineGridGT6))
    shapValsTestGT6 = exp6(testDataFineGridGT6[0])

    shap.plots.beeswarm(shapValsTestGT6[:, :, 1], max_display=11, show=False)
    plot = plt.gcf()
    plot.savefig(beesGT6PltPath, dpi=2000, bbox_inches='tight')


if __name__ == "__main__":
    feature_importance_pipeline()