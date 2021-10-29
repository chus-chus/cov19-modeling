""" Main script for the modelling pipeline """

from src.modeling.pipelineSteps.featureImportance import feature_importance_pipeline
from src.modeling.pipelineSteps.fineTuning import fine_tuning_pipeline
from src.modeling.pipelineSteps.forwardFeatureSelection import forward_feature_selection_pipeline
from src.modeling.pipelineSteps.training import training_pipeline

if __name__ == '__main__':
    training_pipeline()
    fine_tuning_pipeline()
    forward_feature_selection_pipeline()
    feature_importance_pipeline()
