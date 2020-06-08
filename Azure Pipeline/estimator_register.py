import os
import sys
import argparse

from azureml.core import Run
from azureml.core.model import Model
from azureml.pipeline.steps import HyperDriveStep, HyperDriveStepRun


run= Run.get_context()
run_id = run.parent.id

parent_run = Run(experiment = run.experiment,run_id=run_id)


model = parent_run.register_model(model_name='xrayml_pipeline1', model_path='outputs/weights.best.dense_generator_callback.hdf5')