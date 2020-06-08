##########################  Workspace Setup  ##############################
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import azureml.core
from azureml.core import Workspace

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()
print(ws.name, ws.location, ws.resource_group, sep='\t')

experiment_name = 'Estimator_Pipeline_Experiment'

from azureml.core import Experiment
exp = Experiment(workspace=ws, name=experiment_name)

##########################################################################################

##########################  CPU Compute Cluster Setup  ###############################

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "cpu-cluster"

try:
    CPU_compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D13', 
                                                           max_nodes=2)

    # create the cluster
    CPU_compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    CPU_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=10)

# use get_status() to get a detailed status for the current cluster. 
print(CPU_compute_target.get_status().serialize())
##########################################################################################

##########################  GPU Compute Cluster Setup  ###############################

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# choose a name for your cluster
cluster_name = "gpu-cluster"

try:
    GPU_compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=4)

    # create the cluster
    GPU_compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    GPU_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=10)

# use get_status() to get a detailed status for the current cluster. 
print(GPU_compute_target.get_status().serialize())
##########################################################################################

############################# Dataset Reference Creation #################################

from azureml.core import Dataset

xrayimage_dataset = Dataset.get_by_name(ws, name='xray_image_ds')
traindata_dataset = Dataset.get_by_name(ws, name='train_data_ds')
validdata_dataset = Dataset.get_by_name(ws, name='valid_data_ds')
testdata_dataset = Dataset.get_by_name(ws, name='test_data_ds')
traintarget_dataset = Dataset.get_by_name(ws, name='train_target_ds')
validtarget_dataset = Dataset.get_by_name(ws, name='valid_target_ds')
testtarget_dataset = Dataset.get_by_name(ws, name='test_target_ds')

##########################################################################################

############################# Run Configuration Setup #################################

from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages = ['keras<=2.3.1','pandas','matplotlib',
                                                                                            'opencv-python','azure-storage-blob==2.1.0','tensorflow-gpu==2.0.0',
                                                                                            'azureml','azureml-core','azureml-dataprep',
                                                                                           'azureml-dataprep[fuse]','azureml-pipeline'])

##########################################################################################

############################# Pythonscript for preprocessing ###################################

from azureml.core import Workspace,Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep

import os
#script_folder = os.path.join(os.getcwd(), "PreProcessing")

print("Pipeline SDK-specific imports completed")

ws = Workspace.from_config()
datastore = Datastore.get(ws,"xray_datastore")

PreProcessingData = PipelineData("PreProcessingData", datastore=datastore)
 
preprocessing_step = PythonScriptStep(name="preprocessing_step",
                                      script_name="estimator_data_preprocessing.py", 
                                      compute_target=CPU_compute_target, 
                                      runconfig = run_config,
                                      #source_directory = script_folder,
                                      source_directory = 'train',
									  inputs=[xrayimage_dataset.as_named_input('xrayimage_dataset').as_mount('/temp/xray_images'),
                                              traindata_dataset.as_named_input('traindata_dataset'),
                                              validdata_dataset.as_named_input('validdata_dataset'),
                                              testdata_dataset.as_named_input('testdata_dataset'),
                                              traintarget_dataset.as_named_input('traintarget_dataset'),
                                              validtarget_dataset.as_named_input('validtarget_dataset'),
                                              testtarget_dataset.as_named_input('testtarget_dataset')],
                                      arguments=['--PreProcessingData', PreProcessingData], 
                                      outputs = [PreProcessingData],
                                      allow_reuse=True)

print("preprocessing_step")

##########################################################################################

############################# Pythonscript for Training ###################################

from azureml.pipeline.steps import EstimatorStep
from azureml.train.dnn import TensorFlow
import os
script_folder = os.path.join(os.getcwd(), "Train")


est = TensorFlow(source_directory = script_folder, 
                    compute_target = GPU_compute_target,
                    entry_script = 'estimator_training.py',
                    pip_packages = ['keras<=2.3.1','matplotlib','opencv-python','azure-storage-blob==2.1.0','tensorflow-gpu==2.0.0'],
                    conda_packages = ['scikit-learn==0.22.1'],
                    use_gpu = True )


est_step = EstimatorStep(name="Estimator_Train", 
                         estimator=est, 
                         estimator_entry_script_arguments=['--PreProcessingData', PreProcessingData],
                         inputs=[PreProcessingData],
                         runconfig_pipeline_params=None,
                         compute_target=compute_target)
                         
##########################################################################################

############################# Pythonscript for Model Registration ###################################

from azureml.pipeline.steps import PythonScriptStep 
import os
script_folder = os.path.join(os.getcwd(), "Register")

register_step = PythonScriptStep(name = "register_step",
                    script_name= "estimator_register.py",
                    runconfig = run_config,
                    source_directory = script_folder,
                    compute_target=CPU_compute_target 
                    )

##########################################################################################

############################# Define Run Sequence ###################################

est_step.run_after(preprocessing_step)
register_step.run_after(est_step)

##########################################################################################

############################# Build Pipeline ###################################

from azureml.pipeline.core import Pipeline
pipeline = Pipeline(workspace = ws,steps=[preprocessing_step,est_step,register_step])

##########################################################################################

############################# Validate Pipeline ###################################

pipeline.validate()
print("Pipeline validation complete")

##########################################################################################

############################# Submit Pipeline ###################################

pipeline_run = Experiment(ws, 'Pipeline_Experiment').submit(pipeline,pipeline_parameters={})
print("Pipeline is submitted for execution")

##########################################################################################

############################# Publish Pipeline ###################################

published_pipeline = pipeline_run.publish(name="My_New_Pipeline", description="My Published Pipeline Description")

##########################################################################################
