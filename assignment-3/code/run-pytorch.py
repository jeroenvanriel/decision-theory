import os
import shutil

from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.environment import EnvironmentReference
from azureml.core import ScriptRunConfig

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

project_folder = './breakout'
shutil.copytree('dqn', project_folder, dirs_exist_ok=True)

cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', max_nodes=4)

    # Create the cluster with the specified name and configuration
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # Wait for the cluster to complete, show the output log
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

env = Environment.get(name='pytorch-gpu', workspace=ws, version='3')
details = env.get_image_details(ws)
env = Environment.from_docker_image('pytorch-gpu', details.image)

src = ScriptRunConfig(source_directory=project_folder,
                      script='train.py',
                      compute_target=compute_target,
                      environment=env)

run = Experiment(ws, name='breakout').submit(src)
run.wait_for_completion(show_output=True)

print(run.get_portal_url())

