import os
import subprocess
from platform import node
from clearml import Task
from clearml.automation import PipelineController
from config import *
import random

def compare_metrics_and_publish_best(**kwargs):
    from clearml import OutputModel
    # Keep track of best node details
    current_best = dict()

    # For each incoming node, compare against current best
    for node_name, training_task_id in kwargs.items():
        # Get the original task based on the ID we got from the pipeline
        task = Task.get_task(task_id=training_task_id)
        accuracy = task.get_reported_single_value(name="Accuracy")
        model_id = task.get_models()['output'][0].id
        # Check if accuracy is better than current best, if so, overwrite current best
        if accuracy > current_best.get('accuracy', 0):
            current_best['accuracy'] = accuracy
            current_best['node_name'] = node_name
            current_best['model_id'] = model_id
            print(f"New current best model: {node_name}")

    # Print the final best model details and log it as an output model on this step
    print(f"Final best model: {current_best}")
    OutputModel(name="best_pipeline_model", base_model_id=current_best.get('model_id'), tags=['best_model'])

def execute_run(cmd,dir):
    import subprocess
    pipe = subprocess.Popen(cmd,shell=True,cwd=dir, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (out, error) = pipe.communicate()
    print(out, error)
    pipe.wait()

def push_to_git(branch_name= None):
    cwd = os.getcwd()
    execute_run("git status",cwd)
    execute_run(f"git checkout -b {branch_name}",cwd)
    execute_run("git add .",cwd)
    execute_run("git commit -m 'test'",cwd)
    execute_run(f"git push --set-upstream origin {branch_name}",cwd)




if __name__ == "__main__":
    """pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PROJECT_NAME,
        version='0.0.1'
    )

    pipe.set_default_execution_queue('default')
    pipe.add_parameter('trails', 3)
    pipe.add_parameter("query","SELECT * FROM loan_dataframe")

    pipe.add_step(
        name='ingest_data',
        base_task_project=PROJECT_NAME,
        base_task_name='data_ingestion',
    )
    pipe.add_step(
        name='preprocess_data',
        parents=['ingest_data'],
        base_task_project=PROJECT_NAME,
        base_task_name='preprocess_data',
    )
    training_nodes = []
    for i in range(pipe.get_parameters()['trails']):
        node_name = f'model_training_{i}'
        training_nodes.append(node_name)
        pipe.add_step(
            name=node_name,
            parents=['preprocess_data'],
            base_task_project=PROJECT_NAME,
            base_task_name='model_training',
            parameter_override={'General/num_estimators': random.randint(0, 100),
                                'General/max_depth': random.randint(0, 10),
                                }
        )

    pipe.add_function_step(
        name='select_best_model',
        parents=training_nodes,
        function=compare_metrics_and_publish_best,
        function_kwargs={node_name: '${%s.id}' % node_name for node_name in training_nodes},
        monitor_models=["best_pipeline_model"],
        post_execute_callback=
    )

    # for debugging purposes use local jobs
    #pipe.start_locally(run_pipeline_steps_locally=True)
    # Starting the pipeline (in the background)
    pipe.start()

    print('Done!')"""
    push_to_git("test_2")
