import time
import random
from clearml import Task
from tqdm import tqdm

import config

task = Task.init(
    project_name=config.PROJECT_NAME,
    task_name='ci_cd_dummy1',
    reuse_last_task_id=False
)

task_list = task.get_tasks(project_name=config.PROJECT_NAME)
task_list = list(set([task_list[task_performed].name for task_performed in range(len(task_list))]))

for tasks_performed in range(len(task_list)):
    task.get_logger().report_scalar(
        title="List of Tasks",
        series="Series 1",
        iteration=tasks_performed,
        value=0
    )
    print(task_list[tasks_performed])
