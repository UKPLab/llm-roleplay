# Getting Started with UrarTU

Welcome to the UrarTU framework! This guide will lead you through the essential steps to initiate your project with UrarTU. We'll employ a simple text classification project as an example and walk you through it from start to finish.

For installation instructions and additional information, please visit the project's GitHub page: **[UrarTU GitHub Repository](https://github.com/tmynn/urartu)**.

![Figure 1: Schematic Layout of the UrarTU.](https://github.com/tmynn/urartu/assets/23078323/33bd1271-d3a5-4f07-b206-f45c711ca0d9)

Figure 1: Schematic Layout of the UrarTU.

## **Setting Up Your Configuration**

The first step in any UrarTU project is to create a configuration template. This template forms the foundation for your experiment's development process. Follow these steps to create your configuration file:

1. Create a new **`.yaml`** file within the **`configs`** directory.
2. Name the file **`text_classifier.yaml`** to indicate it's for text classification.

Here's a basic structure for your **`text_classifier.yaml`** configuration file:

```yaml
# @package _global_
action_name: text_classification

aim:
  repo: ./

action_config:
  workdir: ./
  experiment_name: Text Classification

  tasks:
    - task: text_classification
      dataset:
        name: imdb
        split: test
      model:
        name: facebook/bart-large-mnli
        cache_dir: ""
        dtype: torch.float16
      metric:
        name: accuracy
```

Don’t worry about the configs inside of the `tasks` argument for now. Their purpose will become evident in the upcoming sections.

This is a general configuration file for your text classification project. However, if multiple team members are working on the same project and have their specific configurations, follow these steps:

1. Create a directory at the same level as the **`configs`** directory and name it **`configs_{username}`**, where **`{username}`** is your OS username.
2. Copy the content of the general configuration file and paste it into the **`configs_{username}`** directory.
3. Customize the specific settings as needed. Suppose I prefer my Aim repository to be a remote URL rather than a local path.

 To achieve this, I've created a custom configuration that's unique to my setup:

```yaml
# @package _global_
action_name: text_classification

aim:
  repo: "aim://0.0.0.0:53800"

action_config:
  workdir: ./
  experiment_name: Text Classification

  tasks:
    - task: text_classification
      dataset:
        name: imdb
        split: test
      model:
        name: facebook/bart-large-mnli
        cache_dir: ""
        dtype: torch.float16
      metric:
        name: accuracy
```

## Enabling SLURM

With just a few straightforward SLURM configuration parameters, we can seamlessly submit our action to the SLURM system. To achieve this, let's make use of the pre-defined **`slurm`** configuration within the **`urartu/urartu/config/main.yaml`** file and customize it as needed:

> **Note: These configurations are fully functional for our SLURM cluster.**
> 

```yaml
slurm:
  # Whether or not to run the job on SLURM
  use_slurm: true
  # Name of the job on SLURM
  name: "text_classification"
  # Comment of the job on SLURM
  comment: "text classifiation job"
  # Partition of SLURM on which to run the job. This is a required field if using SLURM.
  partition: "gpu"
  account: "ukp-researcher"
  # Where the logs produced by the SLURM jobs will be output
  log_folder: "/mnt/beegfs/work/tamoyan/text_classification/.slurm_logs"
  # Maximum number of hours / minutes needed by the job to complete. Above this limit, the job might be pre-empted.
  time_hours: 1
  time_minutes: 0
  # Additional constraints on the hardware of the nodes to allocate (example 'volta' to select a volta GPU)
  constraint: ""
  # GB of RAM memory to allocate for each node
  mem_gb: 40
  # TCP port on which the workers will synchronize themselves with torch distributed
  port_id: 40050
  # Number of CPUs per GPUs to request on the cluster.
  num_cpu_per_proc: 4
  # Number of GPUs per node to request on the cluster.
  num_gpu_per_node: 4
  # Number of nodes to request on the cluster.
  num_nodes: 1
  # Number of processes per node to request on the cluster.
  num_proc_per_node: 1
  # Any other parameters for slurm (e.g. account, hint, distribution, etc.,) as dictated by submitit.
  # Please see https://github.com/facebookincubator/submitit/issues/23#issuecomment-695217824.
  additional_parameters: {}
```

Setting the **`use_slurm`** argument to **`true`** activates SLURM job submission. The other arguments align with familiar **`sbatch`** command options.

## **Creating the Action File**

Next, let's create the action file that will use the parsed configuration to kickstart your work. Follow these steps:

1. Navigate to the **`urartu/actions`** directory.
2. Create a new Python file named **`text_classifier.py`**. Ensure that the file name matches the **`action_name`** specified in your configuration for automatic recognition.

Inside **`text_classifier.py`**, define a main method with the following arguments:

```python
from aim import Run
from omegaconf import DictConfig

def main(cfg: DictConfig, aim_run: Run):
    text_classifier = TextClassifier(cfg, aim_run)
    text_classifier.run()
```

The **`cfg`** parameter will contain overridden parameters, and **`aim_run`** is an instance of our Aim run for tracking progress.

## **Implementing the `TextClassifier` Class**

Now, let's create the **`TextClassifier`** class:

```python
from urartu.common.action import Action

class TextClassifier(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def run(self):
        # Your code goes here
```

Ensure that **`TextClassifier`** inherits from the abstract **`Action`** class. From this point forward, you have full control to implement your text classification logic.

```python
from tqdm import tqdm

from urartu.common.dataset import Dataset
from urartu.common.metric import Metric
from urartu.common.model import Model

class TextClassifier(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def run(self):
        for task_cfg in self.action_cfg.tasks:
            dataset = Dataset.get_dataset(task_cfg.dataset)
            pipe, tokenizer = Model.get_pipe(task_cfg.model)
            metric = Metric.get_metric(task_cfg.metric)

            for idx, sample in tqdm(enumerate(dataset)):
                premise = sample["text"]
                hypothesis = {"negative": 0, "positive": 1}
                output = pipe(premise, list(hypothesis.keys()))

                model_prediction = output["scores"].index(max(output["scores"]))
                label = dataset["label"][idx]

                metric.add(predictions=model_prediction, references=label)

            final_score = metric.compute()
            self.aim_run.track(
                {"final_score": final_score},
                context={"subset": task_cfg.dataset.get("subset")},
                step=idx,
            )
```

Here, we utilize the HuggingFace pipeline for zero-shot classification to predict text labels in the dataset while measuring accuracy between predicted and gold labels. We then track the score using Aim.

Once you've completed these steps, you can easily run UrarTU from the command line by specifying **`text_classification`** as the **`action_config`**

```bash
urartu action_config=text_classification
```

### Batch Execution with Multiple Configurations

You can streamline your experimentation by using Hydra's `--multirun` flag, allowing you to submit multiple runs with different parameters all at once. For example, if you need to run the same script with various model `dtype`s, follow these steps:

1. Add a Hydra sweeper configuration at the end of your config file:
    
```yaml
hydra:
  sweeper:
    params:
      ++action_config.tasks.0.model.dtype: torch.float32, torch.bfloat16
```
    
The double plus sign (**`++`**) will append this configuration to the existing one, resulting in three runs with **`action_config.tasks.0.model.dtype`** set to **`torch.float16`**, **`torch.float32`**, and **`torch.bfloat16`**.
    
2. Execute the following command to start the batch runs:
    
```bash
urartu --multirun action_config=text_classification
```
    

This approach simplifies the process of running experiments with various configurations, making it easier to explore and optimize your models.

## Monitoring the progress of the run

To monitor your experiment's progress and view tracked metadata, simply initiate Aim with the following command:

```bash
aim up
```

You can expect a similar experience as demonstrated in the following image:

https://github.com/tmynn/urartu/assets/23078323/11705f35-e3df-41f0-b0d1-42eb846a5921


## **Resources**

UrarTU is built upon a straightforward combination of three widely recognized libraries. For more in-depth information on how each of these libraries operates, please consult their respective GitHub repositories:

- **Hydra**: **[GitHub Repository](https://github.com/facebookresearch/hydra), [Getting started | Hydra](https://hydra.cc/docs/1.3/intro/)**
- **Submit**: **[GitHub Repository](https://github.com/facebookincubator/submitit)**
- **Aim**: **[GitHub Repository](https://github.com/aimhubio/aim)**

These repositories provide detailed insights into the inner workings of each library.

# Future Steps

While UrarTU currently serves as a versatile experiment running framework, we're considering a more focused direction. Our vision includes narrowing its scope to cater specifically to Natural Language Processing (NLP) tasks. This involves defining a comprehensive NLP pipeline with meaningful abstractions, akin to the illustrative figure below:

![source: [https://github.com/tmynn/tmynNLP](https://github.com/tmynn/tmynNLP)](https://user-images.githubusercontent.com/23078323/231715470-3e394909-51f8-4bd0-8dd9-a2561387ec39.png)

source: [https://github.com/tmynn/tmynNLP](https://github.com/tmynn/tmynNLP)

This transition aims to enhance UrarTU's utility and efficiency in the realm of NLP, aligning it more closely with our needs.
