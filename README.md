<!--- BADGES: START --->
[![arXiv](https://img.shields.io/badge/arXiv-TO.DO-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![GitHub - License](https://img.shields.io/github/license/UKPLab/roleplay)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Python Version](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
<!--- BADGES: END --->


# LLM Roleplay: Simulating Human-Chatbot Interaction

Roleplay is a Python package that provides an easy method for generating goal-oriented, persona-based multi-turn dialogues, simulating diverse human-chatbot interactions.
This repository contains the code and data for the LLM Roleplay method, as presented in the paper [LLM Roleplay: Simulating Human-Chatbot Interaction](https://arxiv.org). It includes all the experiment codes and the necessary data to replicate them, as described in the paper.


## More About Roleplay
<div>
The development of chatbots requires collecting a large number of human-chatbot dialogues to reflect the breadth of users' sociodemographic backgrounds and conversational goals.
However, the resource requirements to conduct the respective user studies can be prohibitively high and often only allow for a narrow analysis of specific dialogue goals and participant demographics.
In this paper, we propose LLM-Roleplay: a goal-oriented, persona-based method to automatically generate diverse multi-turn dialogues simulating human-chatbot interaction.
LLM-Roleplay can be applied to generate dialogues with any type of chatbot and uses large language models (LLMs) to play the role of textually described personas.
To validate our method we collect natural human-chatbot dialogues from different sociodemographic groups 
and conduct a human evaluation to compare real human-chatbot dialogues with our generated dialogues.
We compare the abilities of state-of-the-art LLMs in embodying personas and holding a conversation and find that our method can simulate human-chatbot dialogues with a high indistinguishability.
</div>

<p align="center">
  <img width="350" alt="roleplay-schema" src="https://github.com/UKPLab/roleplay/assets/23078323/c456327d-d95c-41d0-acd1-f75fefeaf18d">
</p>




The LLM Roleplay (roleplay) codebase is built upon the [UrarTU framework](https://github.com/tamohannes/urartu) (version 2). For detailed insights into its structure, please refer to the [Getting Started Guide](https://github.com/tamohannes/urartu/blob/master/getting_started.md).

## Installation
Getting started with roleplay is a breeze! 💨 Just follow these steps to set up the necessary packages and create a local package called `roleplay`:

- Clone the repository: `git clone git@github.com:UKPLab/roleplay.git`
- Navigate to the project directory: `cd roleplay`
- Execute the magic command: `pip install .`

🪄 After running the previous command, `roleplay` will install the required packages including the latest version of `urartu` (>=2.0) and make it ready to use.
Plus, an alias will be created, allowing you to access roleplay from any directory on your operating system effortlessly:

```bash
urartu --help
```

Now, to register `roleplay` under the corresponding name in `urartu` we need to run the following command by providing the path where the module is located, for more info refere to [UrarTU's documentation](https://pypi.org/project/urartu/):
```bash
urartu register --name=roleplay --path=PATH_TO_ROLEPLAY/roleplay
```

After this you can run `urartu -h` again to see the available modules under `launch` command and make sure that `roleplay` is present there.



### Exploring the Experiments

Before diving into using `roleplay`, let's set up [Aim](https://github.com/aimhubio/aim). This tool will track our experiment metadata and generated dialogues, storing them locally on our system.

Let's start the Aim server to store all the metadata and dialogues of our experiments. By default, it will run on port `53800`. Use this command to get it running:

```bash
aim server
```

Since we are running the Aim server on our local machine, we will use the address: `aim://0.0.0.0:53800`. For remote tracking, refer to [Track experiments with aim remote server](https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html).

To explore the wealth of metrics that Aim captures effortlessly, follow these steps:
- Navigate to the directory containing the `.aim` repository.
- Run the command that sparks the magic:
```bash
aim up
```


## Usage

Let's get started with generating dialogues using the `roleplay` action. The process is simple: just provide the name of the configuration file containing the action, followed by the action name itself. For the `roleplay` action, we'll initiate it by using the Mistral 8x7B model as the inquirer. 🎇

```bash
urartu launch --name=roleplay action_config=roleplay +action_config/task/model_inquirer=mixtral +action_config/task/model_responder=llama action_config.task.model_inquirer.api_token="YOUR_TOKEN"
```

The `action_config` parameter specifies which configuration file to use to run the action. Afterward, we define the configuration file for the inquirer using the `model_inquirer` argument and set the configuration for the responder with the `model_responder` argument.

To execute the command on a Slurm cluster, modify the `roleplay/configs/action_config/generate_dialogues.yaml` file with the corresponding fields, and then use the same command to run the job. For more details on how to edit the configuration files, please refer to the upcoming sections.

> **Huggingface Authentication**
> You might need to log in to HuggingFace to authenticate your use of Mistral 8x7B. To do this, use the `huggingface-cli` login command and provide your access token.
> To obtain a HuggingFace access token, refer to [User access tokens](https://huggingface.co/docs/hub/en/security-tokens).


## Configs: Tailoring Your Setup

The default configs which shape the way of configs are defined in `urartu` under `urartu/config` directory:
- `urartu/config/main.yaml`: This core configuration file sets the foundation for default settings, covering all available keys within the system.
- `urartu/config/action_config` Directory: A designated space for specific action configurations.
For more see the structure of [UrarTU](https://github.com/tamohannes/urartu).

### Crafting Customizations

You have two flexible options for tailoring your configurations in `roleplay`. 

1. **Custom Config Files**: To simplify configuration adjustments, `roleplay` provides a dedicated `configs` directory where you can store personalized configuration files. These files seamlessly integrate with Hydra's search path. The directory structure mirrors that of `urartu/config`. You can define project-specific configurations in specially named files.
The `generate_dialogues.yaml` file within the `configs` directory houses all the configurations specific to our `roleplay` project, with customized settings.

    - **Personalized User Configs**: To further tailor configurations for individual users, create a directory named `configs_{username}` at the same level as the `configs` directory, where `{username}` represents your operating system username (check out `configs_tamoyan` for an example). The beauty of this approach is that there are no additional steps required. Your customizations will smoothly load and override the default configurations, ensuring a seamless and hassle-free experience. ✨

    The order of precedence for configuration overrides is as follows: `urartu/config`, `roleplay/configs`, `roleplay/configs_{username}`, giving priority to user-specific configurations.

2. **CLI Approach**: For those who prefer a command-line interface (CLI) approach, `urartu` offers a convenient method. You can enhance your commands with specific key-value pairs directly in the CLI. For example, modifying your working directory path is as simple as:

    ```bash
    urartu launch --name=roleplay action_config=roleplay action_config.workdir=PATH_TO_WORKDIR
    ```

Choose the method that suits your workflow best and enjoy the flexibility `urartu` provides for crafting custom configurations.


### Effortless Launch

With `urartu`, launching actions is incredibly easy, offering you two options. 🚀

- **Local Marvel:** This option allows you to run jobs on your local machine, right where the script is executed.
- **Cluster Voyage:** This choice takes you on a journey to the Slurm cluster. By adjusting the `slurm.use_slurm` setting in `roleplay/configs/action_config/generate_dialogues.yaml`, you can easily switch between local and cluster execution.

Enjoy the flexibility to choose the launch adventure that best suits your needs and goals!



---
You're all set to dive into goal-oriented, persona-based, diverse, and multi-turn dialogue generation with `roleplay`! 🌟 If you encounter any issues or have suggestions, feel free to open an issue for assistance. 😊



## Cite

Please use the following citation:

```
% todo
@article{anonymous,
  title={LLM Roleplay: Simulating Human-Chatbot Interaction},
  author={Hovhannes Tamoyan, Hendrik Schuff, Iryna Gurevych},
  journal={axiv},
  year={2024}
}
```



## Contacts

[Hovhannes Tamoyan](mailto:hovhannes.tamoyan@tu-darmstadt.de), [Hendrik Schuff](schuff@ukp.tu-darmstadt.de)

Please feel free to contact us if you have any questions or need to report any issues.

## Links

[UKP Lab Homepage](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt Website](https://www.tu-darmstadt.de/index.en.jsp)


## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
