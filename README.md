<!--- BADGES: START --->

[![arXiv](https://img.shields.io/badge/ACL-Conference-0052CC?logo=acl&logoColor=white&style=for‑the‑badge)](https://aclanthology.org/2025.sicon-1.1/)
[![arXiv](https://img.shields.io/badge/arXiv-red?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.03974)
[![PyPI - Package Version](https://img.shields.io/pypi/v/llm-roleplay?logo=pypi&style=flat&color=orange)](https://pypi.org/project/llm-roleplay/)
[![GitHub - License](https://img.shields.io/github/license/UKPLab/roleplay)](https://opensource.org/licenses/Apache-2.0)
[![PyPI - Python Version](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

<!--- BADGES: END --->

# LLM Roleplay: Simulating Human-Chatbot Interaction

Roleplay is a Python package that provides an easy method for generating goal-oriented, persona-based multi-turn dialogues, simulating diverse human-chatbot interactions.
This repository contains the code and data for the LLM Roleplay method, as presented in the paper [LLM Roleplay: Simulating Human-Chatbot Interaction](https://aclanthology.org/2025.sicon-1.1/). It includes all the experiment codes and the necessary data to replicate them, as described in the paper.

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
  <img width="350" alt="llm-roleplay-schema" src="https://github.com/UKPLab/llm-roleplay/assets/23078323/c456327d-d95c-41d0-acd1-f75fefeaf18d">
</p>

The LLM Roleplay (llm-roleplay) codebase is built upon the [UrarTU framework](https://github.com/tamohannes/urartu) (version 2). For detailed insights into its structure, please refer to the [Getting Started Guide](https://github.com/tamohannes/urartu/blob/master/getting_started.md).

## Installation

Getting started with llm-roleplay is a breeze! 💨 Just follow these steps to set up the necessary packages and create a local package called `llm-roleplay`:

- Clone the repository: `git clone git@github.com:UKPLab/llm-roleplay.git`
- Navigate to the project directory: `cd llm-roleplay`
- Execute the magic command: `pip install .`

🪄 After running the previous command, `llm-roleplay` will install the required packages including the latest version of `urartu` (>=2.0) and make it ready to use.
Plus, an alias will be created, allowing you to access llm-roleplay from any directory on your operating system effortlessly:

```bash
urartu --help
```

### Exploring the Experiments

Before diving into using `llm-roleplay`, let's set up [Aim](https://github.com/aimhubio/aim). This tool will track our experiment metadata and generated dialogues, storing them locally on our system.

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

Let's get started with generating dialogues using the `llm-roleplay` action. The process is simple: just provide the name of the configuration file containing the action, followed by the action name itself. For the `llm-roleplay` action, we'll initiate it by using the Mistral 8x7B model as the inquirer. 🎇

Step 1: Navigate to the project directory
```bash
cd llm_roleplay
```

Step 2: Run the dialogue generation command
```bash
urartu action_config=dialogue_generator aim=aim slurm=slurm +action_config/task/model_inquirer=mixtral +action_config/task/model_responder=llama action_config.task.model_inquirer.api_token="YOUR_TOKEN"
```

🔐 Don’t forget to replace "YOUR_TOKEN" with your actual API token!

The `aim` and `slurm` configs read the Aim and Slurm configurations from `aim` and `slurm` files which are located in `llm_roleplay/configs_{username}/aim/aim.yaml` and `llm_roleplay/configs_{username}/slurm/slurm.yaml` respectively. The `action_config` parameter specifies which configuration file to use to run the action. Afterward, we define the configuration file for the inquirer using the `model_inquirer` argument and set the configuration for the responder with the `model_responder` argument.

To execute the command on a Slurm cluster, modify the `llm_roleplay/configs_{username}/slurm/slurm.yaml` file with the corresponding fields, and then use the same command to run the job. For more details on how to edit the configuration files, please refer to the upcoming sections.

> **Huggingface Authentication**
> You might need to log in to HuggingFace to authenticate your use of Mistral 8x7B. To do this, use the `huggingface-cli` login command and provide your access token.
> To obtain a HuggingFace access token, refer to [User access tokens](https://huggingface.co/docs/hub/en/security-tokens).

## Configs: Tailoring Your Setup

The default configs which shape the way of configs are defined in `urartu` under `urartu/config` directory:

- `urartu/config/main.yaml`: This core configuration file sets the foundation for default settings, covering all available keys within the system.
- `urartu/config/action_config` Directory: A designated space for specific action configurations.
  For more see the structure of [UrarTU](https://github.com/tamohannes/urartu).

### Crafting Customizations

You have two flexible options for tailoring your configurations in `llm-roleplay`.

1.  **Custom Config Files**: To simplify configuration adjustments, `llm-roleplay` provides a dedicated `configs` directory where you can store personalized configuration files. These files seamlessly integrate with Hydra's search path. The directory structure mirrors that of `urartu/config`. You can define project-specific configurations in specially named files.
    The `dialogue_generator.yaml` file within the `configs` directory houses all the configurations specific to our `llm-roleplay` project, with customized settings.

        - **Personalized User Configs**: To further tailor configurations for individual users, create a directory named `configs_{username}` at the same level as the `configs` directory, where `{username}` represents your operating system username (check out `configs_tamoyan` for an example). The beauty of this approach is that there are no additional steps required. Your customizations will smoothly load and override the default configurations, ensuring a seamless and hassle-free experience. ✨

        The order of precedence for configuration overrides is as follows: `urartu/config`, `llm_roleplay/configs`, `llm_roleplay/configs_{username}`, giving priority to user-specific configurations.

2.  **CLI Approach**: For those who prefer a command-line interface (CLI) approach, `urartu` offers a convenient method. You can enhance your commands with specific key-value pairs directly in the CLI. For example, modifying your working directory path is as simple as:

    ```bash
    urartu action_config=dialogue_generator action_config.workdir=PATH_TO_WORKDIR
    ```

Choose the method that suits your workflow best and enjoy the flexibility `urartu` provides for crafting custom configurations.

### Effortless Launch

With `urartu`, launching actions is incredibly easy, offering you two options. 🚀

- **Local Marvel:** This option allows you to run jobs on your local machine, right where the script is executed.
- **Cluster Voyage:** This choice takes you on a journey to the Slurm cluster. By adjusting the `slurm.use_slurm` setting in `llm_roleplay/configs/action_config/dialogue_generator.yaml`, you can easily switch between local and cluster execution.

Enjoy the flexibility to choose the launch adventure that best suits your needs and goals!

---

You're all set to dive into goal-oriented, persona-based, diverse, and multi-turn dialogue generation with `llm-roleplay`! 🌟 If you encounter any issues or have suggestions, feel free to open an issue for assistance. 😊

## Cite

Please use the following citation:

```
@inproceedings{tamoyan-etal-2025-llm,
    title = "{LLM} Roleplay: Simulating Human-Chatbot Interaction",
    author = "Tamoyan, Hovhannes  and
      Schuff, Hendrik  and
      Gurevych, Iryna",
    editor = "Hale, James  and
      Deuksin Kwon, Brian  and
      Dutt, Ritam",
    booktitle = "Proceedings of the Third Workshop on Social Influence in Conversations (SICon 2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.sicon-1.1/",
    pages = "1--26",
    ISBN = "979-8-89176-266-4",
}
```

## Contacts

[Hovhannes Tamoyan](mailto:hovhannes.tamoyan@tu-darmstadt.de), [Hendrik Schuff](schuff@ukp.tu-darmstadt.de)

Please feel free to contact us if you have any questions or need to report any issues.

## Links

[UKP Lab Homepage](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt Website](https://www.tu-darmstadt.de/index.en.jsp)

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
