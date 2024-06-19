# LLM Roleplay: Simulating Human-Chatbot Interaction

The LLM Roleplay (roleplay) codebase is built upon the [UrarTU framework](https://github.com/tamohannes/urartu). For detailed insights into its structure, please refer to the [Getting Started Guide](https://github.com/tamohannes/urartu/blob/master/getting_started.md).

## Installation
Getting started with roleplay is a breeze! 💨 Just follow these steps to set up the necessary packages and create a local package called `roleplay`:

- Clone the repository: `git clone git@github.com:tamohannes/llm_roleplay.git`
- Navigate to the project directory: `cd roleplay`
- Execute the magic command: `pip install .`


Adding a touch of convenience! 🪄 After running the previous command, `roleplay` will be set up and ready to use. Plus, an alias will be created, allowing you to access roleplay from any directory on your operating system effortlessly:

```bash
roleplay --help
```
<!-- > **Note for Usage on Slurm System**
> For an enhanced experience with the Slurm job cancellation process, it is recommended to utilize a specific fork of the `submitit` package available at: [https://github.com/tamohannes/submitit](https://github.com/tamohannes/submitit). This fork includes the `ResumableSlurmJob.on_job_fail` callback, which allows the incorporation of additional functionality within this callback to ensure a graceful job termination. -->



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
roleplay action_config=roleplay +action_config/task/model_inquirer=mixtral action_config.task.model_inquirer.api_token="YOUR_TOKEN"
```

The `action_config` parameter specifies which configuration file to use to run the action. After that, we specify the configuration file for the inquirer with the `model_inquirer` argument.

To execute the command on a Slurm cluster, configure the `roleplay/config/main.yaml` file with the corresponding fields, and then use the same command to run the job. For more details on how to edit the configuration files, please refer to the upcoming sections.

> **Huggingface Authentication**
> You might need to log in to HuggingFace to authenticate your use of Mistral 8x7B. To do this, use the `huggingface-cli` login command and provide your access token.
> To obtain a HuggingFace access token, refer to [User access tokens](https://huggingface.co/docs/hub/en/security-tokens).


## Configs: Tailoring Your Setup

The default configs which shape the way of configs are defined under `roleplay/config` directory:
- `roleplay/config/main.yaml`: This core configuration file sets the foundation for default settings, covering all available keys within the system.
- `roleplay/config/action_config` Directory: A designated space for specific action configurations.

### Crafting Customizations

You have two flexible options for tailoring your configurations in `roleplay`. 

1. **Custom Config Files**: To simplify configuration adjustments, `roleplay` provides a dedicated `configs` directory where you can store personalized configuration files. These files seamlessly integrate with Hydra's search path. The directory structure mirrors that of `roleplay/config`. You can define project-specific configurations in specially named files. For instance, the `roleplay.yaml` file within the `configs` directory can house all the configurations specific to your `roleplay` project, with customized settings.

    - **Personalized User Configs**: To further tailor configurations for individual users, create a directory named `configs_{username}` at the same level as the `configs` directory, where `{username}` represents your operating system username (check out `configs_tamoyan` for an example). The beauty of this approach is that there are no additional steps required. Your customizations will smoothly load and override the default configurations, ensuring a seamless and hassle-free experience. ✨

    The order of precedence for configuration overrides is as follows: `roleplay/config`, `configs`, `configs_{username}`, giving priority to user-specific configurations.

2. **CLI Approach**: For those who prefer a command-line interface (CLI) approach, `roleplay` offers a convenient method. You can enhance your commands with specific key-value pairs directly in the CLI. For example, modifying your working directory path is as simple as:

    ```bash
    roleplay action_config=roleplay action_config.workdir=PATH_TO_WORKDIR
    ```

Choose the method that suits your workflow best and enjoy the flexibility `roleplay` provides for crafting custom configurations.


### Effortless Launch

With `roleplay`, launching actions is incredibly easy, offering you two options. 🚀

- **Local Marvel:** This option allows you to run jobs on your local machine, right where the script is executed.
- **Cluster Voyage:** This choice takes you on a journey to the Slurm cluster. By adjusting the `slurm.use_slurm` setting in `config/main.yaml`, you can easily switch between local and cluster execution.

Enjoy the flexibility to choose the launch adventure that best suits your needs and goals!



---
You're all set to dive into goal-oriented, persona-based, diverse, and multi-turn dialogue generation with `roleplay`! 🌟 If you encounter any issues or have suggestions, feel free to open an issue for assistance. 😊



## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Hovhannes Tamoyan},
  title     = {LLM Roleplay: Simulating Human-Chatbot Interaction},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```
