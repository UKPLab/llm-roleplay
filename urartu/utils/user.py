import importlib
import os
import pwd


def get_current_user() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def get_user_config_module(file_path):
    config_path = f"configs.configs_{get_current_user}"
    config_path_hydra = config_path.replace(".", "/")
    config_file = file_path.split("/")[-1].split(".")[0]

    config_module = importlib.import_module(
        f"{config_path}.{config_file}", package=__name__
    )

    return config_module, config_path_hydra
