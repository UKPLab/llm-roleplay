from aim import Run
from omegaconf import DictConfig


class Action:
    def __init__(self, cfg: DictConfig, aim_run: Run):
        self.cfg = cfg
        self.action_cfg = cfg.action_config
        self.aim_run = aim_run
