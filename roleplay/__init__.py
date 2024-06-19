import hydra
from aim import Run
from omegaconf import OmegaConf

from roleplay.utils.launcher import launch, launch_on_slurm
from roleplay.utils.slurm import is_submitit_available


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(args):
    cfg = OmegaConf.create(OmegaConf.to_container(args, resolve=True, enum_to_str=True))

    aim_run = Run(
        repo=cfg.aim.repo,
        experiment=cfg.action_config.experiment_name,
    )
    aim_run.set("cfg", cfg, strict=False)

    if cfg.slurm.use_slurm:
        assert is_submitit_available(), "Please 'pip install submitit' to schedule jobs on SLURM"

        launch_on_slurm(
            action_name=cfg.action_name,
            cfg=cfg,
            aim_run=aim_run,
        )
    else:
        launch(action_name=cfg.action_name, cfg=cfg, aim_run=aim_run)

    if aim_run.active:
        aim_run.close()


if __name__ == "__main__":
    main()
