import time
from pathlib import Path
from typing import Dict

from aim import Run
from iopath.common.file_io import g_pathmgr

from roleplay.utils.io import makedir
from roleplay.utils.job import ResumableJob, ResumableSlurmJob


def create_submitit_executor(cfg: Dict):
    import submitit

    log_folder = Path(cfg.slurm.log_folder).joinpath(str(time.time()))
    makedir(log_folder)
    assert g_pathmgr.exists(log_folder), f"Specified cfg.slurm.log_folder={log_folder} doesn't exist"
    assert cfg.slurm.partition, "slurm.PARTITION must be set when using slurm"

    executor = submitit.AutoExecutor(folder=log_folder)
    timeout_min = cfg.slurm.time_hours * 60 + cfg.slurm.time_minutes
    executor.update_parameters(
        name=cfg.slurm.name,
        slurm_comment=cfg.slurm.comment,
        slurm_partition=cfg.slurm.partition,
        slurm_account=cfg.slurm.account,
        slurm_constraint=cfg.slurm.constraint,
        timeout_min=timeout_min,
        nodes=cfg.slurm.num_nodes,
        cpus_per_task=cfg.slurm.num_cpu_per_proc * cfg.slurm.num_proc_per_node,
        tasks_per_node=cfg.slurm.num_proc_per_node,
        gpus_per_node=cfg.slurm.num_gpu_per_node,
        slurm_mem=f"{cfg.slurm.mem_gb}G",
        mem_gb=cfg.slurm.mem_gb,
        slurm_additional_parameters=cfg.slurm.additional_parameters,
    )
    return executor


def launch_on_slurm(cfg: Dict, action_name: str, aim_run: Run):
    executor = create_submitit_executor(cfg)
    trainer = ResumableSlurmJob(action_name=action_name, cfg=cfg, aim_run=aim_run)

    job = executor.submit(trainer)
    print(f"SUBMITTED: {job.job_id}")

    return job


def launch(cfg: Dict, action_name: str, aim_run: Run):
    trainer = ResumableJob(action_name=action_name, cfg=cfg, aim_run=aim_run)
    trainer()
