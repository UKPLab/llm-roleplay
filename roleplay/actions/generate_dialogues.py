import os
from pathlib import Path

import hydra
import jsonlines
import torch
from aim import Run, Text
from omegaconf import DictConfig
from tqdm import tqdm
from urartu.common.action import Action
from urartu.common.dataset import Dataset

from roleplay.common.persona import Persona


class Roleplay(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def track(self, prompt, name, context=None):
        self.aim_run.track(
            Text(prompt),
            name=name,
            context=context,
        )

    def main(self):
        self.aim_run["num_no_prompts"] = 0
        self.aim_run["num_multiple_prompts"] = 0
        self.aim_run["num_non_coherent"] = 0
        self.aim_run["num_regenerate_worked"] = 0
        self.aim_run["num_self_replies"] = 0
        self.aim_run["num_non_coherent_model_responder"] = 0
        self.aim_run["personas"] = {}

        task_cfg = self.action_cfg.task

        records_dir = Path(self.action_cfg.workdir).joinpath(
            "dialogs",
            f"{task_cfg.model_inquirer.name.split('/')[-1]}",
            str(self.aim_run.hash),
        )
        os.makedirs(records_dir, exist_ok=True)

        dataset = Dataset.get_dataset(task_cfg.dataset)
        personas = Persona.get_personas(task_cfg.persona)

        model_inquirer = hydra.utils.instantiate(
            task_cfg.model_inquirer.type, task_cfg.model_inquirer, "model_inquirer"
        )
        model_responder = hydra.utils.instantiate(
            task_cfg.model_responder.type, task_cfg.model_responder, "model_responder"
        )

        model_inquirer.spec_tokens = task_cfg.spec_tokens
        model_responder.spec_tokens = task_cfg.spec_tokens
        model_inquirer.aim_run = self.aim_run
        model_responder.aim_run = self.aim_run

        for idx, sample in tqdm(
            enumerate(dataset.dataset), total=len(dataset.dataset), desc="samples"
        ):
            for persona, persona_hash in tqdm(personas, desc="personas", leave=False):
                self.aim_run["personas"][persona_hash] = persona

                model_inquirer.history = []
                model_responder.history = []
                dialog = []
                raw_dialog = []

                instructions = [
                    instruct.lstrip().rstrip()
                    for instruct in sample[task_cfg.dataset.input_key].split("\n")
                ]

                if self.action_cfg.task.model_inquirer.regenerate_tries:
                    regeneratinon_idx = 0
                inquirer_generate_cfg = None
                responder_output = None
                turn = 0
                with tqdm(total=task_cfg.num_turns, desc="turns", leave=False) as pbar:
                    while turn < task_cfg.num_turns:
                        pbar.set_postfix(turn=turn + 1)
                        # ------------------------------------------ Model A ------------------------------------------
                        inquirer_prompt = model_inquirer.get_prompt(
                            turn=turn,
                            response_msg=responder_output,
                            persona=persona,
                            instructions=instructions,
                        )

                        self.track(
                            prompt=inquirer_prompt,
                            name="inquirer_input",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )
                        inquirer_output, _ = model_inquirer.generate(
                            prompt=inquirer_prompt,
                            generate_cfg=(
                                inquirer_generate_cfg
                                if inquirer_generate_cfg
                                else self.action_cfg.task.model_inquirer.generate
                            ),
                        )
                        if not inquirer_output:
                            break
                        self.track(
                            prompt=inquirer_output,
                            name="inquirer_output",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )

                        # --------------------- if model_inquirer failed to provide coherent text ---------------------
                        if model_inquirer.is_non_coherent(inquirer_output):
                            self.aim_run["num_non_coherent"] += 1
                            break

                        # --------------------- if model_inquirer wants to stop the dialog ---------------------
                        if model_inquirer.stop_dialog(inquirer_output):
                            break

                        inquirer_output_extract, num_prompts = (
                            model_inquirer.extract_prompt(prompt=inquirer_output)
                        )

                        if self.action_cfg.task.model_inquirer.regenerate_tries:
                            # --------------------- if model_inquirer failed to provide prompt ---------------------
                            if inquirer_output_extract is None:
                                if (
                                    regeneratinon_idx
                                    < self.action_cfg.task.model_inquirer.regenerate_tries
                                ):
                                    inquirer_generate_cfg = (
                                        model_inquirer.get_generation_cfg()
                                    )
                                    regeneratinon_idx += 1
                                    continue
                                else:
                                    self.aim_run["num_no_prompts"] += 1
                                    break
                            else:
                                if regeneratinon_idx != 0:
                                    self.aim_run["num_regenerate_worked"] += 1
                                    regeneratinon_idx = 0
                                    inquirer_generate_cfg = None

                        if inquirer_output_extract is None:
                            self.aim_run["num_no_prompts"] += 1
                            break

                        self.track(
                            prompt=inquirer_output_extract,
                            name="inquirer_output_extract",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "num_prompts": num_prompts,
                                "persona_hash": persona_hash,
                            },
                        )

                        # As the context for model_inquirer is getting bigger much faster -> Starts answering it's own questions
                        # To prevent this keep in the inquirer_history only the output prompt(the thing that model_responder will see).
                        model_inquirer.update_history(
                            prompt=inquirer_prompt,
                            output_extract=inquirer_output_extract,
                        )

                        # ------------------------------------------ Model B ------------------------------------------

                        responder_prompt = model_responder.get_prompt(
                            turn=turn, response_msg=inquirer_output_extract
                        )

                        self.track(
                            prompt=responder_prompt,
                            name="responder_input",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )
                        responder_output, responder_model_output_template = (
                            model_responder.generate(
                                prompt=responder_prompt,
                                generate_cfg=self.action_cfg.task.model_responder.generate,
                            )
                        )
                        if not responder_output:
                            break
                        self.track(
                            prompt=responder_output,
                            name="responder_output",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )

                        # --------------------- if model_responder failed to provide coherent text ---------------------
                        if model_responder.is_non_coherent(responder_output):
                            self.aim_run["num_non_coherent_model_responder"] += 1
                            break

                        model_responder.update_history(
                            prompt=responder_prompt,
                            output_extract=responder_model_output_template,
                        )

                        # --------------------------------------- Save the dialog ---------------------------------------
                        dialog.append(
                            {
                                "turn": turn,
                                "model_inquirer": inquirer_output_extract,
                                "model_responder": responder_output,
                            }
                        )
                        raw_dialog.append(inquirer_output_extract)
                        raw_dialog.append(responder_output)

                        torch.cuda.empty_cache()
                        turn += 1
                        pbar.update(1)

                with jsonlines.open(
                    records_dir.joinpath(f"{self.cfg.seed}.jsonl"), mode="a"
                ) as writer:
                    writer.write(
                        {
                            "persona": persona,
                            "sample": sample,
                            "num_turns": turn,
                            "dialog": dialog,
                        }
                    )


def main(cfg: DictConfig, aim_run: Run):
    roleplay = Roleplay(cfg, aim_run)
    roleplay.main()
