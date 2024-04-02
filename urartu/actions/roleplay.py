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
from urartu.common.persona import Persona


class Roleplay(Action):
    def __init__(self, cfg: DictConfig, aim_run: Run) -> None:
        super().__init__(cfg, aim_run)

    def track(self, prompt, name, context=None):
        self.aim_run.track(
            Text(prompt),
            name=name,
            context=context,
        )

    def run(self):
        self.aim_run["num_no_prompts"] = 0
        self.aim_run["num_multiple_prompts"] = 0
        self.aim_run["num_non_coherent"] = 0
        self.aim_run["num_regenerate_worked"] = 0
        self.aim_run["num_self_replies"] = 0
        self.aim_run["num_non_coherent_model_B"] = 0
        self.aim_run["personas"] = {}

        task_cfg = self.action_cfg.task

        records_dir = Path(self.action_cfg.workdir).joinpath(
            "dialogues",
            f"{task_cfg.model_A.name.split('/')[-1]}",
            str(self.aim_run.hash),
        )
        os.makedirs(records_dir, exist_ok=True)

        dataset = Dataset.get_dataset(task_cfg.dataset)
        personas = Persona.get_personas(task_cfg.persona)

        model_A = hydra.utils.instantiate(
            task_cfg.model_A.type, task_cfg.model_A, "model_A"
        )
        model_B = hydra.utils.instantiate(
            task_cfg.model_B.type, task_cfg.model_B, "model_B"
        )

        model_A.spec_tokens = task_cfg.spec_tokens
        model_B.spec_tokens = task_cfg.spec_tokens
        model_A.aim_run = self.aim_run
        model_B.aim_run = self.aim_run

        for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="samples"):
            for persona, persona_hash in tqdm(personas, desc="personas", leave=False):
                self.aim_run["personas"][persona_hash] = persona

                model_A.history = []
                model_B.history = []
                dialogue = []
                raw_dialogue = []

                instructions = [
                    instruct.lstrip().rstrip()
                    for instruct in sample["instruction"].split("\n")
                ]

                if self.action_cfg.task.model_A.regenerate_tries:
                    regeneratinon_idx = 0
                A_generate_cfg = None
                turn = 0
                with tqdm(total=task_cfg.num_turns, desc="turns", leave=False) as pbar:
                    while turn < task_cfg.num_turns:
                        pbar.set_postfix(turn=turn + 1)
                        # ------------------------------------------ Model A ------------------------------------------
                        A_prompt = model_A.get_prompt(
                            turn=turn,
                            response_msg=B_output if "B_output" in locals() else None,
                            persona=persona,
                            instructions=instructions,
                        )

                        self.track(
                            prompt=A_prompt,
                            name="A_input",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )
                        A_output, _ = model_A.generate(
                            prompt=A_prompt,
                            generate_cfg=(
                                A_generate_cfg
                                if A_generate_cfg
                                else self.action_cfg.task.model_A.generate
                            ),
                        )
                        if not A_output:
                            break
                        self.track(
                            prompt=A_output,
                            name="A_output",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )

                        # --------------------- if model_A failed to provide coherent text ---------------------
                        if model_A.is_non_coherent(A_output):
                            self.aim_run["num_non_coherent"] += 1
                            break

                        # --------------------- if model_A wants to stop the dialogue ---------------------
                        if model_A.stop_dialogue(A_output):
                            break

                        A_output_extract, num_prompts = model_A.extract_prompt(
                            prompt=A_output
                        )

                        if self.action_cfg.task.model_A.regenerate_tries:
                            # --------------------- if model_A failed to provide prompt ---------------------
                            if A_output_extract is None:
                                if (
                                    regeneratinon_idx
                                    < self.action_cfg.task.model_A.regenerate_tries
                                ):
                                    A_generate_cfg = model_A.get_generation_cfg()
                                    regeneratinon_idx += 1
                                    continue
                                else:
                                    self.aim_run["num_no_prompts"] += 1
                                    break
                            else:
                                if regeneratinon_idx != 0:
                                    self.aim_run["num_regenerate_worked"] += 1
                                    regeneratinon_idx = 0
                                    A_generate_cfg = None

                        if A_output_extract is None:
                            self.aim_run["num_no_prompts"] += 1
                            break

                        self.track(
                            prompt=A_output_extract,
                            name="A_output_extract",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "num_prompts": num_prompts,
                                "persona_hash": persona_hash,
                            },
                        )

                        # As the context for model_A is getting bigger much faster -> Starts answering it's own questions
                        # To prevent this keep in the A_history only the output prompt(the thing that model_B will see).
                        model_A.update_history(
                            prompt=A_prompt, output_extract=A_output_extract
                        )

                        # ------------------------------------------ Model B ------------------------------------------

                        B_prompt = model_B.get_prompt(
                            turn=turn, response_msg=A_output_extract
                        )

                        self.track(
                            prompt=B_prompt,
                            name="B_input",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )
                        B_output, B_model_output_template = model_B.generate(
                            prompt=B_prompt,
                            generate_cfg=self.action_cfg.task.model_B.generate,
                        )
                        if not B_output:
                            break
                        self.track(
                            prompt=B_output,
                            name="B_output",
                            context={
                                "sample_id": idx,
                                "turn": turn,
                                "persona_hash": persona_hash,
                            },
                        )

                        # --------------------- if model_B failed to provide coherent text ---------------------
                        if model_B.is_non_coherent(B_output):
                            self.aim_run["num_non_coherent_model_B"] += 1
                            break

                        model_B.update_history(
                            prompt=B_prompt, output_extract=B_model_output_template
                        )

                        # --------------------------------------- Save the Dialogue ---------------------------------------
                        dialogue.append(
                            {
                                "turn": turn,
                                "model_A": A_output_extract,
                                "model_B": B_output,
                            }
                        )
                        raw_dialogue.append(A_output_extract)
                        raw_dialogue.append(B_output)

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
                            "dialogue": dialogue,
                        }
                    )


def main(cfg: DictConfig, aim_run: Run):
    roleplay = Roleplay(cfg, aim_run)
    roleplay.run()
