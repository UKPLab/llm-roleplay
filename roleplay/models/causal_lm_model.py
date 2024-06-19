import random
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from roleplay.common.device import AUTO_DEVICE
from roleplay.common.model import Model


class CausalLMModel(Model):
    SELF_REPLY_TOKENS = {
        "llama": "[INST",
        "vicuna": "### Human:",
    }

    def __init__(self, cfg, role) -> None:
        super().__init__(cfg, role)

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.name,
            cache_dir=self.cfg.cache_dir,
            device_map=AUTO_DEVICE,
            torch_dtype=eval(self.cfg.dtype),
            token=self.cfg.api_token,
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.name)
        self.model.eval()

    def get_prompt(self, turn, response_msg=None, persona=None, instructions=None):
        if self.role == "model_inquirer":
            assert persona is not None, "persona cannot be None"
            assert instructions is not None, "instructions cannot be None"

            if turn == 0:
                return (
                    self.conv_template.first_turn_input.replace(
                        self.spec_tokens.persona_placeholder, persona
                    )
                    .replace(
                        self.spec_tokens.objective_placeholder,
                        f"{instructions[0]}",
                        # f"{instructions[0][0].lower()}{instructions[0][1:]} {sample['input']}",
                    )
                    .replace(
                        self.spec_tokens.conv_stop_placeholder,
                        self.spec_tokens.conv_stop_token,
                    )
                )
            else:
                assert response_msg is not None, "response_msg cannot be None"

                if len(instructions) > 1 and turn < len(instructions):
                    response_forwarding = (
                        self.conv_template.mid_response_forwarding.replace(
                            self.spec_tokens.next_prompt, instructions[turn]
                        )
                    )
                else:
                    response_forwarding = (
                        self.conv_template.response_forwarding.replace(
                            self.spec_tokens.next_prompt, ""
                        )
                    )

                return self.conv_template.n_th_turn_input.replace(
                    self.spec_tokens.user_msg,
                    response_forwarding.replace(
                        self.spec_tokens.response_placeholder, response_msg
                    ).replace(
                        self.spec_tokens.conv_stop_placeholder,
                        self.spec_tokens.conv_stop_token,
                    ),
                )
        elif self.role == "model_responder":
            assert response_msg is not None, "response_msg cannot be None"

            if turn == 0:
                return self.conv_template.first_turn_input.replace(
                    self.spec_tokens.objective_placeholder,
                    response_msg,
                )
            else:
                return self.conv_template.n_th_turn_input.replace(
                    self.spec_tokens.user_msg, response_msg
                )
        else:
            raise NotImplemented(f"unknown role: {self.role}")

    def generate(self, prompt: str, generate_cfg):
        self.model.eval()
        model_prompt = prompt
        if self.history:
            model_prompt = f'{"".join(self.history)}{prompt}'
        prompt_tokenized = self.tokenizer.encode(model_prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            output_tokenized = self.model.generate(prompt_tokenized, **generate_cfg)

        output = self.tokenizer.decode(output_tokenized[0], skip_special_tokens=True)

        output_o = (
            output.replace(str(self.tokenizer.bos_token), "")
            .replace(str(self.tokenizer.eos_token), "")
            .strip()
        )

        model_prompt_o = (
            model_prompt.replace(str(self.tokenizer.bos_token), "")
            .replace(str(self.tokenizer.eos_token), "")
            .strip()
        )

        turn_response = output_o.replace(model_prompt_o, "", 1)

        # ----------------------------------- prevent potential self-reply -----------------------------------
        for self_reply_token in CausalLMModel.SELF_REPLY_TOKENS.values():
            if self_reply_token in turn_response:
                turn_response = turn_response.split(self_reply_token)[0]
                self.aim_run["num_self_replies"] += 1

        turn_response = turn_response.lstrip()
        model_output_template = self.conv_template.model_output.replace(
            self.spec_tokens.model_answer, turn_response
        )

        del output_tokenized

        if not turn_response:
            print(turn_response)
            generate_cfg["temperature"] = round(random.uniform(0.6, 1.0), 1)
            self.generate(prompt, generate_cfg)

        return turn_response, model_output_template

    def update_history(self, prompt, output_extract):
        if self.role == "model_inquirer":
            self.history.append(f'{prompt} "{output_extract}"')
        elif self.role == "model_responder":
            self.history.append(f"{prompt}{output_extract}")
        else:
            raise NotImplemented(f"unknown role: {self.role}")
