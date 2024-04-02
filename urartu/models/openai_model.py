from typing import Tuple, Union

import tiktoken
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from urartu.common.model import Model


class OpenAIModel(Model):
    def __init__(self, cfg, role) -> None:
        super().__init__(cfg, role)

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.model = AzureChatOpenAI(
            deployment_name=self.cfg.name,
            openai_api_type=self.cfg.openai_api_type,
            openai_api_version=self.cfg.openai_api_version,
            azure_endpoint=self.cfg.azure_openai_endpoint,
            openai_api_key=self.cfg.azure_openai_api_key,
        )

    def get_prompt(self, turn, response_msg, persona=None, instructions=None):
        if self.role == "model_A":
            assert persona is not None, "persona cannot be None"
            assert instructions is not None, "instructions cannot be None"

            if turn == 0:
                self.sys_prompt = self.conv_template.system_prompt.replace(
                    self.spec_tokens.persona_placeholder, persona
                ).replace(
                    self.spec_tokens.conv_stop_placeholder,
                    self.spec_tokens.conv_stop_token,
                )
                prompt = self.conv_template.first_turn_input.replace(
                    self.spec_tokens.objective_placeholder,
                    f"{instructions[0]}",
                    # f"{instructions[0][0].lower()}{instructions[0][1:]} {sample['input']}",
                )

                return prompt
            else:
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
                    ),
                )
        elif self.role == "model_B":
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

    def generate(self, prompt: Union[str, Tuple[str, str]], generate_cfg):
        if not self.history:
            self.history = [
                SystemMessage(content=self.sys_prompt),
                HumanMessage(content=prompt),
            ]
        else:
            self.history.append(HumanMessage(content=prompt))

        num_history_words = sum([self._get_num_tokens(item.content) for item in self.history])
        if generate_cfg.max_new_tokens + num_history_words > self.cfg.context_length:
            delta = generate_cfg.max_new_tokens + num_history_words - self.cfg.context_length
            i = 1
            while delta > 0:
                len_human_utterance = self._get_num_tokens(self.history[i].content)
                len_aiassistant_utterance = self._get_num_tokens(self.history[i+1].content)
                delta -= (len_human_utterance + len_aiassistant_utterance)
                i += 2
            del self.history[1: i]
        try:
            turn_response = self.model(self.history)
        except Exception as e:
            print(e)
            return None, None

        model_output_template = self.conv_template.model_output.replace(
            self.spec_tokens.model_answer, turn_response.content
        )

        return turn_response.content, model_output_template

    def update_history(self, prompt, output_extract):
        if self.role == "model_A":
            self.history.append(AIMessage(content=f'{prompt} "{output_extract}"'))
        elif self.role == "model_B":
            self.history.append(AIMessage(content=f"{prompt}{output_extract}"))
        else:
            raise NotImplemented(f"unknown role: {self.role}")

    def _get_num_tokens(self, string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
