import copy
import logging
import random
import re
import string
from typing import Any, Dict, List

from roleplay.common.device import DEVICE


class Model:
    def __init__(self, cfg: List[Dict[str, Any]], role=None):
        self.cfg = cfg
        self.conv_template = cfg.conv_template
        self.spec_tokens = None
        self.aim_run = None
        self.model = None
        self.tokenizer = None
        self.role = role
        self.history = []
        self._load_model()

    def _load_model(self):
        raise NotImplementedError("method '_load_model' is not implemented")

    def get_prompt(self, turn, response_msg, persona=None, instructions=None):
        raise NotImplementedError("method 'get_prompt' is not implemented")

    def generate(self, prompt):
        raise NotImplementedError("method 'generate' is not implemented")

    def update_history(self, prompt, output_extract):
        raise NotImplementedError("method 'update_history' is not implemented")

    def extract_prompt(self, prompt: str) -> str:
        if '"' in prompt:
            prompts = re.findall(r'"((?:.|[\n\t\r\b\\"])*?)"', prompt)
            if len(prompts) > 1:
                self.aim_run["num_multiple_prompts"] += 1
                logging.warning(f"More than one ({len(prompts)}) prompt detected!")
                return prompts[self.conv_template.idx_of_possible_prompt], len(prompts)
            elif prompts == []:
                logging.warning(f"No prompt detected!")
                return None, 0
            else:
                return prompts[0], len(prompts)
        else:
            logging.warning(f"No prompt detected!")
            return None, 0

    def stop_dialog(self, prompt):
        translator = str.maketrans("", "", string.punctuation)
        prompt_first_token = (
            re.split(r"\s+|\n", prompt.strip())[0].strip().translate(translator).strip()
        )
        prompt_last_token = (
            re.split(r"\s+|\n", prompt.strip())[-1]
            .strip()
            .translate(translator)
            .strip()
        )
        if (
            self.spec_tokens.conv_stop_token == prompt_first_token
            or self.spec_tokens.conv_stop_token == prompt_last_token
            or self.spec_tokens.conv_stop_token.capitalize() == prompt_first_token
            or self.spec_tokens.conv_stop_token.capitalize() == prompt_last_token
        ):
            return True
        return False

    def is_non_coherent(self, text):
        """
        max_n: from 2-grams to max_n-grams
        r: number of consecutive repititions
        """
        words = text.split()
        for n in range(2, self.cfg.non_coherent_max_n + 1):
            n_grams = []
            for i in range(len(words)):
                n_gram = tuple(words[i : i + n])
                if n_grams and len(n_grams) >= max(self.cfg.non_coherent_r, n):
                    if n_grams[-1] == n_gram or n_grams[-n] == n_gram:
                        last_rs = n_grams[-self.cfg.non_coherent_r :]
                        if (
                            len(last_rs) == self.cfg.non_coherent_r
                            and len(set(last_rs)) == 1
                        ):
                            return True
                        last_rs = n_grams[-n::-n][: self.cfg.non_coherent_r]
                        if (
                            len(last_rs) == self.cfg.non_coherent_r
                            and len(set(last_rs)) == 1
                        ):
                            return True
                n_grams.append(n_gram)
        return False

    def get_generation_cfg(self) -> Dict[str, Any]:
        A_generate_cfg = copy.deepcopy(self.cfg.generate)

        A_generate_cfg["do_sample"] = True
        A_generate_cfg["top_k"] = random.randint(5, 50)
        A_generate_cfg["penalty_alpha"] = random.random()
        A_generate_cfg["num_beams"] = random.randint(4, 10)
        A_generate_cfg["temperature"] = random.uniform(0.5, 1)

        return A_generate_cfg

    @staticmethod
    def collate_tokenize(data, tokenizer, dataset_cfg):
        input_batch = []
        for element in data:
            if isinstance(element[dataset_cfg.input_key], list):
                input_text = " ".join(element[dataset_cfg.input_key])
            else:
                input_text = element[dataset_cfg.input_key]
            input_batch.append(input_text)
        tokenized = tokenizer(
            input_batch, padding="longest", truncation=True, return_tensors="pt"
        )
        tokenized.to(DEVICE)
        return tokenized
