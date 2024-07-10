import importlib
import json
import unittest
from unittest.mock import patch

from aim import Run
from aim.storage.context import Context
from llm_roleplay.actions.dialogue_generator import DialogueGenerator
from omegaconf import OmegaConf
from urartu.common.device import Device


class TestRoleplay(unittest.TestCase):
    def setUp(self):
        # Configuration and Run mock setup
        self.cfg = OmegaConf.create(
            {
                "action_name": "test_roleplay",
                "aim": {"repo": "tmp"},
                "action_config": {
                    "workdir": "tmp",
                    "experiment_name": "test experiment for roleplay",
                    "device": "auto",
                    "task": {
                        "num_turns": 2,
                        "model_inquirer": {
                            "type": {"_target_": "llm_roleplay.models.causal_lm_model.CausalLMModel"},
                            "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                            "cache_dir": "",
                            "dtype": "torch.float16",
                            "non_coherent_max_n": 4,
                            "non_coherent_r": 2,
                            "regenerate_tries": None,
                            "api_token": None,
                            "generate": {"do_sample": True, "max_new_tokens": 1000},
                            "conv_template": {
                                "first_turn_input": (
                                    "[INST]\n"
                                    'You are <PERSONA>. You will start a conversation with an assistant. If you accomplish your final goal during the conversation only say "<CONV_STOP>".\n\n'
                                    "Your ultimate goal is as follows: <OBJECTIVE>. What prompt will you use to direct the assistant toward achieving your goal? Please provide the prompt within double quotes. Use simple language, keep the prompts brief, and be on point. Do not greet the assistant. Maintain a casual style; avoid being overly friendly, don't say thank you. [/INST]"
                                ),
                                "n_th_turn_input": "[INST] <USER_MSG> [/INST]",
                                "model_output": "<MODEL_ANSWER>",
                                "response_forwarding": (
                                    'If the assistant didn\'t help you achieve your goal, ask follow-up or clarification questions within double quotes. Be suspicious, curious, and demanding. Keep it simple, brief, and to the point. Stay casual; avoid being overly friendly. Assistant response: \n\n"<RESPONSE>".'
                                ),
                            },
                            "idx_of_possible_prompt": 0,
                        },
                        "model_responder": {
                            "type": {"_target_": "llm_roleplay.models.pipeline_model.PipelineModel"},
                            "name": "models--llama-2-hf/13B-Chat",
                            "cache_dir": "",
                            "dtype": "torch.float16",
                            "non_coherent_max_n": 5,
                            "non_coherent_r": 2,
                            "api_token": None,
                            "generate": {"max_new_tokens": 4000},
                            "conv_template": {
                                "first_turn_input": "[INST] <<SYS>>\n    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n    <</SYS>>\n\n    <OBJECTIVE> [/INST]",
                                "n_th_turn_input": "[INST] <USER_MSG> [/INST]",
                                "model_output": "<MODEL_ANSWER>",
                            },
                        },
                        "dataset": {
                            "type": {"_target_": "llm_roleplay.datasets.hf_datasets.HFDatasets"},
                            "input_key": "instruction",
                            "data": {
                                "instruction": [
                                    "You want to know how fast you run different distances. You use a stopwatch to measure the time it takes you to complete a 50-meter, 100-meter, and 200-meter race. You want to know how can you calculate your speed for each race? Based on that, you also want to calculate how many calories you burned during each race."
                                ]
                            },
                        },
                        "persona": {
                            "prompt": "<AGE>-year-old <RACE> individual with a <GENDER> gender identity, holding <EDUCATION> and English <NATIVE_ENGLISH> your native language",
                            "fixed": [
                                {
                                    "person": {
                                        "age": "a 25 to 34",
                                        "race": "White",
                                        "gender": "Male",
                                        "education": "Master's degree",
                                        "native_english": "is not",
                                    }
                                }
                            ],
                        },
                        "spec_tokens": {
                            "persona_placeholder": "<PERSONA>",
                            "objective_placeholder": "<OBJECTIVE>",
                            "response_placeholder": "<RESPONSE>",
                            "conv_stop_placeholder": "<CONV_STOP>",
                            "conv_stop_token": "FINISH",
                            "user_msg": "<USER_MSG>",
                            "model_answer": "<MODEL_ANSWER>",
                            "next_prompt": "<NEXT_PROMPT>",
                            "bos_token": "<BOS>",
                            "sep_token": "<SEP>",
                        },
                    },
                },
                "seed": 42,
            }
        )

        self.sample_inquirer_output = '''"Hey assistant, I'm looking to measure my running speed and calculate..."'''
        self.sample_responder_output = '''"Hello! I'd be happy to help you with your questions about measuring..."'''
        self.aim_run = Run(repo=self.cfg.aim.repo, experiment=self.cfg.action_config.experiment_name)
        self.aim_run.set("cfg", self.cfg, strict=False)

    def test_tracking_calls(self):
        dialogue_generator = DialogueGenerator(self.cfg, self.aim_run)
        self.assertEqual(
            self.aim_run["cfg"]["action_name"],
            self.cfg.action_name,
            "Action name in AIM run config does not match the expected value",
        )

        dialogue_generator.track(self.sample_inquirer_output, "test_inquirer_input")
        text_seq = self.aim_run.get_text_sequence("test_inquirer_input", context=Context({}))
        text_record = next(iter(text_seq.data), None)
        self.assertIsNotNone(text_record, "No text records found in AIM run tracking")

    @patch("llm_roleplay.models.openai_model.OpenAIModel.generate")
    @patch("llm_roleplay.models.pipeline_model.PipelineModel.generate")
    @patch("llm_roleplay.models.causal_lm_model.CausalLMModel.generate")
    @patch("llm_roleplay.models.openai_model.OpenAIModel._get_model")
    @patch("llm_roleplay.models.pipeline_model.PipelineModel._get_model")
    @patch("llm_roleplay.models.causal_lm_model.CausalLMModel._get_model")
    def test_initialization(
        self,
        mock_get_model_clm,
        mock_get_model_pipe,
        mock_get_mode_openai,
        mock_generate_clm,
        mock_generate_pipe,
        mock_generate_openai,
    ):
        dialogue_generator = DialogueGenerator(self.cfg, self.aim_run)
        dialogue_generator.initialize()

        self.assertTrue(hasattr(dialogue_generator, "task_cfg"), "dialogue_generator is missing 'task_cfg' attribute")
        self.assertIsNotNone(dialogue_generator.task_cfg, "'task_cfg' attribute is None")

        self.assertTrue(
            hasattr(dialogue_generator, "records_dir"), "dialogue_generator is missing 'records_dir' attribute"
        )
        self.assertIsNotNone(dialogue_generator.records_dir, "'records_dir' attribute is None")

        self.assertTrue(hasattr(dialogue_generator, "dataset"), "dialogue_generator is missing 'dataset' attribute")
        self.assertIsNotNone(dialogue_generator.dataset, "'dataset' attribute is None")
        self.assertEqual(
            dialogue_generator.dataset.dataset.num_rows,
            len(self.cfg.action_config.task.dataset.data.instruction),
            "Number of rows in dataset does not match expected value",
        )

        self.assertTrue(hasattr(dialogue_generator, "personas"), "dialogue_generator is missing 'personas' attribute")
        self.assertIsNotNone(dialogue_generator.personas, "'personas' attribute is None")
        self.assertEqual(
            len(dialogue_generator.personas),
            len(self.cfg.action_config.task.persona.fixed),
            "Mismatch in number of fixed personas",
        )

        self.assertTrue(
            hasattr(dialogue_generator, "model_inquirer"), "dialogue_generator is missing 'model_inquirer' attribute"
        )
        self.assertIsNotNone(dialogue_generator.model_inquirer, "'model_inquirer' attribute is None")
        class_path = self.cfg.action_config.task.model_inquirer.type._target_
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        assert isinstance(dialogue_generator.model_inquirer, getattr(module, class_name)), f"The 'model_inquirer' should be an instance of {class_name} from {module_name}, but got {type(dialogue_generator.model_inquirer).__name__}"

        self.assertTrue(
            hasattr(dialogue_generator, "model_responder"), "dialogue_generator is missing 'model_responder' attribute"
        )
        self.assertIsNotNone(dialogue_generator.model_responder, "'model_responder' attribute is None")
        class_path = self.cfg.action_config.task.model_responder.type._target_
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        assert isinstance(dialogue_generator.model_responder, getattr(module, class_name)), f"The 'model_responder' should be an instance of {class_name} from {module_name}, but got {type(dialogue_generator.model_responder).__name__}"


    @patch("llm_roleplay.models.openai_model.OpenAIModel.generate")
    @patch("llm_roleplay.models.pipeline_model.PipelineModel.generate")
    @patch("llm_roleplay.models.causal_lm_model.CausalLMModel.generate")
    @patch("llm_roleplay.models.openai_model.OpenAIModel._get_model")
    @patch("llm_roleplay.models.pipeline_model.PipelineModel._get_model")
    @patch("llm_roleplay.models.causal_lm_model.CausalLMModel._get_model")
    @patch("torch.cuda.empty_cache")
    def test_resource_management(
        self,
        mock_empty_cache,
        mock_get_model_clm,
        mock_get_model_pipe,
        mock_get_mode_openai,
        mock_generate_clm,
        mock_generate_pipe,
        mock_generate_openai,
    ):
        Device.set_device(self.cfg.action_config.device)
        self.assertEqual(
            Device.get_device(),
            self.cfg.action_config.device,
            "Device configuration does not match the expected setting",
        )

        dialogue_generator = DialogueGenerator(self.cfg, self.aim_run)
        dialogue_generator.initialize()

        dialogue_generator.model_inquirer.generate.return_value = (self.sample_inquirer_output, None)
        dialogue_generator.model_responder.generate.return_value = (self.sample_responder_output, None)

        records_dir = dialogue_generator.generate()
        self.assertTrue(records_dir.is_dir(), "Generated records directory does not exist")
        self.assertTrue(
            (records_dir / f"{self.cfg.seed}.jsonl").exists(), "Expected jsonl file not found in records directory"
        )
        self.assertTrue(
            str(records_dir).startswith("tmp/dialogs"), "Records directory path does not start with 'tmp/dialogs'"
        )

        mock_empty_cache.assert_called()

    @patch("llm_roleplay.models.openai_model.OpenAIModel.generate")
    @patch("llm_roleplay.models.pipeline_model.PipelineModel.generate")
    @patch("llm_roleplay.models.causal_lm_model.CausalLMModel.generate")
    @patch("llm_roleplay.models.openai_model.OpenAIModel._get_model")
    @patch("llm_roleplay.models.pipeline_model.PipelineModel._get_model")
    @patch("llm_roleplay.models.causal_lm_model.CausalLMModel._get_model")
    def test_dialogue_generation(
        self,
        mock_get_model_clm,
        mock_get_model_pipe,
        mock_get_mode_openai,
        mock_generate_clm,
        mock_generate_pipe,
        mock_generate_openai,
    ):
        dialogue_generator = DialogueGenerator(self.cfg, self.aim_run)
        dialogue_generator.initialize()

        dialogue_generator.model_inquirer.generate.return_value = (self.sample_inquirer_output, None)
        dialogue_generator.model_responder.generate.return_value = (self.sample_responder_output, None)

        records_dir = dialogue_generator.generate()

        line_count = 0
        with (records_dir / f"{self.cfg.seed}.jsonl").open("r", encoding="utf-8") as file:
            for line in file:
                file_content = json.loads(line)
                line_count += 1

        self.assertEqual(
            line_count,
            len(self.cfg.action_config.task.persona.fixed),
            "Generated line count does not match the number of fixed personas",
        )
        self.assertEqual(
            file_content["num_turns"],
            self.cfg.action_config.task.num_turns,
            "Number of turns in dialogue does not match expected number",
        )

        for utterance in file_content["dialog"]:
            self.assertIsNotNone(utterance["model_inquirer"], "Model inquirer response is None")
            self.assertIsNotNone(utterance["model_responder"], "Model responder response is None")

        self.assertEqual(
            dialogue_generator.model_inquirer.generate.call_count,
            self.cfg.action_config.task.num_turns,
            "Inquirer model was not called the expected number of times",
        )
        self.assertEqual(
            dialogue_generator.model_responder.generate.call_count,
            self.cfg.action_config.task.num_turns,
            "Responder model was not called the expected number of times",
        )


if __name__ == "__main__":
    unittest.main()
