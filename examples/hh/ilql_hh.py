import math
import os
import re

import numpy as np
import tritonclient.grpc as client_util
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import trlx
from trlx.data.configs import TRLConfig


triton_host = os.environ.get("TRITON_HOST", "localhost:8001")
triton_model = os.environ.get("TRITON_MODEL", "gptj-rm-static")


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def split_dialog(dialog):
    dialog = re.split(r"(\n\nHuman: |\n\nAssistant: )", dialog)[1:]
    return ["".join(dialog[:-1]), dialog[-1]]


def preprocess(sample):
    sample["prompt_output"] = [
        split_dialog(sample["chosen"]),
        split_dialog(sample["rejected"]),
    ]
    sample["reward"] = [1, -1]
    return sample


def main(hparams={}):
    config_path = os.path.join(os.path.dirname(__file__), os.environ.get("CONFIG_PATH"))
    default_config = yaml.safe_load(open(config_path))
    config = TRLConfig.update(default_config, hparams)

    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"
    client = client_util.InferenceServerClient(url=triton_host, verbose=False)

    def reward_fn(samples):
        samples = [s + reward_tokenizer.eos_token for s in samples]
        input = reward_tokenizer(samples, padding=True, max_length=1024)

        mbs = 24
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = np.array(input.input_ids[batch_ixs], dtype=np.int32)
            attention_mask = np.array(input.attention_mask[batch_ixs], dtype=np.int8)

            inputs = [
                prepare_tensor("input_ids", input_ids),
                prepare_tensor("attention_mask", attention_mask),
            ]

            result = client.infer(triton_model, inputs)
            rewards = result.as_numpy("rewards")
            if rewards is None:
                raise RuntimeError("No output data")

            last_ixs = attention_mask.sum(-1, keepdims=True) - 1
            returns = np.take_along_axis(rewards, last_ixs, -1)
            out.extend(returns.flatten())

        return out

    # dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base").map(preprocess)
    # dataset = load_dataset("Anthropic/hh-rlhf").map(preprocess)
    # prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    # rewards = sum(dataset["train"]["reward"], [])
    # test_dataset = load_dataset(
    #     "Anthropic/hh-rlhf", data_dir="helpful-base", split="test"
    # ).map(preprocess)
    # eval_prompts = [sample[0][0] for sample in test_dataset["prompt_output"]][:256]

    def preprocess_static(sample):
        sample["prompt_output"] = [
            [
                sample["prompt"] + "Assistant: ",
                sample["chosen"][len("Assistant: "):]
            ],
            [
                sample["prompt"] + "Assistant: ",
                sample["rejected"][len("Assistant: "):]
            ],
        ]
        sample["reward"] = [1, -1]

        return sample

    dataset = load_dataset("Dahoas/rm-static").map(preprocess_static)
    prompts_outputs = sum(dataset["train"]["prompt_output"], [])
    rewards = sum(dataset["train"]["reward"], [])
    eval_prompts = [sample[0][0] for sample in dataset["test"]["prompt_output"]][:32]

    # def preprocess_labeled(sample):
    #     sample["prompt_output"] = split_dialog(sample["response"])
    #     return sample

    # dataset = load_dataset("Dahoas/reward-labeled-static").map(preprocess_labeled)
    # prompts_outputs = dataset["train"]['prompt_output']
    # rewards = dataset["train"]['reward']

    trlx.train(
        dataset=(prompts_outputs, rewards),
        config=config,
        eval_prompts=eval_prompts,
        metric_fn=lambda xs: {"rewards": reward_fn(xs)},
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )

if __name__ == "__main__":
    import json
    import sys

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
