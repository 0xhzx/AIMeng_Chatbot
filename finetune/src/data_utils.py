# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM
import datasets
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import pandas as pd

def extract_content(chat_str):
    chat_str = chat_str.split("\n\n")
    chat_str = " ".join(chat_str)
    messages = chat_str.split(" ")

    json_list = []

    for i in range(len(messages)):
        if messages[i] == "Human:":
            content = " ".join(messages[i+1:messages.index("Assistant:", i)])
            json_list.append({"role": "human", "content": content})
        elif messages[i] == "Assistant:":
            try:
                content = " ".join(messages[i+1:messages.index("Human:", i)])
            except:
                content = " ".join(messages[i+1:])
            json_list.append({"role": "assistant", "content": content})
    return json_list

def preprocess_rlhf_data(save_dir):
    dataset = load_dataset("Unified-Language-Model-Alignment/Anthropic_HH_Golden")
    dataset = pd.DataFrame(dataset["train"]).applymap(extract_content)
    dataset["prompt"] = dataset["chosen"].apply(lambda x: x[0]["content"])
    dataset.to_pickle(save_dir)



def load_dataset_sft(tokenizer, data_dir):
    def formatting_prompts_func(examples):
        convos = examples["conversation"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )

    dataset = datasets.Dataset.from_pandas(pd.read_pickle(data_dir))
    dataset = dataset.map(formatting_prompts_func, batched = True)
    data_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset =  data_split['train']
    val_dataset = data_split['test']
    return train_dataset, val_dataset

def load_dataset_rlhf(tokenizer, data_dir):
    
    def formatting_prompts_func(examples):

        convos = examples["rejected"]
        rejected = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]

        convos = examples["chosen"]
        chosen = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]

        convos = [[{"role": "user", "content": prompt}] for prompt in examples["prompt"]]

        prompt = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]



        return { "rejected" : rejected, "chosen" : chosen, "prompt" : prompt}


    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml",
        map_eos_token = True, # Maps <|im_end|> to </s> instead
    )

    # data_dir = "/home/featurize/work/AIPI590-chatbot/src/rlhf.pkl"
    dataset = datasets.Dataset.from_pandas(pd.read_pickle(data_dir))
    dataset = dataset.map(formatting_prompts_func, batched = True)
    data_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset =  data_split['train']
    val_dataset = data_split['test']
    return train_dataset, val_dataset




