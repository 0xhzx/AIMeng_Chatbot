# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM
import datasets
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
import pandas as pd



def load_dataset(tokenizer, data_dir):
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