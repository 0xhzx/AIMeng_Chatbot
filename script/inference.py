import torch
from unsloth import FastLanguageModel

from transformers import TextStreamer

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

def inference_method1(max_seq_length = 2048, dtype = None, load_in_4bit = True, alpaca_prompt = alpaca_prompt):
    print("Inference method 1 - local")
    
    QA_model, QA_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Pot-l/mistral-7b-bnb-4bit-QA", # YOUR MODEL YOU USED FOR TRAINING
        # model_name = "unsloth/mistral-7b-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
        # model_name = "mistralai/Mistral-7B-v0.1",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(QA_model) 
    
    inputs = QA_tokenizer(
    [
        alpaca_prompt.format(
            "", # instruction
            "what is the avg cost of a master's program in the US?", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    text_streamer = TextStreamer(QA_tokenizer)
    _ = QA_model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024)
    
    return _

def inference_method2(alpaca_prompt = alpaca_prompt):
    print("Inference method 2")
    
    import requests

    API_URL = "https://uai0gg1o5neqj4xh.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept" : "application/json",
        "Authorization": "Bearer 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",
        "Content-Type": "application/json" 
        }

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": alpaca_prompt.format(
            "", # instruction
            "Give me an introduction of Duke University.", # input
            "", # output - leave this blank for generation!
        )
    })
    
    return output
    
