from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread


def infer(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(input_ids = inputs,use_cache = True, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1000, do_sample=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return responses[0]