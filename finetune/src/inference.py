from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread
from unsloth.chat_templates import get_chat_template


# model放在外面
model_path = "HongxuanLi/test_chatbot"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference


def infer(messages):
# messages = [
#     {"role": "user", "content": "Hi"},
# ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    
    outputs = model.generate(input_ids = inputs, max_new_tokens = 500, use_cache = True, pad_token_id=tokenizer.eos_token_id)
    responses = tokenizer.batch_decode(outputs)
    
    return responses




#     streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"errors": "ignore"})
#     thread = Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, "max_new_tokens": 20})
#     thread.start()

#     output_text = ""
#     for incremental_text in streamer:
#         output_text += incremental_text
#         print(output_text)


