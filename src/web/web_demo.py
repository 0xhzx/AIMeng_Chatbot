import sys
sys.path.append('src/')

import json
import streamlit as st
import os
from dotenv import load_dotenv
import warnings
import requests
import re
from rag.rag import create_database, search_similar_text, make_embeddings
from openai import OpenAI
# from unsloth import FastLanguageModel
# from transformers import TextStreamer
import torch
from transformers import TextStreamer


# load env
load_dotenv()
hf_key = os.getenv("HF_KEY")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=openai_key)
index = create_database()


API_URL = "https://uai0gg1o5neqj4xh.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
	"Accept" : "application/json",
	"Authorization": f"Bearer {hf_key}",
	"Content-Type": "application/json" 
}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()




warnings.filterwarnings("ignore")

st.set_page_config(page_title="Duke Chatbot")
st.title("Duke Chatbot")

# @st.cache_resource
# def init_model():
#     max_seq_length = 2048
#     dtype = None
#     load_in_4bit = True
#     QA_model, QA_tokenizer = FastLanguageModel.from_pretrained(
#         model_name = "Pot-l/mistral-7b-bnb-4bit-QA", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )
#     FastLanguageModel.for_inference(QA_model) # Enable native 2x faster inference
#     return QA_model, QA_tokenizer



# clear history messages
def clear_chat_history():
    del st.session_state.messages

# initialize the chat history
def init_chat_history():
    with st.chat_message("assistant", avatar="https://i.ibb.co/nk0rNwF/These-March-Madness-emojis-are-awesome-removebg-preview.png"):
        st.markdown("Hi, I am an assistant for answering some questions about Duke.")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = "🙋‍♂️" if message["role"] == "user" else "https://i.ibb.co/nk0rNwF/These-March-Madness-emojis-are-awesome-removebg-preview.png"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages





def main():
    # QA_model, QA_tokenizer = init_model()
    st.image('https://upload.wikimedia.org/wikipedia/commons/e/e6/Duke_University_logo.svg', width=200)
    messages = init_chat_history()

    if user_input := st.chat_input("Shift + Enter for switching a new line, Enter for sending"):
        with st.chat_message("user", avatar='🧑'):
            st.markdown(user_input)
        messages.append({"role": "user", "content": user_input})
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""
        user_embeddings = make_embeddings(client, [user_input])[0]
        rag_context = search_similar_text(index, user_embeddings)
        prompt_prefix = "When you are asked about the question about Duke University or AIPI program, you can provide the answer based on the following information. Else, tell the users that they can ask you questions about Duke."
        # rag_context = prompt_prefix + rag_context
        print("rag is:",rag_context)
        # inputs = QA_tokenizer(
        # [
        #     alpaca_prompt.format(
        #         rag_context, # instruction
        #         user_input, # input
        #         "", # output - leave this blank for generation!
        #     )
        # ], return_tensors = "pt").to("cuda")
        # outputs = QA_model.generate(**inputs, max_new_tokens = 1024, use_cache = True)
        # decoded_output = QA_tokenizer.batch_decode(outputs)

        # method 2 for streaming generation
        # text_streamer = TextStreamer(QA_tokenizer)
        # _ = QA_model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024)

        # decoded_output = QA_tokenizer.batch_decode(_)
        # print(decoded_output)
        # generated_text = decoded_output[0]

        generated_text = query({
            "inputs": alpaca_prompt.format(
                rag_context if rag_context else prompt_prefix, # instruction
                user_input, # input
                "", # output - leave this blank for generation!
            ),
            "parameters": {}
        })
        # print(generated_text)
        match = re.search(r'### Response:\n(.*?)(?=###|$)', generated_text[0]['generated_text'], re.DOTALL)
        extracted_text = ''
        if match:
            extracted_text = match.group(1).strip().rstrip('</s>')
            extracted_text = re.sub(r' +', ' ', extracted_text) # continues spaces remove
            # extracted_text = re.sub(r'(?<=\n) {4}', '', extracted_text)
            # extracted_text = re.sub(r'\b\d+\.\b', '', extracted_text)
            print("The extracted texts are:", extracted_text)
        else:
            print("Not found")

        with st.chat_message("assistant", avatar="https://i.ibb.co/nk0rNwF/These-March-Madness-emojis-are-awesome-removebg-preview.png"):
            placeholder = st.empty()
            placeholder.markdown(extracted_text)
            # torch.cuda.empty_cache()
        messages.append({"role": "assistant", "content": extracted_text})
        st.button("Clear Chat", on_click=clear_chat_history)
    
    
if __name__ == "__main__":
    main()