import torch
import wandb, os
from train import finetune, rlhf_dpo
import argparse
from pathlib import Path
from unsloth import FastLanguageModel
import torch

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--logging", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=2.5e-5)#2.5e-5
    parser.add_argument("--output_dir", type=str, default="/home/featurize/work/AIPI590-chatbot/outputs")
    parser.add_argument("--data_root_dir", type=str, default="/home/featurize/work/AIPI590-chatbot/data")
    parser.add_argument("--stage", type=str, default="rlhf")
    args = parser.parse_args()

    if args.stage == "finetune":
        data_dir = Path(args.data_root_dir) / "train_chat.pkl"
    elif args.stage == "rlhf":
        data_dir = Path(args.data_root_dir) / "rlhf.pkl"
    elif args.stage == "evaluation":
        data_dir = Path(args.data_root_dir) / "test_chat.pkl"
    
    
    
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # model quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, 
        max_seq_length = args.max_seq_length,
        dtype = args.dtype,
        load_in_4bit = args.load_in_4bit
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    if args.stage == "finetune":
        finetune(model, tokenizer, data_dir, args)
    elif args.stage == "rlhf":
        rlhf_dpo(model, tokenizer, data_dir, args)
    elif args.stage == "evaluation":
        pass

    
if __name__ == "__main__":
    
    os.environ["WANDB_API_KEY"]= "2f548926b1a03960bd0e22b44bf54dbd530a7b50"
    wandb_project = "mistral-finetune-chat"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
        
    main()
    
    
    