import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from huggingface_hub import login
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)
from datasets import load_dataset
from trl import SFTTrainer


def train_mistral_model(huggingface_token, base_model, dataset_path, save_model):
    # Login to Hugging Face
    login(token=huggingface_token)

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Bits and Bytes configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Disable cache and enable gradient checkpointing
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    # tokenizer = tokenizer.add_bos_token
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure PEFT with LoRA
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=256,
        # use_rslora=True,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir="results",
        num_train_epochs=50,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none"
    )

    # Initialize and train the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()

    # Save the trained model and tokenizer
    trainer.model.save_pretrained(save_model)
    trainer.tokenizer.save_pretrained(save_model)


# Example usage
train_mistral_model(
    huggingface_token='Your API Token',
    base_model="mistralai/Mistral-7B-Instruct-v0.2",
    dataset_path="dataset/story.json",
    save_model="models/Mistral-7b-v2-story_50a_r256"
)
