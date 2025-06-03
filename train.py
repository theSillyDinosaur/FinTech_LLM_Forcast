from dataset import seq_beforeHug
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
import datasets

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", quantization_config=bnb_config)
tokenizer.pad_token = tokenizer.eos_token  # Important for training

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
train_dataset_hug = datasets.Dataset.from_list(seq_beforeHug["train"])
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_hug,  # Use Dataset format
    args=TrainingArguments(
        output_dir="/content/drive/MyDrive/vicuna-bestseller-lora-1abbr",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_dir="./logs",
        fp16=True,
        save_strategy="steps",
        save_steps=100
    )
)

trainer.train()