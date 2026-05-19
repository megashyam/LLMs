
! pip install numpy
! pip install pandas
! pip install transformers
! pip install datasets
! pip install torch
! pip install tqdm
! pip install peft
! pip install bitsandbytes
! pip install einops

from peft import peft_model, LoraConfig, get_peft_model

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import torch
import matplotlib.pyplot as plt
import re
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig

device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

pip install -U transformers

tokenizer=AutoTokenizer.from_pretrained('microsoft/phi-1_5', trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token

bnb=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)


model=AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-1_5',
    trust_remote_code=True,
    device_map='auto',
    quantization_config=bnb
)

lora_config=LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=['q_proj','k_proj','v_proj','dense', 'fc1','fc2'],
    lora_dropout=0.05,
    bias="none",
    task_type='CAUSAL_LM'
)

model=get_peft_model(model, lora_config)

model.print_trainable_parameters()

data=load_dataset('BI55/MedText', split='train')

df=data.to_pandas()
df

df['text'] = df.apply(lambda x: f"prompt: {x['Prompt']} completition: {x['Completion']}", axis=1)

def tok(x):
  tokens=tokenizer(x['text'], padding=True, truncation=True, max_length=512)
  return tokens

data=Dataset.from_pandas(df)

tok_data=data.map(tok, batched=True, remove_columns=data.column_names)

training_args=TrainingArguments(
    output_dir='phi_1_5-finemeds',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    save_strategy='epoch',
    logging_steps=100,
    max_steps=10,
    num_train_epochs=10
)


model


trainer = Trainer(
    model=model,
    train_dataset=tok_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

model.save_pretrained('megumind/phi-1_5-meds-qlora')
