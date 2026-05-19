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
from datasets import  load_dataset, Dataset
from sympy import pretty
import torch
import matplotlib.pyplot as plt
import re
import time

from huggingface_hub import notebook_login
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig

device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

tokenizer=AutoTokenizer.from_pretrained('microsoft/phi-1_5', trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-1_5',
    trust_remote_code=True,
)

model

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

df['text']

def tok(x):
  tokens=tokenizer(x['text'], padding=True, truncation=True, max_length=512)
  return tokens

data=Dataset.from_pandas(df)

tok_data=data.map(tok, batched=True, remove_columns=data.column_names)

tok_data

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

trainer=Trainer(
    model=model,
    train_dataset=tok_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

model.save_pretrained('megumind/phi-1_5-meds-lora')


trained_model=AutoModelForCausalLM.from_pretrained('megumind/phi-1_5-meds-lora')

inputs = tokenizer('prompt: A 60-year-old woman with a history of breast cancer presents with new onset of joint pain and morning stiffness in her hands. She mentions that she started on a new medication for her cancer two months ago. What could be the potential cause? ', return_tensors="pt", return_attention_mask=False)

inputs=inputs.to(device)

trained_model=trained_model.to(device)

outputs = trained_model.generate(**inputs, max_length=512,num_return_sequences=1, temperature=0.7 )



text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(text)
