from peft import LoraConfig, get_peft_model

import pandas as pd
from datasets import  load_dataset, Dataset
import torch

from huggingface_hub import notebook_login
from transformers import AutoTokenizer, Seq2SeqTrainingArguments ,DataCollatorForSeq2Seq , Seq2SeqTrainer, AutoModelForSeq2SeqLM

device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

print(f'Device: {device}')

model_name='google/pegasus-large'

tokenizer=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset=load_dataset('potsawee/podcast_summary_assessment')

df=pd.DataFrame(dataset['evaluation'])
df.drop(columns=df.columns[2:],inplace=True)

lora_config=LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['k_proj','v_proj','q_proj','out_proj', 'fc1','fc2'],
    lora_dropout=0.05,
    bias="none",
    task_type='SEQ_2_SEQ_LM'
)

model.enable_input_require_grads()
model=get_peft_model(model, lora_config)

model.print_trainable_parameters()

def preprocess_data(data):
    inputs=data['transcript']
    summary=data['summary']
    model_inputs=tokenizer(inputs, truncation=True, padding="longest", return_tensors="pt")

    targets=tokenizer(summary, truncation=True,max_length=128, padding="max_length", return_tensors="pt")

    prep_data={
        'input_ids': model_inputs['input_ids'].to(device),
        'attention_mask': model_inputs['attention_mask'].to(device),
        'labels':targets['input_ids'].to(device)
    }

    return prep_data

dataset=Dataset.from_pandas(df)

tokenized_data=dataset.map(preprocess_data, batched=True,remove_columns=['transcript','summary'])


training_args=Seq2SeqTrainingArguments(
    output_dir='./path',
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=24,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    weight_decay=0.01,
    fp16=False,
    lr_scheduler_type='cosine',
    num_train_epochs=4,
    logging_steps=100
)

trainer=Seq2SeqTrainer(
    model=model,
    train_dataset=tokenized_data,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)

)

trainer.train()


# def generate_outputs(input, model):
#     model=model.to(device)
#     tokens=tokenizer(input, truncation=True, padding="longest",max_length=512, return_tensors="pt").to(device)
#     outputs=model.generate(**tokens, num_return_sequences=1 )
#     decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return decoded


# trainer.save_model("path_to_file")
# tokenizer.save_pretrained("path_to_file")