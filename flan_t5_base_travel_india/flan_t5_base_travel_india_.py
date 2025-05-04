
get_ipython().system(' pip install numpy')
get_ipython().system(' pip install pandas')
get_ipython().system(' pip install transformers')
get_ipython().system(' pip install datasets')
get_ipython().system(' pip install torch')
get_ipython().system(' pip install tqdm')
get_ipython().system(' pip install gradio')

import pandas as pd
import numpy as np
from datasets import  load_dataset
import torch
import matplotlib.pyplot as plt
import re
import time
import gc
import gradio as gr

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#from google.colab import drive
#drive.mount('/content/drive')


device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))




dataset=load_dataset('cyberblip/Travel_india')
model_name='google/flan-t5-base'



df=pd.DataFrame(dataset['train'])



tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token


# In[7]:


pre_tuned_model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_output(text, model=model):

    model=model.to(device)


    text='Question: '+text
    inputs=tokenizer(text, return_tensors='pt').to(device)

    outputs=model.generate(
    input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
            max_length=700,
    num_return_sequences=1,
    do_sample=True,
    top_k=8,
    top_p=0.95,
    temperature=0.7,
    )

    decoded=tokenizer.decode(outputs[0],skip_special_tokens=True)
    decoded=f'{text} \n Answer: {decoded}'


    return decoded


# In[39]:


s='What are the popular dishes in Mumbai?'
text=generate_output(s,pre_tuned_model)
print(text)


# In[26]:


s='Plan a 5-day itinerary for Delhi.'
text=generate_output(s,pre_tuned_model)
print(text)


# In[30]:


s=' What is the best time to visit Goa?'
text=generate_output(s,pre_tuned_model)
print(text)




class CutomDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=32):
        self.labels=df.columns
        self.tokenizer=tokenizer
        self.data=df.to_dict(orient='records')
        self.max_length=max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt=self.data[index]['prompt']
        instruction=self.data[index]['instruction']
        output=self.data[index]['output']

        input_text=f"Instruction: {instruction} Prompt: {prompt}"
        output_text=f"{output}"

        input_tokens=tokenizer(input_text, return_tensors='pt',max_length=self.max_length, padding='max_length', truncation=True)
        output_tokens=tokenizer(output_text, return_tensors='pt',max_length=self.max_length, padding='max_length', truncation=True)

        return {
            "input_ids":input_tokens['input_ids'],
            "attention_mask":input_tokens['attention_mask'],
            "target":output_tokens['input_ids']
        }



dataset=CutomDataset(df, tokenizer)
print(dataset)


# In[14]:


train_size=int(.8* len(dataset))
val_size=len(dataset)-train_size


# In[15]:


train_data, val_data=random_split(dataset, [train_size, val_size])


# In[16]:


batch_size=8
num_epochs=10
gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


# In[17]:


train_loader=DataLoader(train_data, batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val_data, batch_size=batch_size)


# In[18]:


model=model.to(device)
criterion= torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4)


# In[19]:


results=pd.DataFrame(columns=['epoch','transformer','batch_size','gpu','train_loss','val_loss','epc_dur'])


# In[20]:


for e in range(num_epochs):
    start_time=time.time()
    model.train()
    ep_train_loss=0

    train_iter=tqdm(train_loader, desc=f'Train epoch {e+1}/{num_epochs} Batch Size: {batch_size}, Transformer: {model_name}')

    for batch in train_iter:
        optimizer.zero_grad()

        inputs=batch['input_ids'].squeeze(1).to(device)
        attention_masks=batch['attention_mask'].squeeze(1).to(device)
        targets=batch['target'].squeeze(1).to(device)

        outputs=model(input_ids=inputs, attention_mask=attention_masks
                      ,labels= targets)

        loss=outputs.loss
        loss.backward()
        optimizer.step()

        train_iter.set_postfix({'Training loss': loss.item()})
        ep_train_loss+=loss.item()

    avg_ep_train_loss=ep_train_loss/len(train_iter)

    #validation
    model.eval()

    ep_val_loss=0
    total_loss=0

    val_iter=tqdm(val_loader, desc=f'val epoch {e+1}/{num_epochs}')

    with torch.no_grad():
        for batch in val_iter:
            inputs=batch['input_ids'].squeeze(1).to(device)
            attention_masks=batch['attention_mask'].squeeze(1).to(device)
            targets=batch['target'].squeeze(1).to(device)

            outputs=model(input_ids=inputs,attention_mask=attention_masks, labels= targets)
            loss=outputs.loss

            total_loss+=loss

            val_iter.set_postfix({'Validation Loss':loss.item()})
            ep_val_loss+=loss.item()

    avg_ep_val_loss=ep_val_loss/len(val_iter)

    end_time=time.time()

    epoch_dur=end_time-start_time

    new_row={
        'epoch':e+1,
        'transformer':model_name,
        'batch_size':batch_size,
        'gpu':gpu,
        'train_loss':avg_ep_train_loss,
        'val_loss':avg_ep_val_loss,
        'epc_dur':epoch_dur
    }

    results.loc[len(df)]=new_row

    print(f'Epoch: {e+1}, validation loss: {total_loss/len(val_loader)}')




print(results)




s='What are the popular dishes in Mumbai?'
text=generate_output(s,model)
print(text)


# In[23]:


s='Plan a 5-day itinerary for Delhi.'
text=generate_output(s,model)
print(text)


# In[24]:


s=' What is the best time to visit Goa?'
text=generate_output(s,model)
print(text)


# In[61]:


interface = gr.Interface(
    fn=generate_output,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here...", label=' Enter Question'),
    title='Flan-T5-Indian Itinerary Q&A' ,
    outputs=gr.Textbox(label="Generated Answer")
)

interface.launch()


# In[ ]:


save_path_m= "/content/drive/MyDrive//travel_india_finetuned_flan-t5-base'"
save_path_t= "/content/drive/MyDrive//travel_india_finetuned_flan-t5-bas-tokenizer'"


# In[ ]:


model.save_pretrained(save_path_m)
tokenizer.save_pretrained(save_path_t)

