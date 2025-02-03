import torch

from huggingface_hub import login

from transformers import AutoTokenizer ,AutoModelForSeq2SeqLM
import adapters

import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import base64
import os
import logging

login("hf_token")


logging.basicConfig(level=logging.DEBUG)


st.set_page_config(layout="wide")

device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
logging.debug(f" Device: {device}")


@st.cache_resource 
def load_model():
	check_point = "google/pegasus-large"
	model = AutoModelForSeq2SeqLM.from_pretrained(check_point)
	tokenizer = AutoTokenizer.from_pretrained(check_point)
	adapter_name = "megumind/pegasus_podcast_summarizer"
	model.load_adapter(adapter_name)

	model=model.to(device)
	logging.debug(f"Model:{model} and tokenizer loaded")

	return model, tokenizer

model, tokenizer= load_model()

def generate(text, model):
    tokens=tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(device)
    outputs=model.generate(**tokens, num_return_sequences=1, temperature=0.7)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    del tokens, outputs
    torch.cuda.empty_cache()

    return decoded


def pegasus_pipeline(filepath):
	input_text = file_prep(filepath)
	result = generate(input_text, model)
	return result[0]


def file_prep(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=80)
    texts=text_splitter.split_documents(pages)
    final_texts=""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts


@st.cache_data
def display_file(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def main():
    st.title("Pegasus Podcast Summarizer")

    choice = st.sidebar.selectbox("Choice:", ["Summarize Text", "Summarize Document"])
    if choice == "Summarize Text":
        st.subheader("Summarize Podcasts using Fine Tuned Pegasus")
        input_text = st.text_area("Enter text here")
        if input_text is not None:
            if st.button("Summarize Text"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("Your Input Text")
                    st.info(input_text)
                with col2:
                    st.markdown("Summary Result")
                    result = generate(input_text, model)
                    st.success(result[0])

    elif choice == "Summarize Document":
        st.subheader("Summarize Document using Fine Tuned Pegasus")
        input_file = st.file_uploader("Upload your document here", type=['pdf'])
        if input_file is not None:
            if st.button("Summarize Document"):
                filepath = os.path.join('data', input_file.name)
                with open(filepath, 'wb') as f:
                    f.write(input_file.getbuffer())

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.info("File uploaded successfully")
                    display_file(filepath)
                with col2:
                    st.markdown("**Summary Result**")
                    try:
                        summary = pegasus_pipeline(filepath)
                        st.success(summary)
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")

if __name__=='__main__':
  main()
