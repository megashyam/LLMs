# Pegasus podcast summarizer app

This repository contains a Streamlit app summarizes podcast transcripts using a Fine Tuned Pegasus Large model. The fine-tuning process involves LoRA (Low-Rank Adaptation), adapting large pre-trained models with significantly fewer parameters, making it both computationally efficient and memor -efficient. The Streamlit-based user interface allows users to input podcast transcripts as text or upload a document to receive summaries. The integration of LoRA for fine-tuning ensures that large models can be used on standard hardware, making this solution accessible and scalable for various tasks.

## Features
- **Abstractive Summarization**:  Using advanced models to produce human-like summaries of podcast content.
- **Efficient Fine-Tuning**:   Utilizing LoRA to adapt the model for podcast-specific data with lower computational resources.
- **Interactive Streamlit Interface**:   Enabling summarization of podcast transcripts via an easy-to-use web app by either uploading a document or typing.
- **Custom Dataset**:  Fine-tuning performed on a podcast summary dataset to enhance performance on real-world podcast content.
  


<br>

## Demo:

![Project Demo:](https://github.com/megashyam/LLMs/blob/main/Pegasus%20podcast%20summarizer/data/demo.gif)

<br>

### LoRA Integration
The Pegasus large model is fine-tuned using LoRA (Low-Rank Adaptation). LoRA adapts large pre-trained models with fewer trainable parameters, making the fine-tuning process more efficient and reducing memory usage.

With PEFT, LoRA modifies the model’s low-rank matrix layers while keeping the pre-trained weights mostly frozen. This process leads to faster and more memory-efficient training. The model was tuned on the HuggingFace [potsawee/podcast_summary_assessment](https://huggingface.co/datasets/potsawee/podcast_summary_assessment) dataset.






