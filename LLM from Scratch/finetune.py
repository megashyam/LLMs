import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import GPT2WeightLoader
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
A Simple Finetuning setup for the custom GPT 
based on the AGNews Dataset

"""

# Setup
tokenizer = tiktoken.get_encoding("gpt2")  # GPT-2 BPE tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select device

# GPT-2 configs
GPT_CONFIG_355M = {
    "vocab_size": 50257,
    "context_len": 1024,
    "embed_dim": 1024,
    "n_heads": 16,
    "n_layer": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_len": 1024,
    "embed_dim": 768,
    "n_heads": 12,
    "n_layer": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


# Load AG News dataset
df1 = pd.read_parquet("data/train-00000-of-00001.parquet")
df2 = pd.read_parquet("data/test-00000-of-00001.parquet")

# Class mapping
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# Custom Dataset
class AGNews(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        # Pre-tokenize text samples
        self.input_ids = [tokenizer.encode(t) for t in self.data["text"]]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        inps = torch.tensor(self.input_ids[index])  # tokenized text
        tars = torch.tensor(self.data.iloc[index]["label"])  # classification label
        return inps, tars


# Collate function for batching
def preprocess_batch(batch, device, max_length=None, pad_id=50256):
    """
    Pad/truncate a batch of variable-length sequences and move to device.
    batch: list of (tokens, label) pairs
    """
    # Find longest sequence in the batch
    max_len = len(max(batch, key=lambda x: x[0].size())[0])
    input_ids, target_ids = [], []

    for b, tars in batch:
        # padding size for current sequence
        pad_size = max_len - len(b)
        pad = torch.full((pad_size,), pad_id)
        b = torch.cat([b, pad])  # apply right padding

        # Truncate if max_length exists
        if max_length is not None:
            b = b[:max_length]

        input_ids.append(b)
        target_ids.append(tars)

    # Stack into batch tensors
    inputs = torch.stack(input_ids).to(device)
    targets = torch.stack(target_ids).to(device)
    return inputs, targets


# Wrap collate function with partial
custom_collate_func = partial(preprocess_batch, device=device, max_length=1024)


# DataLoader setup
batch_size = 10

train_dataset = AGNews(df1, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_func,
    shuffle=True,
    drop_last=True,
)

val_dataset = AGNews(df2, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate_func,
    shuffle=False,
    drop_last=False,
)


# Tokenizer helper functions
def text2token(text, tokenizer):
    enc = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    enc = torch.tensor(enc).unsqueeze(0)
    return enc


def token2text(tokens, tokenizer):
    dec = tokens.squeeze(0)
    return tokenizer.decode(dec.tolist())


# Inference
def final_generation(prompt):
    token_ids = gpt.generate(
        tokens=text2token(prompt, tokenizer).to(device),
        max_new_tokens=100,
        context_size=GPT_CONFIG_124M["context_len"],
        top_k=50,
        temperature=1,
    )
    print("Output text:\n", token2text(token_ids, tokenizer))


# Load pretrained GPT-2 weights
model_dir = "gpt2/124M/"
torch.manual_seed(123)

loader = GPT2WeightLoader(model_dir, GPT_CONFIG_124M, device=device)
gpt = loader.load_weights().to(device)

# Freeze all parameters initially
for param in gpt.parameters():
    param.requires_grad = False


# Modify model for classification
num_classes = 4  # AG News has 4 categories

# Replace LM head with classification head
gpt.out_head = nn.Linear(GPT_CONFIG_124M["embed_dim"], out_features=num_classes).to(
    device
)

# Unfreeze last transformer block and final normalization for finetuning
for param in gpt.transformer_layers[-1].parameters():
    param.requires_grad = True

for param in gpt.final_norm.parameters():
    param.requires_grad = True


def classification_accuracy(data_loader, model, device):
    model.eval()
    correct_preds, total_samples = 0, 0
    step = 0

    for i, (inps, labels) in enumerate(tqdm(data_loader, f"Step {step+1} [Accuracy]")):
        inps, labels = inps.to(device), labels.to(device)

        with torch.no_grad():
            h = model(inps)  # forward pass
            # Index of last non-padding token for each sequence
            last_idx = (inps != 50256).sum(dim=1) - 1
            outs = h[torch.arange(h.size(0), device=h.device), last_idx, :]

        preds = torch.argmax(outs, dim=-1)
        total_samples += preds.shape[0]
        correct_preds += (labels == preds).sum().item()
        step += 1

    return (correct_preds / total_samples) * 100


# Evaluate before training
print(
    "Model Accuracy Prefinetuning:",
    classification_accuracy(val_loader, gpt.to(device), device),
    "%",
)

# Deep copy for training
model = copy.deepcopy(gpt)


# Training function
def train_model(model, num_epochs, optimizer, acc_freq=2, pad_id=50256):
    avg_train_loss, avg_val_loss, train_acc, val_acc = [], [], [], []
    torch.manual_seed(123)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, f"Epoch {epoch+1}/{num_epochs} [Training]")

        for inps, labels in pbar:
            optimizer.zero_grad()

            h = model(inps)
            last_idx = (inps != pad_id).sum(dim=1) - 1
            outs = h[torch.arange(h.size(0), device=h.device), last_idx, :]

            loss = F.cross_entropy(outs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Train Batch Loss": f"{loss.item():.4f}"})

        avg_train_loss.append(train_loss / len(train_loader))
        print(f"Train loss {avg_train_loss[-1]:.3f}")

        # Validation phase
        model.eval()
        val_loss = 0
        pbar = tqdm(val_loader, f"Epoch {epoch+1}/{num_epochs} [Validation]")

        for inps, labels in pbar:
            with torch.no_grad():
                h = model(inps)
                last_idx = (inps != pad_id).sum(dim=1) - 1
                outs = h[torch.arange(h.size(0), device=h.device), last_idx, :]

            loss = F.cross_entropy(outs, labels)
            val_loss += loss.item()
            pbar.set_postfix({"Val Batch Loss": f"{loss.item():.4f}"})

        avg_val_loss.append(val_loss / len(val_loader))
        print(f"Val loss {avg_val_loss[-1]:.3f}")

        # Accuracy reporting
        if epoch % acc_freq == 0:
            trnacc = classification_accuracy(train_loader, model, device)
            valacc = classification_accuracy(val_loader, model, device)

        print(f"Training accuracy: {trnacc*100:.2f}% | ", end="")
        print(f"Validation accuracy: {valacc*100:.2f}%")

        train_acc.append(trnacc)
        val_acc.append(valacc)

        return avg_train_loss, avg_val_loss, train_acc, val_acc


def plot_metrics():
    _, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].plot(avg_train_loss, label="Train Loss", color="r")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("train_Loss")
    ax[0].grid(True)
    ax[0].set_title("Training Loss")
    ax[0].legend()

    ax[0].plot(avg_val_loss, label="Val Loss", color="b")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Val Loss")
    ax[0].set_title("Validation Loss")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(train_acc, label="Train Acc", color="r")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("train_acc")
    ax[1].grid(True)
    ax[1].set_title("Training acc")
    ax[1].legend()

    ax[1].plot(val_acc, label="Val acc", color="g")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Val acc")
    ax[1].set_title("Validation acc")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()


def classify_review(text, model, tokenizer, device, max_length=None, pad_tok_id=50256):

    enc = tokenizer.encode(text)

    # Get max sequence length
    model_context_len = model.pos_emb.weight.shape[0]

    if max_length is None:
        max_length = model_context_len

    # Truncate sequence if longer than context length
    enc = enc[: min(max_length, model_context_len)]

    # Pad sequence with pad token
    enc += [pad_tok_id] * (max_length - len(enc))
    enc = torch.tensor(enc).unsqueeze(0).to(device)
    x = enc

    with torch.no_grad():
        h = model(x)
        last_idx = (x != pad_tok_id).sum(dim=1) - 1
        outs = h[torch.arange(h.size(0)), last_idx, :]

    preds = outs.argmax(dim=-1).item()
    msg = id2label[preds]

    print(text + ":", msg.upper())
    print("---" * 40)


# Train model
avg_train_loss, avg_val_loss, train_acc, val_acc = [], [], [], []
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 10

avg_train_loss, avg_val_loss, train_acc, val_acc = train_model(
    model, num_epochs, optimizer
)

# Plot training graphs
plot_metrics()

text = df1.iloc[np.random.randint(0, len(df1))]["text"]
classify_review(text, model, tokenizer, device)
