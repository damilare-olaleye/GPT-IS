import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch.cuda.amp as amp
import json


# Hyperparameters
batch_size = 64
block_size = 50
max_iters = 1500
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_head = 12
n_layer = 12
dropout = 0.1

# Set random seed
torch.manual_seed(4200)

# Load text data
with open('index.json', 'r') as f:
    text = json.load(f)
    
# Extract input and output keys
inputs = [obj['content'] for obj in text if 'content' in obj]
outputs = [obj['output'] for obj in text if 'output' in obj]

# Define an empty list to store user input and output
# conversation = []

# Create vocabulary inputs
chars = sorted(list(set(''.join(inputs))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(l):
    return ''.join(itos.get(i, '?') for i in l)

# Create vocabulary outputs
chars = sorted(list(set(''.join(outputs))))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encodeOutput(s):
    return [stoi.get(c, 0) for c in s]

def decodeOutput(l):
    return ''.join(itos.get(i, '?') for i in l)


# Train and validation splits
data = torch.tensor(encode(inputs), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Write train and validation data to text files
with open('train_data.txt', 'w', encoding='utf-8') as f:
    f.write(decode(train_data.tolist()))

with open('val_data.txt', 'w', encoding='utf-8') as f:
    f.write(decode(val_data.tolist()))


def get_batch(split):
    data = train_data if split == 'train' else val_data
    max_index = len(data) - block_size
    if max_index < 0:
        raise ValueError("block_size is larger than the length of the data.")
    ix = torch.randint(0, max_index, (batch_size,))
    x = []
    y = []
    for i in ix:
        x_seq = data[i:i + block_size]
        y_seq = data[i + 1:i + block_size + 1]
        if len(x_seq) < block_size:
            # Pad the sequence with zeros
            x_seq = torch.cat([x_seq, torch.zeros(block_size - len(x_seq), dtype=torch.long)])
            y_seq = torch.cat([y_seq, torch.zeros(block_size - len(y_seq), dtype=torch.long)])
        x.append(x_seq)
        y.append(y_seq)
    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)
    return x, y


# Loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split=split)
            logits, loss = model.forward(X)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Load entity recognition pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

class Block(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.attn_drop(self.attn(x))
        x = x + self.resid_drop(self.mlp(self.ln1(x)))
        x = self.ln2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.out = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, L, E = x.shape
        h = self.key(x).view(B, L, self.n_head, E // self.n_head).transpose(1, 2)  # (B, n_head, L, E // n_head)
        g = self.query(x).view(B, L, self.n_head, E // self.n_head).transpose(1, 2)  # (B, n_head, L, E // n_head)
        v = self.value(x).view(B, L, self.n_head, E // self.n_head).transpose(1, 2)  # (B, n_head, L, E // n_head)

        scores = torch.matmul(h, g.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.n_embd / self.n_head))  # (B, n_head, L, L)
        attn = F.softmax(scores, dim=-1)  # (B, n_head, L, L)
        context = torch.matmul(attn, v)  # (B, n_head, L, E // n_head)
        context = context.transpose(1, 2).contiguous().view(B, L, E)  # (B, L, n_head * (E // n_head))
        out = self.out(context)  # (B, L, E)

        return out


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class nanoGPT3(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, labels):
        tokens = self.token_embedding_table(input_ids)  # (batch_size, block_size, n_embd)
        positions = self.position_embedding_table(torch.arange(input_ids.size(1)).to(input_ids.device))  # (1, block_size, n_embd)
        x = tokens + positions  # (batch_size, block_size, n_embd)
        x = self.blocks(x)  # Pass through the sequential blocks
        x = self.ln_f(x)  # (batch_size, block_size, n_embd)
        logits = self.lm_head(x)  # (batch_size, block_size, vocab_size)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))  # scalar

        return logits, loss


model = nanoGPT3(vocab_size, n_embd, n_head, block_size, n_layer)
model = model.to(device)


def save_conversation(conversation, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation, f)

def load_conversation(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        conversation = json.load(f)
    return conversation

def generate_text(model, conversation, max_length=100, temperature=1.0):
    model.eval()
    input_ids = torch.tensor(encodeOutput(conversation[-1]['content']), dtype=torch.long).unsqueeze(0).to(device)
    y = torch.zeros((1, input_ids.shape[1]), dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model.forward(input_ids, y)  # Include the y argument
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            y = torch.cat((y, next_token), dim=1)  # Update the y tensor
            if next_token == 0:
                break
        generated_ids = input_ids.squeeze(0).tolist()
        generated_text = decodeOutput(generated_ids)
        return generated_text

def train_model(model, train_data, val_data, max_epochs=20, batch_size=64, learning_rate=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(max_epochs):
        train_loss = 0.0
        val_loss = 0.0
        num_batches = len(train_data) // batch_size
        for i in range(num_batches):
            X, Y = get_batch("train")
            optimizer.zero_grad()
            logits, loss = model.forward(X, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= num_batches
        with torch.no_grad():
            num_val_batches = len(val_data) // batch_size
            if num_val_batches == 0:
                num_val_batches = 1
            for i in range(num_val_batches):
                X_val, Y_val = get_batch("val")
                val_logits, val_loss = model.forward(X_val, Y_val)
                val_loss += val_loss.item()
            val_loss /= num_val_batches
        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def interact(model, conversation):
    model.eval()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', ':wq', 'q', 'clear']:
            print("Exiting...")
            break
        conversation.append({"role": "user", "content": user_input})
        generated_text = generate_text(model, conversation)
        conversation.append({"role": "assistant", "output": generated_text})
        print("Assistant:", generated_text)

        # Perform sentiment analysis on user input
        sentiment = sentiment_pipeline(user_input)[0]
        entity = ner_pipeline(user_input)
        
        print("Sentiment:", sentiment)
        
        save_conversation(conversation, "conversation.json")

# Training the model
train_model(model, train_data, val_data, max_epochs=20, batch_size=batch_size, learning_rate=learning_rate)

# Start the conversation
conversation = load_conversation("conversation.json")
interact(model, conversation)
