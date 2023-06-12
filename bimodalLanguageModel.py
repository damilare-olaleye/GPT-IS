import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import pipeline
import torch.cuda.amp as amp

# Hyperparameters
batch_size = 50
block_size = 50
max_iters = 100
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 500
n_embd = 160
n_head = 4
n_layer = 4
dropout = 0.2

# Set random seed
torch.manual_seed(1337)

# Define a dictionary to store user input and output
conversation = {'input': [], 'output': []}

# Load text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s):
    encoded = []
    for c in s:
        if c in stoi:
            encoded.append(stoi[c])
        else:
            encoded.append(-1)  # or any other default value you prefer
    return encoded

def decode(l):
    decoded = []
    for i in l:
        if i in itos:
            decoded.append(itos[i])
        else:
            decoded.append('?')  # or any other default character you prefer
    return ''.join(decoded)

# Train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Write train and validation data to text files
with open('train_data.txt', 'w', encoding='utf-8') as f:
    f.write(decode(train_data.tolist()))

with open('val_data.txt', 'w', encoding='utf-8') as f:
    f.write(decode(val_data.tolist()))
    
# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss estimation
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Load entity recognition pipeline
ner_pipeline = pipeline("ner")

# Function to perform sentiment analysis on user input
def perform_sentiment_analysis(input_text):
    result = sentiment_pipeline(input_text)
    sentiment = result[0]['label']
    return sentiment

# Function to perform entity recognition on user input
def perform_entity_recognition(input_text):
    result = ner_pipeline(input_text)
    entities = [entity['entity'] for entity in result]
    return entities


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, T)
        output_val = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return output_val
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out 

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# Function to save user input and output
def save_conversation(input_text, output_text):
    conversation['input'].append(input_text)
    conversation['output'].append(output_text)
    
# Function to generate text or message based on user input
def generate_response(input_text):
    model.eval()
    input_tokens = torch.tensor(encode(input_text), dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = model.generate(input_tokens, max_new_tokens=block_size)[0].tolist()
    output_text = decode(generated_tokens)

    sentiment = perform_sentiment_analysis(input_text)
    entities = perform_entity_recognition(input_text)

    # Customize response based on sentiment and entities
    if sentiment == 'POSITIVE':
        response = "That sounds great!"
    elif sentiment == 'NEGATIVE':
        response = "I'm sorry to hear that."
    else:
        response = "Hmm, I'm not sure how to respond."

    response += " Here are some entities I found in your input: " + str(entities)

    save_conversation(input_text, output_text)
    return response, output_text


# Function to train the model using user input and output
def train_model():
    model.train()
    xb, yb = get_batch('train')

    # Perform the forward and backward pass under autocast
    with amp.autocast():
        logits, loss = model(xb, yb)

    # Scale the loss and perform backpropagation using the scaler
    scaler.scale(loss).backward()

    # Perform the optimizer step using the scaler
    scaler.step(optimizer)
    optimizer.zero_grad(set_to_none=True)
    scaler.update()
    
  
    
# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
    
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # apply one head of self-attention. (B, T, C)
        logits = self.lm_head(x) # (B , T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = idx_next.view(-1, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx

model = BigramLanguageModel()
model = model.to(device)

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print("Using DataParallel for multi-GPU training.")
    model = nn.DataParallel(model)

# Create optimizer and wrap it with GradScaler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = amp.GradScaler()

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    train_model()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=1000)[0].tolist()
generated_text = decode(generated_tokens)
save_conversation('', generated_text)
# print(generated_text)
open('output.txt', 'w').write(generated_text)

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pt')

# Save the conversation data
torch.save(conversation, 'conversation_data.pt')


