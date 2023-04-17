import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


# hyperparameters
batch_size = 32 # How many independent samples to process at once
block_size = 8 # Maximum context length for predictions
epochs = 10000 # How many times to iterate over the entire dataset
eval_interval = 300
lr = 1e-3 # Learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
eval_iters = 100
n_embed = 32 # Embedding size

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # each token also reads off the logits for the next token from a position embedding table
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # the final logits are a linear combination of the two
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape # (B, T) Batch, Time
        tok_emb = self.token_embedding_table(idx) # (B, T, C) Batch, Time, Channel
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) Time, Channel
        x = tok_emb + pos_emb # (B, T, C) Batch, Time, Channel
        logits = self.lm_head(x) # (B, T, vocab_size) Batch, Time, vocab_size
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # (B, T, C) Batch, Time, Channel
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    


def load_shakespeare():
    # read it in to inspect it
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_encoder_decoder(text):
    chars = ''.join(sorted(list(set(text))))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s] # Take in a string, output a list of ints
    decode = lambda l: ''.join([itos[i] for i in l]) # Take in a list of ints, output a string
    return encode, decode, vocab_size


def split_trn_val(data, pct=0.9):
    n = int(len(data)*pct)
    train = data[:n]
    val = data[n:]
    return train, val


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_losses(trn_data, val_data, m):
    losses = {}
    m.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        if split == 'train':
            data = trn_data
        elif split == 'val':
            data = val_data
        losses[split] = 0
        for _ in range(eval_iters):
            X_batch, Y_batch = get_batch(data, block_size = block_size, batch_size = batch_size)
            logits, loss = m(X_batch, Y_batch)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    m.train() # Set model back to training mode
    return losses


def run_model():

    txt = load_shakespeare()
    encode, decode, vocab_size = create_encoder_decoder(txt)
    data = torch.tensor(encode(txt), dtype=torch.long)
    trn_data, val_data = split_trn_val(data, pct=0.9)
    
    split = 'train'
    if split == 'train':
        data = trn_data
    elif split == 'val':
        data = val_data
    X_batch, Y_batch = get_batch(data, block_size = 8, batch_size = 4)

    m = BigramLanguageModel(vocab_size)
    m = m.to(device)
    logits, loss = m(X_batch, Y_batch)

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss_history = []
    for epoch in range(epochs):
        X_batch, Y_batch = get_batch(data, block_size = 8, batch_size = batch_size)
        logits, loss = m(X_batch, Y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % eval_interval == 0:
            losses = estimate_losses(trn_data=trn_data, val_data=val_data, m=m)
            print(f"Epoch {epoch}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
    plt.plot(loss_history)
    plt.show()
    print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))

if __name__ == '__main__':
    run_model()
