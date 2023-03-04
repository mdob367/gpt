import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
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
    chars = ''.join(sorted(list(set(txt))))
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
    return x, y



if __name__ == '__main__':
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
    logits, loss = m(X_batch, Y_batch)

    print(logits.shape)
    print(loss)

    print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
