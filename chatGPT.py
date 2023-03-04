import torch

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
    return encode, decode


def split_trn_val(data, pct=0.9):
    n = int(len(data)*pct)
    train = data[:n]
    val = data[n:]
    return train, val


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])

def train_model(train_data, batch_size = 4, block_size=9):
    torch.manual_seed(42)
    counter = 0
    for n in range(len(train_data)-block_size):
        X = train_data[n:n+block_size]
        Y = train_data[n+1:n+block_size+1]
        for t in range(block_size):
            context = X[:t+1]
            target = Y[t]
            print(context, target)
            counter+=1
            if counter>20:
                exit()


if __name__ == '__main__':
    txt = load_shakespeare()
    encode, decode = create_encoder_decoder(txt)
    data = torch.tensor(encode(txt), dtype=torch.long)
    trn_data, val_data = split_trn_val(data, pct=0.9)
    
    train_model(trn_data)