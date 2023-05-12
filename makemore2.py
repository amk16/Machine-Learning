import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(''.join(words))) 
stoi = {s:i+1 for i,s in enumerate(chars)}

stoi['.'] = 0

itos = {i+1:s for s,i in stoi.items() }
vocab_size = len(itos)
#Build the dataset
def build_dataset(words):
    X,Y = [],[]

    context_size = 3
    for w in words:
        for char in w+'.':
            ix = stoi[char]
            context = [0] * context_size
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

