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

#Preparation for MLP
block_size = 3 #number of characters to be inputted at once
n_embd = 10 # dimensionality for character vectors
n_hidden = 200 # number of neurons in hidden layer

g = torch.Generator().manual_seed(2147483647)
C = torch.randn(vocab_size,n_embd)
W1 = torch.randn((n_embd* block_size, n_hidden),generator=g) * (5/3) / ((n_embd * block_size)**0.5)
W2 = torch.randn((n_hidden,vocab_size),generator =g) * 0.01
b2 = torch.randn(vocab_size,generator=g) * 0
    
    #Batch Norm parameters
bngain = torch.ones((1,n_hidden))
bnbias = torch.zeros((1,n_hidden))
bnmean_running = torch.zeros((1,n_hidden))
bnstd_running = torch.ones((1,n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

#Training

max_steps = 200,000
batch_size = 32
lossi = []

for i in range(max_steps):

    #minibatch construct
    indices = randint(0,Xtr.shape[0],(batch_size,), generator=g)
    Xb, Yb = Xtr[indices],Ytr[indices]





