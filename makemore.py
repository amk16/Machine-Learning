import torch
import torch.nn.functional as F
names = open('names.txt','r').read().splitlines()

#Creating a mapping of alphabets to ints and ints to letters including start/end special character

chars = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
    
#Count based approach
# 1. Get the counts for the bigrams in the data file and put them in a map
    
bigrams = {}
for name in names:
# Adding special start and end characters
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        # Appending the bigram into a map and increasing the count according to appearance
        bigrams[bigram] = bigrams.get(bigram,0) +1

#2. Getting the counts of the bigrams and storing them in a 2D torch tensor
#Populating 2D tensor with bigram counts

N = torch.zeros((27,27), dtype=torch.int32)
    
for name in names:
    chs= ['.'] + list(name) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        N[idx1,idx2] += 1
  
#Normalizing the counts in the 2D tensor row wise

p = (N+1).float()
p /= p.sum(1,keepdims=True) #Set keepdims to true to maintain dimensions of tensor

#3. Sample from p to create potential names one letter at a time
g = torch.Generator().manual_seed(2147483647)
    
for i in range(5):
    out = []
    ix = 0
    while True:
        l = p[ix]

        ix = torch.multinomial(l,num_samples=1,replacement=True,generator=g).item()
        char = itos[ix]
        
        out.append(char)
        if ix == 0:
            break

        
#Neural Net approach, using a single layer neural net to predict potential new names
#1.create training set of bigrams

xs = []
ys = []
num = 0
for name in names:
    ch = ['.'] + list(name)+ ['.']
    for ch1,ch2 in zip(ch,ch[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)
        num+= 1

xs = torch.tensor(xs)
ys = torch.tensor(ys)

#2.Initialize a layer of 27 neurons with 27 weights each

W = torch.randn((27,27),generator=g,requires_grad=True )

#3. Use one-hot encoding approach to make the xs tensor capable of being an input to the NN

xenc = F.one_hot(xs,num_classes=27).float()

#4. Forward Pass

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

prob = []
for i in range(100):
    logits =  xenc @ W
    counts = logits.exp()
    prob = counts / counts.sum(1,keepdims=True)
    #print(f'print the sum : {prob[0].sum()}')
    loss = -prob[torch.arange(num),ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())

#5. Backward pass

    W.grad = None
    loss.backward()

#6 Update the weights

    W.data += -50 * W.grad

#7 Sample from nn probablities

for i in range(5):
    outNN = []
    ixNN = 0
    while True:
        rowNN = prob[ixNN]
        ixNN = torch.multinomial(rowNN,num_samples=1,replacement=True,generator=g).item()
        charNN = itos[ixNN]
        outNN.append(charNN)
        
        if ixNN == 0:
            break
    print(''.join(outNN))














