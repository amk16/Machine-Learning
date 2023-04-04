import random
from engine import Value
class Neuron:

    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))

    def activate(self,x):
        act = sum((weight*input for weight,input in zip(self.w,x)),self.bias)
        out = act.__tanh__()
        return out
    def parameters(self):
        return self.w + [self.bias]
          
        

class Layer:
    def __init__(self,nin,nout):
        
        self.neurons= [Neuron(nin) for _ in range(nout)]
        self.out = []
        
    def activate(self,x):
        out = [n.activate(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        params=[]
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

class MLP:
    def __init__(self,nin,nouts):  
        sz = [nin] + nouts     
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

    def activate(self,x):
        for layer in self.layers:
            x = layer.activate(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
def learn(nin,nouts,xs,ys,iterations):


    perceptron = MLP(nin,nouts)
    
    for x in range(iterations): 
        preds = [perceptron.activate(x) for x in xs]

        loss = sum((y-pred)**2 for y,pred in zip(ys,preds))
        print(x,loss)
        for p in perceptron.parameters():
            p.grad = 0
        loss.backward()

        for p in perceptron.parameters():
            
            p.data += -0.1  * p.grad


    

def main():
    #these values can be anything, last value of the nouts array should be one to signify the output layer
    #each array in xs should be the length of nin (i.e nin=3 then len(xs[0])=3)
    nin = 3
    nouts= [4,4,1]
    xs = [
        [2.0,3.0,-1.0],
        [3.0,-1.0,0.5],
        [0.5,1.0,1.0],
        [1.0,1.0,-1.0],

    ]
    ys = [1.0,-1.0,-1.0,1.0]

    learn(nin,nouts,xs,ys,1000)


    """
    print(perceptron.layers[0].neurons[0].w[0].grad)

    print(preds)
    print(perceptron.parameters())
    print(len(perceptron.parameters()))
    """
   
   

    
    

if __name__ == "__main__":
    main()


        
