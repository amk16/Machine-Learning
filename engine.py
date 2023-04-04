import math
class Value:
    def __init__(self,scalar,_children=(), _op=''):
        self.data = scalar
        self.grad = 0
        self.prev = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        if not self.prev and not other.prev:
            out = Value(self.data + other.data, (), '+')
        else:
            out = Value(self.data + other.data, (self, other), '+')

            def _backwards():
                self.grad += 1.0 * out.grad
                other.grad += 1.0 * out.grad
            out._backward = _backwards

        return out
    """
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')

        def _backwards():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backwards
        
        return out
    """
    def __tanh__(self):
         
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out
    def __relu__(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')

        def _backwards():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backwards

        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backwards():
            self.grad += (other*self.data**(other-1)) * out.grad
        out._backward = _backwards

        return out
    
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()



    def __neg__(self):
        return self*-1
    def __radd__(self,other):
        return self + other

    def __sub__(self,other):
        return self + (-other)
    def __rsub__(self,other):
        return other + (-self)
    def __rmul__(self,other):
        return self * other 
    

    def __truediv__(self,other):
        return self * other**-1
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data} grad={self.grad} op={self._op})"

            

"""        
a = Value(3.0)
b = Value(4.0)
c = Value(5.0)
print(a,b,c)
print(a+b)
print(a*b)
print(c+a)
print(a+b+c)
print(a/b)
"""