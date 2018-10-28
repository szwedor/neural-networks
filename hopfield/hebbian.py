import numpy as np

class hebbian():
    def _activation(self,value):
        if value > 0:
            return 1
        return -1
    def _synchronous(self,states):
        return self.activation(np.dot(states,self.neurons))
    def _asynchronous(self,states):
        for ind in np.random.permutation(self.size[0]):
            x= np.dot(states,self.neurons[ind].reshape(states.shape[1],1))
            states[:,ind]= self._activation(x)
        return states 
    def __init__(self,neuron_no,synchronous):
        self.neurons = np.zeros((neuron_no,neuron_no),dtype=int)
        self.size = (neuron_no,neuron_no)
        self.activation = np.vectorize(self._activation)
        if(synchronous):
            self.update = self._synchronous
        else:
            self.update = self._asynchronous

    def train(self, trainSet):
        for sample in trainSet:
            v = np.reshape(sample.values,(self.size[0],1))
            self.neurons+=np.dot(v,v.T)
        self.neurons = self.neurons / self.size[0]
        for i in range(self.size[0]):
            self.neurons[i,i]=0
        
    def predict(self, sample):
        states = sample.values.reshape(self.size[0],1).T
        prev = 0
        prevprev=0
        while not np.array_equal(prevprev,states):
            prevprev = prev
            prev =states
            states = self.update(states)
        return states
    def visualize(self):
        pass