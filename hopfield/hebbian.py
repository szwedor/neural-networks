import numpy as np
import collections

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.widgets import Button
from itertools import count
Neuron = collections.namedtuple('Neuron', 'id node_no layer_no bias')

class Index():
        def __init__(self,graphs):
                self.graphs = graphs
                self.ind = 0
        def next(self):
                if self.ind >= len(self.graphs)-1:
                        return self.ind
                self.ind +=1
                return self.ind 
        def prev(self):
                if self.ind <1:
                        return self.ind
                self.ind -=1
                return self.ind

class hebbian():
    def _activation(self,value):
        if value > 0:
            return 1
        return -1
    def _synchronous(self,states):
        self.history.append(states.copy())
        return self.activation(np.dot(states,self.neurons))
    def _asynchronous(self,states):
        self.history.append(states.copy())
        for ind in np.random.permutation(self.size[0]):
            
            x= np.dot(states,self.neurons[ind].reshape(states.shape[1],1))
            states[:,ind]= self._activation(x)
        return states 
    def __init__(self,size,synchronous):
        neuron_no = size[0]*size[1]
        self.orgsize=size
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
        self.history = []
        while not np.array_equal(prevprev,states):
            prevprev = prev
            prev =states
            states = self.update(states)
        self.history.append(states.copy())
        return states
    def visualize(self):
        self.ind = Index(self.history)
        self.im=None
        self.text= None
        def n(event):
            update(self.ind.next())
        def p(event):
            update(self.ind.prev())
        def update(i):
            self.im.set_data(np.resize(self.history[i],self.orgsize))
            self.text.remove()
            self.text=plt.text(0.01,0.01,str(i))
            plt.draw()
        def draw():
            self.ax = plt.subplot()
            self.im = self.ax.imshow(np.resize(self.history[0],self.orgsize))
            self.text=plt.text(0.01,0.01,str(0))
            axnext = plt.axes([0.5, 0.0, 0.1, 0.075])
            next = Button(axnext, 'next', hovercolor='green')
            axprev = plt.axes([0.4, 0.0, 0.1, 0.075])
            prev = Button(axprev, 'prev', hovercolor='green')
            next.on_clicked(n)
            prev.on_clicked(p)
            plt.show()   

        draw()