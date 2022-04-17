import numpy as np
import time

class Neurogine:
    def __init__(self, layers):
        if(len(layers)<2):
            raise Neuroerror("Layer count must greater then 2!")
        np.random.seed(1)
        self.nVroot = None
        self.layers = layers
        self.lastCalculatedLayer = []
        self.synapses = []
        self.synapses_stored = []
        for lid in range(0,len(layers)-1):
            nlayer = layers[lid]
            alayer = layers[lid+1]
            self.synapses.append(
                2*np.random.random((nlayer,alayer))-1
            )
    def updateStoredSynapses(self):
        self.synapses_stored = np.copy(self.synapses)
    def loadVisualizer(self,visualizer):
        self.nVroot=visualizer
        self.nVroot.load(self)
    def updateVisualizer(self):
        self.nVroot.update()
    def sigmoid(self,x,d=False):
        if(d): return x*(1-x)
        return 1/(1+np.exp(-x))
    def relu(self,mat,der=False):
        if(der): return  (mat>0)*1
        return np.multiply(mat,(mat>0))
    def getIOShape(self):
        return (self.layers[0],self.layers[-1])
    def think(self,Input,withStored=False):
        dF  = Input
        synapses = self.synapses_stored if withStored else self.synapses
        for item in range(0,len(self.synapses)):
            dF = self.sigmoid(np.dot(dF,self.synapses[item])) 
        return dF
    def thinkLayered(self,Input):
        dF  = Input
        layers = [Input]
        for item in range(0,len(self.synapses)):
            dF = self.sigmoid(np.dot(dF,self.synapses[item])) 
            layers.append(dF)
        self.lastCalculatedLayer = layers
        return dF
    def trainRelu(self,X,y,iteration=100000,learningRate=0.6):
        for it in range(0,iteration):
            #think 
            dF  = X
            layers = [X]
            for item in range(0,len(self.synapses)):
                dF = self.relu(np.dot(dF,self.synapses[item])) 
                layers.append(dF)
            self.lastCalculatedLayer = layers
            Error = y-layers[-1]
            fD = Error*self.relu(layers[-1],True)
            self.synapses[-1]+= layers[-2].T.dot(fD)*learningRate
            for lid in reversed(range(0,len(self.synapses)-1)):
                fD = fD.dot(self.synapses[lid+1].T)*self.relu(layers[lid+1],True)
                self.synapses[lid]+= layers[lid].T.dot(fD)*learningRate
    def train(self,X,y,iteration=100000,learningRate=0.6):
        for it in range(0,iteration):
            #think 
            dF  = X
            layers = [X]
            for item in range(0,len(self.synapses)):
                dF = self.sigmoid(np.dot(dF,self.synapses[item])) 
                layers.append(dF)
            Error = y-layers[-1]
            self.lastCalculatedLayer = layers
            fD = Error*self.sigmoid(layers[-1],True)
            self.synapses[-1]+= layers[-2].T.dot(fD)*learningRate
            for lid in reversed(range(0,len(self.synapses)-1)):
                fD = fD.dot(self.synapses[lid+1].T)*self.sigmoid(layers[lid+1],True)
                self.synapses[lid]+= layers[lid].T.dot(fD)*learningRate
    def trainMSE(self,X,y,iteration=100000,learningRate=0.6,nrg=False):
        # print("Input",X) 
        # print("Synapses",[repr(i.shape) for i in self.synapses])
        # print("Output",y) 
        for it in range(0,iteration):
            #think 
            dF  = X
            layers = [X]
            for item in range(0,len(self.synapses)):
                dF = self.sigmoid(np.dot(dF,self.synapses[item])) 
                layers.append(dF)
            self.lastCalculatedLayer = layers
            Error = (y-layers[-1])**2
            fD = Error*self.sigmoid(layers[-1],True)
            self.synapses[-1]+= layers[-2].T.dot(fD)*learningRate
            for lid in reversed(range(0,len(self.synapses)-1)):
                fD = fD.dot(self.synapses[lid+1].T)*self.sigmoid(layers[lid+1],True)
                self.synapses[lid]+= layers[lid].T.dot(fD)*learningRate

def main():
    import random
    vs = NeuroVisualizer()
    ne = Neurogine([
        2,4,1
    ])
    ne.loadVisualizer(vs)
    vs.VisualizeAsync()
    Input = np.array([
            [0,0],
            [0,1],
            [1,0],
            [1,1]
        ])
    Output = np.array([
            [0],
            [1],
            [1],
            [0]
        ])
    ne.train(Input,Output,10000,0.1)
    for i in range(0,16):
        print(ne.thinkLayered([Input[i%len(Input)]]))
        time.sleep(.5)
if __name__ == '__main__':
    main()
    
    