import numpy as np
import pygame as pg
import threading
import time
import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

class NeuroVisualizer:
    def __init__(self):
        self.isLoaded = False
        self.layerCircleMarginBetweenHorizontal = 250
        self.layerCircleMarginBetweenVertical = 5
        self.WeightColorStrength = 0.5
        self.layerCircleDiameter = 16
        self.connectionThickness = 1
        self.CircleThickness = 1
        self.innerPadding = 10
        self.Done = False
    def calculateWindowSize(self):
        if(self.nrg):
            layerCount = len(self.nrg.layers)
            maxLayerCount = max(self.nrg.layers)
            self.width = self.innerPadding*2+layerCount*self.layerCircleDiameter+(layerCount-1)*self.layerCircleMarginBetweenHorizontal
            self.height = self.innerPadding*2+maxLayerCount*self.layerCircleDiameter+(maxLayerCount-1)*self.layerCircleMarginBetweenVertical
            return (self.width,self.height)
        return (200,200)
    def load(self,nrg):
        self.nrg = nrg
    def tickEvents(self):
        for event in pg.event.get():
            # self.EventExecutor(event)
            if event.type == pg.QUIT:
                print("Quitting.")
                self.Done=True
                pg.quit()
    def VisualizeAsync(self):
        threading.Thread(target=self.mainloop_).start()
    def mainloop_(self):
        pg.init()
        pg.display.set_caption('Neurogine')
        self.wh = self.calculateWindowSize()
        print("Visualizing neurogine on %sx%s"%(self.wh))
        self.display = pg.display.set_mode(self.wh)
        while not self.Done:
            self.tickEvents()
            self.update()
            time.sleep(.013)
    def drawNeuros(self):
        # print(len(self.nrg.lastCalculatedLayer))
        if(len(self.nrg.lastCalculatedLayer)>0):
            mxl = max(self.nrg.layers)
            lc = len(self.nrg.layers)
            ckx = lambda i:self.innerPadding+(i)*(self.layerCircleDiameter+self.layerCircleMarginBetweenHorizontal)+(self.layerCircleDiameter)//2
            cky = lambda j,i:0.5*(mxl-self.nrg.layers[i])*(self.layerCircleDiameter+self.layerCircleMarginBetweenVertical)+((self.innerPadding+((j)*(self.layerCircleDiameter+self.layerCircleMarginBetweenVertical)+(self.layerCircleDiameter)//2)))
            for i in range(0,lc):
                for j in range(0,self.nrg.layers[i]):
                    # print("Actv",self.nrg.lastCalculatedLayer)
                    activation = self.nrg.lastCalculatedLayer[i][-1][j]
                    cx = ckx(i)
                    cy = cky(j,i)
                    pg.draw.circle(
                        self.display,
                        (255*(abs(1-abs(activation))),255*(abs(activation)),0),
                        (cx,cy),
                        self.layerCircleDiameter//2,
                        0
                    )
                    pg.draw.circle(
                        self.display,
                        (255,255,255),
                        (cx,cy),
                        self.layerCircleDiameter//2,
                        self.CircleThickness
                    )
                    if(i!=lc-1):
                        for wx in range(0,self.nrg.layers[i+1]):
                            wx_value = self.nrg.synapses[i][j][wx]
                            kcxsg = self.nrg.sigmoid(self.WeightColorStrength*wx_value)
                            kcxsr = 1-self.nrg.sigmoid(self.WeightColorStrength*wx_value)
                            pg.draw.line(
                                self.display,
                                (kcxsr*255,kcxsg*255,0),
                                (cx+self.layerCircleDiameter//2,cy),
                                (ckx(i+1)-self.layerCircleDiameter//2,cky(wx,i+1)),
                                self.connectionThickness
                            )
                        # print(self.nrg.layers[i+1],"------")
    def update(self):
        self.drawNeuros()
        pg.display.update()
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
    
    