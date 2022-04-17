import pygame as pg
import threading
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