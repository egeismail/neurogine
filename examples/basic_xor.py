import numpy as np
from neurogine.neurogine import Neurogine
ne = Neurogine([
        2,4,1
    ])
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

print(ne.think(Input[0]))
print(ne.think(Input[1]))
print(ne.think(Input[2]))
print(ne.think(Input[3]))