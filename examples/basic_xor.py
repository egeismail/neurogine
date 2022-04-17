import numpy as np
from neurogine import Neurogine
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