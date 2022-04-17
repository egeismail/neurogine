
<h1 align="center">Neurogine</h1>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/egeismail/neurogine)
[![GitHub Issues](https://img.shields.io/github/issues/septillioner/TEOPS.svg)](https://github.com/egeismail/neurogine/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/septillioner/TEOPS.svg)](https://github.com/egeismail/neurogine/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---


## 📝 Table of Contents

- [About](#about)
- [Examples](#examples)
- 1. [XOR Training](#xor_training)
-  [License](#examples)
-  [Author](#authors)
---
## 🧐 About <a name = "about"></a>

Simplest, visual and flexible neural network builder

### Prerequisites

Just Python 3.6.0>

---
## ✒ Examples <a name = "examples"></a>
### XOR Training <a name = "xor_training"></a>
```python
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
```
### Output
```bash

>python basic_xor.py
[0.09579394]
[0.91582867]
[0.91082701]
[0.09591084]
```

---
## ✍️ Authors <a name = "authors"></a>

- [@egeismail](https://github.com/egeismail) - Idea & Initial work

See also the list of [contributors](https://github.com/egeismail/neurogine) who participated in this project.

---
## 📜License

MIT


