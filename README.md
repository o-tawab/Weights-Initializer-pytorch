# Weight Initializer For pytorch Models
This is a class to make initializing the weights easier in pytorch.

## How to use
First, few imports
```python
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from weight_initializer import Initializer
```
Then, we can define a simple model
```python
# Simple model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```
After that all what we need to do is to instantiate the model and call the weight initializer. You can pass whatever arguments you need to pass to the weight initializer.
```python
net = Model()  # instantiate the model

# to apply xavier_uniform:
Initializer.initialize(model=net, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

# or maybe normal distribution:
Initializer.initialize(model=net, initialization=init.normal, mean=0, std=0.2)
```