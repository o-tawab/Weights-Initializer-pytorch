import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from weight_initializer import Initializer


# Sample model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


def main():
    net = Model()  # instantiate the model

    # to apply xavier_uniform:
    Initializer.initialize(model=net, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

    # or maybe normal distribution:
    Initializer.initialize(model=net, initialization=init.normal, mean=0, std=0.2)


if __name__ == '__main__':
    main()
