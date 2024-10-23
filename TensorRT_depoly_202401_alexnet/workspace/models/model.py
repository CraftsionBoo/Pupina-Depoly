import torch
import torch.nn as nn 
import utils
print = utils.logger.info

import torch.utils.model_zoo as model_zoo 
from collections import OrderedDict

class Alexnet(nn.Module):
    def __init__(self, num_classes:int=10, dropout:float=0.5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # 32x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 64x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 128x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 256x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 256x7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)          # 256x3x3
        )
        self.classifer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128*3*3, 512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.feature(x)       # 256x3x3
        x = x.view(x.size(0), -1) #2304
        x = self.classifer(x)     # 10
        return x

def alexnet(pretrained=False, model_root=None, **kwards):
    model = Alexnet(**kwards)
    if pretrained:
        model.load_state_dict(torch.load(model_root + "/alexnet_pretrianed.pth"))
    return model


# 全链接
model_urls = {
    'mnist' : "http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth"
}

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_classes):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i,n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU(inplace=True)
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_classes)
        self.model = nn.Sequential(layers)
        print(self.model)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        assert x.size(1) == self.input_dims
        return self.model.forward(x)

def mlp(input_dims=784, n_hiddens=[256,256], n_classes=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_classes)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict if isinstance(m, nn.Module) else m 
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model