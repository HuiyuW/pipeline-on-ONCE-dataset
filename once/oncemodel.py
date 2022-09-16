from abstract.model import IModel
from torchvision import models
import torch.nn as nn
import torch





class MultiOutputModel(IModel):
    def __init__(self, path):
        super().__init__()
        self.n_weather_classes = 3
        self.n_period_classes = 4

        resnet18 = models.resnet18(pretrained=True)

        self.base_model = nn.Sequential(*(list(resnet18.children())[0:8]))
        
  
        last_channel = models.resnet18().fc.in_features 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.weather = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=self.n_weather_classes)
        )
        self.period = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=self.n_period_classes)
        )
        self.load_state_dict(torch.load(path))

        


    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        return {
            'weather': self.weather(x),
            'period': self.period(x)
        }

