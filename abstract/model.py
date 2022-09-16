from torch.nn import Module
from typing import Any



class IModel(Module):    

    def forward(self, inputs: Any)-> Any:
        raise NotImplementedError


