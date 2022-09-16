from typing import Dict, List
import torch

class IDataStruct():
    data: Dict[str,Dict[str,torch.Tensor]]
    set_types: List[str]
    data_types: List[str]

class ADataStruct(IDataStruct):
    def __init__(self, set_types: List[str], data_types: List[str]) -> None:
        self.data = dict()
        self.set_types = set_types
        self.data_types = data_types
        for set_type in set_types:
            self.data[set_type] = dict()
            for data_type in data_types:
                self.data[set_type][data_type] = torch.tensor([])





