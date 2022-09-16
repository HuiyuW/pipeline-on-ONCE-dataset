from abstract.dataset import IDataSet,ADataSet
import torch
from typing import Dict

class IPreprocess():
    dataset:IDataSet
    set_type:str

    # required by the dataloader
    def __init__(self) -> None: 
        raise NotImplementedError
    # required by the dataloader
    def __getitem__(self, index: int)-> Dict[str,torch.Tensor]:
        raise NotImplementedError
    # required by the dataloader
    def __len__(self) -> int:
        raise NotImplementedError
    def batch_preprocess(self, index: int)-> Dict[str,torch.Tensor]:
        raise NotImplementedError
    def before_train(self)-> None:
        raise NotImplementedError
    def get_set(self,set_type)-> 'IPreprocess':
        raise NotImplementedError



class APreprocess(IPreprocess):
    def __init__(self, dataset:ADataSet,set_type:str) -> None:
        self.dataset = dataset
        self.set_type = set_type

    def __getitem__(self, index)-> Dict[str,torch.Tensor]:
        return self.batch_preprocess(index)
    def get_set(self, set_type)-> 'APreprocess': 
        self.set_type = set_type
        return self


