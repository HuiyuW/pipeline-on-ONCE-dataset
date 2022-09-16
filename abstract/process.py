from abstract.preprocess import APreprocess
from abstract.model import IModel
from abstract.optimizer import IOptimizer

class IProcess():   
    preprocess_block: APreprocess
    model : IModel
    optimizer: IOptimizer

    def process(self)-> None:
        raise NotImplementedError
    
    def save(self)-> None:
        raise NotImplementedError 

