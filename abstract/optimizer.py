from abstract.preprocess import APreprocess
from abstract.model import IModel

class IOptimizer():
    def train(self, preprocess_block: APreprocess, model: IModel, save)-> IModel:
        raise NotImplementedError

    def eval(self, preprocess_block: APreprocess, model: IModel, save)-> None:
        raise NotImplementedError

