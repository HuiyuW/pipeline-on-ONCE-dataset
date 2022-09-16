from abstract.process import IProcess
from abstract.optimizer import IOptimizer
from abstract.model import IModel
from abstract.preprocess import APreprocess
import torch
import os
import numpy as np

class onceprocess(IProcess):
    def __init__(self,model:IModel,optimizer:IOptimizer,preprocess_block:APreprocess) -> None:
        super().__init__()
        self.model = model 
        self.optimizer = optimizer
        self.preprocess_block = preprocess_block

    def process(self)-> None:
        
        trainedmodel = self.optimizer.train(preprocess_block= self.preprocess_block, model=self.model, save = self.save)

        self.optimizer.test(preprocess_block= self.preprocess_block, model=trainedmodel,save = self.save)

        
    
    def save(self)-> None:

        new_folder = './results/'
        self.mkdir(new_folder)
        model_parameter_saved_name = new_folder + 'ONCE_' + str(np.random.randint(0,100)) +'_model_parameter.pkl'
        torch.save(self.model.state_dict(), model_parameter_saved_name) 
        
    def mkdir(self,path):
    
        folder = os.path.exists(path)
    
        if not folder:                   
            os.makedirs(path)           
            print ("---  new folder...  ---")
            print ("---  OK  ---")
    
        else:
            print ("---  There is this folder!  ---")


