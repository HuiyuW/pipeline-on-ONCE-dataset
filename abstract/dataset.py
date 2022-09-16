from abstract.datastruct import ADataStruct, IDataStruct



class IDataSet():   
    data_struct: IDataStruct 
    def load(self) -> None:
        raise NotImplementedError
    
    def update(self) -> None:
        raise NotImplementedError

    def forward(self)-> ADataStruct:
        raise NotImplementedError    

class ADataSet(IDataSet):
    def __init__(self, data_struct: ADataStruct) -> None:
        self.data_struct = data_struct  

    def forward(self) -> ADataStruct:
        return self.data_struct
