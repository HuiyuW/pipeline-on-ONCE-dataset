from abstract.datastruct import ADataStruct


class SupervisedData(ADataStruct):
    def __init__(self) -> None:
        set_types = ['Train', 'Test', 'Validation']
        data_types = ['Input', 'Output']
        super().__init__(set_types, data_types)


class UnsupervisedData(ADataStruct):
    def __init__(self) -> None:
        set_types = ['Train', 'Test', 'Validation']
        data_types = ['Input']
        super().__init__(set_types, data_types)