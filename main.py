from once.oncedatastruct import SupervisedData
 from once.oncedataset import OnceSupervisedDataSet
from once.oncepreprocess import OncePreprocess
# from torch.utils.data import DataLoader
from once.onceoptimizer import onceoptimizer
from once.oncemodel import MultiOutputModel
from once.onceprocess import onceprocess
import time

def main():
    since = time.time()
    oncedatasetpath = 'D:/Dataset/ONCEdata/'
    onceinstancedatastruct = SupervisedData()
    onceinstancedataset = OnceSupervisedDataSet(path = oncedatasetpath, data_struct = onceinstancedatastruct)
    # data_struct = oncedataset.forward()
    # print(oncedataset)
    onceinstancepreprocess = OncePreprocess(dataset = onceinstancedataset)
    onceinstancepreprocess.before_train()
    trainedmodelpath = './results/ONCE_18model_parameter.pkl' # load pretrain model parameters
    onceinstancemodel = MultiOutputModel(path = trainedmodelpath)

    onceinstanceoptimizer = onceoptimizer()
    onceinstanceprocess = onceprocess(model= onceinstancemodel,optimizer=onceinstanceoptimizer,preprocess_block=onceinstancepreprocess)
    onceinstanceprocess.process()

    time_elapsed = time.time() - since
    print("All compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    









if __name__ == '__main__':
    main()