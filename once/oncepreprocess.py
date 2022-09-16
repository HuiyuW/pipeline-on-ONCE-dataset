from abstract.preprocess import APreprocess
from abstract.dataset import ADataSet
from torchvision import transforms

class OncePreprocess(APreprocess):
    def __init__(self, dataset:ADataSet,set_type='Train'): 
        super().__init__(dataset,set_type)
        self.datastr = dataset.forward()
        self.set_type = set_type


    def before_train(self) -> None:
        allImage = self.datastr.data['Train']['Input']
        allLabel = self.datastr.data['Train']['Output']
        test_split = 0.2
        test_size = int(test_split * allImage.shape[0])
        testImage = allImage[:test_size]
        testLabel = (allLabel[0][:test_size],allLabel[1][:test_size])

        comsize =  allImage.shape[0] - test_size
        val_split = 0.25
        val_size = int(val_split * comsize)
        valImage = allImage[test_size:test_size+val_size]
        valLabel = (allLabel[0][test_size:test_size+val_size],allLabel[1][test_size:test_size+val_size])
        train_size = comsize - val_size
        trainImage = allImage[-train_size:]
        trainLabel = (allLabel[0][-train_size:],allLabel[1][-train_size:])
        self.datastr.data['Train']['Input'] = trainImage 
        self.datastr.data['Train']['Output'] = trainLabel
        self.datastr.data['Test']['Input'] = testImage
        self.datastr.data['Test']['Output'] = testLabel
        self.datastr.data['Validation']['Input'] = valImage
        self.datastr.data['Validation']['Output'] = valLabel


    def batch_preprocess(self, image):
        image = image.numpy()
        transform = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])
        return transform(image)


    def countNum(self):
        label_weathers = self.datastr.data[self.set_type]['Output'][0]
        label_periods = self.datastr.data[self.set_type]['Output'][1]
        label_weathers = label_weathers.numpy()
        label_periods = label_periods.numpy()


        dict_p = {}
        for key in label_periods:
            dict_p[key] = dict_p.get(key, 0) + 1 
        dict_w = {}
        for key in label_weathers:
            dict_w[key] = dict_w.get(key, 0) + 1


        return dict_w, dict_p

    def __getitem__(self, index):
        
        image = self.datastr.data[self.set_type]['Input'][index]

        img = self.batch_preprocess(image)
        label_weather = self.datastr.data[self.set_type]['Output'][0][index]
        label_period = self.datastr.data[self.set_type]['Output'][1][index]
        label_weather = label_weather.numpy().tolist()
        label_period = label_period.numpy().tolist()
        dict_data = {
            'img': img,
            'labels': {
                'weather_labels': label_weather,
                'period_labels': label_period
            }
        }


        return dict_data

    def __len__(self):
        return self.datastr.data[self.set_type]['Input'].shape[0]


    def get_set(self, phase): 
        self.set_type = phase
        return self 

    
class PREv2(OncePreprocess): 
    def pre_train(self):
        pass