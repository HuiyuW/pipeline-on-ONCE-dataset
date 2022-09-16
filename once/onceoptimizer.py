from abstract.optimizer import IOptimizer
from abstract.model import IModel
from abstract.preprocess import APreprocess
from once.creterion import ConfusionMatrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot as plt
import numpy as np


class Aonceoptimizer(IOptimizer):
    def __init__(self,epoch,batch_size):
        self.epoch = epoch
        self.batch_size = batch_size  
        super().__init__()


class onceoptimizer(Aonceoptimizer):
    def __init__(self,epoch=15,batch_size=64):
        self.epoch = epoch
        self.batch_size = batch_size  
        super().__init__(epoch,batch_size)

    def train(self, preprocess_block: APreprocess, model: IModel, save) -> IModel:
        since = time.time()

        device = torch.device('cuda')
        trainDataloader = DataLoader(dataset = preprocess_block.get_set('Train'),batch_size=self.batch_size, shuffle=False)
        n_train_samples = len(trainDataloader)
        trainDictW = preprocess_block.countNum()[0]
        trainDictP = preprocess_block.countNum()[1]
        train_wacc_history = []
        train_pacc_history = []
        train_loss_history = []


        class_sample_counts_w = [trainDictW[0], trainDictW[1], trainDictW[2]]
        weights_w = 1. / torch.tensor(class_sample_counts_w, dtype=torch.float)
        weights_w = weights_w.to(device)
        class_sample_counts_p = [trainDictP[0], trainDictP[1], trainDictP[2], trainDictP[3]]
        weights_p = 1. / torch.tensor(class_sample_counts_p, dtype=torch.float)
        weights_p = weights_p.to(device)
        criterion_p = nn.CrossEntropyLoss(weight=weights_p)  
        criterion_w = nn.CrossEntropyLoss(weight =weights_w)


        for epoch in range(self.epoch):
            total_loss = 0
            accuracy_weather = 0
            accuracy_period = 0
            for batch in trainDataloader:
                periodloss = 0
                weatherloss = 0
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.zero_grad()
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                model.to(device)
                output = model(img.to(device))


                periodloss = criterion_p(output['period'], target_labels['period_labels'])
                weatherloss = criterion_w(output['weather'], target_labels['weather_labels'])
                loss_train = periodloss + weatherloss


                total_loss += loss_train.item()
                batch_accuracy_weather, batch_accuracy_period= \
                    self.calculate_metrics(output, target_labels)
                accuracy_weather += batch_accuracy_weather
                accuracy_period += batch_accuracy_period
                loss_train.backward()
                optimizer.step()
            print("epoch {:4d}, loss: {:.4f}, weather: {:.4f}, period: {:.4f}".format(
                epoch,
                total_loss / n_train_samples,
                accuracy_weather / n_train_samples,
                accuracy_period / n_train_samples))
            train_loss_history.append(total_loss / n_train_samples)
            train_wacc_history.append(accuracy_weather / n_train_samples)
            train_pacc_history.append(accuracy_period / n_train_samples)


        time_elapsed = time.time() - since
        save()
        print("Train compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
        self.plotacc(train_wacc_history,train_pacc_history)
        self.plotloss(train_loss_history)
        self.val(preprocess_block = preprocess_block,model = model)
        return model

    def val(self,preprocess_block: APreprocess, model:IModel) ->None:
        since = time.time()
        model.eval()

        confusion_p = ConfusionMatrix(num_classes=4)
        confusion_w = ConfusionMatrix(num_classes=3)
        device = torch.device('cuda')
        model.to(device)
        classes_p = ('morning', 'noon', 'afternoon','night')
        classes_w = ('sunny', 'cloudy', 'rainy')
        trainDictW = preprocess_block.countNum()[0]
        trainDictP = preprocess_block.countNum()[1]
        class_correct_p = list(0. for i in range(4))
        class_correct_w = list(0. for i in range(3))
        class_total_p = list(0. for i in range(4))
        class_total_w = list(0. for i in range(3))


        class_sample_counts_w = [trainDictW[0], trainDictW[1], trainDictW[2]]
        weights_w = 1. / torch.tensor(class_sample_counts_w, dtype=torch.float)
        weights_w = weights_w.to(device)
        class_sample_counts_p = [trainDictP[0], trainDictP[1], trainDictP[2], trainDictP[3]]
        weights_p = 1. / torch.tensor(class_sample_counts_p, dtype=torch.float)
        weights_p = weights_p.to(device)
        criterion_p = nn.CrossEntropyLoss(weight=weights_p)  
        criterion_w = nn.CrossEntropyLoss(weight =weights_w)

        with torch.no_grad():
            avg_loss = 0
            accuracy_weather = 0
            accuracy_period = 0
            valDataloader = DataLoader(dataset = preprocess_block.get_set('Validation'),batch_size=self.batch_size, shuffle=False)


            for batch in valDataloader:
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                output = model(img.to(device))

                periodloss = criterion_p(output['period'], target_labels['period_labels'])
                weatherloss = criterion_w(output['weather'], target_labels['weather_labels'])
                loss_test = periodloss + weatherloss

                avg_loss += loss_test.item()
                batch_accuracy_weather, batch_accuracy_period = \
                    self.calculate_metrics(output, target_labels)

                accuracy_weather += batch_accuracy_weather
                accuracy_period += batch_accuracy_period
                _, preds_p = torch.max(output['period'], 1)
                _, preds_w = torch.max(output['weather'], 1) 
                confusion_p.update(preds_p.cpu().numpy(), target_labels['period_labels'].cpu().numpy())  
                confusion_w.update(preds_w.cpu().numpy(), target_labels['weather_labels'].cpu().numpy())  

                c_p = (preds_p == target_labels['period_labels']).squeeze()   
                for i in range(len(target_labels['period_labels'])):
                    label_p = target_labels['period_labels'][i]
                    class_correct_p[label_p] += c_p[i].item()
                    class_total_p[label_p] += 1
                c_w = (preds_w == target_labels['weather_labels']).squeeze()   
                for i in range(len(target_labels['weather_labels'])):
                    label_w = target_labels['weather_labels'][i]
                    class_correct_w[label_w] += c_w[i].item()
                    class_total_w[label_w] += 1


        n_samples = len(valDataloader)
        avg_loss /= n_samples
        accuracy_weather /= n_samples
        accuracy_period /= n_samples
        print('-' * 72)
        print("Val  loss: {:.4f}, weather: {:.4f}, period: {:.4f}\n".format(
            avg_loss, accuracy_weather, accuracy_period))
        for i in range(4):
            print('Accuracy of %5s : %2d %%' %(classes_p[i],100*class_correct_p[i]/class_total_p[i]))
        print("-"*10)
        for i in range(3):
            print('Accuracy of %5s : %2d %%' %(classes_w[i],100*class_correct_w[i]/class_total_w[i]))
        acc_p,table_p = confusion_p.summary()   
        acc_w,table_w = confusion_w.summary()
        print("period creterion")
        print(table_p)
        print("weather creterion")
        print(table_w)
        confusion_p.plot()
        confusion_w.plot()
        time_elapsed = time.time() - since
        print("Val compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))



    def test(self,preprocess_block: APreprocess, model:IModel, save) ->None:
        since = time.time()
        model.eval()
        confusion_p = ConfusionMatrix(num_classes=4)
        confusion_w = ConfusionMatrix(num_classes=3)
        device = torch.device('cuda')
        model.to(device)
        classes_p = ('morning', 'noon', 'afternoon','night')
        classes_w = ('sunny', 'cloudy', 'rainy')
        trainDictW = preprocess_block.countNum()[0]
        trainDictP = preprocess_block.countNum()[1]
        class_correct_p = list(0. for i in range(4))
        class_correct_w = list(0. for i in range(3))
        class_total_p = list(0. for i in range(4))
        class_total_w = list(0. for i in range(3))


        class_sample_counts_w = [trainDictW[0], trainDictW[1], trainDictW[2]]
        weights_w = 1. / torch.tensor(class_sample_counts_w, dtype=torch.float)
        weights_w = weights_w.to(device)
        class_sample_counts_p = [trainDictP[0], trainDictP[1], trainDictP[2], trainDictP[3]]
        weights_p = 1. / torch.tensor(class_sample_counts_p, dtype=torch.float)
        weights_p = weights_p.to(device)
        criterion_p = nn.CrossEntropyLoss(weight=weights_p)  
        criterion_w = nn.CrossEntropyLoss(weight =weights_w)

        with torch.no_grad():
            avg_loss = 0
            accuracy_weather = 0
            accuracy_period = 0
            testDataloader = DataLoader(dataset = preprocess_block.get_set('Test'),batch_size=self.batch_size, shuffle=False)


            for batch in testDataloader:
                img = batch['img']
                target_labels = batch['labels']
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                output = model(img.to(device))

                periodloss = criterion_p(output['period'], target_labels['period_labels'])
                weatherloss = criterion_w(output['weather'], target_labels['weather_labels'])
                loss_test = periodloss + weatherloss

                avg_loss += loss_test.item()
                batch_accuracy_weather, batch_accuracy_period = \
                    self.calculate_metrics(output, target_labels)

                accuracy_weather += batch_accuracy_weather
                accuracy_period += batch_accuracy_period
                _, preds_p = torch.max(output['period'], 1)
                _, preds_w = torch.max(output['weather'], 1) 
                confusion_p.update(preds_p.cpu().numpy(), target_labels['period_labels'].cpu().numpy())  
                confusion_w.update(preds_w.cpu().numpy(), target_labels['weather_labels'].cpu().numpy())  

                c_p = (preds_p == target_labels['period_labels']).squeeze()   
                for i in range(len(target_labels['period_labels'])):
                    label_p = target_labels['period_labels'][i]
                    class_correct_p[label_p] += c_p[i].item()
                    class_total_p[label_p] += 1
                c_w = (preds_w == target_labels['weather_labels']).squeeze()   
                for i in range(len(target_labels['weather_labels'])):
                    label_w = target_labels['weather_labels'][i]
                    class_correct_w[label_w] += c_w[i].item()
                    class_total_w[label_w] += 1


        n_samples = len(testDataloader)
        avg_loss /= n_samples
        accuracy_weather /= n_samples
        accuracy_period /= n_samples
        print('-' * 72)
        print("Test  loss: {:.4f}, weather: {:.4f}, period: {:.4f}\n".format(
            avg_loss, accuracy_weather, accuracy_period))
        for i in range(4):
            print('Accuracy of %5s : %2d %%' %(classes_p[i],100*class_correct_p[i]/class_total_p[i]))
        print("-"*10)
        for i in range(3):
            print('Accuracy of %5s : %2d %%' %(classes_w[i],100*class_correct_w[i]/class_total_w[i]))
        acc_p,table_p = confusion_p.summary()   
        acc_w,table_w = confusion_w.summary()
        print("period creterion")
        print(table_p)
        print("weather creterion")
        print(table_w)
        confusion_p.plot()
        confusion_w.plot()
        time_elapsed = time.time() - since
        print("Test compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))


    def calculate_metrics(self,output, target):
        _, predicted_weather = output['weather'].cpu().max(1)
        gt_weather = target['weather_labels'].cpu()

        _, predicted_period = output['period'].cpu().max(1)
        gt_period = target['period_labels'].cpu()

        with warnings.catch_warnings():  
            warnings.simplefilter("ignore")
            accuracy_weather = accuracy_score(y_true=gt_weather.numpy(), y_pred=predicted_weather.numpy())
            accuracy_period = accuracy_score(y_true=gt_period.numpy(), y_pred=predicted_period.numpy())

        return accuracy_weather, accuracy_period

    def plotacc(self,whistory,phistory):
        fig = plt.figure(1) 
        plt.title("Train Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,self.epoch+1),whistory,label='Train weather Acc')
        plt.plot(range(1,self.epoch+1),phistory,label='Train period Acc')
        # plt.ylim((0,1.))
        plt.xticks(np.arange(1, self.epoch+1))
        plt.legend()
        pic_acc_name = './results/Once_' + str(np.random.randint(0,100)) +'_acc.png'
        plt.savefig(pic_acc_name,bbox_inches='tight')

    def plotloss(self,losshistory):
        fig = plt.figure(2) 
        plt.title("Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1,self.epoch+1),losshistory,label='Train loss')
        # plt.ylim((0,1.))
        plt.xticks(np.arange(1, self.epoch+1))
        plt.legend()
        pic_loss_name = './results/Once_' + str(np.random.randint(0,100)) +'_loss.png'
        plt.savefig(pic_loss_name,bbox_inches='tight')

