import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F



class Img_Dataset(Dataset):
    
    def __init__(self, img_x, img_y):
        self.img_x = torch.Tensor(img_x)
        self.img_y = torch.Tensor(img_y)
        
    def __getitem__(self, index):
        return (self.img_x[index], self.img_y[index])


    def __len__(self):
        return len(self.img_x)
    
    

class Image_Classifier(torch.nn.Module):
    def __init__(self):
        super(Image_Classifier, self).__init__()
        
        self.Z1 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.A1 = torch.nn.ReLU(self.Z1)   
        self.P1 = torch.nn.MaxPool2d(2, stride=2)
        
        self.Z2 = torch.nn.Conv2d(12, 18, 4, stride=1, padding=1)
        self.A2 = torch.nn.ReLU(self.Z2)
        self.P2 = torch.nn.MaxPool2d(2, stride=2)
        self.linear1 = torch.nn.Linear(4050, 720, bias=True)
        self.A3 = torch.nn.ReLU(self.linear1)
        self.linear2 = torch.nn.Linear(720, 100, bias=True)
        self.A4 = torch.nn.ReLU(self.linear2)
        self.linear3 = torch.nn.Linear(100, 6, bias=True)
        

            
    def forward(self, data):

        layer1_1 = self.Z1(data)
        
        layer1_2 = self.A1(layer1_1)
        
        layer1_3 = self.P1(layer1_2)

        
        layer2_1 = self.Z2(layer1_3)

        layer2_2 = self.A2(layer2_1)
        layer2_3 = self.P2(layer2_2)
       
        flatten = layer2_3.view(layer2_3.size()[0], 4050 )
       
        lin_out_1 = self.linear1(flatten)
        layer3_1 = self.A3(lin_out_1)
        layer3_2 = self.A4(self.linear2(layer3_1))
        out = self.linear3(layer3_2)
    
        return F.log_softmax(out)
        
    
def main():
    torch.manual_seed(100)
    train_data = h5py.File('Data/train_signs.h5', 'r')
    train_x, train_y = train_data['train_set_x'], train_data['train_set_y']
    train_dataset = Img_Dataset(train_x, train_y)
    train_dl = DataLoader(train_dataset, batch_size=16, drop_last = True, shuffle=True)
    
    test_data = h5py.File('Data/test_signs.h5', 'r')
    test_x, test_y = test_data['test_set_x'], test_data['test_set_y']
    test_dataset = Img_Dataset(test_x, test_y)
    test_dl = DataLoader(test_dataset, batch_size=16, drop_last=True, shuffle=False)
    
    model = Image_Classifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss()
    
    print('****** Training Model ****************')
    for i in range(10):
        loss_avg = []
        acc_avg = []
        model.train()
        for batch in tqdm(train_dl):
            #print(batch[0])

            batch_dat = batch[0].permute(0, 3, 1, 2)

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            predict = model.forward(batch_dat)

            #print('****PREDICTION***********')
            #print(predict)

            pred = np.argmax(predict.detach().numpy(), axis=1)

            count = sum(sum([pred == batch[1].long().detach().numpy()]))


            acc = count/len(pred)
            acc_avg.append(acc)
            loss = criterion(predict, batch[1].long())

            #print(loss)
            loss_avg.append(loss.detach().cpu())
            loss.backward()

            optimizer.step()  
    

        print('Avg Train Loss for Epoch : '+str(i)+" is "+str(np.mean(loss_avg)))
        print('Avg Train accuracy for Epoch : '+str(i)+" is "+str(np.mean(acc_avg)))
    ### Validation after each epoch 

        model.eval()
        val_acc = []
        for val_batch in test_dl:
            val_dat = val_batch[0].permute(0, 3, 1, 2)
            with torch.no_grad():
                predict = model.forward(val_dat)
                pred = np.argmax(predict.detach().numpy(), axis=1)
                count = sum(sum([pred == val_batch[1].long().detach().numpy()]))
                acc = count/len(pred)
                val_acc.append(acc)

        print('Avg Validation accuracy for Epoch : '+str(i)+" is "+str(np.mean(val_acc)))
        
    torch.save(model, "image_model.pt")

if __name__ == '__main__':
    main()
