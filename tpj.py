import torch
class TPJ1(torch.nn.Module):
    def __init__(self):
        super (TPJ1, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.relu1 = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm2d(32)
        torch.nn.init.xavier_uniform_(self.cnn1.weight)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.cnn2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2)
        self.relu2 = torch.nn.ReLU()
        self.norm2 = torch.nn.BatchNorm2d(64)
        torch.nn.init.xavier_uniform_(self.cnn2.weight)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fcrelu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4096, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fcrelu(out)
        out = self.fc2(out)
        return out


import torch.nn as nn
import math
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class TPJ2(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(TPJ2, self).__init__()
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)  #modify first-parameter/channel: 3-->1
        block = BasicBlock
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


import random
import math
class RandomErasing(object):
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        return img

from torchvision.transforms import *
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5, resample=False, expand=False, center=None),   #john
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),   #john
    transforms.RandomGrayscale(0.333),   #john
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    RandomErasing(probability=0.5, sl=0.01, sh=0.4, r1=0.3, mean=[0.4914]),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    RandomErasing(probability=0.5, sl=0.01, sh=0.4, r1=0.3, mean=[0.4914]),
])
'''
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
'''
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def adjust_learning_rate(optimizer, epoch, learning_rate):
    factor = [0.1, 0.1, 0.2]
    epochs = [100, 300, 700]
    if epoch in epochs:
        print('epoch=',epoch,'index=',epochs.index(epoch),'factor=',factor[epochs.index(epoch)])
        learning_rate *= factor[epochs.index(epoch)]
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    return learning_rate

#John: 上帝首先是个思想家或哲学家，不同的问题解决的难度不一样，动用的资源不一样,这个是用简单网络测试后估算, you can test Distinguish-Difficulty between two-classes use a mini-network, for Distinguish-Difficulty, the one-one boosting is a better solution!
generalization_table = {  #[letter, greater]
'[0, 1]':98.50, '[0, 2]':97.90, '[0, 3]':-1.00, '[0, 4]':-1.00, '[0, 5]':-1.00, '[0, 6]':-1.00, '[0, 7]':-1.00, '[0, 8]':-1.00, '[0, 9]':-1.00, 
                '[1, 2]':99.00, '[1, 3]':-1.00, '[1, 4]':-1.00, '[1, 5]':99.50, '[1, 6]':-1.00, '[1, 7]':-1.00, '[1, 8]':-1.00, '[1, 9]':-1.00,
                                '[2, 3]':-1.00, '[2, 4]':-1.00, '[2, 5]':-1.00, '[2, 6]':-1.00, '[2, 7]':-1.00, '[2, 8]':-1.00, '[2, 9]':-1.00,
                                                '[3, 4]':-1.00, '[3, 5]':-1.00, '[3, 6]':-1.00, '[3, 7]':-1.00, '[3, 8]':-1.00, '[3, 9]':-1.00,
                                                                '[4, 5]':-1.00, '[4, 6]':-1.00, '[4, 7]':-1.00, '[4, 8]':-1.00, '[4, 9]':-1.00,
                                                                                '[5, 6]':-1.00, '[5, 7]':-1.00, '[5, 8]':-1.00, '[5, 9]':-1.00,
                                                                                                '[6, 7]':-1.00, '[6, 8]':-1.00, '[6, 9]':-1.00,
                                                                                                                '[7, 8]':-1.00, '[7, 9]':-1.00,
                                                                                                                                '[8, 9]':-1.00,
}
def surpass_generalization_limitation(pickup_group, generalization):
    pickup_group_key = str(pickup_group)
    if not generalization_table.__contains__(pickup_group_key):
        pickup_group_key = str([pickup_group[1],pickup_group[0]])
    limitation = generalization_table[pickup_group_key]
    if limitation==-1.00:
        if generalization < 95.0:  #default
            return False
        else:
            return True
    else:
        if generalization < limitation:
            return False
        else:
            return True


import os
def load_model(pickup_group):
    path = os.path.join('subnetwork', str(pickup_group))
    if os.path.exists(path):
        files = os.listdir(path)
        if len(files) > 0:
            files.sorted(reverse=True)
            model = torch.load(files[0])  #torch.save(model.state_dict(), 'params.pkl')
            return model
    return None

def save_model(pickup_group, generalization_accuracy, model):
    path = os.path.join('subnetwork', str(pickup_group))
    if not os.path.exists(path):
        os.makedirs(path)
    file = os.path.join(path, str(generalization_accuracy))
    if not os.path.exists(file):
        torch.save(model, file)    #torch.save(model.state_dict(), 'params.pkl')

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable
def mysubnet(pickup_group, train_loader, test_loader):   
    tpj = load_model(pickup_group)
    if not tpj:
        tpj = TPJ2(num_classes=10, depth=28, widen_factor=4)   #depth: 6n+4    #tpj = TPJ1()
    mynet = tpj.to(device)
    optimizer_classify = torch.optim.Adam(mynet.parameters(), lr=0.01)   #John： lr can not be too big, of course, you can your scheduler-lr&mb
    distance_classify= torch.nn.CrossEntropyLoss()   #onehot=False
    #distance_classify= torch.nn.MSELoss()           #onehot=True
    archived = False
    epochs = 22
    #learning_rate = 0.1
    for epoch in range(epochs):
        correct = 0
        total = 0
        loops = int(len(train_loader.dataset) / train_loader.batch_size )
        mynet.train()
        #learning_rate = adjust_learning_rate(optimizer_classify, epoch, learning_rate)
        for batch, (image, label) in enumerate(train_loader):
            image = Variable(image).to(device)
            label = Variable(label).to(device)

            predicted = mynet(image)
            
            #print('train: label=',label)
            classify_loss = distance_classify(predicted, label)

            optimizer_classify.zero_grad()
            classify_loss.backward()
            optimizer_classify.step()

            _, predict = torch.max(predicted, 1)
            correct += (predict == label).sum().item()        
            total += len(predict)               
            
            accumulated_accuracy = correct/total*100
            print('mysubnet:  epoch=%d/%d'%(epoch,epochs),' batch=%d/%d'%(batch,loops),'   accuracy=%.2f'%accumulated_accuracy, '    pickup_group=',pickup_group, '    ', end='\r')

            validate_times = 10 
            if accumulated_accuracy>99.0 and batch % int(loops/validate_times)==0:  #re-stat periodically
                generalization = myvalidate(mynet, test_loader)   #you can use a decay-policy to validate once train-accuracy greater than a threshold, not every big-train
                save_model(pickup_group, generalization, mynet)
                mynet.train()    #John:  MUST must MUST must              
                print()
                print('mysubnet:  generalization=%.2f'%generalization, '  ', end='\n')
                if surpass_generalization_limitation(pickup_group, generalization):
                    archived = True
                    break
                else:
                    pass
        print()
        if archived:
            break

    return (mynet,pickup_group)

def myvalidate(mynet, test_loader):
    correct = 0
    total = 0
    mynet.eval()
    for batch, (image, label) in enumerate(test_loader):
        image = Variable(image).to(device)
        label = Variable(label).to(device)

        predicted = mynet(image)

        _, predict = torch.max(predicted, 1)
        correct += (predict == label).sum().item()        
        total += len(predict) 
    return correct/total*100 

#John：上帝是个思想家+哲学家： 一个脑袋区分很多类很难，但独立区分2类，3类还是很容易，；对于特别难区分的，多训练多增强  #2-classes combin, 3-classes combin, bin-tree combin, tree-combin, any combin
def myinit(mymatrix, key, pickup_group):
    from fashion import Fashion
    import torch.utils.data as data
    train_loader = data.DataLoader(Fashion(root='./fashion', train=True, transform=transform_train, pickup_group=pickup_group), batch_size=120, shuffle=True, num_workers=1)
    test_loader = data.DataLoader(Fashion(root='./fashion', train=False, transform=transform_test, pickup_group=pickup_group), batch_size=100, shuffle=True, num_workers=1)
    mymatrix[key]=mysubnet(pickup_group, train_loader, test_loader) 

import numpy as np
def myvote(mynet_group, test_loader):
    mynet = mynet_group[0]
    group = mynet_group[1]
    mynet.eval()
    votes = []
    for batch, (image, label) in enumerate(test_loader):
        image = Variable(image).to(device)
        label = Variable(label).to(device)

        predicted = mynet(image)

        logits, predict = torch.max(predicted, 1)
        vote = predict.detach().cpu().numpy()
        votes.append(vote)
        break
    return np.array(votes).flatten()

def myfull():  #6 4 2  0
    classes = 3
    mymatrix = {}
    for x in range(0, classes-1):
        for y in range(x+1, classes):
            key = 'pk:'+str(x)+'-'+str(y)
            myinit(mymatrix, key, [x,y])
    #
    from fashion import Fashion
    import torch.utils.data as data    
    for test_class_id in range(classes):    
        test_loader = data.DataLoader(Fashion(root='./fashion', train=False, transform=transform_test, pickup_group=[test_class_id]), batch_size=1, shuffle=True, num_workers=1)
        test_count = 1000
        test_correct = 0
        for count in range(test_count):
            votes_all = []        
            for x in range(0, classes-1):
                for y in range(x+1, classes):
                    key = 'pk:'+str(x)+'-'+str(y)
                    votes = myvote(mymatrix[key],test_loader)
                    votes_all.append(votes)
                    #print("test_class_id=",test_class_id, 'key=',key, 'votes=',votes.flatten())
            #print('votes_all=',votes_all)
            votes_all = np.array(votes_all)
            final_voted = sorted([(np.sum(votes_all==i),i) for i in set(votes_all.flat)])
            final_voted = final_voted[-1][-1]
            #print('final_voted=',final_voted)
            if final_voted == test_class_id:
                test_correct += 1
        print("test_class_id=",test_class_id, 'test_correct=',test_correct)
        print()

if __name__ == '__main__':
    myfull()
