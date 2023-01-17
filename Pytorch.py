from code import interact
from multiprocessing.util import close_all_fds_except
import torch
from torchvision import transforms
from PIL import Image
import os
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

torch.zeros(1).cuda()


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 normalize])
#TARGET: [isCat, isDog]
train_data_list = []
target_list = []
train_data = []
waited = False
files = listdir('CatDog/train/')
for i in range(len(listdir('catdog/train/'))):
    if len(train_data) == 58 and not waited:
        waited = True
        continue
    f = random.choice(files)
    files.remove(f)
    img = Image.open("CatDog/train/" + f)
    img_tensor = transforms(img)#(3,256,256)
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        print('Loaded batch ', len(train_data), 'of ', int(len(listdir('catdog/train/'))/64))
        print('Percentage Done: ', 100*len(train_data)/int(len(listdir('catdog/train/'))/64), '%')
        if len(train_data) > 450:
            break
class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=5)
        self.fc1 = nn.Linear(18816, 9408)
        self.fc2 = nn.Linear(9408, 3456)
        self.fc3 = nn.Linear(3456, 1000)
        self.fc4 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1,18816)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return torch.sigmoid(x)

model = Netz()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data),
                   100. * batch_id / len(train_data), loss.data))
        batch_id = batch_id + 1



def test():
    model.eval()
    files = listdir('catdog/train/')
    f = random.choice(files)
    img = Image.open('catdog/train/' + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim=True)[1])
    print(out.data)
    outData = out.data.max(1, keepdim=True)[1]
    return outData, img.filename
   




for epoch in range(1,50):
    train(epoch)

isTrueCat = 0
isFalseCat = 0
isTrueDog = 0
isFalseDog = 0

TestLauf = int(input("Wie viele Durchgaenge soll getestet werden?: "))
for i in range(TestLauf):
    testFunc = test()
    print(testFunc[1])
    if "cat" in testFunc[1][7:]:
        if testFunc[0] == 0:
            isTrueCat += 1
        else:
            isFalseCat += 1
    elif "dog" in testFunc[1][7:]:
        if testFunc[0] == 1:
            isTrueDog += 1
        else:
            isFalseDog += 1
ywerte = [isTrueCat, isFalseCat, isTrueDog, isFalseDog]
xwerte = ["Cat true", "Cat false", "Dog true", "Dog false"]
plt.bar(xwerte, ywerte)
plt.xlabel("Item found")
plt.ylabel("How often it got found")
plt.title("True: Higher is better; False: lower is better")
plt.show()

# 0 = Katze
# 1 = Hund

