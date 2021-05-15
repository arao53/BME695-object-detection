import argparse, torch.utils.data, glob, os, numpy, PIL, requests, logging, json, torch, matplotlib.pyplot as plt, pandas as pd, random, csv
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as tvt
import torch
from torch import nn
import torch.nn.functional as F

# This was originally written in google colab to access GPU support.
# All plots were made in Colab using this code. This code has been tested in pieces, but not entirely together.
# Please refer to the notebook file if errors persist

## Initialize Seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)


class your_dataset_class(Dataset):
    def __init__(self, path, class_list):
        self.class_list = class_list
        self.folder = path

    def __len__(self):
        g = glob.glob(self.folder + '/**/*.jpg')  # ,'*.jpg')
        return (len(g))

    def __getitem__(self, item):
        g = glob.glob(self.folder + '/**/*.jpg')  # , '*.jpg')

        im = PIL.Image.open(g[item])
        transformer = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        H, W = im.size
        im_array = torch.randint(0, 256, (3, H, W)).type(torch.uint8)
        for i in range(H):
            for j in range(W):
                im_array[:, j, i] = torch.tensor(im.getpixel((i, j)))

        im_scaled = im_array / im_array.max()  # scaled from 0-1
        im_tf = transformer(numpy.transpose(im_scaled.numpy()))

        label = torch.zeros(1)
        for i in range(10):
            if self.class_list[i] in g[item]:
                label = i
        return im_tf, label

parser = argparse.ArgumentParser(description ='HW04 training set')
parser.add_argument('--root_path', type =str, required = True)
parser.add_argument('--class_list', nargs = '*', type =str , required = True)
args, args_other = parser.parse_known_args()

root_path = args.root_path
class_list = args.class_list


train_path = root_path + 'Train'
val_path = root_path + 'Val'

batch_size = 32

train_dataset = your_dataset_class(train_path,class_list)

train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 4,
                                                drop_last=True)

val_dataset = your_dataset_class(val_path, class_list)
val_data_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 4,
                                                drop_last=True)

class TemplateNet1(nn.Module):
  def __init__(self):
    super(TemplateNet1, self).__init__()
    self.conv1 = nn.Conv2d(3,128,3)
    self.conv2 = nn.Conv2d(128, 128, 3)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(31*31*128, 1000)  # vary the input dimension
    #self.fc1 = nn.Linear(14*14*128, 1000)  # ([input - kernal + 2*pad]/stride +1)/pooling
    self.fc2 = nn.Linear(1000, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    #x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1,31*31*128)               # reshape to dim of the 1st fc layer
    #x = x.view(-1,14*14*128)               # reshape to dim of the 1st fc layer
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class TemplateNet2(nn.Module):
  def __init__(self):
    super(TemplateNet2, self).__init__()
    self.conv1 = nn.Conv2d(3,128,3)
    self.conv2 = nn.Conv2d(128, 128, 3)
    self.pool = nn.MaxPool2d(2,2)
    #self.fc1 = nn.Linear(31*31*128, 1000)  # vary the input dimension
    self.fc1 = nn.Linear(14*14*128, 1000)  # ([input - kernal + 2*pad]/stride +1)/pooling
    self.fc2 = nn.Linear(1000, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    #x = x.view(-1,31*31*128)               # reshape to dim of the 1st fc layer
    x = x.view(-1,14*14*128)               # reshape to dim of the 1st fc layer
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class TemplateNet3(nn.Module):
  def __init__(self):
    super(TemplateNet3, self).__init__()
    self.conv1 = nn.Conv2d(3,128,3, padding = 1)
    self.conv2 = nn.Conv2d(128, 128, 3, padding  = 1)
    self.pool = nn.MaxPool2d(2,2)
    #self.fc1 = nn.Linear(32*32*128, 1000)  # vary the input dimension
    #self.fc1 = nn.Linear(14*14*128, 1000)  # ([input - kernal + 2*pad]/stride +1)/pooling
    self.fc2 = nn.Linear(1000, 10)
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    #x = x.view(-1,32*32*128)               # reshape to dim of the 1st fc layer
    x = x.view(-1,14*14*128)               # reshape to dim of the 1st fc layer
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x


def run_code_for_training(net, lrate, mom, epochs):
  device = torch.device('cuda:0')
  net = net.to(device)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr = lrate, momentum = mom)
  loss_tracker = []
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device).long()
      optimizer.zero_grad()
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      loss_tracker = numpy.append(loss_tracker, loss.item())
      print(i, loss.item())
      if (i+1)%500 ==0:
        print("\n[epoch: %d, batch: %5d] Avg batch loss: %.3f" %(epoch + 1, i+1, running_loss/float(500)))
        running_loss = 0.0
  return loss_tracker


lrate = 1e-3
mom = 0.9
epochs = 2

net1 = TemplateNet1()
net2 = TemplateNet2()
net3 = TemplateNet3()

loss_tracker1 = run_code_for_training(net1, lrate, mom, epochs)
save_data1_path = os.path.join(root_path, "loss_net1.csv")
pd.DataFrame(loss_tracker1).to_csv(save_data1_path)
savepath = os.path.join(root_path, "net1.pth")
torch.save(net1, savepath)


loss_tracker2 = run_code_for_training(net2, lrate, mom, epochs)
save_data2_path = os.path.join(root_path, "loss_net2.csv")
pd.DataFrame(loss_tracker2).to_csv(save_data2_path)
savepath = os.path.join(root_path, "net2.pth")
torch.save(net2, savepath)


loss_tracker3 = run_code_for_training(net3, lrate, mom, epochs)
save_data3_path = os.path.join(root_path, "loss_net3.csv")
pd.DataFrame(loss_tracker3).to_csv(save_data3_path)
savepath = os.path.join(root_path, "net3.pth")
torch.save(net3, savepath)