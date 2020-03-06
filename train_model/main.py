import argparse
import numpy as np
import pandas as pd
import itertools
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()  
from tqdm import tqdm
import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils import CustomDataset, compute_img_mean_std
from train import train, validate
from getdata import get_data
from visualize import plot_confusion_matrix, view_samples

device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--samples', type=bool, default=False, help='See sample images')
parser.add_argument('--view_data_counts', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--train', default=True, type=bool, help='train the model')

opt = parser.parse_args()


if opt.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.join('..', 'input','skin-cancer-mnist-ham10000' )
all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}


df_train, df_val = get_data(base_dir, imageid_path_dict)

#normMean,normStd = compute_img_mean_std(all_image_path)
normMean = [0.7630358, 0.54564357, 0.5700475]
normStd = [0.14092763, 0.15261263, 0.16997081]

model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Linear(1280, 7)

#model.to(device)

input_size = 224
# define the transformation of the train images
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(normMean, normStd)])
# define the transformation of the val images
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normMean, normStd)])

# Define training set using train_df and our defined transitions (train_transform)
training_set = CustomDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)
# Same for validation set:
validation_set = CustomDataset(df_val, transform=train_transform)
val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)


model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

if opt.train:
    epoch_num = opt.num_epochs
    best_val_acc = 0
    total_loss_val, total_acc_val = [], []
    for epoch in tqdm(range(1, epoch_num + 1)):
        loss_train, acc_train, total_loss_train, total_acc_train = train(train_loader, model, criterion, optimizer, epoch, device)
        loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch, device)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')

    fig = plt.figure(num = 2)
    fig1 = fig.add_subplot(2,1,1)
    fig2 = fig.add_subplot(2,1,2)
    fig1.plot(total_loss_train, label = 'training loss')
    fig1.plot(total_acc_train, label = 'training accuracy')
    fig2.plot(total_loss_val, label = 'validation loss')
    fig2.plot(total_acc_val, label = 'validation accuracy')
    plt.legend()
    plt.show()

    print ('Evaluating the model')
    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    plot_confusion_matrix(confusion_mtx, plot_labels)
    report = classification_report(y_label, y_predict, target_names=plot_labels)
    print(report)

    label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    plt.bar(np.arange(7),label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')


torch.save(model, 'models/model.pth')