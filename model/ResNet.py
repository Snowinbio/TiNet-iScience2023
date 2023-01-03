import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

# For classification task for texture and edge channels

def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    test_acc_history = []

    train_auc_history = []
    val_auc_history = []
    test_auc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            running_prob = []
            running_label = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                running_prob.extend(outputs[:, 1].cpu().detach().numpy())
                running_label.extend(labels.cpu().detach().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_auc = roc_auc_score(running_label, running_prob)

            print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_auc))

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_auc_history.append(epoch_auc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_auc = epoch_auc
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_auc_history.append(epoch_auc)

            if phase == 'test':
                test_acc_history.append(epoch_acc)
                test_auc_history.append(epoch_auc)

        scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} AUC: {:4f}'.format(best_acc, best_auc))

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, test_acc_history, train_auc_history, val_auc_history, test_auc_history


if __name__ == '__main__':
    resnet = models.resnet18(pretrained=True)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features, 2)

    data_transforms = {
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'train': transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
            ),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    }

    print("Initializing Datasets and Dataloaders...")

    data_dir = "image"
    train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    val_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms['val'])
    test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['val'])

    batch_size = 128
    num_workers = 4
    dataloaders_dict = {
        'train': DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'test': DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet = resnet.to(device)

    params_to_update = resnet.parameters()

    optimizer = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.99))

    criterion = nn.CrossEntropyLoss()

    num_epochs = 100

    model_ft, train_acc, val_acc, test_acc , train_auc, val_auc, test_auc = train_model(resnet, dataloaders_dict, criterion, optimizer, num_epochs)

    torch.save(model_ft, 'resnet_ft.pth')

    tracc = [h.cpu().numpy() for h in train_acc]
    vacc = [h.cpu().numpy() for h in val_acc]
    tsacc = [h.cpu().numpy() for h in test_acc]

    trauc = [h for h in train_auc]
    vauc = [h for h in val_auc]
    tsauc = [h for h in test_auc]

    plt.title("Accuracy & AUC vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy or AUC")
    plt.plot(range(1, num_epochs + 1), tracc, label="Train Acc")
    plt.plot(range(1, num_epochs + 1), vacc, label="Val Acc")
    plt.plot(range(1, num_epochs + 1), tsacc, label="Test Acc")
    plt.plot(range(1, num_epochs + 1), trauc, label="Train AUC")
    plt.plot(range(1, num_epochs + 1), vauc, label="Val AUC")
    plt.plot(range(1, num_epochs + 1), tsauc, label="Test AUC")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 10.0))
    plt.legend()
    plt.savefig("acc_auc.png")
