import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from dataset import *
from HSFNetworks import *
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')


class LinearModel(nn.Module):
    def __init__(self, num_classes, last_dim):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(last_dim + 2 * (last_dim // 2), last_dim)
        self.linear2 = nn.Linear(last_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def train(pretrained_model, classifier, train_loader, criterion, optimizer, device):
    pretrained_model.eval()
    classifier.train()
    running_loss = 0.0
    for data, labels in tqdm(train_loader):
        inputs_spatial = data[0].to(device).float()
        inputs_real = data[1].real.to(device).float()
        inputs_imag = data[1].imag.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        x1, x2, x3 = pretrained_model.get_sfem_outputs(inputs_spatial, inputs_real, inputs_imag)
        x = torch.cat((x1, x2, x3), dim=1)
        
        outputs = classifier(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def test(pretrained_model, classifier, test_loader, criterion, optimizer, device):
    pretrained_model.eval()
    classifier.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, labels in test_loader:
            inputs_spatial = data[0].to(device).float()
            inputs_real = data[1].real.to(device).float()
            inputs_imag = data[1].imag.to(device).float()
            labels = labels.to(device)
            
            optimizer.zero_grad()
            x1, x2, x3 = pretrained_model.get_mamba_outputs(inputs_spatial, inputs_real, inputs_imag)
            x = torch.cat((x1, x2, x3), dim=1)
            outputs = classifier(x)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    return test_loss, accuracy, f1, precision, recall, auc, cm, all_labels, all_probs



parser = argparse.ArgumentParser(description="Configuration for the script")
parser.add_argument('--train_ds_name', type=str, default="NFBS_T1w", help='Dataset name')
parser.add_argument('--test_ds_name', type=str, default="NFBS_T1w", help='Dataset name')
args = parser.parse_args()

dist_type = "all"
batch_size = 32
num_workers = 8
num_epochs = 5
start_epoch = 0
lr = 0.001
num_classes = 21
sample_length = 30
last_dim = 128
cuda_id = 0

best_epoch = 10
model_name = "HSFNet"
root_dir = f'datasets/dist/{args.test_ds_name}/'
device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")


seed_everything()

data_loaders = get_data_loaders(root_dir, args.test_ds_name, dist_type, batch_size, num_workers, sample_length=sample_length, k_fold_splits=5)

pretrained_model = HSFNet(sample_length=sample_length, num_classes=num_classes, last_dim=last_dim).to(device)
model_path = f'models/{model_name}/model_dataset_{args.train_ds_name}_fold_1_epoch_{best_epoch}.pth'
pretrained_model.load_state_dict(torch.load(model_path, map_location='cpu'))
pretrained_model.to(device)

classifier = LinearModel(num_classes=num_classes, last_dim=last_dim).to(device)

for fold_idx, fold_data in enumerate(data_loaders):
    print(f"Fold {fold_idx + 1}:")
    train_loader = fold_data['train']
    test_loader = fold_data['test']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    for eeppch in range(start_epoch, start_epoch+num_epochs):
        print(f"Epoch {eeppch + 1}/{start_epoch+num_epochs}")

        train_loss = train(pretrained_model, classifier, train_loader, criterion, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")


        test_loss, accuracy, f1, precision, recall, auc, cm, all_labels, all_probs = test(pretrained_model, classifier, test_loader, criterion, optimizer, device)
        print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")
        print(f"confusion matrix: {cm.shape} \n{cm}")

    break

# nohup python -u cross_dataset.py --train_ds_name OASIS_T1w --test_ds_name NFBS_T1w > cross_train_OASIS_T1w_test_NFBS_T1w.log 2>&1 &
# nohup python -u cross_dataset.py --train_ds_name OASIS_T1w --test_ds_name IXI_T1w > cross_train_OASIS_T1w_test_IXI_T1w.log 2>&1 &
# nohup python -u cross_dataset.py --train_ds_name OASIS_T1w --test_ds_name ARC_T1w > cross_train_OASIS_T1w_test_ARC_T1w.log 2>&1 &
# nohup python -u cross_dataset.py --train_ds_name OASIS_T1w --test_ds_name QTAB_T1w > cross_train_OASIS_T1w_test_QTAB_T1w.log 2>&1 &
