import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from dataset import *
from HSFNetworks import *
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for data, labels in tqdm(train_loader):
        inputs_spatial = data[0].to(device).float()
        inputs_real = data[1].real.to(device).float()
        inputs_imag = data[1].imag.to(device).float()
        # print(inputs.shape)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs_spatial, inputs_real, inputs_imag)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(train_loader)


def test(model, test_loader, criterion, optimizer, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            inputs_spatial = data[0].to(device).float()
            inputs_real = data[1].real.to(device).float()
            inputs_imag = data[1].imag.to(device).float()
            # print(inputs.shape)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_spatial, inputs_real, inputs_imag)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    # auc = roc_auc_score(np.eye(20)[all_labels], np.eye(20)[all_preds], average='macro', multi_class='ovo') # Change 20 if different number of classes

    return test_loss, accuracy, f1, precision, recall


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for the script")
    
    parser.add_argument('--ds_name', type=str, default="NFBS_T1w", help='Dataset name')
    parser.add_argument('--dist_type', type=str, default="all", help='Distortion type')
    parser.add_argument('--model_name', type=str, default='HSFNet', help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=21, help='Number of classes')
    parser.add_argument('--sample_length', type=int, default=30, help='Sample length')
    parser.add_argument('--last_dim', type=int, default=128, help='Last dimension')
    parser.add_argument('--cuda_id', type=int, default=0, help='Cuda ID number')
    
    return parser.parse_args()


def main():
    seed_everything()

    args = parse_args()
    
    args.root_dir = f'datasets/dist/{args.ds_name}/'
    os.makedirs(f'models/{args.model_name}/', exist_ok=True)
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    
    # Print configuration
    print('*'*20)
    print("Configuration:")
    print(f"Dataset Name: {args.ds_name}")
    print(f"Distortion type: {args.dist_type}")
    print(f"Root Directory: {args.root_dir}")
    print(f"Model Name: {args.model_name}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Starting epoch: {args.start_epoch}")
    print(f"Learning Rate: {args.lr}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Sample Length: {args.sample_length}")
    print(f"Last Dimension: {args.last_dim}")
    print(f"Device: {device}")
    print('*'*20 + '\n\n')

    # Get the dataloader to train the model
    data_loaders = get_data_loaders(args.root_dir, args.ds_name, args.dist_type, args.batch_size, args.num_workers, sample_length=args.sample_length, k_fold_splits=5)

    for fold_idx, fold_data in enumerate(data_loaders):
        print(f"Fold {fold_idx + 1}:")
        train_loader = fold_data['train']
        test_loader = fold_data['test']

        if args.model_name == 'HSFNet':
            model = HSFNet(sample_length=args.sample_length, num_classes=args.num_classes, last_dim=args.last_dim).to(device)
        elif args.model_name == 'HSFNet_spa':
            model = HSFNet_spa(sample_length=args.sample_length, num_classes=args.num_classes, last_dim=args.last_dim).to(device)
        elif args.model_name == 'HSFNet_freq':
            model = HSFNet_freq(sample_length=args.sample_length, num_classes=args.num_classes, last_dim=args.last_dim).to(device)
        elif args.model_name == 'HSFNet_wo_real':
            model = HSFNet_wo_real(sample_length=args.sample_length, num_classes=args.num_classes, last_dim=args.last_dim).to(device)
        elif args.model_name == 'HSFNet_wo_imag':
            model = HSFNet_wo_imag(sample_length=args.sample_length, num_classes=args.num_classes, last_dim=args.last_dim).to(device)
        else:
            raise ValueError(f'Illegal model_name ({args.model_name}).')

        if args.start_epoch > 0:
            model_path = f'models/{args.model_name}/model_dataset_{args.ds_name}_fold_{fold_idx + 1}_epoch_{args.start_epoch}.pth'
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        for epoch in range(args.start_epoch, args.start_epoch+args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.start_epoch+args.num_epochs}")

            train_loss = train(model, train_loader, criterion, optimizer, device)
            print(f"Training loss: {train_loss:.4f}")

            test_loss, accuracy, f1, precision, recall = test(model, test_loader, criterion, optimizer, device)
            print(f"Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
            save_path = f'models/{args.model_name}/'
            model_filename = f'model_dataset_{args.ds_name}_fold_{fold_idx + 1}_epoch_{epoch + 1}.pth'
            model_path = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Model {args.model_name} for fold {fold_idx + 1} with epoch {epoch+1} saved to {model_path}\n")

        print("\n")
        break


if __name__ == "__main__":
    main()


# python main.py --ds_name NFBS_T1w --dist_type all --model_name HSFNet --batch_size 32 --num_workers 8 --num_epochs 20 --start_epoch 0 --lr 0.001 --num_classes 21 --sample_length 30 --last_dim 128 --cuda_id 0
