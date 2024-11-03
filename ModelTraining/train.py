import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter   
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import sys
from tqdm import tqdm
import datetime
import os
import logging
import time
import multiprocessing as mp
# models
sys.path.append('../Model')
from CNNLSTM import CNNLSTM
from ResTCN import TCNSE


from dataset.dataset import NIDSDataset

def getlogger(log_name, log_dir):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    handler1 = logging.FileHandler(log_dir, mode="a+")
    handler2 = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.5, reduction='mean'):
        """
        Binary Focal Loss
        :param alpha: Balancing factor, can be a float or a list [alpha_neg, alpha_pos]
        :param gamma: Modulating factor
        :param reduction: Reduction method for the loss, 'none' | 'mean' | 'sum'
        """
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            assert len(alpha) == 2, "Alpha must be a list or tuple of length 2."
            self.alpha = torch.tensor([alpha[0], alpha[1]])
        else:
            self.alpha = torch.tensor([1 - alpha, alpha])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass
        :param inputs: Model outputs logits, shape (batch_size, 1)
        :param targets: True labels, shape (batch_size, 1)
        :return: Calculated Focal Loss
        """
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        targets = targets.float()
        
        # Calculate sigmoid probabilities
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=1e-6, max=1 - 1e-6)  # Prevent numerical issues
        
        # Calculate pt
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Calculate alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = torch.where(targets == 1, self.alpha[1], self.alpha[0])
        else:
            at = 1.0
        
        # Calculate Focal Loss
        loss = - at * (1 - pt) ** self.gamma * torch.log(pt)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss Implementation.

    Parameters:
        alpha (float or list): Class weights, can be a single float (applies to all classes) or a list of weights for each class.
        gamma (float): Modulating factor gamma >= 0.
        reduction (str): Reduction method for the loss: 'none' | 'mean' | 'sum'.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            raise TypeError('alpha must be float, list, or None')

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.

        Parameters:
            inputs: Model outputs logits, shape (batch_size, num_classes).
            targets: True labels, shape (batch_size,).
        """
        # Check input shape
        if inputs.dim() > 2:
            # If inputs are multi-dimensional, flatten to 2D tensor
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # (N, C, H*W)
            inputs = inputs.permute(0, 2, 1)  # (N, H*W, C)
            inputs = inputs.reshape(-1, inputs.size(-1))  # (N*H*W, C)

        targets = targets.view(-1)  # (N*H*W,)

        # Calculate softmax probabilities
        softmax = F.softmax(inputs, dim=1)

        # Get the predicted probabilities for the true classes
        pt = softmax[range(len(targets)), targets]

        # Calculate log(pt)
        logpt = torch.log(pt + 1e-6)  # Add 1e-6 to prevent numerical issues

        # Calculate alpha_t
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            if self.alpha.dim() == 1:
                at = self.alpha.gather(0, targets)
            else:
                at = self.alpha
            logpt = logpt * at

        # Calculate Focal Loss
        loss = - (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def trainer(model, lr, train_dir, test_dir, log_dir, use_focal_loss=False, binary_classification=True):
    logger = getlogger("train_log", log_dir)

    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Epoch = 40
    Lr = lr
    BatchSize = 2048  # Adjust as needed
    logger.info(f"using model: {model.model_name}")
    logger.info(f"using device: {device}")
    logger.info(f"hyperparameters: Epoch: {Epoch}, Lr: {Lr}, BatchSize: {BatchSize}")

    # **Select loss function**
    if binary_classification:
        criterion = BinaryFocalLoss() if use_focal_loss else nn.BCEWithLogitsLoss()
        logger.info(f"using {'Binary Focal Loss' if use_focal_loss else 'BCEWithLogitsLoss'} as loss function")
    else:
        class_counts = torch.tensor([2643568, 56345, 31479, 249162, 101288, 32431, 177], dtype=torch.float)
        total_samples = class_counts.sum()
        class_freq = class_counts / total_samples  # Sample frequency for each class
        alpha = 1.0 / class_freq
        alpha = alpha / alpha.sum()  # Normalize so that weights sum to 1
        criterion = FocalLoss(alpha=alpha.tolist(), gamma=2.0, reduction='mean')
        logger.info(f"using CrossEntropyLoss as loss function")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Load data
    train_data = NIDSDataset(train_dir, binary_classification=binary_classification)
    logger.info(f"train data loaded successfully, total length: {len(train_data)}")
    train_loader = DataLoader(train_data, batch_size=BatchSize, shuffle=True, num_workers=10)

    test_data = NIDSDataset(test_dir, binary_classification=binary_classification)
    logger.info(f"test data loaded successfully, total length: {len(test_data)}")
    test_loader = DataLoader(test_data, batch_size=BatchSize * 4, shuffle=False, num_workers=10)  # Test set is not shuffled

    # Logging
    log_dir = f'./log/{model.model_name}/{os.path.basename(log_dir)}'
    pth_dir = f'./save/{model.model_name}/{os.path.basename(log_dir)}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pth_dir, exist_ok=True)
    now_time = datetime.datetime.now().strftime('%Y_%m_%d%H_%M_%S')
    writer = SummaryWriter(log_dir=f'{log_dir}/{now_time}')

    model.to(device)
    count = 0
    # Training
    for epoch in range(Epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        logger.info(f"Begin the {epoch + 1} epoch training, learning rate is {scheduler.get_last_lr()}")
        # Log learning rate to tensorboard
        writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

        # Initialize metrics
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        fpr = 0.0
        tnr = 0.0
        for i, data in enumerate(train_loader, 0):
            features, labels = data
            features = features.unsqueeze(1)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features).squeeze()

            # **Process labels and calculate loss**
            if binary_classification:
                labels = labels.float()  # Binary classification labels are float
                loss = criterion(outputs, labels)
            else:
                labels = torch.argmax(labels, dim=1).long()
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        end_time = time.time()
# if (i + 1) % 100 == 0:
        with torch.no_grad():
            # Calculate metrics on the test set
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            test_loss = 0
            for data in test_loader:
                features, labels = data
                features = features.unsqueeze(1)
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features).squeeze()

                # **Process labels and calculate loss**
                if binary_classification:
                    labels = labels.float()
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    # Use sigmoid activation and set threshold
                    preds = torch.sigmoid(outputs) >= 0.5
                    predicted = preds.long()
                else:
                    labels = torch.argmax(labels, dim=1).long()
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    # Use softmax activation and take argmax
                    preds = torch.softmax(outputs, dim=1)
                    predicted = torch.argmax(preds, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            accuracy = 100 * correct / total

            # **Calculate metrics**
            if binary_classification:
                # Calculate binary classification metrics
                tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
                precision = tp / (tp + fp) if (tp + fp) != 0 else 0
                recall = tp / (tp + fn) if (tp + fn) != 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
                tnr = tn / (fp + tn) if (fp + tn) != 0 else 0

                # Write to tensorboard
                writer.add_scalar('test precision', precision, count)
                writer.add_scalar('test recall', recall, count)
                writer.add_scalar('test f1 score', f1, count)
                writer.add_scalar('test FPR', fpr, count)
                writer.add_scalar('test TNR', tnr, count)
                logger.info(f"Accuracy: {accuracy:.3f}%, Precision: {precision:.2f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, FPR: {fpr:.3f}, TNR: {tnr:.3f}")
            else:
                # Calculate multi-class classification metrics
                precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
                logger.info(f"Accuracy: {accuracy:.3f}%, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")

                # If needed, calculate metrics for each class
                class_report = classification_report(all_labels, all_predictions)
                logger.info(f"Classification Report:\n{class_report}")

            # Test set loss
            avg_loss = test_loss / len(test_loader)

            # Write to tensorboard
            writer.add_scalar('test accuracy', accuracy, count)
            writer.add_scalar('test loss', avg_loss, count)
            count += 1

            writer.add_scalar('training loss', loss.item(), count)

            # Save model
            logger.info(f"Epoch {epoch + 1} finished, time cost is {end_time - start_time}s")
            torch.save(model.state_dict(), f'{pth_dir}/{epoch + 1}_{model.model_name}_{accuracy}.pth')
            scheduler.step()

    logger.info("training finished")
    writer.close()


def run_trainer(model_class, lr, train_dir, test_dir, log_dir, model_params=None, use_focal_loss=False, binary_classification=True):
    if model_params:
        model = model_class(**model_params)
    else:
        model = model_class()
    trainer(model, lr, train_dir, test_dir, log_dir, use_focal_loss=use_focal_loss, binary_classification=binary_classification)

def run_models_in_batches(tasks, batch_size=3):
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        processes = []
        for model_class, lr, train_dir, test_dir, log_dir, model_params, use_focal_loss, binary_classification in batch:
            p = mp.Process(target=run_trainer, args=(model_class, lr, train_dir, test_dir, log_dir, model_params, use_focal_loss, binary_classification))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()



def test_models(models, batch_size=10, features=41):
    for model_class, lr, model_params in models:
        try:
            # Initialize model
            model = model_class(**model_params)
            print(f"Initialized {model_class.__name__} successfully.")

            # Create input tensor
            x = torch.randn(batch_size, 1, features)  # (batch_size, 1, features)

            # Forward pass
            output = model(x)
            print(f"{model_class.__name__} forward pass successful. Output shape: {output.shape}")

        except Exception as e:
            print(f"Error with {model_class.__name__}: {e}")
         

if __name__ == "__main__":
    feature_sets = [20, 30, 40, 50, 60, 68]
    models = [
        # (CNNLSTM, 0.01, {}),
        (TCNSE, 0.001, {}, False, True),
    ]
    # test_models(models, features=41)
    tasks = []
    for features in feature_sets:
        for model_class, lr, model_params, use_focal_loss, binary_classification in models:
            if binary_classification:
                train_file = f'/root/autodl-tmp/train_{features}_features.csv'
                test_file = f'/root/autodl-tmp/test_{features}_features.csv'
                log_dir = f'./log/{model_class.__name__}_{features}_focal_loss'
            else:
                train_file = f'/root/autodl-tmp/data/train_{features}_features.csv'
                test_file = f'/root/autodl-tmp/data/test_{features}_features.csv'
                log_dir = f'./log/{model_class.__name__}_{features}_multiclass'

            tasks.append((model_class, lr, train_file, test_file, log_dir, model_params, use_focal_loss, binary_classification))

    run_models_in_batches(tasks, batch_size=2)
    print("All done")