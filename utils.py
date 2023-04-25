import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id<0 or cnn_id>2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model

class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """
    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def project(res):
    max_part = torch.maximum(res, torch.zeros_like(res))
    min_part = torch.minimum(max_part, torch.ones_like(max_part))
    return min_part

def project_eps_ball(res, x, eps):
    max_part = torch.maximum(res, x - eps)
    min_part = torch.minimum(max_part, x + eps)
    return min_part


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    total = len(data_loader)
    correct = 0
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = targets.to(device)

        pred_logits = model(samples)
        pred_labels = pred_logits.max(-1)[1]
        correct += (pred_labels == targets).float().sum()
        total += targets.shape[0]

    return correct / total

def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    all_perturbed_samples = []
    all_targets= []
    for samples, real_targets in data_loader:
        samples = samples.to(device)
        real_targets = real_targets.to(device)

        if targeted:
            targets = (real_targets + torch.randint(1, n_classes, size = real_targets.shape).to(device)) % n_classes
        else:
            targets = real_targets

        perturbed_samples = attack.execute(samples, targets, targeted=targeted)
        all_perturbed_samples.append(perturbed_samples)
        all_targets.append(targets)

    # all_perturbed_samples = torch.stack(all_perturbed_samples)
    # all_targets = torch.stack(all_targets)
    return all_perturbed_samples, all_targets




def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    all_perturbed_samples = []
    all_targets = []
    all_num_queries = []

    for samples, real_targets in data_loader:
        samples = samples.to(device)
        real_targets = real_targets.to(device)

        if targeted:
            targets = (real_targets + torch.randint(1, n_classes, size=real_targets.shape).to(device)) % n_classes
        else:
            targets = real_targets

        perturbed_samples, num_queries = attack.execute(samples, targets, targeted=targeted)
        all_perturbed_samples.append(perturbed_samples)
        all_targets.append(targets)
        all_num_queries.append(num_queries)

    # all_perturbed_samples = torch.stack(all_perturbed_samples)
    # all_targets = torch.stack(all_targets)
    return all_perturbed_samples, all_targets, torch.cat(all_num_queries)

def compute_attack_success(model, stacked_x_adv, stacked_y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    sum_correct = 0
    sum_total = 0

    for x_adv, y in zip(stacked_x_adv, stacked_y):
        x_adv = x_adv.to(device)
        y = y.to(device)

        pred = model(x_adv)
        pred_labels = pred.max(-1)[1]

        if targeted:
            equal_adv_labels = pred_labels == y
            correct = equal_adv_labels.float().sum()
        else:
            unequal_orig_labels = pred_labels != y
            correct = unequal_orig_labels.float().sum()
        sum_total += x_adv.shape[0]
        sum_correct += correct

    return sum_correct / sum_total


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass # FILL ME

def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass # FILL ME

def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass # FILL ME
