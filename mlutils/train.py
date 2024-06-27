import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import numpy as np

def train_epoch_classifcation(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn = torch.nn.CrossEntropyLoss(), device=torch.device('cpu')) -> list[float]:
  '''
  Args:
    model 
      The model being trained
    dataloader
      The training dataloader
    optim
      The optimizer to use for training
    loss_fn
      (defaults to torch.nn.CrossEntropyLoss()) the classifcation loss to use
    device
      The device on which to train on

  Returns:
    losses: a list of losses for every batch in the epoch
  '''
  losses = []
  for (data, label) in dataloader:
    data = data.to(device)
    label = label.to(device)
    logits = model(data)
    loss = loss_fn(logits, label)

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses


def train_epoch_regression(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn = torch.nn.MSELoss(), device=torch.device('cpu'), norm: tuple = (0, 1)) -> list[float]:
  '''
  Args:
    model 
      The model being trained
    dataloader
      The training dataloader
    optim
      The optimizer to use for training
    loss_fn
      (defaults to torch.nn.MSELoss()) the regresison loss to use
    device
      The device on which to train on
    norm : tuple
      A shifting and scaling constant, in that order

  Returns:
    losses: a list of losses for every batch in the epoch
  '''
  losses = []
  for (data, label) in dataloader:
    data = data.to(device).float()
    label = (label.to(device).float() - norm[0]) / norm[1]
    pred = model(data)
    loss = loss_fn(pred, label)

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses

@torch.inference_mode()
def validate_epoch_regression(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn = torch.nn.MSELoss(), device=torch.device('cpu'), norm: tuple = (0, 1)) -> list[float]:
  '''
  Args:
    model 
      The model being trained
    dataloader
      The training dataloader
    optim
      The optimizer to use for training
    device
      The device on which to train on
    norm : tuple
      A shifting and scaling constant, in that order

  Returns:
    losses: a list of losses for every batch in the epoch
  '''
  losses = []
  for (data, label) in dataloader:
    data = data.to(device).float()
    label = (label.to(device).float() - norm[0]) / norm[1]
    pred = model(data)
    loss = loss_fn(pred, label)

    losses.append(loss.item())
  return losses


@torch.inference_mode()
def validate_binary_softmax(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device=torch.device('cpu')):
  """
  Returns summary statistics for a softmax based binary classifcation model

  Args:
    model : torch.nn.Module
      the module being evaluated
    dataloader : torch.utils.data.DataLoader
      the data to evalutate on
    device
      the device to run the model on

  Returns:
    statistics : (TP, TN, FP, FN)
      returns the number of True Positives, True Negatives, False Positives, False Negatives accordingly
  """
  TP, TN, FP, FN = 0, 0, 0, 0
  for (data, label) in tqdm(dataloader, leave=False):
    data = data.to(device)
    label = label.to(device)
    logits = model(data)
    pred = logits.argmax(dim=-1)

    TN += ((pred == 0) & (label == 0)).sum(dim=-1).item()
    FN += ((pred == 0) & (label == 1)).sum(dim=-1).item()
    TP += ((pred == 1) & (label == 1)).sum(dim=-1).item()
    FP += ((pred == 1) & (label == 0)).sum(dim=-1).item()
  return TP, TN, FP, FN


@torch.inference_mode()
def validate_binary_sigmoid(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device=torch.device('cpu')):
  """
  Returns summary statistics for a sigmoid based binary classifcation model

  Args:
    model : torch.nn.Module
      the module being evaluated
    dataloader : torch.utils.data.DataLoader
      the data to evalutate on
    device
      the device to run the model on

  Returns:
    statistics : (TP, TN, FP, FN)
      returns the number of True Positives, True Negatives, False Positives, False Negatives accordingly
  """
  TP, TN, FP, FN = 0, 0, 0, 0
  for (data, label) in tqdm(dataloader, leave=False):
    data = data.to(device)
    label = label.to(device)
    prob = model(data)[..., 0]
    pred = torch.zeros_like(prob)
    pred[prob > 0.5] = 1

    TN += ((pred == 0) & (label == 0)).sum(dim=-1).item()
    FN += ((pred == 0) & (label == 1)).sum(dim=-1).item()
    TP += ((pred == 1) & (label == 1)).sum(dim=-1).item()
    FP += ((pred == 1) & (label == 0)).sum(dim=-1).item()
  return TP, TN, FP, FN


@torch.inference_mode()
def validate_binary_auc(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device=torch.device('cpu')):
    """
    Calculates AUC and displays ROC curve for a binary classification model

    Args:
        model : torch.nn.Module
            the module being evaluated
        dataloader : torch.utils.data.DataLoader
            the data to evaluate on
        device : torch.device
            the device to run the model on

    Returns:
        auc_score : float
            the Area Under the Curve score
    """
    all_probs = []
    all_labels = []
    
    for (data, label) in tqdm(dataloader, leave=False):
        data = data.to(device)
        label = label.to(device)
        probs = torch.softmax(model(data), dim=-1)

        # We take the probability of the positive class (index 1)
        pos_probs = probs[:, 1]
        
        all_probs.extend(pos_probs.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return auc_score