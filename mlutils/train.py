import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch_classifcation(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn = torch.nn.CrossEntropyLoss(), device=torch.device('cpu')) -> list[float]:
  '''
  Args:
    model: The model being trained
    dataloader: The training dataloader
    optim: The optimizer to use for training
    loss_fn: (defaults to torch.nn.CrossEntropyLoss()) the classifcation loss to use
    device: The device on which to train on

  Returns:
    losses: a list of losses for every batch in the epoch
  '''
  losses = []
  for (data, label) in tqdm(dataloader):
    data = data.to(device)
    label = label.to(device)
    logits = model(data)
    loss = loss_fn(logits, label)

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses


def train_epoch_regression(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn: torch.nn.CrossEntropyLoss = torch.nn.MSELoss, device=torch.device('cpu')) -> list[float]:
  '''
  Args:
    model: The model being trained
    dataloader: The training dataloader
    optim: The optimizer to use for training
    loss_fn: the regression loss to use

  Returns:
    losses: a list of losses for every batch in the epoch
  '''
  losses = []
  for (data, label) in tqdm(dataloader):
    data = data.to(device)
    label = label.to(device)
    logits = model(data)
    loss = loss_fn(logits, label.flatten().long())

    optim.zero_grad()
    loss.backward()
    optim.step()

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
  for (data, label) in tqdm(dataloader):
    data = data.to(device)
    label = label.to(device)
    logits = model(data)
    pred = logits.argmax(dim=-1)

    TN += ((pred == 0) & (label == 0)).sum(dim=-1).item()
    FN += ((pred == 0) & (label == 1)).sum(dim=-1).item()
    TP += ((pred == 1) & (label == 1)).sum(dim=-1).item()
    FP += ((pred == 1) & (label == 0)).sum(dim=-1).item()

  return TP, TN, FP, FN