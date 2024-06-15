import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch_classifcation(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn = torch.nn.CrossEntropyLoss()) -> list[float]:
  '''
  Args:
    model: The model being trained
    dataloader: The training dataloader
    optim: The optimizer to use for training
    loss_fn: (defaults to torch.nn.CrossEntropyLoss()) the classifcation loss to use

  Returns:
    losses: a list of losses for every batch in the epoch
  '''
  losses = []
  for (data, label) in iter(dataloader):
    logits = model(data)
    loss = loss_fn(logits, label.flatten().long())

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses


def train_epoch_regression(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn: torch.nn.CrossEntropyLoss = torch.nn.MSELoss) -> list[float]:
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
  for (data, label) in iter(dataloader):
    logits = model(data)
    loss = loss_fn(logits, label.flatten().long())

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses
