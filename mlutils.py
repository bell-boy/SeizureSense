import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch_classifcation(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()) -> list[float]:
  '''
  model: The model being trained
  dataloader: The training dataloader
  optim: The optimizer to use for training
  loss_fn: (optional) the specific instance of nn.CrossEntropyLoss to use 
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