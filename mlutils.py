import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch_classifcation(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.CrossEntropyLoss, optim: torch.optim.Optimizer) -> list[float]:
  losses = []
  for (data, label) in iter(dataloader):
    logits = model(data)
    loss = loss_fn(logits, label.flatten().long())

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses