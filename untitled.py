import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch():
  losses = []
  for (data, label) in iter(paitent_0_dataloader):
    data.to(device)
    label.to(device)

    logits = dumb_model(data.float())
    loss = loss_fn(logits, label.flatten().long())

    optim.zero_grad()
    loss.backward()
    optim.step()

    losses.append(loss.item())
  return losses