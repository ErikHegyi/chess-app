import torch
from typing import Callable


def train(
        model: torch.nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    # Unpack
    x, y = batch

    # Do the forward pass
    prediction: torch.Tensor = model(x)

    # Calculate the loss
    loss: torch.Tensor = criterion(prediction, y)

    # Zero the optimizer gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Gradient descent
    optimizer.step()

    return loss