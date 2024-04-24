from torch import nn
import torch

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
model = ToyModel(5, 3).cuda()

with torch.cuda.amp.autocast():
    # Input tensor
    x = torch.randn(2, 5).cuda()

    # Model parameters
    params = list(model.parameters())
    print("Model Parameters:", params)

    # Output from the first feed-forward layer
    fc1_out = model.fc1(x)
    print("Output of fc1:", fc1_out)

    # Output from LayerNorm
    ln_out = model.ln(fc1_out)
    print("Output of Layer Norm:", ln_out)

    # Model's predicted logits
    y = model(x)
    print("Model's Predicted Logits:", y)

    # Define a target and loss function
    target = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float).cuda()
    loss_fn = nn.MSELoss()

    # Calculate the loss
    loss = loss_fn(y, target)
    print("Loss:", loss)

    # Perform backpropagation to get gradients
    loss.backward()
    
    # Retrieve gradients
    gradients = [param.grad for param in model.parameters()]
    print("Model's Gradients:", gradients)