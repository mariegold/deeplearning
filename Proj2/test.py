import nn
import math
import torch
torch.set_grad_enabled(False)

def generate_dataset(n):
    input = torch.empty(n, 2).uniform_(0, 1)
    target = (input - 0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    return input, target

if __name__ == '__main__':
    train_input, train_target = generate_dataset(1000)
    test_input, test_target = generate_dataset(1000)

    """ Testing, delete later"""
    x = train_input[0:3]
    y = train_target[0]
    #print(x, y)
    aff = nn.Linear(input_dim = 2, output_dim=3)
    #print(aff.w.size(), aff.b.size())
    #print(x.t().size())
    aff.forward(x)
    aff.backward(torch.tensor([[0.5, 1, 0.2],[-0.5, 0, 0.2], [-0.5, 0, 0.2]]))
    #print(aff.backward(torch.tensor([[0.5, 1, 0.2],[-0.5, 0, 0.2]])))
    criterion = nn.LossMSE()
    loss = criterion.forward(torch.tensor([0.5]), torch.tensor([1.]))
    #print(loss)
    #print(criterion.backward())
