from nn import *
import math
import torch
torch.set_grad_enabled(False)

def generate_dataset(n):
    input = torch.empty(n, 2).uniform_(0, 1)
    target = (input - 0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    return input, target

def train_model(model, train_input, train_target, mini_batch_size = 50, nb_epochs = 25, lr = 0.1):
    criterion = LossMSE()
    optimizer = SGD(model.param(), lr)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size): 
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss_gradient = criterion.backward() # Compute gradient of loss with respect to model's outputs
            model.backward(loss_gradient) # Use the loss gradient to accumulate gradient wrt parameters
            optimizer.step() # Perform the update step on parameters 

if __name__ == '__main__':
    train_input, train_target = generate_dataset(1000)
    test_input, test_target = generate_dataset(1000)

    model = Sequential(Linear(2,25), ReLu(), Linear(25,25), ReLu(),  Linear(25,25), ReLu(), Linear(25,25), ReLu(), Linear(25, 2))

    train_model(model, train_input, train_target)

    """ Testing, delete later"""
    # x = train_input[0:3]
    # y = train_target[0]
    #print(x, y)
    # aff = Linear(2,25)
    #print(aff.w.size(), aff.b.size())
    #print(x.t().size())
    #aff.forward(x)
    #aff.backward(torch.tensor([[0.5, 1, 0.2],[-0.5, 0, 0.2], [-0.5, 0, 0.2]]))
    #print(aff.backward(torch.tensor([[0.5, 1, 0.2],[-0.5, 0, 0.2]])))
    # criterion = LossMSE()
    # loss = criterion.forward(torch.tensor([0.5]), torch.tensor([1.]))
    #print(loss)
    #print(criterion.backward())
