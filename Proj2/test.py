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
        cum_loss = 0.0

        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            cum_loss += loss # Accumulate loss per epoch
            model.zero_grad()
            loss_gradient = criterion.backward() # Compute gradient of loss with respect to model's outputs
            model.backward(loss_gradient) # Use the loss gradient to accumulate gradient wrt parameters
            optimizer.step() # Perform the update step on parameters
        print("Epoch {}: training loss = {}" .format(e, cum_loss.item()))

if __name__ == '__main__':
    train_input, train_target = generate_dataset(1000)
    test_input, test_target = generate_dataset(1000)

    model = Sequential(Linear(2,25), ReLu(), Linear(25,25), ReLu(),  Linear(25,25), ReLu(), Linear(25,25), ReLu(), Linear(25, 2))

    train_model(model, train_input, train_target, nb_epochs = 10, mini_batch_size = 50)
