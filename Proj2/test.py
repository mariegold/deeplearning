from nn import *
import math
import torch
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)

def generate_dataset(n):
    input = torch.empty(n, 2).uniform_(0, 1)
    target = (input - 0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    return input, target

def train_model(model, train_input, train_target, mini_batch_size = 50, nb_epochs = 25, lr = 0.001):
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


def compute_nb_errors(model, test_input, test_target, mini_batch_size = 50, plot = False):
    nb_errors = 0
    predictions = []
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model.forward(test_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        predictions.extend(predicted_classes.tolist())
        for k in range(mini_batch_size):
            if test_target[b + k] != predicted_classes[k]:
                nb_errors += 1
    if plot:
        plot_result(test_input, test_target, predictions)
    return nb_errors

def plot_result(test_input, test_target, predictions):
    gt = plt.Circle((0.5, 0.5), (1/(2*math.pi))**0.5, color='grey', alpha = 0.5)
    plt.gca().add_patch(gt)
    plt.scatter(test_input[:,0], test_input[:,1], c = predictions)
    plt.savefig('dataset.png')

if __name__ == '__main__':
    n = 1000

    train_input, train_target = generate_dataset(n)
    test_input, test_target = generate_dataset(n)

    model = Sequential(Linear(2,25), ReLu(), Linear(25,25), ReLu(), Linear(25,25), ReLu(), Linear(25,25), ReLu(), Linear(25, 2))

    train_model(model, train_input, train_target, nb_epochs = 25, mini_batch_size = 1)

    nb_errors_test = compute_nb_errors(model, test_input, test_target, 50, True)
    nb_errors_train = compute_nb_errors(model, train_input, train_target)

    print("Test accuracy: {} " .format(1 - nb_errors_test/n))
    print("Train accuracy: {} " .format(1 - nb_errors_train/n))
