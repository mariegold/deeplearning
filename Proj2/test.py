from nn import *
import math
import torch
import copy
import matplotlib.pyplot as plt
torch.set_grad_enabled(False)

def generate_dataset(n):
    input = torch.empty(n, 2).uniform_(0, 1)
    target = (input - 0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    return input, target

def train_model(model, train_input, train_target, mini_batch_size = 50, nb_epochs = 25, lr = 0.001, sgd = True):
    criterion = LossMSE()
    if sgd: 
        optimizer = SGD(model.param(), lr)
    else:
        optimizer = Adam(model.param(), lr)
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


def compute_nb_errors(model, test_input, test_target, mini_batch_size = 50, plot = False, name = ""):
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
        plot_result(test_input, test_target, predictions, name)
    return nb_errors

def plot_result(test_input, predictions, name):
    gt = plt.Circle((0.5, 0.5), (1/(2*math.pi))**0.5, color='grey', alpha = 0.5)
    plt.gca().add_patch(gt)
    plt.scatter(test_input[:,0], test_input[:,1], c = predictions)
    plt.savefig("dataset_{}.png".format(name))

def performance_estimation(datasets, model):
    n = 1000
    # Parameter grid
    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_sizes = [1, 5, 10, 20, 50, 100]

    param_combinatinos = [(lr, batch_size) 
        for lr in lrs
        for batch_size in batch_sizes]
    
    # For saving mean and std across datasets for each optimizer and hyperparameter combination    
    model_sgd_mean = {} 
    model_sgd_std = {}
    model_adam_mean = {} 
    model_adam_std = {}
    for param_combo in param_combinatinos:
        model_sgd_mean[param_combo] = []
        model_sgd_std[param_combo] = []
        model_adam_mean[param_combo] = []
        model_adam_std[param_combo] = []
        lr, batch_size = param_combo
        # Train with each dataset with the given param combination using SGD and Adam, save accuracy for each dataset
        for train_input, train_target, test_input, test_target in datasets:
            # Deepcopy is necessary so that we train with the model from scratch
            model_sgd = copy.deepcopy(model)
            train_model(model_sgd, train_input, train_target, mini_batch_size = batch_size, nb_epochs=25, lr=lr, sgd = True)
            nb_errors_sgd = compute_nb_errors(model_sgd, test_input, test_target, mini_batch_size = 25)
            model_sgd_mean[param_combo].append(1-nb_errors_sgd/n)

            model_adam = copy.deepcopy(model)    
            train_model(model_adam, train_input, train_target, mini_batch_size = batch_size, nb_epochs=25, lr=lr,  sgd = False)
            nb_errors_adam = compute_nb_errors(model_adam, test_input, test_target, mini_batch_size = 25)
            model_adam[param_combo].append(1-nb_errors_adam/n)
        
        # Compute mean and standard deviation across the datasets for each optimizer and param combo
        model_sgd_scores = torch.FloatTensor(model_sgd_mean[param_combo])
        model_sgd_mean[param_combo] = model_sgd_scores.mean().item()
        model_sgd_std[param_combo] = model_sgd_scores.std().item()

        model_adam_scores = torch.FloatTensor(model_adam_mean[param_combo])
        model_adam_mean[param_combo] = model_adam_scores.mean().item()
        model_adam_std[param_combo] = model_adam_scores.std().item()

    # Return means and standard deviations for each param combo and optimiser
    return model_sgd_mean, model_sgd_std, model_adam_mean, model_adam_std
    

if __name__ == '__main__':
    n = 1000

    train_input, train_target = generate_dataset(n)
    test_input, test_target = generate_dataset(n)
    
    # Testing different activation functions
    model_relu = Sequential(Linear(2,25), ReLu(), Linear(25,25), ReLu(), Linear(25,25), ReLu(), Linear(25,25), ReLu(), Linear(25, 2))
    model_tanh = Sequential(Linear(2,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25,25), Tanh(), Linear(25, 2))
    model_sigmoid = Sequential(Linear(2,25), Sigmoid(), Linear(25,25), Sigmoid(), Linear(25,25), Sigmoid(), Linear(25,25), Sigmoid(), Linear(25, 2))

    models = [model_relu, model_tanh, model_sigmoid]
    activation_names = ["ReLU", "Tanh", "Sigmoid"]
    
    # Datasets for batch size and learning rate tuning (can change number of datasets to arbitrary number)
    datasets = []
    for i in range(4):
        train_input_tune, train_target_tune = generate_dataset(n)
        test_input_tune, test_target_tune = generate_dataset(n)
        datasets.append(train_input_tune, train_target_tune, test_input_tune, test_target_tune)

    # Find best hyperparamters (batch size and learning rate) for each activation function and optimiser combination
    for (model, activation) in zip(models, activation_names):
        model_sgd_mean, model_sgd_std, model_adam_mean, model_adam_std = performance_estimation(datasets, model)
        best_sgd_params = max(model_sgd_mean.items(), key = lambda k : k[1])
        print('Best (lr, mini_batch_size) combination for model with {} activation and SGD optimiser: {}'.format(activation, best_sgd_params)) 
        print('With accuracy: {:.3f} +/- {:.3f}.'.format(model_sgd_mean[best_sgd_params], model_sgd_std[best_sgd_params])) 
        best_adam_params = max(model_adam_mean.items(), key = lambda k : k[1])
        print('Best (lr, mini_batch_size) combination for model with {} activation and Adam optimiser: {}'.format(activation, best_adam_params)) 
        print('With accuracy: {:.3f} +/- {:.3f}.'.format(model_adam_mean[best_adam_params], model_adam_std[best_adam_params])) 
