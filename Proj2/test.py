import nn
import math

def generate_dataset(n):
    input = torch.empty(n, 2).uniform_(0, 1)
    target = (((input - 0.5).pow(2).sum(1)) < (1 / (2*math.pi)**0.5)).long()
    return input, target

if __name__ == '__main__':
    train_input, train_target = generate_dataset(1000)
    test_input, test_target = generate_dataset(1000)
