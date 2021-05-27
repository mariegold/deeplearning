from dlc_practical_prologue import generate_pair_sets
from helpers import *
from models import *

if __name__ == '__main__':

  # -------------------------------------------------------------------------------------
  # When True, trains the best model only
  # No tuning, outputs accuracy for 1 run
  run_best = True
  
  # Set to True if you want to see the results for all models
  # Tunes parameters and outputs accuracy averaged over 10 runs   
  run_full = False
  # -------------------------------------------------------------------------------------

  n = 1000
  lr = 0.1

  if run_full:
    print("Training and tuning, this might take a while...")
    train_tune_evaluate(lr, n)
  if run_best:
    print("Training the best model overall...")
    train_input, train_target, train_classes, test_input, test_target, _ = generate_pair_sets(n)
    model = BaseNetWeightShareAux(batch_normalization = True, dropout = 0)
    train_evaluate_best(model, train_input, train_target, train_classes, test_input, test_target, True, lr, n)

