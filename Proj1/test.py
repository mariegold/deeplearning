from dlc_practical_prologue import generate_pair_sets
from helpers import *
from models import *
import time

if __name__ == '__main__':
  n = 1000
  lr = 0.1

  start = time.time()
  # Generate an initial dataset for parameter tuning
  init_train_input, init_train_target, init_train_classes, init_test_input, init_test_target, _ = generate_pair_sets(n)
  model_base_mean, model_aux_mean, model_ws_mean, model_ws_aux_mean = param_tune(init_train_input, init_train_target, init_train_classes, init_test_input, init_test_target, lr, n)

  best_base_params, init_base_acc = max(model_base_mean.items(), key = lambda k : k[1])
  print('Best (use_bn, dropout rate) combination with BaseNet:', best_base_params) 
  print('Initial dataset accuracy with BaseNet: {:.3f}'.format(init_base_acc)) 

  best_aux_params, init_aux_acc = max(model_aux_mean.items(), key = lambda k : k[1])
  print('Best (use_bn, dropout rate) combination with BaseNetAux:', best_aux_params) 
  print('Initial dataset accuracy with BaseNetAux: {:.3f}'.format(init_aux_acc))

  best_ws_params, init_wc_acc = max(model_ws_mean.items(), key = lambda k : k[1])
  print('Best (use_bn, dropout rate) combination with BaseNetWeightShare:', best_ws_params) 
  print('Initial dataset accuracy with BaseNetWeightShare: {:.3f}'.format(init_wc_acc)) 

  best_ws_aux_params, init_ws_aux_acc = max(model_ws_aux_mean.items(), key = lambda k : k[1])
  print('Best (use_bn, dropout rate) combination with BaseNetWeightShareAux:', best_ws_aux_params) 
  print('Initial dataset accuracy with BaseNetWeightShareAux: {:.3f}'.format(init_ws_aux_acc)) 

  best_base_bn, best_base_dropout = best_base_params
  best_model_base = BaseNet(batch_normalization=best_base_bn, dropout=best_base_dropout)

  best_aux_bn, best_aux_dropout = best_aux_params
  best_model_aux = BaseNetAux(batch_normalization=best_aux_bn, dropout=best_aux_dropout)

  best_ws_bn, best_ws_dropout = best_ws_params
  best_model_ws = BaseNetWeightShare(batch_normalization=best_ws_bn, dropout=best_ws_dropout)

  best_ws_aux_bn, best_ws_aux_dropout = best_ws_aux_params
  best_model_ws_aux = BaseNetWeightShareAux(batch_normalization=best_ws_aux_bn, dropout=best_ws_aux_dropout)

  # Generate 10 datasets for performance estimation
  datasets = []
  for i in range(10):
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n)
    # Standardize dataset
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    datasets.append((train_input, train_target, train_classes, test_input, test_target, test_classes))

  # Find and report means and standard devs
  base_mean, base_std = performance_estimation(datasets, best_model_base, lr, False, n)
  print('BaseNet accuracy: {:.3f} +/- {:.3f}.'.format(base_mean, base_std))
  aux_mean, aux_std = performance_estimation(datasets, best_model_aux, lr, True, n)
  print('BaseNetAux accuracy: {:.3f} +/- {:.3f}.'.format(aux_mean, aux_std))
  ws_mean, ws_std = performance_estimation(datasets, best_model_ws, lr, False, n)
  print('BaseNetWeightShare accuracy: {:.3f} +/- {:.3f}.'.format(ws_mean, ws_std))
  ws_aux_mean, ws_aux_std = performance_estimation(datasets, best_model_ws_aux, lr, True, n)
  print('BaseNetWeightShareAux accuracy: {:.3f} +/- {:.3f}.'.format(ws_aux_mean, ws_aux_std))


  end = time.time()
  print('Execution time: {:.3f}'.format(end - start))

