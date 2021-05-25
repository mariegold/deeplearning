from dlc_practical_prologue import generate_pair_sets
from helpers import *
from models import *

if __name__ == '__main__':
  n = 1000

  # Generate an initial dataset for cross validation
  # Find best hyperparameter settings to use for each model
  cross_val_input, cross_val_target, cross_val_classes, _, _, _ = generate_pair_sets(n)
  model_base_mean, model_aux_mean, model_ws_mean, model_ws_aux_mean = cross_validate(cross_val_input, cross_val_target, cross_val_classes,n)

  best_base_params, cross_base_acc = max(model_base_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNet:', best_base_params) 
  print('Cross validation accuracy with BaseNet: {:.3f}'.format(cross_base_acc)) 

  best_aux_params, cross_aux_acc = max(model_aux_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNetAux:', best_aux_params) 
  print('Cross validation accuracy with BaseNetAux: {:.3f}'.format(cross_aux_acc))

  best_ws_params, cross_wc_acc = max(model_ws_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNetWeightShare:', best_ws_params) 
  print('Cross validation accuracy with BaseNetWeightShare: {:.3f}'.format(cross_wc_acc)) 

  best_ws_aux_params, cross_ws_aux_acc = max(model_ws_aux_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNetWeightShareAux:', best_ws_aux_params) 
  print('Cross validation accuracy with BaseNetWeightShareAux: {:.3f}'.format(cross_ws_aux_acc)) 

  best_base_lr, best_base_bn, best_base_dropout = best_base_params
  best_model_base = BaseNet(batch_normalization=best_base_bn, dropout=best_base_dropout)

  best_aux_lr, best_aux_bn, best_aux_dropout = best_aux_params
  best_model_aux = BaseNetAux(batch_normalization=best_aux_bn, dropout=best_aux_dropout)

  best_ws_lr, best_ws_bn, best_ws_dropout = best_ws_params
  best_model_ws = BaseNetWeightShare(batch_normalization=best_ws_bn, dropout=best_ws_dropout)

  best_ws_aux_lr, best_ws_aux_bn, best_ws_aux_dropout = best_ws_aux_params
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
  base_mean, base_std = performance_estimation(datasets, best_model_base, best_base_lr, False, n)
  print('BaseNet accuracy: {:.3f} +/- {:.3f}.'.format(base_mean, base_std))
  aux_mean, aux_std = performance_estimation(datasets, best_model_aux, best_aux_lr, True, n)
  print('BaseNetAux accuracy: {:.3f} +/- {:.3f}.'.format(aux_mean, aux_std))
  ws_mean, ws_std = performance_estimation(datasets, best_model_ws, best_ws_lr, False, n)
  print('BaseNetWeightShare accuracy: {:.3f} +/- {:.3f}.'.format(ws_mean, ws_std))
  ws_aux_mean, ws_aux_std = performance_estimation(datasets, best_model_ws_aux, best_ws_aux_lr, True, n)
  print('BaseNetWeightShareAux accuracy: {:.3f} +/- {:.3f}.'.format(ws_aux_mean, ws_aux_std))


  # # Generate 10 datasets for performance estimation
  # datasets = []
  # for i in range(10):
  #   train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n)
  #   # Standardize dataset
  #   mu, std = train_input.mean(), train_input.std()
  #   train_input.sub_(mu).div_(std)
  #   test_input.sub_(mu).div_(std)
  #   datasets.append((train_input, train_target, train_classes, test_input, test_target, test_classes))
  
  # # Get mean and std accuracy for each model across datasets for different hyperparamer settings
  # model_base_mean, model_base_std, model_aux_mean, model_aux_std, model_ws_mean, model_ws_std, model_ws_aux_mean, model_ws_aux_std = performance_estimation_param_fit(datasets, n)

  # # Find hyperparameter setting for each model that achieves the best mean accuracy
  # # Print the hyperparameter values, mean accuracy and the corresponding std
  # best_base_params, best_base_acc = max(model_base_mean.items(), key = lambda k : k[1])
  # print(best_base_params)
  # print('Best (lr, use_bn, dropout rate) combination with BaseNet:', best_base_params) 
  # print('Best accuracy with BaseNet: {:.3f} +/- {:.3f}.'.format(best_base_acc, model_base_std[best_base_params])) 

  # best_aux_params, best_aux_acc = max(model_aux_mean.items(), key = lambda k : k[1])
  # print('Best (lr, use_bn, dropout rate) combination with BaseNetAux:', best_aux_params) 
  # print('Best accuracy with BaseNetAux: {:.3f} +/- {:.3f}.'.format(best_aux_acc, model_aux_std[best_aux_params])) 

  # best_ws_params, best_wc_acc = max(model_ws_mean.items(), key = lambda k : k[1])
  # print('Best (lr, use_bn, dropout rate) combination with BaseNetWeightShare:', best_ws_params) 
  # print('Best accuracy with BaseNetWeightShare: {:.3f} +/- {:.3f}.'.format(best_wc_acc, model_ws_std[best_ws_params])) 

  # best_ws_aux_params, best_ws_aux_acc = max(model_ws_aux_mean.items(), key = lambda k : k[1])
  # print('Best (lr, use_bn, dropout rate) combination with BaseNetWeightShareAux:', best_ws_aux_params) 
  # print('Best accuracy with BaseNetWeightShareAux: {:.3f} +/- {:.3f}.'.format(best_ws_aux_acc, model_ws_aux_std[best_ws_aux_params])) 