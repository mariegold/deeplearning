from dlc_practical_prologue import generate_pair_sets
from helpers import performance_estimation

if __name__ == '__main__':
  n = 1000
  # train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n)

  # Generate 10 datasets for performance estimation
  datasets = []
  for i in range(10):
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n)
    # Standardize dataset
    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)
    datasets.append((train_input, train_target, train_classes, test_input, test_target, test_classes))
  
  # Get mean and std accuracy for each model across datasets for different hyperparamer settings
  model_base_mean, model_base_std, model_aux_mean, model_aux_std, model_ws_mean, model_ws_std, model_ws_aux_mean, model_ws_aux_std = performance_estimation(datasets, n)

  # Find hyperparameter setting for each model that achieves the best mean accuracy
  # Print the hyperparameter values, mean accuracy and the corresponding std
  best_base_params, best_base_acc = max(model_base_mean.items(), key = lambda k : k[1])
  print(best_base_params)
  print('Best (lr, use_bn, dropout rate) combination with BaseNet:', best_base_params) 
  print('Best accuracy with BaseNet: {:.3f} +/- {:.3f}.'.format(best_base_acc, model_base_std[best_base_params])) 

  best_aux_params, best_aux_acc = max(model_aux_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNetAux:', best_aux_params) 
  print('Best accuracy with BaseNetAux: {:.3f} +/- {:.3f}.'.format(best_aux_acc, model_aux_std[best_aux_params])) 

  best_ws_params, best_wc_acc = max(model_ws_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNetWeightShare:', best_ws_params) 
  print('Best accuracy with BaseNetWeightShare: {:.3f} +/- {:.3f}.'.format(best_wc_acc, model_ws_std[best_ws_params])) 

  best_ws_aux_params, best_ws_aux_acc = max(model_ws_aux_mean.items(), key = lambda k : k[1])
  print('Best (lr, use_bn, dropout rate) combination with BaseNetWeightShareAux:', best_ws_aux_params) 
  print('Best accuracy with BaseNetWeightShareAux: {:.3f} +/- {:.3f}.'.format(best_ws_aux_acc, model_ws_aux_std[best_ws_aux_params])) 