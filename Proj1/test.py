import dlc_practical_prologue as prologue
from helpers import *
from models import *

if __name__ == '__main__':
  n_pairs = 1000
  train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n_pairs)

  model = BaseNet()
  train_model(model, train_input, train_target)
  nb_errors_test = compute_nb_errors(model, test_input, test_target)
  nb_errors_train = compute_nb_errors(model, train_input, train_target)

  print("Test accuracy: {} " .format(1 - nb_errors_test/n_pairs))
  print("Train accuracy: {} " .format(1 - nb_errors_train/n_pairs))

  model_aux = BaseNetAux()
  train_model_with_aux_loss(model_aux, train_input, train_target, train_classes)
  nb_errors_test_aux = compute_nb_errors_with_aux_loss(model_aux, test_input, test_target)
  nb_errors_train_aux = compute_nb_errors_with_aux_loss(model_aux, train_input, train_target)
  
  print("Test accuracy with aux. loss: {} " .format(1 - nb_errors_test_aux/n_pairs))
  print("Train accuracy with aux. loss: {} " .format(1 - nb_errors_train_aux/n_pairs))

  model_ws = BaseNetWeightShare()
  train_model(model_ws, train_input, train_target)
  nb_errors_test_ws = compute_nb_errors(model_ws, test_input, test_target)
  nb_errors_train_ws = compute_nb_errors(model_ws, train_input, train_target)

  print("Test accuracy with weight share: {} " .format(1 - nb_errors_test_ws/n_pairs))
  print("Train accuracy with weight share: {} " .format(1 - nb_errors_train_ws/n_pairs))

  model_ws_aux = BaseNetWeightShareAux()
  train_model_with_aux_loss(model_ws_aux, train_input, train_target, train_classes)
  nb_errors_test_ws_aux = compute_nb_errors_with_aux_loss(model_ws_aux, test_input, test_target)
  nb_errors_train_ws_aux = compute_nb_errors_with_aux_loss(model_ws_aux, train_input, train_target)
  
  print("Test accuracy with weight share and aux. loss: {} " .format(1 - nb_errors_test_ws_aux/n_pairs))
  print("Train accuracy with weight share and aux. loss: {} " .format(1 - nb_errors_train_ws_aux/n_pairs))
