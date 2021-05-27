# Deep Learning Mini-Projects (May 2021)

*Authors: Marie Biolkova, Minh Tri Pham*

## Project 1

### Usage

To see the training and accuracy of the best model overall (1 run):

```bash
python3 Proj1/test.py 
```

To see the tuning and accuracies (with standard deviations) of all tested models (10 runs), modify `test.py`:

```python
# test.py 
run_full = True
```

and run the same bash command as above. This takes a while to complete (~40 minutes)

## Project 2

### Usage

To see the training and accuracy of the best model trained by SGD and MSE (1 run):

```bash
python3 Proj2/test.py 
```

To see the training and accuracy of the best model overall (1 run), modify `test.py`:

```python
# test.py 
run_best = True
```

and run the same bash command as above. 

To see the tuning and accuracies (with standard deviations) of all tested models (10 runs), set:

```python
# test.py 
run_training = True
```

and run the same bash command as above. This takes a while to complete (~10 minutes).