# fiber_base
Base repo for single molecule imputation based on Fiber-seq

## File descriptions

### Main Files

- data_utils.py: main file to load data
    - creates a tensor of N,C,L for input data and 1,L for target assay
- models.py: directory for main models
    - best working model as of 27th July, 2026: FiberDeep01ResConv1dBlock
- main.py: the entrance point to the code

### Other Important files

- trainer.py: training loop
- args.py: cmd args

### Unimportant files

- all of the rest. Will delete in main release
