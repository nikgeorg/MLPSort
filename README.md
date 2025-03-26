This program trains, validates and sorts samples using the multilayer perceptron on 2 hidden layers (pt2.c) and 3 hidden layers (pt3.c). The number of samples and such settings are set in the #define macros in the beginning of the program.

How to compile:
- For gcc/clang, -lm is necessary as a flag. Then run the executable normally.

There's also a plot.py script for plotting the samples and seeing their changes (NumPy and matplotlib required).

Options (all are defined by the macros in each file) :
- Use ReLU or tanh as activation functions (defined by the USE_TANH macro)
- Number of neurons per hidden layer (in each H1/H2/H3 macro)
- Learning rate of the network (defined by LR)
- Batch size of samples
- Number of (minimum) epochs to be run
