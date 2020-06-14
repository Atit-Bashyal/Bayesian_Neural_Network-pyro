# Bayesian_Neural_Network-pyro


Uses Variational Bayes for finding posteriors for neural network parameters. 

Requires following packages:

PyTorch,Pyro,Numpy,Matplotlib

 
In the training/optimizing step the true posterior probability for the weights is approximated using the distribution given by parameter values and is used for predictions. The modelling technique aslo allows us to sample a new set of weights and parameters for each prediction step.
This effectively means sampling new set of neural networks and averaging final layer output values after which, the class with 
max activation value is the predicted digit. To understand the process better it can be broken down as the following steps

- For an input image, take n samples of neural networks to get n different output values from the last layer.
- Convert those outputs (which are logsoftmaxed) into probabilities.
- Now, given the input image and n probability values for each digit, consider  median of the n probability values as the predectided probability for each digit.
- If the predectided probability is greater than p(any threshold)select the digit as a classification output from the network




