import random
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, cos, pi
import pandas as pd

class Dataset:
    '''
    A class representing a dataset built reading the file whose name is stored in 'datafile'. The first layer of an instance of the SVM class should be a Dataset object.
    
    self.input : is a numpy array containing the inputs. Size is NxD, N number of samples, D dimensions
    self.output : is a numpy array containing the target corresponding at each input. Size is N
    self.input_size : is the number of dimensions D
    self.len : is the number of samples N
    self.indices : is a list containing the different indices of the samples: from 0 to N-1
    '''
    def __init__(self, datafile=None, input_size=0, length=0):
        '''
        Initializing function to build the Dataset.
        
        INPUT:
        - datafile : a path with the name of the file containing the data to be stored in the dataset
        - input_size : D, the bumber of dimensions of each input
        - length : N , the number if input samples
        '''
        
        self.input=[]
        self.output=[]
        
        if datafile:
            datas=pd.read_csv(datafile,delimiter=' ')
            self.input = datas.drop(['y'],axis=1).values
            self.output = datas['y'].values
            self.input_size = self.input.shape[1] # Number of input dimensions
            self.len = self.input.shape[0] # Number of samples in the dataset (accessible through len(dataset))
            self.indices = list(range(self.len)) # List of indices used to pick samples in a random order
        else :
            self.input_size = input_size # Number of input dimensions
            self.len = length # Number of samples in the dataset (accessible through len(dataset))
            self.indices = list(range(self.len)) # List of indices used to pick samples in a random order
class SVM:
    '''
    A class representing a Support Vector Machine (SVM) used to build SVM models
    '''
    def __init__(self, train_input, train_output, test_input, test_output):
        # infile: SVM description file, dataset: Dataset object
        self.train_input = train_input # Training input
        self.train_output = train_output # Training output
        self.test_input = test_input # Testing input
        self.test_output = test_output # Testing output
        self.test_plot = list() # Plot for test dataset
        input_dimensions= train_input.shape[0]
        self.w = np.random.rand(input_dimensions+1)-0.5 # self.w[-1] is actually b
        self.print_step=None


    def train(self, n_iterations, lambda_w, print_every_k=None,verbose=False):
        '''
        Train function for an SVM model using PEGASOS.
        If verbose is TRUE print the accuracy every 'print_every_k' iterations

        INPUTS:
        - self containing in particular:
            self.train_dataset with inside
                self.train_dataset.input -> inputs in a numpy NxD array, N number of sample, D dimensions
                self.train_dataset.output -> outputs in a numpy X array
            self.w -> weights of the model initialized at random before. Length D+1, because 1 is the bias
        - n_iterations -> total number of epochs
        - lambda_w -> lambda in PEGASOS
        - print_every_k -> compute the accuracy of the model every 'print_every_k' epochs
        - verbose -> if TRUE, in addition of computing the accuracy, the model also print it every 'print_every_k' epochs
        '''

    
        if not print_every_k:
            print_every_k = max(1, int(n_iterations/50))
            self.print_step=print_every_k
            # For n_iterations epochs
            for i in range(n_iterations): # equivalent for t in {1,...,T}
            
            # ! REMEMBER TO UPDATE self.w AT THE END OF EVERY ITERATION
            # ! self.train_dataset.input DOES NOT CONTAIN THE BIAS : a '1' in the dimension D+1
                # REMBER TO ADD THE '1' OF THE BIAS TO YOUR x VECTOR OR YOU WILL HAVE A SIZE MISMATCH
                # ADD THE BIAS IN THE LAST POSITION OF YOUR x VECTOR, OTHERWISE THE REST OF THE TRAIN WILL BE WRONG
                # x VECTOR OF [self.train_dataset.input[sample,:], 1]
            
            ################ YOUR CODE HERE #################
                inputs = self.train_input
                outputs = self.train_output
                N, D = inputs.shape # Number of samples and dimensions of a sample vector x
                sample = random.randint(0,N-1) # from random that we imported at the beginning
                x_sample = np.append(inputs[sample,:], 1)
                y_sample = outputs[sample]
                w_i = self.w # self.w = np.random.rand(input_dimensions+1)-0.5 
                eta_i = 1/(lambda_w*(i+1)) #learning rate of update formula

                # Considering 𝜙𝐼, the indicator function of the set:
                if y_sample * np.dot(w_i, x_sample) < 1 :
                    self.w = w_i - eta_i * (lambda_w * w_i - y_sample * x_sample) 

                else :
                    self.w = w_i - eta_i * lambda_w * w_i
            
            
            ################ END OF YOUR CODE ###############
            
                if not i%print_every_k:
                    if verbose:
                        print("Epoch: ", i+1, " out of ", n_iterations)
                        self.print_accuracy()
                    else:
                        self.compute_accuracy()


    def setmode(self, mode):
        '''
        Function used to change between training set and testing set
        '''
        if mode == "train":
            self.dataset = self.train_dataset
        elif mode == "test":
            self.dataset = self.test_dataset
        else:
            print("Unknown mode!")

    def print_accuracy(self):
        '''
        Print accuracy of neural network on current dataset
        '''
        print("Accuracy:", 100*self.compute_accuracy(), "%")

    def compute_accuracy(self):
        '''
        Compute accuracy of neural network on train and test dataset and return accuracy on test dataset
        '''
        # Compute accuracy on training set
        self.setmode("train")
        x=np.concatenate((self.dataset.input,np.ones((self.dataset.len,1))),axis=1)
        accuracy=np.mean(np.sign(x @ self.w.T) == self.dataset.output.T)
        self.train_plot.append(accuracy)

        # Compute accuracy on testing set
        self.setmode("test")
        x=np.concatenate((self.dataset.input,np.ones((self.dataset.len,1))),axis=1)
        self.test_plot.append(np.mean(np.sign(x @ self.w.T) == self.dataset.output.T))

        # Do not forget to go back to the training set!
        self.setmode("train")

        return accuracy

    def reset_plot(self):
        '''
        Reset plot
        '''
        self.train_plot = list()
        self.test_plot = list()

    def make_plot(self):
        '''
        Print plot
        '''
        plt.plot([x*self.print_step for x in range(len(self.train_plot))], self.train_plot, 'r-',label='training accuracy')
        plt.plot([x*self.print_step for x in range(len(self.test_plot))], self.test_plot, 'g-', label='test accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch number')
        plt.axis([0, self.print_step*(len(self.train_plot)-1), 0, 1.05])
        plt.legend()
        plt.show()

