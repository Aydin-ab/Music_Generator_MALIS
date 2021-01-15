import numpy as np
from math import exp
from scipy.stats import logistic
import matplotlib.pyplot as plt # For plotting the results


class Layer:
    """
    A class used to represent a Layer of a Neural Network

    ...

    Attributes
    ----------
    W : list
        the incoming weights
    b : list
        the biases
    a : list
        the activations
    z : list
        the outputs
    s : list
        the gradient of the incoming weights
    d_b : list
        the gradient of the biases
    d_a : list 
        the gradient of the activations
    d_z : list
        the gradient of the outputs
    """


    def __init__(self):
        """
        Initialise all attributes to an empty list
        """
        self.W=[] # self.W = the incoming weights
        self.b=[] # self.b = the biases
        self.a=[] # self.a = the activations
        self.z=[] # self.z = the outputs
        self.d_W=[] # self.d_W = the gradient of the incoming weights
        self.d_b=[] # self.d_b = the gradient of the biases
        self.d_a=[] # self.d_a = the gradient of the activations
        self.d_z=[] # self.d_z = the gradient of the outputs
        self.feature = ""

class MLP(Layer):
    """
    A class used to represent a Neural Network

    ...

    Attributes
    ----------
    self.layer : Layer
        a formatted string to print out what the animal says
    losses : list
        losses during training (for each epochs)
    accuracy : list
        accuracies during training (for each epochs)
    feature : string
        the feature being predicted (Number of notes, Notes or Octaves)


    Methods
    -------
    sigmoid(a)
        Sigmoid activation function. It can work with single inputs or vectors or matrices.
    d_sigmoid(a)
        Derivative of sigmoid activation function. It can work with single inputs or vectors or matrices.
    forward(x)
        Forward function. From input layer to output layer. Input can be 1D or 2D.
    loss(y_hat, y)
        Compute the loss between y_hat and y! they can be 1D or 2D arrays!
    accuracy(y_hat, y)
        Compute the accuracy between y_hat and y
    backpropagation(x,y,y_hat,learning_rate)
        Backpropagate the error from last layer to input layer and then update the parameters
    training(x,y,learning_rate,num_epochs,verbose=False, print_every_k=1)
        Training your Neural Network
    plot()
        plot the training score (when the Neural Network was constructed) and the testing score
    """


    def __init__(self, neurons_per_layer):
        '''
        Create the weight matrices for each layer following the neurons_per_layer vector.
        It initializes also the loss and accuracy vector
        
        self.layer[0].W contains the weights which connect input layer 1 with 1st hidden layer. Dimensions [n_1st,n_input]
        self.layer[0].b contains the biases of 1st hidden layer
        self.layer[0].a contains the activation of 1st hidden layer
        self.layer[0].z contains the outputs of 1st hidden layer
        self.layer[0].d_W contains the derivative of loss w.r.t the weights which connect input layer 1 with 1st hidden layer. Dimensions [n_1st,n_input]
        self.layer[0].d_b contains the derivative of loss w.r.t the biases of 1st hidden layer
        self.layer[0].d_a contains the derivative of loss w.r.t the activations of 1st hidden layer
        self.layer[0].d_z contains the derivative of loss w.r.t the outputs of 1st hidden layer
        self.layer[1].W contains the weights which connect 1st hidden layer with 2nd hidden layer. Dimensions [n_2nd,n_1st]
        self.layer[1].b contains the biases of 2nd hidden layer
        ecc...
        self.weights[n] contains the weights which connect nth hidden layer with output layer. Dimensions Dimensions [n_nth,n_output]
        self.biases[n] contains the biases of output layer
        ...
        
        INPUT : 
        - neurons_per_layer : numpy array containing the number of neurons in
            [ input layer, hidden layer1, hidden layer 2, ..., output layer ]
        '''


        super().__init__()
        
        self.layer={}
        
        for i in range(0,len(neurons_per_layer)-1) :
            self.layer[i]=Layer()
            self.layer[i].W=(10**(-1))*np.random.randn(neurons_per_layer[i+1],neurons_per_layer[i])
            self.layer[i].b=np.zeros((1,neurons_per_layer[i+1]))
            self.layer[i].a=np.zeros((1,neurons_per_layer[i+1])) 
            self.layer[i].z=np.zeros((1,neurons_per_layer[i+1]))
            self.layer[i].d_W=np.zeros((neurons_per_layer[i+1],neurons_per_layer[i]))
            self.layer[i].d_b=np.zeros((1,neurons_per_layer[i+1])) 
            self.layer[i].d_a=np.zeros((1,neurons_per_layer[i+1]))
            self.layer[i].d_z=np.zeros((1,neurons_per_layer[i+1]))

        self.M_1 = 0 # Error on the number of chords
        self.M_2 = 0 # Error on the classes of notes
        self.M_3 = 0 # Error on the octave

        self.losses=[]
        self.accuracies=[]
        
    def sigmoid(self, a) :
        """ Sigmoid activation function. It can work with single inputs or vectors or matrices.

        Parameters
        ----------
        a : float or ndarray of dimension 2 or more inputs

        Output
        ----------
        sigmoid of a or each of each of its element, keeping the shape of a
        """
        
        # logistic.cdf from scipy is used for stability instead of exponential functions
        
        return np.array(logistic.cdf(a)) 
    
    def d_sigmoid(self, a) :
        """ Sigmoid activation function. It can work with single inputs or vectors or matrices.

        Parameters
        ----------
        a : float, ndarray of dimension 2 or more
            Input
        
        Output
        ----------
        derivative of the sigmoid of a or each of each of its element, keeping the shape of a
        """

        ################# YOUR CODE HERE ####################
        
        if np.isscalar(a) : # If a is a scalar then we just calculate normally
            sig = self.sigmoid(a)
            d_sig = sig*(1-sig)
            a = d_sig
        
        else : # If a is a numpy list or array    
            # I used the numpy.nditer to modify a whether it is a numpy list or array
            # https://numpy.org/doc/stable/reference/arrays.nditer.html Section " Modifying Array values"
            with np.nditer(a, op_flags=['readwrite']) as it: # We iterate through every element of the list / array
                for element in it : 
                    sig = self.sigmoid(element) # Sigmoid of the element
                    d_sig = sig*(1-sig) # Derivative sigmoid of the element using the sigmoid properties
                    element[...] = d_sig # We modify the element
            
        return a
        
        ################ END OF YOUR CODE HERE ##############
    
    
    def forward(self, x) :        
        """ Forward function. From input layer to output layer. Input can be 1D or 2D.

        Parameters
        ----------
        x : numpy array of size NxD
            Where N is the number of samples, D is the number of input dimensions referred
        
        Output
        ----------
        y_hat : numpy array of size NxC
            Where C is the number of classes
        """

        
        ################# YOUR CODE HERE ####################
        
        ######### Please note that I am using the notation of the online chapter you gave in the lab2
        ######### This means : self.layer[l].z in the weighted input vector of the layer l
        #########              self.layer[l].a = MLP.sigmoid(z) is the output vector of the layer l
        
        L = len(self.layer) # Number of layers
        z = x # Initialisation : "output" of the first layer, l = 0, are the inputs
        for i in range(L) :
            z = self.layer[i].b + z @ self.layer[i].W.T # Computing the weighted input for each layer
            self.layer[i].z = z
            a = self.sigmoid(z) # Computing the output for each layer
            self.layer[i].a = a 
            z=a # The weighted input of the next layer is the output of the current layer
            
        y_hat = self.layer[L-1].a # The last output is y_hat 
        
        ################ END OF YOUR CODE HERE ##############

        return y_hat


    def loss(self, y_hat, y) :
        """
        Compute the loss between y_hat and y! they can be 1D or 2D arrays!
        
        Parameters
        ----------
        y_hat : numpy array of size NxC 
            Where N is the number of samples,  C the number of bits. It contains the estimated values of y
        y : numpy array of size NxC 
            Where N is number of samples, C the number of bits. It contains the correct values of the samples
        
        Output
        ----------:
        L : float
            MSE loss
        """
        

        # compute the mean square loss between y_hat and y
        
        ################# YOUR CODE HERE ####################
        N = len(y_hat) # Number of samples
        B = len(y_hat[0]) # Number of bits in a chord

        loss = 0
        for j in range(N) :
            for i in range(B) :
                    loss += (y_hat[j][i] - y[j][i])**2
     
        loss_average = loss/N

        return loss_average
        ################ END OF YOUR CODE HERE ##############
 

    def accuracy(self, y_hat,y) :
        """
        Compute the accuracy between y_hat and y

        Parameters
        ----------
        y_hat : numpy array of size NxC
            Where N is the number of samples, C is the number of bits. It contains the estimated values of y
        y : numpy array of size NxC 
            Where N is the number of samples, C is the number of bits. It contains the correct values of the samples
        
        Outputs
        ----------
        acc : float
            the accuracy value between 0 and 1
        """


        ################# YOUR CODE HERE ####################

        N = len(y) # Number of samples
        C = len(y[0]) # Number of bits in a chord
        correct_samples = 0 # Number of correct samples initialised to 0 
        for n in range(N) : # We iterate for each samples
            if self.feature == "N" or self.feature == "O" :
                # We check with a boolean correct_bits if each bit is correctly predicted
                correct_bits = True
                for c in range(C) :
                    if y_hat[n][c] >= 0.5 and y[n][c] == 0 :
                        correct_bits = False
                    elif y_hat[n][c] < 0.5 and y[n][c] == 1 :
                        correct_bits = False
                # Therefore , correct_bits is True if every predicted bits is > 0.5 or < 0.5 respectively with the correct bits being 1 or 0
                if correct_bits :
                    correct_samples += 1
            elif self.feature == "NN" :
                if np.argmax(y_hat[n]) == np.argmax(y[n]):
                    correct_samples += 1 # We count every correct samples
        
        acc = correct_samples / N # Accuracy
        return acc

        ################ END OF YOUR CODE HERE ##############
        
    
    def backpropagation(self,x,y,y_hat,learning_rate) :
        '''
        Backpropagate the error from last layer to input layer and then update the parameters

        Parameters
        ----------
        y_hat : numpy array of size NxC
            Where N is the number of samples, C is the number of classes. It contains the estimated values of y
        y : numpy array of size NxC 
            Where N is the number of samples, C is the number of classes. It contains the correct values of the samples

        Outputs
        ----------
        (compute the error at the different levels and for each layer and update them)
        - d_a
        - d_z
        - delta_L
        - delta_l
        - d_W
        - d_b
        '''


        # compute gradients

            ################# YOUR CODE HERE ####################

        ######### I WILL FOLLOW THE ONLINE CHAPTER NOTATION ##########
        ######### z_l is the weighted input vector of the layer l
        ######### a_l = MLP.sigmoid(z_l) is the output vector of layer l with a_L being by definition y_hat (a_l_k is its k'th element)
        ######### grad_L_da_l or grad_L_dw_l or grad_L_db_l are the cost gradient relative to the output vector 
        #########                                           or the weight vectors or the biases vector of the layer l
        ######### d_sig_z_l is the sigmoid derivative vector applied to the vector z_l
        ######### delta_l is the output error vector of the layer l (delta_l_j is its j'th element)
        ######### The used equation are :
        ######### (BP1) : delta_L = grad_L_da_L * d_sig_z_L 
        ######### (BP2) : delta_l = w_above_l.T @ delta_above_l * d_sig_z_l with w_above_l and delta_above_l 
        #########                                                           the weight vectors and the output error vector 
        #########                                                           of the layer above l (=l+1)
        ######### (BP3) : grad_L_b_l = delta_l
        ######### (BP4) : grad_L_w_jk = a_below_l_k * delta_l_j     with grad_l_w_jk the grad_L_w element of the j'th line and k'th column
        #########                                                   and a_below_l_k the k'th element of the output vector of the layer l-1


        L = len(self.layer) # Number of layers
        
        ## STEP 1 INITIALISATION : Computing the output error vector delta_L using (BP1) equation
        ## (BP1) : delta_L = grad_L_da_L * d_sig_z_L
        ## Then we will compute the cost gradient relative to the bias and the weight of the last layer
        
        ### The output vector a_L is y_hat :
        a_L = y_hat # = self.layer[L-1].a
        
        ### Computing the cost function derivative relative to a_L
        grad_L_da_L = a_L - y # we derive a norm function
        
        ### Retrieving the vector d_sigmoid(z_L)
        d_sig_z_L = self.d_sigmoid(self.layer[L-1].z)
        
        ### Computing the output error delta_L
        delta_L = grad_L_da_L * d_sig_z_L  
        
        ### Computing the cost gradient relative to the bias 
        grad_L_b_L = delta_L # By (BP3) 
        self.layer[L-1].d_b = grad_L_b_L
        
        ### Retrieving the ouput vector of the layer below
        a_below_L = self.layer[L-2].a
            
        ### Computing the gradient of the cost relative to the weights of the layer l    
        grad_L_w_L = delta_L.T @ a_below_L # By (BP4)
        self.layer[L-1].d_W = grad_L_w_L
                
        ## STEP 2 BACKPROPAGATION : We calculate for each layer, starting from the last layer, the gradient of the cost relative to the biases and the wheights

        delta_above_l = delta_L # We initialise for the loop below
        
        for i in range(L-1) : # We iterate through the layers
            
            l = L-2-i # We work on the layer l starting from the penultimate layer        
            
            if l != 0 :  # We deal with the the first hidden layer later because its cost gradient relative to the weight is defined from x and not the activation 
            ### First, we compute the error vector delta_l of the layer l using (BP2)
            #### (BP2) : delta_l = w_above_l.T @ delta_above_l * d_sig_z_l

            #### Retrieving the weights w_above_l ( = w_l+1)
                w_above_l = self.layer[l+1].W # w_l+1

            #### Computing the vector d_sig_z_l
                d_sig_z_l = self.d_sigmoid(self.layer[l].z)

            #### Computing the error vector delta_l
                delta_l = (delta_above_l @ w_above_l)*d_sig_z_l

            ### Then, we compute the cost gradient relative to the biases and the weights using (BP3) and (BP4)
            ### (BP3) : d_L_b_l = delta_l
            ### (BP4) : d_L_w_jk = a_below_l_k * delta_l_j

            #### Computing the gradient of the cost relative to the biases of the layer l
                grad_L_b_l = np.array(delta_l) # By (BP3)
                self.layer[i].d_b = grad_L_b_l

            #### Retrieving the output vector of the layer l-1
                a_below_l = self.layer[l-1].a

            #### Computing the gradient of the cost relative to the weights of the layer l
                # Note that grad_L_w_l_jk = a_below_l[k] * delta_l[j] by (BP4) equation
                grad_L_w_l = np.array([[a_below_l[0][k] * delta_l[0][j] for k in range(len(a_below_l[0]))] for j in range(len(delta_l[0]))]) # By (BP4)
                self.layer[l].d_W = grad_L_w_l

            ### Restarting the next loop for the layer l-1
                delta_above_l = delta_l 
        
        
        ## STEP 3 the first layer : We deal with the first layer specially because we compute the cost gradient relative to the weights
        ##                          with x and not with a z_l variable as it was for the intermediate layers
        
        delta_1 = delta_above_l # At the end of the for loop above, delta_1 the output error of the first layer is by construction delta_above_l
        d_sig_z_0 = self.d_sigmoid(self.layer[0].z) # The weighted input of the first layer
        delta_0 = (delta_1 @ self.layer[1].W)*d_sig_z_0 # By (BP2)
        grad_L_b_0 = np.array(delta_0) # By (BP3)
        self.layer[0].d_b = grad_L_b_0
        
        grad_L_w_0 = delta_0.T @ x # By (BP4) but we use x there
        self.layer[0].d_W = grad_L_w_0

        
        
        ################ END OF YOUR CODE HERE ##############

        # apply gradients
        # just one for loop passing through all layers is sufficient
        # apply the gradients only to self.layer[i].b and self.layer[i].W

        ################# YOUR CODE HERE ####################

        for i in range(L) :
            self.layer[i].b = self.layer[i].b - learning_rate * self.layer[i].d_b
            self.layer[i].W = self.layer[i].W - learning_rate * self.layer[i].d_W
        
        ################ END OF YOUR CODE HERE ##############
  
        
    def training(self,x,y,learning_rate,num_epochs,verbose=False, print_every_k=1) :
        """
        Training your network
        
        Parameters
        ----------
        x : numpy array of size NxD
            where N is the number of samples, D is the number of features of your input
        y : numpy array of size NxC
            where N is the number of samples, C is the number of classes, with correct values of target
        learning_rate : float
            a numpy scalar containing your learning rate
        num_epochs : float
            a numpy scalar representing the number of epochs with which train your networks
        verbose : boolean = False by default
            a boolean False by default, if True print the training loss and training accuracy values
                    if False only store them
        print_every_k int = 1 by default
            a numpy scalar equal 1 by default, if verbose is True print the result every print_every_k epochs
        """


        accuracy=[]
        loss=[]

        # iterate for num_epochs number of epochs
        for epoch in range(num_epochs) :
            
            # shuffle your training set
            shuffle=np.random.permutation(range(x.shape[0]))
            x_shuffled=x[shuffle,:]
            y_shuffled=y[shuffle,:]
            
            # sample by sample forward and backward through the network using stochastic gradient descent (SGD)
            for sample in range(x.shape[0]) :
                y_hat=self.forward(x_shuffled[sample,:])
                self.backpropagation(x_shuffled[sample,:].reshape(1,x.shape[1]),y_shuffled[sample,:],y_hat,learning_rate)

        # check how is performing the network after each epoch
            # estimate the training labels
            Y_hat=self.forward(x)
            # compute the loss
            loss.append(self.loss(Y_hat,y))
            # compute the accuracy
            accuracy.append(self.accuracy(Y_hat,y))
            
            # if verbose is True print the results every print_every_k
            if ((verbose == True) and (epoch%print_every_k==0)):
                print('Epoch %d : loss = %.5e, accuracy = %.2f %%' %(epoch,loss[epoch],100*accuracy[epoch]))

        self.losses=loss
        self.accuracies=accuracy
            
    
    def plot(self, epochs, learning_rate) :
        """
        Training your network
        
        Parameters
        ----------
        epochs : int
            a numpy scalar representing the number of epochs with which train your networks
        learning_rate : float
            a numpy scalar containing your learning rate
        """


        plt.plot(list(range(epochs)),self.losses,c='r',marker='o',ls='--')
        plt.title("Training Loss")
        plt.xlabel("epochs")
        plt.ylabel("loss value")
        plt.show()

        plt.plot(list(range(epochs)),self.accuracies,c='g',marker='o',ls='--')
        plt.title("Training accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()   