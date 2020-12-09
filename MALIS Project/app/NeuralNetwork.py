import numpy as np
from math import exp
from scipy.stats import logistic

class Layer:
    '''
    The class Layer contains the parameters of each layer. Its initialization make them all empty
    '''
    def __init__(self):
        self.W=[] # self.W = the incoming weights
        self.b=[] # self.b = the biases
        self.a=[] # self.a = the activations
        self.z=[] # self.z = the outputs
        self.d_W=[] # self.d_W = the gradient of the incoming weights
        self.d_b=[] # self.d_b = the gradient of the biases
        self.d_a=[] # self.d_a = the gradient of the activations
        self.d_z=[] # self.d_z = the gradient of the outputs

class MLP(Layer): # Multi Layer Perceptron
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
        '''
        Sigmoid activation function. It can work with single inputs or vectors or matrices.
        '''
        # logistic.cdf from scipy is used for stability instead of exponential functions
        
        return np.array(logistic.cdf(a)) 
    
    def d_sigmoid(self, a) :
        '''
        Derivative of sigmoid activation function. It can work with single inputs or vectors or matrices.
        Return the sigmoid derivative of a
        '''
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
        '''
        Forward function. From input layer to output layer. Input can be 1D or 2D.
        
        INPUTS:
        - x : numpy array of size NxD, where N is the number of samples, D is the number of input dimensions referred as n_input before
        
        OUTPUTS:
        - y_hat : numpy array of size NxC, where C is the number of classes
        '''
        
        '''
        Forward function. From input layer to output layer. Input can handle 1D or 2D inputs.

        INPUTS:
        - x : numpy array of size NxD, where N is the number of samples, D is the number of input dimensions referred as n_input before

        OUTPUTS:
        - y_hat : numpy array of size NxC, where C is the number of classes
        '''
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
            
        y_hat = self.layer[L-1].z # The last output is y_hat 
        
        ################ END OF YOUR CODE HERE ##############

        return y_hat

    
    def loss(self, y_hat, y) :
        '''
        Compute the loss between y_hat and y! they can be 1D or 2D arrays!
        
        INPUTS:
        - y_hat : numpy array of size NxC ,N number of samples,  C number of bits. It contains the estimated values of y
        - y : numpy array of size NxC ,N number of samples, C corresponding to the correct bits for that sample
        
        OUTPUTS:
        - L : MSE loss
        '''
        
        # compute the mean square loss between y_hat and y
        
        ################# YOUR CODE HERE ####################
        N = len(y_hat) # Number of samples
        B = len(y_hat[0]) # Number of bits in a chord

        for j in range(N) :
            error_number_of_notes = 0
            error_classes_of_notes = 0
            error_octaves_of_notes = 0
            for i in range(B) :
                if i in [0,1,2] : # Indexes of the bits of number of notes in a chord
                    error_number_of_notes += (y_hat[j][i] - y[j][i])**2
                elif i in [3,10,17,24,31] :
                    error_octaves_of_notes += (   (y_hat[j][i] - y[j][i])**2
                                                + (y_hat[j][i+1] - y[j][i+1])**2
                                                + (y_hat[j][i+2] - y[j][i+2])**2  
                                              ) # With indexes of the bits of the classes
                elif i in [6,13,20,27,34] :
                    error_classes_of_notes += (   (y_hat[j][i] - y[j][i])**2 
                                                + (y_hat[j][i+1] - y[j][i+1])**2 
                                                + (y_hat[j][i+2] - y[j][i+2])**2
                                                + (y_hat[j][i+3] - y[j][i+3])**2 
                                              ) # With indexes of the bits of the octaves

            #M_1 = 1000 # Error on the number of chords is very higly punished
            #M_2 = 100 # Error on the classes of notes is higly punished
            #M_3 = 1 # Error on the octave is relatively acceptable, if the octaves are next to each other it's ok

            loss = (  0.5 * self.M_1 * error_number_of_notes 
                    + 0.5 * self.M_2 * error_classes_of_notes 
                    + 0.5 * self.M_3 * error_octaves_of_notes
                   )
            loss_average = loss/N

        return loss_average
        ################ END OF YOUR CODE HERE ##############
    

    def grad_loss(self, y_hat, y) :
        '''
        Compute the DERIVATIVE loss between y_hat and y! they can be 1D or 2D arrays!
        
        INPUTS:
        - y_hat : numpy array of size NxC ,N number of samples,  C number of bits. It contains the estimated values of y
        - y : numpy array of size NxC ,N number of samples, C corresponding to the correct bits for that sample
        
        OUTPUTS:
        - grad_L : MSE loss
        '''
        
        # compute the mean square loss between y_hat and y
        
        ################# YOUR CODE HERE ####################
        N = len(y_hat) # Number of samples
        B = len(y_hat[0]) # Number of bits in a chord
        grad_loss_average = [[]] # Grad vector

        for j in range(N) :
            for i in range(B) :
                if i in [0,1,2] : # Indexes of the bits of number of notes in a chord
                    bit_error_number_of_notes = (y_hat[j][i] - y[i])
                    grad_loss_average[0].append( self.M_1 * bit_error_number_of_notes / N)
                elif i in [3,10,17,24,31] : # Indexes of the bits of the octaves
                    for k in range(3) : # Number of bits of the octaves
                        bit_error_octaves = (y_hat[j][i + k] - y[i + k])
                        grad_loss_average[0].append( self.M_2 * bit_error_octaves / N)
                elif i in [6,13,20,27,34] :
                    for k in range(4) : # Number of bits of the classes
                        bit_error_classes = (y_hat[j][i + k] - y[i + k])
                        grad_loss_average[0].append( self.M_2 * bit_error_classes / N)

        return np.array(grad_loss_average)
        ################ END OF YOUR CODE HERE ##############


    def accuracy(self, y_hat,y) :
        '''
        Compute the accuracy between y_hat and y

        INPUTS:
        - y_hat : numpy array of size NxC, C number of bits. It contains the estimated values of y
        - y : numpy array of size NxC with correct values of y

        OUTPUTS:
        - acc : the accuracy value between 0 and 1
        '''
        ################# YOUR CODE HERE ####################

        N = len(y) # Number of samples
        C = len(y[0]) # Number of bits in a chord
        correct_samples = 0 # Number of correct samples initialised to 0 
        for n in range(N) : # We iterate for each samples
            # We check with a boolean correct_bits if each bit is correctly predicted
            correct_bits = True
            for c in range(C) :
                if y_hat[n][c] >= 0.5 and y[n][c] == 0 :
                    correct_bits = False
                elif y_hat[n][c] < 0.5 and y[n][c] == 1 :
                    correct_bits = False
            # Therefore , correct_bits is True if every predicted bits is > 0.5 or < 0.5 respectively with the correct bits being 1 or 0
            if correct_bits :
                correct_samples += 1 # We count every correct samples
        
        acc = correct_samples / N # Accuracy
        return acc

        ################ END OF YOUR CODE HERE ##############
        
    
    def backpropagation(self,x,y,y_hat,learning_rate) :
        '''
        Backpropagate the error from last layer to input layer and then update the parameters

        INPUTS:
        - y_hat : numpy array of size NxC, C number of classes. It contains the estimated values of y
        -y : numpy array of size NxC with correct values of y

        OUTPUTS: (compute the error at the different levels and for each layer)
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
        grad_L_da_L = self.grad_loss(a_L, y) # we derive a norm function
        
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
        '''
        Training your network
        
        INPUTS:
        - x : numpy array of size NxD, D number of features of your input
        - y : numpy array of size NxC, C number of classes, with correct values of target
        - learning_rate : a numpy scalar containing your learning rate
        - num_epochs : a numpy scalar representing the number of epochs with which train your networks
        - verbose : a boolean False by default, if True print the training loss and training accuracy values
                    if False only store them
        - print_every_k : a numpy scalar equal 1 by default, if verbose is True print the result every print_every_k epochs

        OUTPUTS: /
        '''
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
                test = x_shuffled[sample,:]
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
            
