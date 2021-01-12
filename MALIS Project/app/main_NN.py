    # -*- coding: utf-8 -*-
    """
    Created on Wed Oct 28 14:35:13 2020

    @author: Aydin
    """


    ########## This is a test algorithm using piano only ##########
    # The first chapters of the music21 documentations is well-enough to understand all the code
    # Available here http://web.mit.edu/music21/doc/index.html

    # Library for understanding music
    import music21 as mu

    # This searchs for an external software to display or play music. MuseScore 3 is good.
    # Will annoy you with questions at the beginning...
    # To go through them fast press Enter ==> Enter ==> n + enter ==> n + Enter ==> n + Enter ==> Enter 
    # Comment this after the first run 
    #mu.configure.run()


    # Useful functions to manipulate and adapt music data to our models
    import utils as u

    # Array Processing
    import numpy as np



    ################################### USER SETTINGS #####################################
    file_input = '/Users/aydinabiar/Desktop/MALIS Project/mozart_samples/mid/lacrimosa_original.mid' # path to the input music file in .mid
    feature = "O" # Feature can be "NN" for Number of Notes, "N" for Notes and "O" for Octaves
    NNlist = [45,30,15] # Layers of the NN. Last layer is 5 for "NN", 20 for "N" or 15 for "O"
    epochs=150
    learning_rate=0.01

    chords, initial_stream = u.read_midi(file_input)


    ####### STEP 1 : Create samples 
    ## Put "NN" for number of notes, "N" for notes and "O" for octave (pas encore fait l'octave) 
    x_train, y_train, x_test, y_test = u.make_samples(chords, feature) # From Utils



    ####### STEP 2 : Implement the Neural Network

    # Multi-Layer Perceptron class
    from NeuralNetwork import MLP


    # To avoid training the same Neural Network again and again after successive runs, 
    #   we save the Neural Network in a file by using the pickle library
    # Then, if we run this code without changing the Neural Network ,
    #   (which means the layers, not the epochs or learning_rate)
    #   we don't have to redo the training and we just load the Neural Network from the pickle file

    # Need Path class to create file for the Neural Network
    from pathlib import Path
    # Need pickle to save and load files
    import pickle
    # For listing down the file names (for the pickle)
    import os


    # To differientate Neural Networks by their layers

    pickle_name = ''
    for layer in NNlist :
        pickle_name += str(layer) + '_'
    # Path to save or load the Neural Network
    pickle_path = Path("/Users/aydinabiar/Desktop/MALIS Project/app/pickles/"+ pickle_name + '.pk') # Put your own

    if pickle_path.is_file(): # If File exists
        # Load Neural Network back to memory 
        with open(pickle_path, 'rb') as fi:
            NN1 = pickle.load(fi)

    else : # Or if it's the first time we make that Neural Network
        # Print advancement True or False, every k steps
        verbose=True
        print_every_k=10

        # Initialization of the NN.
        # 45 is the dimension of the input vector x. 
        # 5 is the dimension of the output vector y.
        # The rest are the dimension of the hidden layers
        NN1 = MLP(NNlist)

        print('TRAINING')

        # Training
        NN1.training(x_train,y_train,learning_rate,epochs,verbose,print_every_k)

        # Open and save the Neural Netork in a pickle file
        with open(pickle_path, 'wb') as fi:
            # dump the Neural Network into the file
            pickle.dump(NN1, fi)


    # Compute the training loss and accuracy after having completed the training
    #y_hat=NN1.forward(x_train)
    #print('final : loss = %.3e , accuracy = %.2f %%'%(NN1.loss(y_hat,y_train),100*NN1.accuracy(y_hat,y_train)))


    ####### STEP 3 : Testing and Plotting
    
    # Test
    print('\nTEST')
    y_hat=NN1.forward(x_test)
    print('loss = %.3e , accuracy = %.2f %%\n'%(NN1.loss(y_hat,y_test),100*NN1.accuracy(y_hat,y_test)))

        

    ### To plot the loss and accuracies over time
    NN1.plot(epochs, learning_rate)




    '''
    # Last 9 consecutive chords of the piece of music
    last_chords = chords[len(chords)-9 :]

    # Generated chords
    predicted_future_stream = generate_music_stream(last_chords, 100)

    # Load the prediction
    predicted_future_stream.write('midi', fp='/Users/aydinabiar/Desktop/MALIS Project/results/'+pickle_name+'.mid')

    # Show the prediction in MuseScore 3 for example
    predicted_future_stream.show()

    # Append the generated chords to the initial piece of music
    final_stream = initial_stream
    for chord in predicted_future_stream :
        final_stream.append(chord)

    # Load the final music (initial + prediction)
    #final_stream.write('midi', fp=output_path)
    #     
    '''
