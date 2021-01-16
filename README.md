# Music_Generator_MALIS
School project for the MALIS course about generating music with Machine Learning

In the "app" folder :
- "NeuralNetwork.py" has the model of our neural network with the definition of our loss, accuracy, backpropagation etc. It comes from the lab 2.

- "main_NN.py" is the file to run to test our three Neural Networks or train your own. It also contains at the end the algorithm generating the music. Specify the path to the mozart .mid file, the feature to predict (Number of Note = "NN", Classes of notes = "N", Octaves of the notes = "O", then the epochs and the learning_rate. To change the Neural Networks' layers, you can do it in the get_list_layers(feature) method in the utils.py file.
Don't forget to adapt the path variables to your computer, specially for the input MIDI file (and the output MIDI file if you want to save the generated music) at the beggining of the "main.py" code AND the pickle files path later in the code.
The last part of the script is about generating the music so if you are not interested about it, you can comment it.

- "main_SVM.py" is the file to run if you want to train and test our Multi Classifiers SVM with 4 different kernels (linear, polynomial, radial, sigmmoid). It only predicts the Number of Notes.

- "utils.py" contains utility functions useful to process music data and to build our program. The first method is reading a .mid file and creating a raw Dataset. The next methods are functions to create computable Datasets and run them through the Neural Networks. Then we have methods that create datasets specifically for the SVMs. Finally, we have methods that are used to generate the predicted music by repeatingly running our Neural Networks.


 In the "pickles" folder :
 - It contains pickle files that saves the Neural Network models and makes them persistent. So we don't have to create and train the same model every time we run "main.py". However, it will not take into account changes in epochs and learning rate so you will have to delete the pickle file of your Nerual Network and train it again with the new epochs and learning rate.
- It already contains pickle files corresponding to Neural Networks of layers [45, 25, 5] for Number of Notes, [45, 28, 20] for Classes of notes, [45, 30, 15] for Octaves of notes.


In the "results" folder :
- It contains the generated music made by the "main_NN.py" last part of the code. Files are in .mid type. In order to convert them to .mp3, you can use this website https://solmire.com/midi-to-mp3 (don't forget to tick the "Full Length" at Step 8 before converting)
- It already contains the results for 100 generated notes made in the middle of Mozart's Lacrimosa music, in both .mid and .mp3 file
