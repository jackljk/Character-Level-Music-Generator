from util import *
from constants import *
from train import *
from dataLoader import *
import torch.optim as optim
import torch.nn as nn
import sys
import os


def train(model, data, data_val, char_idx_map, config, device):

    """
    Train the provided model using the specified configuration and data.

    Parameters:
    - model (nn.Module): The neural network model to be trained
    - data (list): A list of training data sequences
    - data_val (list): A list of validation data sequences
    - char_idx_map (dict): A dictionary mapping characters to their corresponding indices
    - config (dict): A dictionary containing configuration parameters for training:
    - device (torch.device): The device (e.g., "cpu" or "cuda") on which the model is located

    Returns:
    - losses (list): A list containing training losses for each epoch
    - v_losses (list): A list containing validation losses for each epoch
    """

    # Extracting configuration parameters
    N_EPOCHS = config["no_epochs"]
    LR = config["learning_rate"]
    SAVE_EVERY = config["save_epoch"]
    MODEL_TYPE = config["model_type"]
    HIDDEN_SIZE = config["hidden_size"]
    DROPOUT_P = config["dropout"]
    SEQ_SIZE = config["sequence_size"]
    CHECKPOINT = 'ckpt_mdl_{}_ep_{}_hsize_{}_dout_{}'.format(MODEL_TYPE, N_EPOCHS, HIDDEN_SIZE, DROPOUT_P)

    model = model.to(device) # Transfer the model to the device (CPU/GPU)

    optimizer = optim.Adam(model.parameters(), lr=LR) # Initialize the optimizer

    lossfunc = nn.CrossEntropyLoss().to(device) # Initialize the loss function

    # Lists to store training and validation losses over the epochs
    train_losses, validation_losses = [], []

    # Training over epochs
    for epoch in range(N_EPOCHS):
        # Initialize epoch training and validation losses
        epoch_train_loss, epoch_val_loss = 0, 0
        
        # TRAIN: Train model over training data
        for i in range(len(data)):
            '''
                - For each song:
                    - Zero out/Re-initialise the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.) (Done for you)
                    - Zero out the gradient (Done for you)
                    - Get a random sequence of length: SEQ_SIZE from each song (check util.py)
                    - Iterate over sequence characters : 
                        - Transfer the input and the corresponding ground truth to the same device as the model's
                        - Do a forward pass through the model
                        - Calculate loss per character of sequence
                    - backpropagate the loss after iterating over the sequence of characters
                    - update the weights after iterating over the sequence of characters
                    - Calculate avg loss for the sequence
                - Calculate avg loss for the training dataset and 


            '''
            loss = 0 # Initialize loss for the sequence
            model.init_hidden() # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
            model.zero_grad()   # Zero out the gradient
            song_slice = get_random_song_slice(data[i], SEQ_SIZE)  # Get a random sequence of length: SEQ_SIZE 
            
            # Iterate over sequence characters
            for j in range(len(song_slice) - 1):
                char_input = characters_to_tensor([song_slice[j]], char_idx_map).to(device) # Transfer the input to the same device as the model's
                targ_output = characters_to_tensor([song_slice[j+1]], char_idx_map).to(device) # Transfer the ground truth to the same device as the model's
                #print(char_input.shape, targ_output.shape)
                ypred, _ = model.forward(char_input) # Do a forward pass through the model
                loss += lossfunc(ypred, targ_output) # Calculate loss per character of sequence
            
            # backpropagate the loss after iterating over the sequence of characters
            loss.backward()
            # update the weights after iterating over the sequence of characters
            optimizer.step()
                
			# Calculate avg loss for the sequence
            avg_loss_per_sequence = loss.item()/SEQ_SIZE
            # Add the avg loss for the sequence to the epoch training loss
            epoch_train_loss += avg_loss_per_sequence
            

            # Display progress
            msg = '\rTraining Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/len(data)*100, i, avg_loss_per_sequence)
            sys.stdout.write(msg)
            sys.stdout.flush()

        print() # Print a new line after each epoch

		# Append the avg loss on the training dataset to train_losses list for the current epoch
        train_losses.append(epoch_train_loss/len(data))


        # VAL: Evaluate Model on Validation dataset
        model.eval() # Put in eval mode (disables batchnorm/dropout) !
        with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
            # Iterate over validation data
            for i in range(len(data_val)):
                '''
                    - For each song:
                        - Zero out/Re-initialise the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.) (Done for you)
                        - Get a random sequence of length: SEQ_SIZE from each song- Get a random sequence of length: SEQ_SIZE from each song (check util.py)
                        - Iterate over sequence characters : 
                            - Transfer the input and the corresponding ground truth to the same device as the model's
                            - Do a forward pass through the model
                            - Calculate loss per character of sequence
                        - Calculate avg loss for the sequence
                    - Calculate avg loss for the validation dataset and 
                '''

                model.init_hidden() # Zero out the hidden layer (When you start a new song, the hidden layer state should start at all 0’s.)
				
				
                song_slice = get_random_song_slice(data_val[i], SEQ_SIZE) # Get a random sequence of length: SEQ_SIZE from each song
                loss = 0 # Initialize loss for the sequence
    
                for j in range(len(song_slice) - 1):
					# Transfer the input to the same device as the model's
                    char_input = characters_to_tensor([song_slice[j]], char_idx_map).to(device) # Transfer the input to the same device as the model's
                    targ_output = characters_to_tensor([song_slice[j+1]], char_idx_map).to(device) # Transfer the ground truth to the same device as the model's
                    ypred, activation = model.forward(char_input) # Do a forward pass through the model
                    loss += lossfunc(ypred, targ_output) # Calculate loss per character of sequence


				# Calculate avg loss for the sequence
                avg_loss_per_sequence = loss.item()/SEQ_SIZE
    
				# Add the avg loss for the sequence to the epoch validation loss
                epoch_val_loss += avg_loss_per_sequence





                # Display progress
                msg = '\rValidation Epoch: {}, {:.2f}% iter: {} Loss: {:.4}'.format(epoch, (i+1)/len(data_val)*100, i, avg_loss_per_sequence)
                sys.stdout.write(msg)
                sys.stdout.flush()

            print()


        # Append the avg loss on the validation dataset to validation_losses list
        validation_losses.append(epoch_val_loss/len(data_val))

        model.train() #TURNING THE TRAIN MODE BACK ON !

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        # Save checkpoint.
        if (epoch % SAVE_EVERY == 0 and epoch != 0)  or epoch == N_EPOCHS - 1:
            print('=======>Saving..')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss/len(data_val),
                }, './checkpoint/' + CHECKPOINT + '.t%s' % epoch)

    return train_losses, validation_losses