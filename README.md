# PA3-Instructions
1. In your terminal, cd to the location where this folder is located. 
2. You can then run any config file in the configs folder into the model by doing: `python main.py config_{specify which config file}.py` in the terminal. 
4. After you run a model, a plot of the training and validation losses get saved in a plots folder, and checkpoints of the model will be saved in the checkpoints folder. 
5. To generate music for a given model without training, set the path of the checkpoint model in the config file, and also set the evaluation mode to true in the config. This will save the generated song as a .txt file.
6. To generate a heatmap for a given neuron, set the generate heatmap config to true, and set the neuron number in the generate.py file. Running main with evaluation mode will generate a heatmap for the specified path model, and save the heatmap as a png file.

