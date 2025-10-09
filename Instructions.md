# General Instructions - Training

When training you need to be change the following things:
- In /home/asavari/foundation_model/configs/model/baseline.yaml
    - name - Name it as per the Loss function you are using, If you are using Focal Loss name it Focal. If you running with the same loss function(say Focal) again but with a slight change in some parameter, name it Focal2, this decides what name your run is saved with in ~/foundation_model/checkpoints/ folder
- In /home/asavari/foundation_model/run.sh
    - The dataset name being trained can be changed here.
    - The loss function being used can be changed here(the options available are basically yaml file names)

To Begin training:
- Start a tmux session by running ./tmux.sh from ~/foundation_model dir
- 
- Run "./run.sh" from the ~/foundation_model directory

# File Structure - What contains what files
Once the training ends, the visualization. the metrics report and all will be saved in the ~/foundation_model/checkpoints directory

# Evaluation - On the test set
If you wish to run the evaluation, go to ~/foundation_model/eval.sh, change --config-path param to the folder that contains your config file. After training it will mostly be a folder in ~/foudnation_model/checkpoints based on the name you chose for training. 
