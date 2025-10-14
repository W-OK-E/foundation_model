# General Instructions - Training

When training you need to be change the following things:
- In /home/asavari/foundation_model/configs/model/baseline.yaml
    - name - Name it as per the Loss function you are using, If you are using Focal Loss name it Focal. If you running with the same loss function(say Focal) again but with a slight change in some parameter, name it Focal2, this decides what name your run is saved with in ~/foundation_model/checkpoints/ folder
- In /home/asavari/foundation_model/run.sh
    - The dataset name being trained can be changed here.
    - The loss function being used can be changed here(the options available are basically yaml file names)


To Begin training:
- Start a tmux session by running ./tmux.sh from ~/foundation_model dir
-   If you are training multiple experiments the GPUs can be busy, but since you have two GPUs make use of the fact in the following ways:
    - `/home/asavari/foundation_model/configs/accelerator/gpu.yaml` has device:[0] you can change that to decide on what GPU the experiment runs(You have 0,1)
    - So if it's not possible to fit two experiments on a single GPU, change the device param in gpu.yaml to utilise the free GPU
- Run "./run.sh" from the ~/foundation_model directory

# File Structure - What contains what files
Once the training ends, the visualization. the metrics report and all will be saved in the ~/foundation_model/checkpoints directory

# Evaluation - On the test set
If you wish to run the evaluation, go to ~/foundation_model/eval.sh, change --config-path param to the folder that contains your config file. After training it will mostly be a folder in ~/foudnation_model/checkpoints based on the name you chose for training. 


# The Difference between Multi-Class and Multi-Label Classification(in terms of shape of the target values)
