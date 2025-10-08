# General Instructions - Training

When training you need to be change the following things:
- In /home/asavari/foundation_model/configs/model/baseline.yaml
    - name - Name it as per the Loss function you are using, If you are using Focal Loss name it Focal. If you running with the same loss function(say Focal) again but with a slight change in some parameter, name it Focal2
- In /home/asavari/foundation_model/run.sh
    - Change the dataset name to the dataset being trained on

To Begin training:
- Run "./run.sh" from the ~/foundation_model directory

# File Structure - What contains what files

Once the training ends, the visualization and everything will be saved in the 

