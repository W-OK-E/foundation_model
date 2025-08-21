# foundation_model


Dice and Jackard Index for every modality and model training.

A combined dataset training for IDRID and US_Nerve_Segmentation

Share Visual Results

FOR Idrid dataset:

shape = (2848,4288) 
padded_shape = (2944,4352) #Apply requried reflect padding

For US_Nerve_Seg 
shape = (580,420)
padded_shape = (640,512)

These shapes are required coz at each step in EliteNet, the images are downsampled 7 times by factors of 2
Hence whatever the size of the image we need to take it to the nearest multiple of 128.

Dataloader needs to be modified - Needs to be able to handle these two datasets at input with their unique resolution 


Also at the output we can increase the number of classes for datasets which has less number of classes but then this 
becomes trickier as the number of datasets increases. 

Take the smaller resolution

Three Experiments:

1. IDRID
2. US-Nerve Segmentation
3. Combined - One with different sizes | One where US-Nerve Seg is padded upto the size of IDRID Dataset

TODO: Check if it is possible to load different sized images in one single batch.