1. Streamline the visualization process.
    Requirements for Visualization:
    - Using a fixed Cmap
    - Displaying the image, gt and then all the predictions follow
2. Organize data into split.json files
3. Check whether the transformations are being applied to images and masks both
extract the bg from the gt itself and use that for the prediction 


#After staring cholec:
1. If training loss and visualiations aren't good enough change weightage : 0.75 - Dice and 0.25 for CE 
2. Create and save the visualizations for US_Nerve across the three loss functions
3. IDRID MUlti class and Multi Label with MKConv and Conv2d
#WHEN RUNNING FOR IDRID:
Change all the MKCONV2d to Conv2d