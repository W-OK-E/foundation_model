When passing your data through binary cross entropy funciton, Binary Cross entropy is meant to work only for 2 choices, so either it expects probability or it expects labels 0,1. So the data type has to be float.

On the other hand, cross entropy is meant for multi-label data so it expects labels to be 0,1,2,3... hence the target variables have to be long format


## Argmax removes gradients
So in Binary class segmentation, the model's predictions were of the size B x 2 x H x W and the target masks were only B x H x W which meant that the predictions would have to be also reduced to that same shape, for which I took argma. But remember that argmax removes the gradients. You can not taks gradients of a function.