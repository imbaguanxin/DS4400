# HW2 Xin Guan

## q6:
All my plots and codes are in `q6.ipynb`. open or run it will give out all the results.

## q7:

There are 3 files doing q7: `q7normalize_before_power.ipynb`, `q7power_before_normalize.ipynb`, `q7c.ipynb`.

I posted some question on piazza about the order of normalizing and powering.

I have tried both order since I didn't get promising result from the method that Professor Elhamifar told us to do in class.

In `q7power_before_normalize.ipynb`, I first take powers of the data and then normalize each column which follows that we are told to do in class.
Though I get rather small regression errors, but my plot seems to be problematic. I don't know how to plot the theta.
I just sample from -1 to 1 and apply this vector to the phi function and I get the sampled n+1 dimensional data (with the 1's at the back).
Then apply this matrix to theta.

In `q7power_before_normalize.ipynb`, I first normalize the original data and then take powers of the data.
It was not suggested in class but have better plots. I use the exactly same method to draw the theta as previously described.

In `q7c.ipynb`, the problem c of q7 is written.

I guess there is something wrong with my plotting of theta, but I don't know how to fix it.

I am curious to know which method is better and why. But the only thing that I know now is that they are different since n-power and normalizing is not commutative. 