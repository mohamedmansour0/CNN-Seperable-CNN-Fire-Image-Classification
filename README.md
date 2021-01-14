# CNN-Seperable-CNN-Fire-Image-Classification
Analyzing Fire and Smoke Images by Different Convolutional Neural Network Architectures

# Convolutional Neural Network (CNN.ipnyb)
Convolution neural networks normally use a kernel that has a depth identical to the input,
suppose an image of a size 12 x 12 x 3, subjected to a kernel of size 5 x 5 x 3 that’s 
performing scalar matrix multiplications equivalent to the size of the kernel on every stride,
resulting in 1 number each time.

After going through the kernel over the whole image layers, the resulting matrix will become
a reduced 8 x 8, where 8 = 12 - 5 + 1. Noticing that the resulting matrix has only 1 channel
then we will need 256 number of kernels to get 256 number of channels.
 
In this case of using the conventional convolutional neural network the total number of
multiplication for 256 kernels of dimensions 5 x 5 x 3 that move on an image of 12 x 12 x 3
times. That’s 256 x 3 x 52 x 82 = 1,228,800 multiplications 
This large number with regards to a multi-layer and a multi-epochs architecture for a significantly
large datasets can be very expensive computationally as it’s time complexity will grow exponentially
with respect to the size of the kernel and the image.

# Separable Convolutional Neural Network (SCNN.ipnyb)
The depthwise separable convolution separates the convolution process into two parts: a depthwise 
convolution and a pointwise convolution. Depthwise convolution step has a kernel that consists of 3
separate slices (or same as the depth size of the input), each kernel iterates on only one channel 
of the input resulting in a stacked 8 x 8 x 3 for a structure similar to the previously presented
example of 12 x 12 x 3 image and a kernel of 5 x 5 x 3 size.

Then the intermediate output of the size 8 x 8 x 3 is then subjected to the pointwise convolution
process, it’s mission is to increase the number of its output channels, this is done by using  a
1x1x3 kernel to iterate through every point of the intermediate network,  and after a scalar matrix
multiplication process each kernel will result an 8 x 8 matrix. As figure 5 below 256 kernels were
used to create an output with the desired 256 channels.

The separable convolution complexity is going to be a summation of both of its processes. In the
depthwise convolution, the number of multiplications will be 3 x 52 x 82 = 4,800. Whereas in the
pointwise convolution, we have 3 x 256 x 82 = 49,152 multiplications. Adding them up together will
result in a total of 53,952 multiplications. Which is a fraction of the 1,228,800 multiplications for 
the traditional convolutional neural network method by over 22 times.

The difference between the complexity of each architecture is affected by the square of the kernel size,
and the number of channels in each layer as the following equation.

C = (A/n) + (A/K2)

Whereas A is the number of multiplications by the traditional convolutional neural network with the same
parameters, C  is the number of multiplications by the separable convolutional neural network, K is the 
kernel size and n is the number of channels.
