# Tensorflow Network Wrapper
This python project should make it easier to build, initialize and train Tensorflow models.

## Input
### Images
Images should have the following format: ```[height, width, channels]```. 

## Architecture Shape Format
The parameter architecture_shape specifies the architecture of the network. It is a list of layer tuples which contain information about the layer. 
The shape of a tuple describing the layer should look like this ```(TF_LAYER.Type, name, (layer_parameters))```.

```TF_LAYER.Type``` can be one of the following types:
1. ```TF_LAYER.Dense``` Dense Layer
2. ```TF_LAYER.Dropout``` Dropout
3. ```TF_LAYER.Convolution2D``` 2D Convolution Layer
4. ```TF_LAYER.MaxPooling``` Max Pooling
5. ```TF_LAYER.Normalization``` Local Response Normalization


```name``` should be a unique layer name.


```(layer_parameters)``` is a tuple of parameters for the respective layer:


- **TF_LAYER.Dense**: 
  - ```layer_parameters[0]```: Number of neurons in layer.



- **TF_LAYER.Dropout**: 
    - ```layer_parameters[0]```: Keep probability [0, 1].

- **TF_LAYER.Convolution2D**: 
  - ```layer_parameters[0]```: Kernel shape. This should be a 1-D list of ints with length 4. ```[5, 5, 1, 32]``` is an example for a small 5x5 kernel where first two dimensions are the patch size (size of the kernel), the next is the number of input channels (either 1 or 3), and the last is the number of output channels.
  - ```layer_parameters[1]```: Stride. This should be a 1-D list of ints with length 4. The stride of the sliding window for each dimension of input. The stride of the sliding window for each dimension of input. The strides list should have the following format ```[1, stride_horizontal, stride_vertical, 1]``` so ```strides[0] = strides[3] = 1```. Usually ```stride_horizontal = stride_vertical```.
  - ```layer_parameters[2]``` (_optional_): Padding. Default is ```SAME```. Put in ```None``` if default. Allowed values are ```"SAME", "VALID"```. 



- **TF_LAYER.MaxPooling**: 
  - ```layer_parameters[0]```: Kernel shape. This should be a 1-D list of ints with length 4. ```[1, 2, 2, 1]``` is a common max pooling operation which subsamples / reduces the input size by two with a 2x2 max pooling area.
  - ```layer_parameters[1]```: Stride. This should be a 1-D list of ints with length 4. The stride of the sliding window for each dimension of input. The strides list should have the following format ```[1, stride_horizontal, stride_vertical, 1]``` so ```strides[0] = strides[3] = 1```. Usually ```stride_horizontal = stride_vertical```.
  - ```layer_parameters[2]``` (_optional_): Padding. Default is ```SAME```. Put in ```None``` if default. Allowed values are ```"SAME", "VALID"```

- **TF_LAYER.Normalization**: 
  - ```layer_parameters[0]```(_optional_): Depth Radius. Defaults to 5. 0-D. Half-width of the 1-D normalization window. Put in ```None``` if default.
  - ```layer_parameters[1]```(_optional_): Bias. Defaults to 1. An offset (usually positive to avoid dividing by 0). Put in ```None``` if default.
  - ```layer_parameters[2]``` (_optional_): Alpha. Defaults to 1. A scale factor, usually positive. Put in ```None``` if default.
  - ```layer_parameters[3]``` (_optional_): Beta. An optional float. Defaults to 0.5. An exponent. Put in ```None``` if default.

### Construction

#### Convolutional Layers
The output shape of a convolutional layer can be calculated by using the kernel shape and the stride.
The size of a kernel does not increase or decrease the image size. However, if *overlap* is specified in the strides parameter (```layer_parameters[1][1] , layer_parameters[1][2]```) the image will increase depending on the overlap. An overlap of 1 horizontally will increase the image size for each convolution. 
For ```"SAME"``` the image dimensions are calculated by the ceiled division of the image dimension and the strides + the padding which looks like this:
```python
w_out = ceil(w_in / stride_horizontal)
h_out = ceil(h_in / stride_vertical)

w_padding = (w_out - 1) * stride_horizontal + kernel_w  - w_in
h_padding = (h_out - 1) * stride_vertical + kernel_h  - h_in

w_out += w_padding
h_out += h_padding
```
For an image of size 28x28, a 5x5 kernel would convolve 9 times (padding = ```"SAME"```) for one row therefore increasing the image size from 28x28 to 45x45 (Padding is 17).

### Example:
TODO

## Tweakable Parameters

### Convolutional Layer
#### 1. Weights
initialization is done with a std_dev of ```std_dev / math.sqrt(float(input_shape[0]))```
Is this the right way to do it for conv layers?

#### 2. Biases
The biases are initialized with ```tf.zeros``` in the [Tutorial](https://www.tensorflow.org/tutorials/deep_cnn/#cifar-10_model) they are initialized with a value of 0.1.

#### 3. Pre Initialization
Because of 2.) pre initialization doesn't make sense here. Test if this makes a difference.

### Normalization
Many parameters that are pre set can be tweaked. See [Documentation](https://www.tensorflow.org/api_docs/python/nn/normalization) and the corresponding [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

## TODOs
- [ ] add weight decay.
- [ ] investigate GPU utilization issue.
- [ ] add additional [pooling types](https://www.tensorflow.org/api_docs/python/nn/pooling).
- [ ] add variable dropout for each dropout layer.
- [ ] add support for variable image size by using random cropping like [here](https://github.com/Jorba123/Deep-Food/blob/master/TumFoodCam/classification/deep/batch_iterators.py).