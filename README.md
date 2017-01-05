# Tensorflow Network Wrapper
This python project should make it easier to build, initialize and train Tensorflow models.

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
```TF_LAYER.Dense```: 
1. ```layer_parameters[0]```: Number of neurons in layer.

```TF_LAYER.Dropout```: 
1. ```layer_parameters[0]```: Keep probability [0, 1].

```TF_LAYER.Convolution2D```: 
1. ```layer_parameters[0]```: Kernel shape. This should be a 1-D list of ints with length 4.
2. ```layer_parameters[1]```: Stride. This should be a 1-D list of ints with length 4. The stride of the sliding window for each dimension of input.
3. ```layer_parameters[2]``` (_optional_): Padding. Default is SAME. Put in None if default.

```TF_LAYER.MaxPooling```: 
1. ```layer_parameters[0]```: Kernel shape. This should be a 1-D list of ints with length 4.
2. ```layer_parameters[1]```: Stride. This should be a 1-D list of ints with length 4. The stride of the sliding window for each dimension of input.
3. ```layer_parameters[2]``` (_optional_): Padding. Default is SAME. Put in None if default.

```TF_LAYER.Normalization```: 
1. ```layer_parameters[0]```(_optional_): Depth Radius. Defaults to 5. 0-D. Half-width of the 1-D normalization window. Put in None if default.
2. ```layer_parameters[1]```(_optional_): Bias. Defaults to 1. An offset (usually positive to avoid dividing by 0). Put in None if default.
3. ```layer_parameters[2]``` (_optional_): Alpha. Defaults to 1. A scale factor, usually positive. Put in None if default.
4. ```layer_parameters[3]``` (_optional_): Beta. An optional float. Defaults to 0.5. An exponent. Put in None if default.


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
- [ ] add weight decay
- [ ] investigate GPU utilization issue
- [ ] add additional [pooling types](https://www.tensorflow.org/api_docs/python/nn/pooling).
- [ ] add variable dropout