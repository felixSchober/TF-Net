import os
import os.path
from prediction.tf_model import TensorFlowNet, TF_LAYER
from helper import create_loggers
from examples.cifar10_input import maybe_download_and_extract, distorted_inputs

data_set_path = os.path.join(os.getcwd(), 'data', 'cifar10')

# Download data set
maybe_download_and_extract(data_set_path)

data_set_path = os.path.join(data_set_path, 'cifar-10-batches-bin')

images, labels = distorted_inputs(data_set_path, 100)

print(images)
