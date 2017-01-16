from helper import create_loggers
create_loggers()

from datasets.image_test_data import ImageDataSetLoader, ImageDataSet
from prediction.tf_model import TensorFlowNet, TF_LAYER

loader = ImageDataSetLoader(
    'PATH_TO_DATASET',
    grayscale=True,
    image_size=[28, 28, 1],
    one_hot=False)
loader.load()
X_train, X_test, y_train, y_test = loader.get_test_train_split() 
path = 'PATH_TO_SAVE_TFRECORDS_IN'
train = ImageDataSet(
                    X_train,
                    y_train,
                    name='train',
                    desired_image_size=[28,28,1],
                    grayscale=True, 
                    one_hot=False,
                    data_path=path)
test = ImageDataSet(X_test, 
                    y_test, 
                    name='test',
                    desired_image_size=[28,28,1],
                    grayscale=True, 
                    one_hot=False,
                    data_path=path)

train.export_to_tfrecords()
test.export_to_tfrecords()

net = TensorFlowNet(
    train_data_set=train,
    test_data_set=test, 
    num_classes=train.num_classes, 
    input_shape=[28, 28, 1],
    targets_shape=[-1],
    input_is_image=True,
    batch_size=128, 
    reshape_input_to=[-1, 28, 28, 1],    
    initial_learning_rate=1e-4, 
    architecture_shape=[
        (TF_LAYER.Convolution2D, 'conv1', ([5, 5, 1, 32], [1, 1, 1, 1], 'SAME')), # 5x5 Filters, 32 Features, 1-Stride -> Same output dimension
        (TF_LAYER.MaxPooling, 'pool1', ([1, 2, 2, 1], [1, 2, 2, 1], 'SAME')), # 2x2 max pooling -> Image size is halved (14x14)

        (TF_LAYER.Convolution2D, 'conv2', ([5, 5, 32, 64], [1, 1, 1, 1], 'SAME')), # 5x5 Filters, 64 Features, 1-Stride -> Same output dimension
        (TF_LAYER.MaxPooling, 'pool2', ([1, 2, 2, 1], [1, 2, 2, 1], 'SAME')), # 2x2 max pooling -> Image size is halved (7x7)

        (TF_LAYER.Dense, 'hidden1', 1024)],
    
    max_epochs=500,
    num_epochs_per_decay = 70,
    model_name='Font-Data-Testrun2',
    verbose=False
    )
net.run_training()


    
