from tensorflow.examples.tutorials.mnist import input_data
from prediction.tf_model import TensorFlowNet, TF_LAYER
from helper import create_loggers
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

create_loggers()
net = TensorFlowNet(
    train_data_set=mnist.train,
    test_data_set=mnist.test, 
    num_classes=10, 
    input_shape=784,
    targets_shape=[-1, 10],
    input_is_image=True,
    batch_size=100, 
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
    model_name='Categorical-features'

    )
net.run_training()


#self,
                #train_data_set, 
                #test_data_set, 
                #num_classes, 
                #input_shape,                
                #input_is_image,
                #batch_size, 
                #reshape_input_to=None,
                #initial_learning_rate=0.1, 
                #architecture_shape=[(TF_LAYER.Dense, 1024, 'hidden1'), (TF_LAYER.Dense, 64, 'hidden2'), (TF_LAYER.Dropout, 0.6, 'dropout1')], 
                #log_dir=TF_LOG_DIR, 
                #max_epochs=500,
                #num_epochs_per_decay=150,
                #learning_rate_decay_factor=0.1,
                #model_name=str(int(time.time())),
                #early_stopping_epochs = 50):