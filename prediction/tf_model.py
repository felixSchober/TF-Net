# python imports
from __future__ import division
import math
import time
import os
import os.path
import sys
import logging
from enum import Enum
import errno

# lib imports
import tensorflow as tf
import numpy as np
import colorama

#project inports

logger = logging.getLogger('prediction')

TF_RANDOM_SEED = 42

TF_LOG_CREATE_SUB_DIR = True
TF_LOG_DIR = os.path.join(os.getcwd(), 'log', 'tensorflow')
SUMMARY_EVERY_X_EPOCHS = 1

TF_LAYER = Enum('Layertype', 'Dense Dropout')

def get_weights_variable(dim_x, dim_y, std_dev=1.0, name='weights'):
    weights = tf.Variable(
        tf.truncated_normal([dim_x, dim_y], stddev=std_dev / math.sqrt(float(dim_x))),
        name=name)
    return weights


def get_bias_variable(num_neurons, name='biases'):
    biases = tf.Variable(tf.zeros([num_neurons]), name=name)
    return biases


def get_placeholders(X_feature_vector_length, batch_size, num_classes):
    input_features_placeholders = tf.placeholder(tf.float32, shape=(None, X_feature_vector_length))
    targets_placeholder = tf.placeholder(tf.int32, shape=(None))
    keep_prob_placeholder = tf.placeholder(tf.float32)
    return input_features_placeholders, targets_placeholder, keep_prob_placeholder


def fill_feed_dict(data_set, features_placeholders, targets_placeholders, keep_prob_placeholder, keep_prob, batch_size, shuffle=True):
    features_feed, targets_feed = data_set.next_batch(batch_size, shuffle)

    feed_dict = {
        features_placeholders: features_feed,
        targets_placeholders: targets_feed,
        keep_prob_placeholder: keep_prob
    }
    return feed_dict

def fill_feed_dict_logistic_regression(data_set, features_placeholders, targets_placeholders, batch_size):
    features_feed, targets_feed = data_set.next_batch(batch_size)

    feed_dict = {
        features_placeholders: features_feed,
        targets_placeholders: targets_feed
    }
    return feed_dict

def get_dense_layer(input_tensor, input_shape, layer_size, name, create_summary=True):
    # Layer:
    with tf.variable_scope(name) as scope:
        weights = get_weights_variable(input_shape, layer_size)            
        biases = get_bias_variable(layer_size)
        layer_tensor = tf.nn.relu(tf.matmul(input_tensor, weights) + biases, name=scope.name)
        if create_summary:
            activation_summary(layer_tensor)
    return layer_tensor

def get_dropout_layer(input_tensor, keep_prob_pl, name):
    # Dropout:
    with tf.variable_scope(name) as scope:
        dropout_tensor = tf.nn.dropout(input_tensor, keep_prob_pl)
    return dropout_tensor


def inference(input_shape, output_shape, architecture, features_pl, keep_prob_pl):
        """Build the defect prediction model.

        Args:
            input_shape: Length of the feature vector
            output_shape: num_classes = num_neurons in the last layer
            architecture: List of layers. Format: [(TF_LAYER.Dense|Dropout, SIZE_OF_LAYER, NAME)]
            features_pl: feature placeholder
            keep_prob_pl: dropout keep placeholder

        Returns:
            Logits Tensor.
        """

        # Hidden Layers
        predecessor_tensor = features_pl
        predecessor_shape = input_shape
        for type, size, name in architecture:
            if type == TF_LAYER.Dense:
                layer_tensor = get_dense_layer(predecessor_tensor, predecessor_shape, size, name)
                predecessor_shape = size
            elif type == TF_LAYER.Dropout:
                if keep_prob_pl is None:
                    raise AttributeError('Model contains dropout layer but keep_prob placeholder is not specified.')
                layer_tensor = get_dropout_layer(predecessor_tensor, keep_prob_pl, name)
            predecessor_tensor = layer_tensor            
                    
        # Output:
        with tf.variable_scope('softmax_linear') as scope:
            weights = get_weights_variable(predecessor_shape, output_shape)
            biases = get_bias_variable(output_shape)
            logit_tensor = tf.add(tf.matmul(predecessor_tensor, weights), biases, name=scope.name)
            activation_summary(logit_tensor)
        return logit_tensor


def loss(logit_tensor, targets_pl):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Targets Placeholder. 1-D tensor of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    targets = tf.to_int64(targets_pl)

    # calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit_tensor, targets, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    tf.summary.scalar('loss', cross_entropy_mean)
    return cross_entropy_mean


def evaluate(logit_tensor, targets_pl):
    correct = tf.nn.in_top_k(logit_tensor, targets_pl, 1)

    # get the number of correct entries
    correct_sum = tf.reduce_sum(tf.cast(correct, tf.int32))
    return correct_sum

def accuracy(logits, targets_pl):
    targets = tf.to_int64(targets_pl)
    correct_prediction = tf.equal(tf.arg_max(logits, 1), targets)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'), name='accuracy_mean')
    tf.summary.scalar('accuracy_mean', accuracy)
    return accuracy


def logistic_regression_inference(input_size, output_size, features_pl):
    W = tf.Variable(tf.zeros([input_size, output_size]))
    b = tf.Variable(tf.zeros([output_size]))

    logit = tf.nn.softmax(tf.matmul(features_pl, W) + b)
    activation_summary(logit)
    return logit

def loss_logistic_regression(logit_tensor, targets_pl):
    targets = tf.to_int64(targets_pl)

    cost = tf.reduce_mean(-tf.reduce_sum(targets_pl*tf.log(logit_tensor), reduction_indices=1))
    tf.summary.scalar('cost', cost)
    return cost

def validate_architecture(architecture):
    if architecture is None:
        raise AttributeError('Architecture is None.')
    if len(architecture) == 0:
        raise AttributeError('Architecture does not contain a layer.')

    # validate every layer
    names = []
    for layer in architecture:

        if len(layer) != 3:
            raise AttributeError('Architecture contains invalid entries.')

        type, size, name = layer

        # check type
        if type != TF_LAYER.Dense and type != TF_LAYER.Dropout:
            raise AttributeError('Architecture contains an invalid layer type.')
            
        # check if size is an int
        try:
            size += 1
        except:
            raise AttributeError('Architecture size contains an invalid size.')

        if name == '':
            raise AttributeError('Architecture contains a layer without a name.')

        if name in names:
            raise AttributeError('Architecture contains the same name for two different layers.')
        names.append(name)

        
         

class TensorFlowNet(object):
    """description of class"""

    def __init__(self,
                train_data_set, 
                test_data_set, 
                num_classes, 
                batch_size, 
                initial_learning_rate=0.1, 
                architecture_shape=[(TF_LAYER.Dense, 1024, 'hidden1'), (TF_LAYER.Dense, 64, 'hidden2'), (TF_LAYER.Dropout, 0.6, 'dropout1')], 
                log_dir=TF_LOG_DIR, 
                max_epochs=500,
                num_epochs_per_decay=150,
                learning_rate_decay_factor=0.1,
                model_name=str(int(time.time())),
                early_stopping_epochs = 50):
        self.sess = None
        self.saver = None
        self.train = train_data_set
        self.test = test_data_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        validate_architecture(architecture_shape)
        self.model_architecture = architecture_shape

        self.max_epochs = max_epochs
        self.steps_per_epoch = train_data_set.num_examples // self.batch_size
        self.global_step = 0

        self.initial_learning_rate = initial_learning_rate

        # epochs after which learning rate decays.
        self.num_epochs_per_decay = num_epochs_per_decay
        

        # how much does the learning rate decay after num_epochs_per_decay
        self.learning_rate_decay_factor = learning_rate_decay_factor

        # the last layer of the net used after training for prediction
        self.model = None 

        # Inputs placeholder
        self.features_pl = None

        self.best_train_loss = np.inf
        self.best_train_precission = 0
        self.best_test_loss = np.inf
        self.best_test_precission = 0

        self.last_test_loss_improvement = 0
        self.early_stopping_epochs = early_stopping_epochs

        self.model_name = model_name

        if TF_LOG_CREATE_SUB_DIR:
            self.log_dir = os.path.join(log_dir, self.model_name)
            create_dir_if_necessary(self.log_dir)
        else:
            self.log_dir = log_dir 
        colorama.init()
        

    
    def get_train_op(self, loss_tensor, global_step):

        # decay learning rate based on the number of steps (global_step)
        num_batches_per_epoch = self.train.num_examples / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='learning_rate_decay')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        # apply the gradients to minimize loss.
        # for each minization increment global_step counter
        train_op = optimizer.minimize(loss_tensor, global_step=global_step)

        # add histogram for each trainable variable
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        return train_op

    def get_train_op_logistic_regression(self, loss_tensor, global_step):
        num_batches_per_epoch = self.train.num_examples / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   self.learning_rate_decay_factor,
                                                   staircase=True,
                                                   name='learning_rate_decay')
        tf.summary.scalar('learning_rate', learning_rate)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_tensor)
        return train_op

    

    def do_eval(self, eval_correct_tensor, features_pl, targets_pl, keep_prob_pl, data_set):

        # number of correct predictions
        true_count = 0
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in range(steps_per_epoch):
            feed_dict = fill_feed_dict(data_set, features_pl, targets_pl, keep_prob_pl, 1.0, self.batch_size)
            true_count += self.sess.run(eval_correct_tensor, feed_dict=feed_dict)
        precision = true_count / num_examples
        return num_examples, true_count, precision

    def do_eval_logistic_regression(self, eval_correct_tensor, features_pl, targets_pl, keep_prob_pl, data_set):
        # number of correct predictions
        true_count = 0
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in range(steps_per_epoch):
            feed_dict = fill_feed_dict_logistic_regression(data_set, features_pl, targets_pl, self.batch_size)
            true_count += self.sess.run(eval_correct_tensor, feed_dict=feed_dict)
        precision = true_count / num_examples
        return num_examples, true_count, precision


    

    def run_training(self):
        logger.info('Building NN model. Attributes:')
        logger.debug('\tTrain Feature Shape: {0}'.format(self.train.feature_shape))
        logger.debug('\tTest Feature Shape: {0}'.format(self.test.feature_shape))
        logger.info('\tTrain Zero Error: {0}'.format(self.train.zero_error))
        logger.info('\tTest Zero Error: {0}'.format(self.test.zero_error))
        logger.debug('\tBatch Size: {0}'.format(self.batch_size))
        logger.debug('\tMax Epochs: {0}'.format(self.max_epochs))
        logger.debug('\tNum Classes: {0}'.format(self.num_classes))
        logger.debug('\tInitial Learning rate: {0}'.format(self.initial_learning_rate))
        
        # print model architecture
        logger.debug('Network architecture')
        for type, size, name in self.model_architecture:
            logger.info('\t\t{0} {1}\t: {2}'.format(type, size, name))


        with tf.Graph().as_default():

            tf.set_random_seed(TF_RANDOM_SEED)


            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

            # Generate the input placeholders
            features_pl, targets_pl, keep_prob_pl = get_placeholders(self.train.feature_shape[1], self.batch_size, self.num_classes)
            self.features_pl = features_pl
            self.keep_prob_pl = keep_prob_pl

            # build the model
            try:
                logit_tensor = inference(self.train.feature_shape[1], self.num_classes, self.model_architecture, features_pl, keep_prob_pl)
            except Exception as e:
                logger.exception('Could not create model.')
                raise e
            self.model = logit_tensor

            # add loss tensor to graph
            loss_tensor = loss(logit_tensor, targets_pl)


            # create gradient training op
            train_op = self.get_train_op(loss_tensor, global_step_tensor)

            # add evaluation op for the training and test set.
            eval_correct = evaluate(logit_tensor, targets_pl)
            accuracy_tensor = accuracy(logit_tensor, targets_pl, )

            # Build the summary Tensor based on the TF collection of Summaries.
            summary_tensor = tf.summary.merge_all()

            # add variables initializer
            init = tf.global_variables_initializer()

            # initialize model saver
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            self.sess = tf.Session()

            # initialize a SummaryWriter which writes a log file
            summary_writer_train = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.sess.graph)
            summary_writer_test = tf.summary.FileWriter(os.path.join(self.log_dir, 'test'), self.sess.graph)

            # initialize variables
            self.sess.run(init)

            logger.info('Neural Net is initialized and ready to train.')
            print('\n')
            logger.debug('Step (/100)\tLoss\tDuration') 
            
            print('Epoch\tTrain Loss\tTrain accuracy\t\tTest Loss\tTest accuracy\tDuration')    
            print('=====================================================================================================')      

            # start training    
            start_time = time.time()
            average_train_loss = 0  
            early_stopping = False      
            test_feed_dict = fill_feed_dict(self.test, features_pl, targets_pl, keep_prob_pl, keep_prob=1.0, batch_size=self.test.num_examples, shuffle=False)
            for epoch in range(self.max_epochs):                

                for step in range(self.steps_per_epoch):
                    # fill feed dict with batch
                    train_feed_dict = fill_feed_dict(self.train, features_pl, targets_pl, keep_prob_pl, keep_prob=0.5, batch_size=self.batch_size)

                    # run the model
                    # _: result of train_op (is None)
                    # loss_value: result of loss operation (the actual loss)
                    train_loss_value = -1
                    try:                    
                        _, train_loss_value = self.sess.run(
                            [train_op, loss_tensor],
                            feed_dict=train_feed_dict)
                    except:
                        logger.exception('Could not run train epoch {0} step {1}. Loss Value: {2}'.format(epoch, self.global_step, train_loss_value))

                    assert not np.isnan(train_loss_value), 'Model diverged with loss = NaN'
                    average_train_loss += train_loss_value
                    self.global_step += 1

                    summary_str_train = self.sess.run(summary_tensor, feed_dict=train_feed_dict)
                    summary_str_test = self.sess.run(summary_tensor, feed_dict=test_feed_dict)

                    summary_writer_train.add_summary(summary_str_train, self.global_step)
                    summary_writer_train.flush()
                    summary_writer_test.add_summary(summary_str_test, self.global_step)
                    summary_writer_test.flush()

                # Write summaries SUMMARY_EVERY_X_EPOCHS.
                if epoch % SUMMARY_EVERY_X_EPOCHS == 0:
                    duration = time.time() - start_time
                    start_time = time.time()    


                    # compute detailed stats                    
                    train_feed_dict = fill_feed_dict(self.train, features_pl, targets_pl, keep_prob_pl, keep_prob=1.0, batch_size=self.batch_size, shuffle=False)

                    # don't take the average in the first step
                    if epoch > 0:
                        average_train_loss /= (self.steps_per_epoch * SUMMARY_EVERY_X_EPOCHS)

                    try:
                        train_accuracy_value = self.sess.run([accuracy_tensor], feed_dict=train_feed_dict)[0]
                        test_accuracy_value, test_loss_value = self.sess.run([accuracy_tensor, loss_tensor], feed_dict=test_feed_dict)
                    except:
                        logger.exception('Could not compute train- and test accuracy values in epoch {0}, step {1}.'.format(epoch, self.global_step))
                        train_accuracy_value = test_accuracy_value = -1
                        train_num_examples, train_true_count, train_precision = self.do_eval(eval_correct, features_pl, targets_pl, keep_prob_pl, self.train)
                        test_num_examples, test_true_count, test_precision = self.do_eval(eval_correct, features_pl, targets_pl, keep_prob_pl, self.test)

                        logger.debug('Train: Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(train_num_examples, train_true_count, train_precision))
                        logger.debug('Test: Num examples: {0}\tNum correct: {1}\tPrecision: {2:.4f}'.format(test_num_examples, test_true_count, test_precision))
                    logger.debug('{0}\t\t{1:.4f}\t{2:.5f}'.format(epoch, train_loss_value, duration))                  
                                        
                    # if early stopping is True abort training and write a last summary
                    early_stopping = self.early_stopping(epoch, test_loss_value)

                    
                    # only save checkpoint if the test loss improved
                    if test_accuracy_value > self.best_test_precission:
                        checkpoint_file = os.path.join(self.log_dir, 'model')                    
                        try:
                            self.save_model(checkpoint_file, step)
                        except:
                            logger.exception('Could not save model.')

                    self.print_step_summary_and_update_best_values(epoch, average_train_loss, train_accuracy_value, test_loss_value, test_accuracy_value, duration, colored=True)
                    
                    average_train_loss = 0

                    if early_stopping:
                        print('-----\n\n')
                        logger.info('Early stopping after {0} steps.'.format(epoch))                    
                        break

            logger.info('Training complete.')
            logger.info('Restoring best model.') 
            

            # Restore best model
            ckpt = tf.train.get_checkpoint_state(os.path.join(self.log_dir, 'model'))
            self.saver.restore(self.sess, os.path.join(self.log_dir, 'model'))
            #if ckpt and ckpt.model_checkpoint_path:
            #    print('..')
            #else:
            #    logger.error('Could not restore model. No model found.')

            logger.info('Best Losses: Train {0:.5f} - Test: {1:.5f}'.format(self.best_train_loss, self.best_test_loss)) 
            logger.info('Best Precisions: Train {0:.5f} - Test: {1:.5f}'.format(self.best_train_precission, self.best_test_precission)) 


    def run_training_logistic_regression(self):
        logger.info('Building logistic_regression model. Attributes:')
        logger.info('\tTrain Feature Shape: {0}'.format(self.train.feature_shape))
        logger.info('\tTest Feature Shape: {0}'.format(self.test.feature_shape))
        logger.info('\tTrain Zero Error: {0}'.format(self.train.zero_error))
        logger.info('\tTest Zero Error: {0}'.format(self.test.zero_error))
        logger.info('\tBatch Size: {0}'.format(self.batch_size))
        logger.info('\tMax Steps: {0}'.format(self.max_epochs))
        logger.info('\tNum Classes: {0}'.format(self.num_classes))
        logger.info('\tInitial Learning rate: {0}'.format(self.initial_learning_rate))
        
        with tf.Graph().as_default():

            global_step = tf.Variable(0, trainable=False, name='global_step')

            # Generate the input placeholders
            features_pl, targets_pl, keep_prob_pl = get_placeholders(self.train.feature_shape[1], self.batch_size, self.num_classes)
            self.features_pl = features_pl

            # build the model
            try:
                logit_tensor = logistic_regression_inference(self.train.feature_shape[1], self.num_classes, features_pl)
            except Exception as e:
                logger.exception('Could not create model.')
                raise e
            self.model = logit_tensor

            # add loss tensor to graph
            loss_tensor = loss_logistic_regression(logit_tensor, targets_pl)


            # create gradient training op
            train_op = self.get_train_op_logistic_regression(loss_tensor, global_step)

            # add evaluation op for the training and test set.
            #eval_correct = self.evaluate(logit_tensor, targets_pl)

            # Build the summary Tensor based on the TF collection of Summaries.
            summary_tensor = tf.summary.merge_all()

            # add variables initializer
            init = tf.global_variables_initializer()

            # initialize model saver
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

            self.sess = tf.Session()

            # initialize a SummaryWriter which writes a log file
            summary_writer_train = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.sess.graph)
            summary_writer_test = tf.summary.FileWriter(os.path.join(self.log_dir, 'test'), self.sess.graph)

            # initialize variables
            self.sess.run(init)

            logger.info('Neural Net is initialized and ready to train.')
            print('\n')
            logger.debug('Step (/100)\tLoss\tDuration') 
            
            print('Step (in hundreds)\tTrain Loss\tTrain precission\t\tTest Loss\tTest precission\tDuration')    
            print('=====================================================================================================')      

            # start training    
            start_time = time.time()        
            for step in range(self.max_epochs):                

                # fill feed dict with batch
                train_feed_dict = fill_feed_dict_logistic_regression(self.train, features_pl, targets_pl, batch_size=self.batch_size)
                test_feed_dict = fill_feed_dict_logistic_regression(self.test, features_pl, targets_pl, batch_size=self.batch_size)

                # run the model
                # _: result of train_op (is None)
                # loss_value: result of loss operation (the actual loss)
                train_loss_value = -1
                test_loss_value = -1
                try:
                    _, train_loss_value = self.sess.run(
                        [train_op, loss_tensor],
                        feed_dict=train_feed_dict)
                    test_loss_value = self.sess.run([loss_tensor], feed_dict=test_feed_dict)[0]

                except:
                    logger.exception('Could not run train step {0}. Loss Value: {1}'.format(step, train_loss_value))

                assert not np.isnan(train_loss_value), 'Model diverged with loss = NaN'

                # if early stopping is True abort training and write a last summary
                early_stopping = self.early_stopping(step, test_loss_value)

                # Write summaries
                if step % 100 == 0:
                    duration = time.time() - start_time
                    start_time = time.time()    

                    #train_num_examples, train_true_count, train_precision = self.do_eval_logistic_regression(eval_correct, features_pl, targets_pl, self.train)
                    #test_num_examples, test_true_count, test_precision = self.do_eval_logistic_regression(eval_correct, features_pl, targets_pl, self.test)

                    
                    logger.debug('{0}\t\t{1:.4f}\t{2:.5f}'.format(step/100, train_loss_value, duration))                  
                                        
                    summary_str_train = self.sess.run(summary_tensor, feed_dict=train_feed_dict)
                    summary_str_test = self.sess.run(summary_tensor, feed_dict=test_feed_dict)

                    summary_writer_train.add_summary(summary_str_train, step)
                    summary_writer_train.flush()
                    summary_writer_test.add_summary(summary_str_test, step)
                    summary_writer_test.flush()

                    # only save checkpoint if the test loss improved
                    if self.model_improved(test_loss_value):
                        checkpoint_file = os.path.join(self.log_dir, 'model.ckpt')                    
                        try:
                            self.save_model(checkpoint_file, step)
                        except:
                            logger.exception('Could not save model.')

                    self.print_step_summary_and_update_best_values(step, train_loss_value, -1, test_loss_value, -1, duration, colored=True)

                if early_stopping:
                    print('-----\n\n')
                    logger.info('Early stopping after {0} steps.'.format(step))   
                               
                    break
            logger.info('Training complete.')

            # Restore best model
            ckpt = tf.train.get_checkpoint_state(TF_LOG_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logger.error('Could not restore model. No model found.')

            logger.info('Best Losses: Train {0:.5f} - Test: {1:.5f}'.format(self.best_train_loss, self.best_test_loss)) 
            logger.info('Best Precisions: Train {0:.5f} - Test: {1:.5f}'.format(self.best_train_precission, self.best_test_precission)) 


    def model_improved(self, test_loss_value):
        return test_loss_value < self.best_test_loss


    def early_stopping(self, epoch, test_loss):
        if not self.model_improved(test_loss):
            # when did the model last improve?
            improvement = epoch - self.last_test_loss_improvement
            if improvement > self.early_stopping_epochs:
                return True
        return False


    def predict(self, X):
        X = X.reshape((1,X.shape[0]))
        feed_dict = {self.features_pl: X, self.keep_prob_pl: 1.0}
        softmax = tf.nn.softmax(self.model)
        activations = softmax.eval(session=self.sess, feed_dict=feed_dict)
        y = activations / sum(activations[0])

        # get predicted label
        label = np.argmax(y)
        return label, y[0][label]

    def predict_logistic_regression(self, X):
        X = X.reshape((1,X.shape[0]))
        feed_dict = {self.features_pl: X}
        softmax = tf.nn.softmax(self.model)
        activations = softmax.eval(session=self.sess, feed_dict=feed_dict)
        y = activations / sum(activations[0])

        # get predicted label
        label = np.argmax(y)
        return label, y[0][label]

    def calculate_manual_accuracy(self):
        num_prediction_tests = self.test.num_examples
        X, y = self.test.get_random_elements(-1)
        correct = 0
        for i in range(num_prediction_tests):
            y_hat, prob = self.predict(X[i])
            if y_hat == y[i]:
                correct += 1
                #print('[!] Y: {0} - Y predicted: {1} ({2:.2f})'.format(y[i], y_hat, prob))

        return float(correct) / float(num_prediction_tests)


    def print_step_summary_and_update_best_values(self, epoch, train_loss, train_precission, test_loss, test_precission, duration, colored=True):
        
        # print table header again after every 3000th step
        if epoch % 100 == 0 and epoch > 0:
            print('Epoch\tTrain Loss\tTrain accuracy\t\tTest Loss\tTest accuracy\tDuration')    
           
        tr_l_color = colored_shell_seq('WHITE')
        te_l_color = colored_shell_seq('WHITE')
        tr_p_color = colored_shell_seq('WHITE')
        te_p_color = colored_shell_seq('WHITE')

        if train_loss < self.best_train_loss:
            self.best_train_loss = train_loss
            tr_l_color = colored_shell_seq('GREEN')        

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.last_test_loss_improvement = epoch
            te_l_color = colored_shell_seq('GREEN')

        if train_precission > self.best_train_precission:
            self.best_train_precission = train_precission
            tr_p_color = colored_shell_seq('GREEN')

        if test_precission > self.best_test_precission:
            self.best_test_precission = test_precission
            te_p_color = colored_shell_seq('GREEN')

        train_string = tr_l_color + '{0:.5f}\t\t'.format(train_loss) + tr_p_color + '{0:.2f}%\t\t\t'.format(train_precission*100)
        test_string = te_l_color + '{0:.5f}\t\t'.format(test_loss) + te_p_color + '{0:.2f}%\t\t'.format(test_precission*100)

        print('{0}\t'.format(epoch) + train_string + test_string + colored_shell_seq('WHITE') + '{0:.3f}'.format(duration))
              


    def add_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)

    def save_model(self, checkpoint_file, step):
        if self.saver is None:
            raise Exception('Could not save model because saver is not initialized. Models can only be saved during training. Model dump: ' + str(self))
        save_path = self.saver.save(self.sess, checkpoint_file)       

        
    def load_model(self, file_name):
        if self.saver is None:
            raise Exception('Could not load model because saver is not initialized. Model dump: ' + str(self))
        self.saver.restore(self.sess, file_name)
        logger.info('Model was restored.')

def activation_summary(x):

      """Helper to create summaries for activations.

      Creates a summary that provides a histogram of activations.
      Creates a summary that measures the sparsity of activations.

      Args:
        x: Tensor
      Returns:
        nothing
      """

      tensor_name = x.op.name
      tf.summary.histogram(tensor_name + '/activations', x)
      tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
      #tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
      #tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def add_loss_summaries(loss_tensor):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [loss_tensor])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [loss_tensor]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op



# helper methods
def create_dir_if_necessary(path):
    """ Save way for creating a directory (if it does not exist yet). 
    From http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary

    Keyword arguments:
    path -- path of the dir to check
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def colored_shell_seq(color):
    if color == 'RED':
        return '\033[31m'
    elif color == 'GREEN':
        return '\033[32m'
    elif color == 'WHITE':
        return '\033[37m'
