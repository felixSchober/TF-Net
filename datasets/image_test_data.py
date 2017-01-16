import logging
import os
from os import walk
import os.path

from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from imageflow.imageflow import convert_images, distorted_inputs, inputs
import cv2

from datasets.test_data import DataSet
from helper import equalize_image_size, crop_to_square, check_if_file_exists

logger = logging.getLogger('io')

def to_one_hot(y):
    """Transform multi-class labels to binary labels
        The output of to_one_hot is sometimes referred to by some authors as the
        1-of-K coding scheme.
        Parameters
        ----------
        y : numpy array or sparse matrix of shape (n_samples,) or
            (n_samples, n_classes) Target values. The 2-d matrix should only
            contain 0 and 1, represents multilabel classification. Sparse
            matrix can be CSR, CSC, COO, DOK, or LIL.
        Returns
        -------
        Y : numpy array or CSR matrix of shape [n_samples, n_classes]
            Shape will be [n_samples, 1] for binary problems.
        classes_ : class vector extraceted from y.
        """
    lb = LabelBinarizer()
    lb.fit(y)
    Y = lb.transform(y)
    return (Y.base, lb.classes_)

def load_image(file_name, grayscale):
    if grayscale:
        return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(file_name, cv2.IMREAD_COLOR)

class ImageDataSetLoader(object):

    def __init__(self, 
                 data_set_path,
                 grayscale,
                 image_size,
                 one_hot=True,
                 seed=42,
                 create_vector_mapping=True):
        self.data_set_path = data_set_path

        self.__X = []
        self.__y = []
        self.one_hot = one_hot
        self.grayscale = grayscale
        self.image_size = image_size

        self.data_set_name = data_set_path.split('/')[-1]

        if self.grayscale:
            self.image_size.append(1)
        else:
            self.image_size.append(3)
        self.seed = seed
        self.create_vector_mapping = create_vector_mapping

        # initialized after 'initialize()
        self.initialized = False
        self.num_classes = -1
        self.num_samples = -1
        self.classes = []
        self.file_name_dict = {}

    def load(self):

        # Create list of folders in the root path
        folder_list = []
        try:
            folder_list = next(walk(self.data_set_path))[1]
        except:
            logger.exception("Could not iterate through root folder of dataset. - Path: {0}.".format(self.data_set_path))
        
        # in case the test data folder does not contain folders only images
        if len(folder_list) == 0:
            logger.warning("Unusual dataset. - Path: {0}.".format(self.data_set_path))
            folder_list = [""]   
            
        self.num_classes = len(folder_list)
        self.classes = folder_list     

        logger.debug('Found {0} different classes.'.format(self.num_classes))

        self.file_name_dict = {y_: [] for y_ in self.classes}

        class_token = 0        

        # fill file_name_dict with image filenames
        for y_ in self.file_name_dict.keys():
            try:
                _, _, self.file_name_dict[y_] = walk(os.path.join(self.data_set_path, y_)).__next__()
            except:
                logger.exception("Could not iterate through folder {0}.".format(os.path.join(self.data_set_path, y_)))

            # filter by extension.  Only allow jpg, JPG, Jpg, png, PNG, Png
            class_files = [os.path.join(self.data_set_path, y_, file) for file in self.file_name_dict[y_] if file.endswith(('.jpg','.Jpg','.JPG','.png','.Png','.PNG', '.tif', '.TIF'))]
            self.file_name_dict[y_] = class_files

            # append all class files to the actual test data vectors
            self.__X.extend(class_files)

            # create y depending if one_hot or not
            if self.one_hot:
                y_label = np.zeros((self.num_classes))
                y_label[class_token] = 1

                # repeat y num class_files times so that for each example a row
                # is added
                y_label = np.tile(y_label, [len(class_files), 1])
            else:
                y_label = [class_token] * len(class_files)
            self.__y.extend(y_label)
            class_token += 1

        # convert __y to numpy array
        self.__y = np.array(self.__y)
        self.__X = np.array(self.__X)
        self.num_samples = self.__y.shape[0]
        logger.debug('Data set is initialized with {0} samples.'.format(self.num_samples))
        self.initialized = True


    def get_test_train_split(self, test_ratio=0.2, random_seed=42, stratify=True):
        if not self.initialized:
            raise Exception('Data is not loaded. Call load() first.')

        logger.debug('Generating train/test split with test_size {0} and random_state {1}'.format(test_ratio, random_seed))

        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__y, test_size=test_ratio, random_state=random_seed, stratify=self.__y)
            logger.debug('Stratify: True')
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.__X, self.__y, test_size=test_ratio, random_state=random_seed)
        logger.debug('Testset Split:')
        logger.debug('\tTrain Shape: {0} - {1}'.format(X_train.shape, y_train.shape))
        logger.debug('\tTest Shape: {0} - {1}'.format(X_test.shape, y_test.shape))
        return X_train, X_test, y_train, y_test



class ImageDataSet(DataSet):

    

    def __init__(self, 
                 filenames,
                 targets,                 
                 name,
                 desired_image_size,
                 grayscale,
                 one_hot,
                 data_path):
        super().__init__(features=[],
                                targets=targets,
                                name=name,
                                one_hot=one_hot)  
                
        self.__X = filenames   
        self.grayscale = grayscale     
        self.is_tfrecords_dataset = True
        self.data_set_path = os.path.join(data_path, self.name)

        if desired_image_size[0] != desired_image_size[1]:
            raise AttributeException('Only square image sizes supported.')
           
        self.image_size = desired_image_size
        
        

    def get_random_elements(self, num_elements=1, seed=42):
        if not seed is None:
            np.random.seed(seed=seed)
        
        # return all elements if number of elements is -1
        if num_elements == -1:
            return self.__X, self.__y

        # get indices of elements to sample
        indices = np.random.choice(np.arange(self.__num_examples), size=num_elements, replace=False)

        X_sample = self.__X[indices]
        y_sample = self.__y[indices]

        return X_sample, y_sample

    def get_images(self, start, end):
        filenames = self.__X[start:end]

    def export_to_tfrecords(self):
            
        # check if tfrecord was already written
        if check_if_file_exists(self.data_set_path + '.tfrecords'):
            logger.info('tfrecords were already generated: ' + self.data_set_path)
            return self.data_set_path  
  
        # load images
        logger.info('Loading images.')
        images = []
        for filename in self.__X:
            images.append(load_image(filename, self.grayscale))

        images = self._fit_images_to_square_size(images, self.image_size[0])

        images_np = np.array(images)

        # assume images are all the same size
        reshape_to = (-1, *self.image_size)
        images_np = images_np.reshape(reshape_to)
        
        logger.info('{0} - Image Feature Vector shape: {1}'.format(self.name, images_np.shape))
        
        imageflow.convert_images(images_np, self.targets, self.data_set_path)
        logger.info('Saved .tfrecords in {0}'.format(self.data_set_path + '.tfrecords'))
        return self.data_set_path

    def _fit_images_to_square_size(self, images, side_length):
        image_size = side_length * side_length
        for i in range(len(images)):
            image = images[i]
            h, w = image.shape[:2]
            if h != w:
                image = crop_to_square(image)
                if image is None:
                    logger.warning('Could not crop image to square image. Aspect ratio might be too big or too small')
               
            # reduce / increase size
            images[i] = equalize_image_size(image, image_size)
        return images
    
    def get_distorted_image_tensor(self, batch_size, num_epochs, num_threads=5):
        return distorted_inputs(filename=self.data_set_path + '.tfrecords', 
                                          batch_size=batch_size,
                                          num_epochs=num_epochs,
                                          num_threads=num_threads, 
                                          imshape=self.image_size,
                                          num_examples_per_epoch=128,
                                          flatten=False)

    def get_normal_image_tensor(self, batch_size, num_epochs, num_threads=5):
        return inputs(filename=self.data_set_path + '.tfrecords', 
                                          batch_size=batch_size,
                                          num_epochs=num_epochs,
                                          num_threads=num_threads, 
                                          imshape=self.image_size,
                                          flatten=False)
