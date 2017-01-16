import logging

from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import LabelBinarizer

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

class DataSet(object):

    def __init__(self, 
                 features,
                 targets,
                 name,
                 one_hot):

        self.name = name
        
        self.__X = features
        self.__y = targets

        self.__epochs_completed = 0
        self.__index_in_epoch = 0

        # will be set in initialize 
        self.__num_examples = targets.shape[0]

        self.one_hot = one_hot

        if one_hot:
            self.__y = to_one_hot(self.__y)[0]

        self.is_tfrecords_dataset = False
        

    @property
    def features(self):
        return self.__X

    @property
    def feature_shape(self):
        return self.__X.shape

    @property
    def targets(self):
        return self.__y

    @property
    def num_examples(self):
        return self.__num_examples

    @property
    def num_classes(self):
        return np.max(self.__y) + 1

    @property
    def epochs_completed(self):
        return self.__epochs_completed

    @property
    def most_frequent_class(self):
        counts = np.bincount(self.__y)
        return np.argmax(counts), np.max(counts)

    @property
    def zero_error(self):
        return 0
        # how many occurences has the most frequent class?
        _, fc = self.most_frequent_class
        return fc / self.__num_examples

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


    def next_batch(self, batch_size, shuffle_data):
        start = self.__index_in_epoch
        self.__index_in_epoch += batch_size

        # Current epoch is finished (used all examples)
        if self.__index_in_epoch > self.__num_examples:
            self.__epochs_completed += 1

            # reshuffle data for next epoch
            if shuffle_data:
                self.__X, self.__y = shuffle(self.__X, self.__y)
            start = 0
            self.__index_in_epoch = batch_size

            # make sure batch size is smaller than the actual number of examples
            assert batch_size <= self.__num_examples
        end = self.__index_in_epoch
        return self.__X[start:end], self.__y[start:end]