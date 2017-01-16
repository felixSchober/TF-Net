from enum import Enum
import uuid
import os
import errno
import cv2
from math import sqrt


TF_LAYER = Enum('Layer_type', 'Dense Dropout Convolution2D MaxPooling Normalization')


def get_uuid():
    """ Generates a unique string id."""

    x = uuid.uuid1()
    return str(x)

def colored_shell_seq(color):
    if color == 'RED':
        return '\033[31m'
    elif color == 'GREEN':
        return '\033[32m'
    elif color == 'WHITE':
        return '\033[37m'

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

def get_unique_layer_name(layer_type):
    return str(layer_type) + '_' + get_uuid()


def create_loggers():
    import logging
    # setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='run.log', filemode='w', level=logging.DEBUG)

    # create loggers
    logger_io = logging.getLogger('main')
    logger_io.setLevel(logging.DEBUG)

    logger_prediction = logging.getLogger('prediction')
    logger_prediction.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch_pred = logging.StreamHandler()
    ch_pred.setLevel(logging.INFO)

    ch_io = logging.StreamHandler()
    ch_io.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch_io.setFormatter(formatter)
    ch_pred.setFormatter(formatter)

    # add ch to logger
    logger_io.addHandler(ch_io)
    logger_prediction.addHandler(ch_pred)

def tensor_shape_to_list(tensor_shape):
    output = []
    try:
        for dim in tensor_shape:
            output.append(dim.value)
    except:
        return ['?']
    return output

def crop_to_square(image):
    """ Crops the square window of an image around the center."""

    if image is None:
        return None
    w, h = (image.shape[1], image.shape[0])
    w = float(w)
    h = float(h)

    # only crop images automatically if the aspect ratio is not bigger than 2 or not smaller than 0.5
    aspectRatio = w / h
    if aspectRatio > 3 or aspectRatio < 0.3:
        return None
    if aspectRatio == 1.0:
        return image
    
    # the shortest edge is the edge of our new square. b is the other edge
    a = min(w, h)
    b = max(w, h)

    # get cropping position
    x = (b - a) / 2.0

    # depending which side is longer we have to adjust the points
    # Heigth is longer
    if h > w:
        upperLeft = (0, x)        
    else:
        upperLeft = (x, 0)
    cropW = cropH = a    
    return crop_image(image, upperLeft[0], upperLeft[1], cropW, cropH)

def equalize_image_size(image, size):
    """ Resizes the image to fit the given size."""

    if image is None:
        return None

    # image size
    w, h = (image.shape[1], image.shape[0])
        
    if (w*h) != size:
        # calculate factor so that we can resize the image. The total size of the image (w*h) should be ~ size.
        # w * x * h * x = size
        ls = float(h * w)
        ls = float(size) / ls
        factor = sqrt(ls)
        image = resize_image(image, factor)
    return image

def crop_image(image, x, y, w, h):
    """ Crops an image.

    Keyword arguments:
    image -- image to crop
    x -- upper left x-coordinate
    y -- upper left y-coordinate
    w -- width of the cropping window
    h -- height of the cropping window
    """

    # crop image using np slicing (http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)
    image = image[y: y + h, x: x + w]
    return image

def resize_image(image, resizeFactor):
    """ Resize image by a positive resizeFactor."""

    return cv2.resize(image, (0,0), fx=resizeFactor, fy=resizeFactor)

def check_if_file_exists(path):
    """ Cecks if a file exists."""

    return os.path.exists(path)