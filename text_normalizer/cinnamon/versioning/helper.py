# Common and base helper to load data from version mini-system
# @author: _john
# ==============================================================================
import cv2


class BaseHelper(object):

    def postprocess_single_X(self, *args, **kwargs):
        """Post-process X"""
        if len(args) == 1:
            return args[0]
        else:
            return args
    
    def postprocess_single_y(self, *args, **kwargs):
        """Post-process y"""
        if len(args) == 1:
            return args[0]
        else:
            return args
    
    def postprocess_batch(self, *args, **kwargs):
        """Post-process whole batch"""
        if len(args) == 1:
            return args[0]
        else:
            return args


class SingleLabelImageBaseHelper(object):

    def postprocess_single_X(self, image):
        """Post-process X. Basically load the image
        
        # Arguments
            image [str]: the path to image
        
        # Returns
            [np array]: the loaded image
        """
        return cv2.imread(image, cv2.IMREAD_GRAYSCALE)
