# Load data
# @TODO: sort sequence by length
# @author: _john
# ==============================================================================
import os

import dataloader.utils.constants as constants
from dataloader.cinnamon.versioning.helper import BaseHelper


class SingleLabelLoader(object):
    """Load image data that has only a single label.

    Example: line OCR images.

    # Arguments
        data_folder [str]: the path that contain data
        train [str]: the path to train version file
        val [str]: the path to validation version file
        test [str]: the path to test version file
        helper [Helper object]: the helper that customize the output data
    """

    def __init__(self, data_folder, train, val=None, test=None, helper=None):
        """Initialize the object"""
        # test object
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.idx_train, self.idx_val, self.idx_test = 0, 0, 0

        self.X_train, self.y_train = self.load(train, data_folder)

        if isinstance(val, str):
            self.X_val, self.y_val = self.load(val, data_folder)

        if isinstance(test, str):
            self.X_test, self.y_test = self.load(test, data_folder)

        self.helper = BaseHelper() if helper is None else helper

    def load(self, version_path, data_folder):
        """Load filepath and label from version text file

        # Arguments
            version_path [str]: the path to version file
            data_folder [str]: the path that contain data

        # Returns
            [list of str]: the list of path X
            [list of str]: the list of label y
        """
        X, y = [], []
        with open(version_path, 'r') as f_in:
            for each_line in f_in:
                each_line = each_line.strip()
                components = each_line.split(',')
                path = os.path.join(data_folder, components[0])
                label = ','.join(components[1:])

                X.append(path)
                y.append(label)

        if len(X) != len(y):
            raise ValueError('incompatible X and y amounts {} vs {}'
                .format(len(X), len(y)))

        return X, y

    def get_batch(self, batch_size, mode=constants.TRAIN_MODE, debug=False,
        *args, **kwargs):
        """Get a batch of data

        # Arguments
            batch_size [int]: the batch size
            mode [int]: whether train, validation or test
            debug [bool]: whether to return debug information
        """
        X, y = [], []
        debug_info = []

        while len(X) < batch_size:

            if mode == constants.TRAIN_MODE:
                each_X = self.X_train[self.idx_train]
                each_y = self.y_train[self.idx_train]
                self.idx_train = (self.idx_train + 1) % len(self.X_train)
            elif mode == constants.TEST_MODE:
                each_X = self.X_test[self.idx_test]
                each_y = self.y_test[self.idx_test]
                self.idx_test = (self.idx_test + 1) % len(self.X_test)
            elif mode == constants.VALIDATION_MODE:
                each_X = self.X_val[self.idx_val]
                each_y = self.y_val[self.idx_val]
                self.idx_val = (self.idx_val + 1) % len(self.X_val)
            else:
                raise AttributeError('unknown mode, look at `utils.constants`')

            X.append(self.helper.postprocess_single_X(each_X))
            y.append(self.helper.postprocess_single_y(each_y))
            debug_info.append(each_X)

        if debug:
            return self.helper.postprocess_batch(X, y), debug_info

        return self.helper.postprocess_batch(X, y)

    def get_batch_iter(self, batch_size, mode=constants.TRAIN_MODE, debug=False,
        *args, **kwargs):
        """This is an iteration wrapper on top of `get_batch`. Can refer to
        `get_batch` method to understand about provided options"""

        if mode == constants.TRAIN_MODE:
            self.idx_train = 0
            iterations = len(self.X_train) // batch_size
        elif mode == constants.TEST_MODE:
            self.idx_test = 0
            iterations = len(self.X_test) // batch_size
        elif mode == constants.VALIDATION_MODE:
            self.idx_test = 0
            iterations = len(self.X_val) // batch_size
        else:
            raise AttributeError('unknown mode, look at `utils.constants`')

        for _ in range(iterations):
            yield self.get_batch(batch_size, mode=mode, debug=debug,
                                 *args, **kwargs)
