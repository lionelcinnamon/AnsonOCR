# Utility to examine data created using versionings
# @author: _john
# ==============================================================================
import os
from collections import defaultdict

import numpy as np


def load_label(version_file):
    """Load the label from version file

    # Arguments
        version_file [str]: the path to version files
    
    # Returns
        [list of str]: list of labels
    """
    labels = []
    with open(version_file, 'r') as f_in:
        for each_line in f_in:
            components = each_line.strip().split(',')
            labels.append(','.join(components[1:]))
    
    return labels


def get_character_counts(labels):
    """Get the character count from labels

    # Arguments
        labels [list of str]: all label text files
    
    # Returns:
        [dict]: the character count, where key is character and value is count
    """
    count = defaultdict(int)
    for each_line in labels:
        for each_character in each_line:
            count[each_character] += 1
    
    return count


def divide_train_val_test(version_train, output_folder, validation=True):
    """Divide the train, validation and test

    There are several schemes when dividing train, validation and test sets:
        - Just mix in and randomized
        - Test and non-test: (1) test contains a lot of images not from the same
            distribution as non-test, (2) non-test will be used also as k-fold
            cross-validation.
        - Test, val and train: (1) test contains a lot of images not from the
            same distribution as val and train, (2) val contains several
            images not in the same distribution with train.

    The test set should be used as a final test set that evaluates model
    generalization performance. If we use that final test set too often,
    we will risk overfit that test set. As a result, we should create our own
    train, validation and test set from the train set.

    In this scheme, we want the resulting test set to contain more images
    that not in the distribution of the training and validation images. And
    since we might want to efficiently optimize the training data, we can use
    cross-validation.

    All files -> Separate test and non-test -> For non-test, create validation.

    # Arguments
        version_train [str]: the path to version train text
        output_folder [str]: the folder to store splitted versions
        validation [bool]: wheter to construct validation

    # Returns
        {dict of train}
        {dict of validation}
        {dict of test}
    """
    filepaths, labels = [], []
    projects_count = defaultdict(int)
    with open(version_train, 'r') as f_in:
        for each_line in f_in:
            each_line = each_line.strip()
            components = each_line.split(',')

            filepaths.append(components[0])
            labels.append(','.join(components[1:]))

            project = components[0].split('/')[0]
            projects_count[project] += 1

    test, non_test = {}, {}
    test_determinant = np.percentile(list(projects_count.values()), 25)
    for filepath, label in zip(filepaths, labels):
        project = filepath.split('/')[0]
        if (projects_count[project] < test_determinant and
            np.random.random() > 0.1):
            test[filepath] = label
        elif (projects_count[project] > test_determinant and
            np.random.random() < 0.05):
            test[filepath] = label
        else:
            non_test[filepath] = label

    print('Dumping out test set...')
    with open(os.path.join(output_folder, 'test.txt'), 'w') as f_out:
        for filepath, label in test.items():
            f_out.write('{},{}\n'.format(filepath, label))

    if not validation:
        print('Dumping out train set...')
        with open(os.path.join(output_folder, 'train.txt'), 'w') as f_out:
            for filepath, label in non_test.items():
                f_out.write('{},{}\n'.format(filepath, label))

        return non_test, test

    train, validation = {}, {}
    for filepath, label in non_test.items():
        if np.random.random() > 0.2:
            train[filepath] = label
        else:
            validation[filepath] = label

    print('Dumping out train set...')
    with open(os.path.join(output_folder, 'train.txt'), 'w') as f_out:
        for filepath, label in train.items():
            f_out.write('{},{}\n'.format(filepath, label))

    print('Dumping out validation set...')
    with open(os.path.join(output_folder, 'validation.txt'), 'w') as f_out:
        for filepath, label in validation.items():
            f_out.write('{},{}\n'.format(filepath, label))

    return train, test, validation
