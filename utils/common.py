from enum import Enum


# Constant for data
class InputType(Enum):
    gray_scale = 1
    rgb = 2
    binary = 3


ImageMode = {"L": 1,
             "RGB": 2,
             "1": 3,
             1: "L",
             2: "RGB",
             3: "1"}


# Constant for model
class OptimizerType(Enum):
    adam = "adam"
    adadelta = "adadelta"
    rmsprop = "rmsprop"


class FeatureExtractorType(Enum):
    VGG = "VGG"
