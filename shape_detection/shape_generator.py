"""Module to generate various randomised shapes.

Module generates shapes with randomised anchor points and colours.

Usage example:
    1. gen = ShapeGenerator(N_x=100, N_y=100, N_channels=3)
    2. image = gen.generate_random(
        colouring=Colouring.SINGLE_CHANNEL,
        shape_type=None
    )
"""


import numpy as np
from enum import Enum
from typing import Union

class Colouring(Enum):
    SINGLE_CHANNEL = 1
    SINGLE_COLOUR = 2
    RANDOM_PIXELS = 3

class ShapeTypes(Enum):
    CIRCLE = 1
    CROSS = 2
    FOUR_CORNERS = 3
    HOURGLASS = 4
    LINE = 5
    PARALLEL_LINES = 6
    TRIANGLE = 7


class ShapeGenerator(object):
    """Generator class to get a follow-up item."""
    # TODO: Make this generator-style?

    def __init__(self, N_x=100, N_y=100, N_channels=3):
        """Sets static values for the images to be created.
        
        Parameters:
            N_x: Number of pixels in image's x-axis.
            N_y: Number of pixels in image's y-axis.
            channels: Number of colour-channels. Three seems like a good number. ;)
        """
        self.N_x = N_x
        self.N_y = N_y
        self.N_channels = N_channels

    def generate_random(self, colouring: Union[Colouring, np.ndarray] = None, shape_type: ShapeTypes = None) -> np.ndarray:
        """Generates a random shape.
        
        Chooses a random shape (unless a shape is specified) with the chosen colouring.
        Returns a numpy array of shape (N_x, N_y, N_channels).

        Parameters:
            colouring: Determines the colouring mode for the image.
                       Colouring.SINGLE_CHANNEL uses only one, randomly chosen single channel with a random value.
                       Colouring.SINGLE_COLOUR randomly generates one multi-channel colour to use.
                       Colouring.RANDOM_PIXELS generates a random colour for each pixel.
                       If given an array of shape (N_channels,), then this colour will be used for all pixels.
                       If None, then a random colouring will be chosen.
            shape_type: Determines the type of the shape. Please check ShapeTypes above to check which shapes
                        are available. They are rather self-explanatory (I hope). If you want to visually inspect
                        them, check out the script `draw_shapes.py` in the same folder.
        """
        pass

    def colour(self, colouring: np.ndarray) -> np.ndarray:
        """Colours a mask.
        
        Takes a 2D-mask of shape (N, M) as input and adds "colours" according to the number of channels.
        Output will be a 3D-array of shape (N, M, C).
        """
        return None

    def generate_circle(self, colouring: np.ndarray) -> np.ndarray:
        return None

    def generate_cross(self, colouring: np.ndarray) -> np.ndarray:
        return None

    def generate_four_corners(self, colouring: np.ndarray) -> np.ndarray:
        return None

    def generate_hourglass(self, colouring: np.ndarray) -> np.ndarray:
        return None

    def generate_line(self, colouring: np.ndarray) -> np.ndarray:
        return None

    def generate_parallel_lines(self, colouring: np.ndarray) -> np.ndarray:
        return None

    def generate_triangle(self, colouring: np.ndarray) -> np.ndarray:
        return None

    @staticmethod
    def scale_translate_and_rotate(image: np.ndarray) -> np.ndarray:
        """Method to randomly scale, translate and rotate an image.
        
        Use this if you feel that the generator-process above is a bit too easy
        for the model (it places the points in fixed coordinates, after all).

        This method is a wrapper around other methods to rescale, translate and rotate a
        shape to have more variety in the output.
        
        Please note that this methods works on the 3D-array, not the 2-D-shape-mask!
        This is to make use of the all the ready-made PIL-functions to manipulate images.
        However, since we want to stay with numpy for as long as possible, we will not
        give back a PIL-image, but the resulting 3D-array.
        """
        return image