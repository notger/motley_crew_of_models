"""Module to generate various randomised shapes.

Module generates shapes with randomised anchor points and colours.

Usage example:
    1. gen = ShapeGenerator(N_x=100, N_y=100)
    2. image = gen.generate_random(
        colouring=Colouring.SINGLE_CHANNEL,
        shape_type=None
    )
"""

import random
import numpy as np
from enum import Enum
from typing import Union, Tuple

from PIL import Image, ImageDraw

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
    """Generator class to get a follow-up item.
    
    Please note that how this whole generation works is that it generates a mask/canvas
    with black background, on which the PIL-draw functions will be executed.

    Then we will fill in the colours later, as we want to have very special colouring
    options which are not supported by default (see colouring-function).

    For this reason, for now, only a black background is allowed, though nothing keeps
    you from later adding an inversion function or replacing black with any other colour.
    """
    # TODO: Make this generator-style?
    # TODO: The frequent calls to self.colour_in(im, colour) smell like a generator is in order!

    def __init__(self, N_x: int = 256, N_y: int = 256):
        """Sets static values for the images to be created.
        
        Parameters:
            N_x: Number of pixels in image's x-axis.
            N_y: Number of pixels in image's y-axis.
        """
        self.N_x = N_x
        self.N_y = N_y
        self.background = (0, 0, 0)
        self.shape_drawing_colour = (1, 1, 1)

        # Generate the lookup for the generator-function available:
        self.generator_function_lookup = {
            ShapeTypes.CIRCLE: self.generate_circle,
            ShapeTypes.CROSS: self.generate_cross,
            ShapeTypes.FOUR_CORNERS: self.generate_four_corners,
            ShapeTypes.HOURGLASS: self.generate_hourglass,
            ShapeTypes.LINE: self.generate_line,
            ShapeTypes.PARALLEL_LINES: self.generate_parallel_lines,
            ShapeTypes.TRIANGLE:self.generate_triangle
        }
        # Generate a list to choose randomly from, to not have to do that with every call again:
        self.generator_function_candidates = list(self.generator_function_lookup.values())

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
        # First determine a shape, if we haven't gotten one passed:
        if shape_type is None:
            generator_function = random.choice(self.generator_function_candidates)
        else:
            try:
                generator_function = self.generator_function_lookup[shape_type]
            except:
                raise ValueError(f'Parameter shape_type has to be either None or of ShapeTypes, not {shape_type}')

        # Then determine the colouring (we do this old-school style, as I currently still write in 3.8):
        if colouring == Colouring.SINGLE_CHANNEL:
            colour = np.zeros((3,), dtype=np.uint8)
            colour[np.random.randint(3)] = np.random.randint(255) + 1

        elif colouring == Colouring.SINGLE_COLOUR:
            colour = np.random.randint(256, size=3, dtype=np.uint8)

        elif colouring == Colouring.RANDOM_PIXELS:
            # Please note that we have to generate the pixelwise random colours
            # with flipped axis, so N_y first, then N_x, due to how PIL handles these things:
            colour = np.random.randint(256, size=(self.N_y, self.N_x, 3), dtype=np.uint8)

        elif type(colouring) == np.ndarray:
            colour = colouring

        else:
            raise ValueError(f"Colouring parameter has to be of type Colouring, not {colouring}")

        return generator_function(colour)

    def get_canvas(self) -> Tuple:
        """Creates an image and a draw object to work on."""
        im = Image.new('RGB', (self.N_x, self.N_y), self.background)
        draw = ImageDraw.Draw(im)
        return im, draw

    def colour_in(self, image: Image, colours: np.ndarray) -> np.ndarray:
        """Colours a mask.
        
        Takes an image of shape (N, M, 3) as input and adds "colours".
        
        Parameters:
            image: PILImage-object, which will act as a mask. So it should be binary in nature,
                   containing zeros where no colour should be and ones where colours should be.
            colours: numpy-array either of shape (N, M, 3) or (1, 1, 3) which contains the colour
                     values before masking. Please note that the data type should be uint8, so that
                     the result will be a valid image.
        """
        if colours.dtype == np.uint8:
            return Image.fromarray(
                np.array(image) * colours
            )
        else:
            raise TypeError('Colours array has the wrong data type. Please make sure it is np.uint8.')

    def generate_circle(self, colouring: np.ndarray) -> np.ndarray:
        im, draw = self.get_canvas()

        # Generate a center-point which should lie somewhere in the middle.
        center = (
            np.random.uniform(low=im.size[0]//4, high=(im.size[0] * 3) // 4),
            np.random.uniform(low=im.size[1]//4, high=(im.size[1] * 3) // 4)
        )

        # Determine the max radius:
        max_radius = min(
            center[0], center[1], im.size[0] - center[0], im.size[1] - center[1]
        )

        # Determine the actual radius:
        radius = np.random.uniform(low=max_radius // 4, high=(max_radius * 3) // 4)

        # Draw the circle:
        draw.ellipse(
            (
                (center[0] - radius, center[1] - radius),
                (center[0] + radius, center[1] + radius)
            ), fill=self.background, outline=self.shape_drawing_colour
        )

        return self.colour_in(im, colouring)

    @staticmethod
    def four_points(im: Image):
        """Creates four coordinates for four points, where one lies in each quadrant.
        
        Parameters:
            im: PIL.Image which determines the quadrants.

        Returns:
            Tuple in the ordering: top_left, top_right, bottom_left, bottom_right
        """
        # Create the coordinates for four points, one in each quadrant:
        low_x = np.random.uniform(low=0, high=im.size[0]//2 - 1, size=2)
        high_x = np.random.uniform(low=im.size[0]//2, high=im.size[0] - 1, size=2)
        low_y = np.random.uniform(low=0, high=im.size[1]//2 - 1, size=2)
        high_y = np.random.uniform(low=im.size[1]//2, high=im.size[1] - 1, size=2)

        return (low_x[0], low_y[0]), (high_x[0], low_y[1]), (low_x[1], high_y[0]), (high_x[1], high_y[1])


    def generate_cross(self, colouring: np.ndarray) -> np.ndarray:
        im, draw = self.get_canvas()

        top_left, top_right, bottom_left, bottom_right = self.four_points(im)

        draw.line((top_left, bottom_right), width=1, fill=self.shape_drawing_colour)
        draw.line((top_right, bottom_left), width=1, fill=self.shape_drawing_colour)

        return self.colour_in(im, colouring)

    def generate_four_corners(self, colouring: np.ndarray) -> np.ndarray:
        im, draw = self.get_canvas()

        top_left, top_right, bottom_left, bottom_right = self.four_points(im)

        # Draw the four lines:
        draw.line((top_left, top_right), width=1, fill=self.shape_drawing_colour)
        draw.line((top_right, bottom_right), width=1, fill=self.shape_drawing_colour)
        draw.line((bottom_right, bottom_left), width=1, fill=self.shape_drawing_colour)
        draw.line((bottom_left, top_left), width=1, fill=self.shape_drawing_colour)

        return self.colour_in(im, colouring)

    def generate_hourglass(self, colouring: np.ndarray) -> np.ndarray:
        im, draw = self.get_canvas()

        top_left, top_right, bottom_left, bottom_right = self.four_points(im)

        # Draw the four lines:
        draw.line((top_left, bottom_right), width=1, fill=self.shape_drawing_colour)
        draw.line((top_left, top_right), width=1, fill=self.shape_drawing_colour)
        draw.line((top_right, bottom_left), width=1, fill=self.shape_drawing_colour)
        draw.line((bottom_left, bottom_right), width=1, fill=self.shape_drawing_colour)

        return self.colour_in(im, colouring)

    def generate_line(self, colouring: np.ndarray) -> np.ndarray:
        im, draw = self.get_canvas()

        x_coords = np.random.uniform(low=0, high=im.size[0], size=(2))
        y_coords = np.random.uniform(low=0, high=im.size[1], size=(2))

        draw.line(
            (
                (x_coords[0], y_coords[0]),
                (x_coords[1], y_coords[1])
            ),
            width=1, fill=self.shape_drawing_colour
        )

        return self.colour_in(im, colouring)

    def generate_parallel_lines(self, colouring: np.ndarray) -> np.ndarray:
        im, draw = self.get_canvas()

        x_coords = np.random.uniform(low=im.size[0]//10, high=(im.size[0] * 9) // 10, size=(2))
        y_coords = np.random.uniform(low=im.size[1]//10, high=(im.size[1] * 9) // 10, size=(2))

        x_translation = np.random.uniform(low=im.size[0]//20, high=im.size[0]//10)
        y_translation = np.random.uniform(low=im.size[1]//20, high=im.size[1]//10)

        draw.line(
            (
                (x_coords[0], y_coords[0]),
                (x_coords[1], y_coords[1])
            ),
            width=1, fill=self.shape_drawing_colour
        )
        draw.line(
            (
                (x_coords[0] + x_translation, y_coords[0] + y_translation),
                (x_coords[1] + x_translation, y_coords[1] + y_translation)
            ),
            width=1, fill=self.shape_drawing_colour
        )

        return self.colour_in(im, colouring)

    def generate_triangle(self, colouring: np.ndarray) -> np.ndarray:
        im, draw =self.get_canvas()

        points = self.four_points(im)
        choice = np.random.choice(list(range(4)), size=3, replace=False)
        triangle_points = tuple(points[c] for c in choice)

        draw.polygon(triangle_points, outline=self.shape_drawing_colour, fill=self.background)

        return self.colour_in(im, colouring)

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
        # TODO: Shouldn't there be some code here that actually does something? ;)
        return image