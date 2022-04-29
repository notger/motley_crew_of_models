"""Helper script to draw shapes and visually inspect them."""
import numpy as np
from shape_generator import ShapeGenerator

s = ShapeGenerator()

im = s.generate_four_corners(
    np.asarray((150, 50, 0), dtype=np.uint8)
)
im.show()