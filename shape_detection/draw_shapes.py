"""Helper script to draw shapes and visually inspect them."""
import numpy as np
from shape_generator import ShapeGenerator, Colouring


if __name__ == '__main__':
    s = ShapeGenerator()

    #im = s.generate_parallel_lines(
    #    np.asarray((150, 50, 0), dtype=np.uint8)
    #
    im, shape_type = s.generate_random(colouring=Colouring.RANDOM_PIXELS)
    print(f'Generated {shape_type}')
    im.show()
