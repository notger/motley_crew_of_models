"""Unit tests for the shape generator.

As most of the tests have to done by visual inspection, tests here 
are a bit bland (after all, the generator is supposed to create human 
interpretable shapes who will be used to train a machine-interpretation-model; 
so we can't test the accuracy of the shapes other than visually)
"""
import unittest
from shape_generator import ShapeGenerator, Colouring


class TestShapeGenerator(unittest.TestCase):

    def test_correct_size_of_generated_images(self):
        N_x, N_y = 128, 314
        generator = ShapeGenerator(N_x, N_y)

        im = generator.generate_random(colouring=Colouring.RANDOM_PIXELS)

        self.assertEqual((N_x, N_y), im.size)
