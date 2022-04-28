"""Unit tests for the shape generator.

As most of the tests have to done by visual inspection, tests here 
are a bit bland (after all, the generator is supposed to create human 
interpretable shapes who will be used to train a machine-interpretation-model; 
so we can't test the accuracy of the shapes other than visually)
"""
import unittest
from shape_generator import ShapeGenerator