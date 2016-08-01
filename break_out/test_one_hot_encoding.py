from unittest import TestCase
from mountain_car import one_hot_encoding
import logging

logging.basicConfig(level=logging.INFO)

class TestMountainCar(TestCase):
    def test_one_hot_encoding(self):
        speed_range = [-0.07, 0.07]
        position_range = [-1.2, 0.6]
        print(one_hot_encoding([0.59, 0.069], position_range, speed_range, 10))

        self.fail()
