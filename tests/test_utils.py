# import pytest
import numpy as np
import timewizard.util as twu
import unittest


class Testtwutils(unittest.TestCase):
    def test_check_symmetric(self):
        self.assertTrue(twu.check_symmetric(np.zeros((5, 5))))
        self.assertTrue(twu.check_symmetric(np.eye(5)))
        self.assertFalse(twu.check_symmetric(np.arange(25).reshape((5, 5))))

    def test_to_np_array(self):
        a = ["a", "b", "c"]
        b = [1, 2, 3]

        a1 = twu.castnp(a)  # note unpacking of 1-item tuple
        a2 = np.array(a)
        self.assertTrue(np.all(a1 == a2))

        a1, b1 = twu.castnp(a, b)
        a2, b2 = (np.array(a), np.array(b))
        self.assertTrue(np.all(a1 == a2))
        self.assertTrue(np.all(b1 == b2))

        (a1,) = twu.castnp(2)
        a2 = np.array([2])
        self.assertEqual(a1, a2)

    def test_issorted(self):
        self.assertTrue(twu.issorted(np.arange(10)))
        self.assertTrue(twu.issorted(np.array([0, 1, 5, 10, 10.1, np.inf])))
        self.assertFalse(
            twu.issorted(np.array([0, 1, np.nan, 2]))
        )  # nans are neither greater than or less than other numbers
        self.assertFalse(twu.issorted(np.arange(10)[::-1]))
