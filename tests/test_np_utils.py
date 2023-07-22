# import pytest
import numpy as np
import moseq_fo as mf
import moseq_fo.util.np_utils as npu
import unittest
import unittest
import pdb


class TestNpUtils(unittest.TestCase):
    def test_check_symmetric(self):
        self.assertTrue(npu.check_symmetric(np.zeros((5, 5))))
        self.assertTrue(npu.check_symmetric(np.eye(5)))
        self.assertFalse(npu.check_symmetric(np.arange(25).reshape((5, 5))))

    def test_to_np_array(self):
        a = ["a", "b", "c"]
        b = [1, 2, 3]

        (a1,) = npu.castnp(a)  # note unpacking of 1-item tuple
        a2 = np.array(a)
        self.assertTrue(np.all(a1 == a2))

        a1, b1 = npu.castnp(a, b)
        a2, b2 = (np.array(a), np.array(b))
        self.assertTrue(np.all(a1 == a2))
        self.assertTrue(np.all(b1 == b2))

        (a1,) = npu.castnp(2)
        a2 = np.array([2])
        self.assertEqual(a1, a2)

    def test_issorted(self):
        self.assertTrue(npu.issorted(np.arange(10)))
        self.assertTrue(npu.issorted(np.array([0, 1, 5, 10, 10.1, np.inf])))
        self.assertFalse(
            npu.issorted(np.array([0, 1, np.nan, 2]))
        )  # nans are neither greater than or less than other numbers
        self.assertFalse(npu.issorted(np.arange(10)[::-1]))
