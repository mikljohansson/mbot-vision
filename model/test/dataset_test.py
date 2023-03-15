import unittest

from dataset import is_next_frame, StridedSampler


class TestDataset(unittest.TestCase):
    def test_strided_sampler(self):
        self.assertEquals(list(StridedSampler(list(range(12)), 4)), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])

    def test_is_next_frame(self):
        self.assertTrue(is_next_frame('test/abc_def_0001.png', 'test/abc_def_0002.png'))
        self.assertFalse(is_next_frame('test/abc_def_0001.png', 'test/abc_def_0003.png'))
        self.assertFalse(is_next_frame('test/abc_foo_0001.png', 'test/abc_def_0002.png'))

if __name__ == '__main__':
    unittest.main()
