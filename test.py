

from gpumeanshift import filter
import unittest
import numpy as np
import cv2
from cv2.cv import PyrMeanShiftFiltering

class TestGPUMeanShiftFilter(unittest.TestCase):
	def SetUp(self):
		self.path = "data/star.jpg"
	def test_emptyNDArray(self):
		empty_array = np.array([], dtype = np.uint8)
		self.assertRaises(TypeError, filter, *[empty_array, 12, 12])
	def test_StringInput(self):
		self.assertRaises(TypeError, filter, *['string_input', 12, 12])
	def test_2dimsNDarray(self):
		self.assertRaises(TypeError, filter, *[np.ones((5,5), dtype = np.uint8), 12, 12])
	def test_5dimsNDarray(self):
		self.assertRaises(TypeError, filter, *[np.ones((5,5,5,5,5), dtype = np.uint8), 12, 12])
	def test_MeanShift(self):
		self.path = "data/star.jpg"
		img = cv2.imread(self.path, cv2.IMREAD_COLOR)
		img_cpu = cv2.cv.LoadImage(self.path)
		out_cpu = cv2.cv.CloneImage(img_cpu)
		PyrMeanShiftFiltering(img_cpu, out_cpu, 12, 12)
		out_array_cpu = np.asarray(out_cpu[:,:])
		out_gpu = filter(img, 12, 12)[:,:,0:3]
		self.assertAlmostEqual( out_gpu.all(), out_array_cpu.all())

if( __name__ == "__main__"):
	unittest.main()
