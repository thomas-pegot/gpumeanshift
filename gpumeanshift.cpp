//#include <Python.h>

#include <boost/python.hpp>

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "conversion.h"


using namespace cv;
using namespace std;
namespace py = boost::python;

PyObject*
filter(PyObject *array, int Range, int Spatial)
{

	cv::Mat processedImage ;
	NDArrayConverter cvt;
	
	cv::Mat src = cvt.toMat(array);

	cv::Mat dst_cpu;
	cv::cvtColor(src, dst_cpu, CV_BGR2BGRA);

	cv::gpu::GpuMat d_cvImage(dst_cpu);
	cv::gpu::GpuMat d_processedImage(dst_cpu);

	cv::gpu::meanShiftFiltering(d_cvImage, d_processedImage, Range, Spatial);

	d_processedImage.download(processedImage);
	PyObject* OutPut =  cvt.toNDArray(dst_cpu);

	return OutPut;
}

static void init()
{
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(filter)
{
	init();
	py::def("filter", filter );
}
