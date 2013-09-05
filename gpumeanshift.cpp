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
filter(PyObject *array, int Range = 10, int Spatial = 10)
{
	if(Range < 0 || Spatial < 0)
	{
		PyErr_SetString(PyExc_ValueError, "Spatial and range radius must be greater than 0\n");
		py::throw_error_already_set();	
	}

	if(!PyArray_Check(array))
	{
		PyErr_SetString(PyExc_TypeError, "First argument is not a numpy array.");
		py::throw_error_already_set();	
	}

	int typenum = PyArray_TYPE(array), new_typenum = typenum;

	int type = typenum == NPY_UBYTE ? CV_8U :
		typenum == NPY_USHORT ? CV_16U :
		typenum == NPY_FLOAT ? CV_32F : -1;
	if( type < 0 )
	{
		PyErr_SetString(PyExc_TypeError, "Type is not supported");
		py::throw_error_already_set();	
	}

	int ndims = PyArray_NDIM(array);
	if(ndims >= 5)
	{
		PyErr_SetString(PyExc_TypeError, "dimensions is to high");
		py::throw_error_already_set();	

	}else if(ndims < 3)
	{
		PyErr_SetString(PyExc_TypeError, " dimension to low: need 3 or 4 dims");
		py::throw_error_already_set();	
	}

	cv::Mat processedImage ;
	NDArrayConverter cvt;
	
	cv::Mat src = cvt.toMat(array);

	cv::Mat dst_cpu;

	if(ndims == 4)
		dst_cpu = src.clone();
	else
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

BOOST_PYTHON_MODULE(gpumeanshift)
{
	init();
	py::def("filter", filter );
}
