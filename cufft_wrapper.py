#!/usr/local/epd/bin/python
#-------------------------------------------fft CUDA wrapper---------------------------

import numpy as np
import sys
import ctypes

CUFFT_Z2Z = 0x69
CUFFT_C2C = 0x29

CUFFT_FORWARD = -1
CUFFT_INVERSE = 1

# General CUFFT error:
class cufftError(Exception):
    """CUFFT error"""
    pass
# Exceptions corresponding to different CUFFT errors:
class cufftInvalidPlan(cufftError):
    """CUFFT was passed an invalid plan handle."""
    pass

class cufftAllocFailed(cufftError):
    """CUFFT failed to allocate GPU memory."""
    pass

class cufftInvalidType(cufftError):
    """The user requested an unsupported type."""
    pass

class cufftInvalidValue(cufftError):
    """The user specified a bad memory pointer."""
    pass

class cufftInternalError(cufftError):
    """Internal driver error."""
    pass

class cufftExecFailed(cufftError):
    """CUFFT failed to execute an FFT on the GPU."""
    pass

class cufftSetupFailed(cufftError):
    """The CUFFT library failed to initialize."""
    pass

class cufftInvalidSize(cufftError):
    """The user specified an unsupported FFT size."""
    pass

class cufftUnalignedData(cufftError):
    """Input or output does not satisfy texture alignment requirements."""
    pass

cufftExceptions = {
    0x1: cufftInvalidPlan,
    0x2: cufftAllocFailed,
    0x3: cufftInvalidType,
    0x4: cufftInvalidValue,
    0x5: cufftInternalError,
    0x6: cufftExecFailed,
    0x7: cufftSetupFailed,
    0x8: cufftInvalidSize,
    0x9: cufftUnalignedData
    }

def cufftCheckStatus(status):
	"""Raise an exception if the specified CUBLAS status is an error."""
	
	if status != 0:
		try:
			raise cufftExceptions[status]
		except KeyError:
			raise cufftError

#----------------------------------------------------------------------------

_libcufft = ctypes.cdll.LoadLibrary('libcufft.so')

# Execution
_libcufft.cufftExecZ2Z.restype = int
_libcufft.cufftExecZ2Z.argtypes = [ctypes.c_uint,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int]

def cufftExecZ2Z(plan, idata, odata, direction):
	"""Execute double precision complex-to-complex transform plan as
	specified by `direction`."""
	status = _libcufft.cufftExecZ2Z(plan, idata, odata, direction)
	cufftCheckStatus(status)

_libcufft.cufftExecC2C.restype = int
_libcufft.cufftExecC2C.argtypes = [ctypes.c_uint,
                                   ctypes.c_void_p,
                                   ctypes.c_void_p,
                                   ctypes.c_int]

def cufftExecC2C(plan, idata, odata, direction):
	"""Execute single precision complex-to-complex transform plan as
	specified by `direction`."""
	status = _libcufft.cufftExecC2C(plan, idata, odata, direction)
	cufftCheckStatus(status)


	
# Destroy Plan

_libcufft.cufftDestroy.restype = int
_libcufft.cufftDestroy.argtypes = [ctypes.c_uint]
def cufftDestroy(plan):
	"""Destroy FFT plan."""
	status = _libcufft.cufftDestroy(plan)
	cufftCheckStatus(status)

# Plan for multiple dimensions

_libcufft.cufftPlanMany.restype = int
_libcufft.cufftPlanMany.argtypes = [ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_void_p,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int,
                                    ctypes.c_int]  
                               
def cufftPlanMany(rank, n, 
                  inembed, istride, idist, 
                  onembed, ostride, odist, fft_type, batch):
    """Create batched FFT plan configuration."""

    plan = ctypes.c_uint()
    status = _libcufft.cufftPlanMany(ctypes.byref(plan), rank, n,
                                     inembed, istride, idist, 
                                     onembed, ostride, odist, 
                                     fft_type, batch)
    cufftCheckStatus(status)
    return plan

# Class plan double precision
#........................... 1D,2D,3D fft...........................................

class Plan_Z2Z:
	"""
	CUFFT plan class.

	This class represents an FFT  plan for CUFFT for complex double precission Z2Z

	Parameters
	----------
	shape : ntuple 
	batch : int
        Number of FFTs to configure in parallel (default is 1).
	"""
	def __init__(self, shape, batch=1):
		self.shape = shape  
		self.batch = batch
		self.fft_type = CUFFT_Z2Z

		if len(self.shape) > 0:
			n = np.asarray(shape, np.int32)
			rank = len(shape)
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              None, 1, 0, None, 1, 0,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform size')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass

#

class Plan_Z2Z_2D_Axis0:
	"""
	fft in axis 0 for a 2D array
	Parameters
	----------
	shape : ntuple of four elements
	"""
	def __init__(self, shape):  
		self.batch = shape[1]
		self.fft_type = CUFFT_Z2Z

		if len(shape) == 2:
			n = np.array([ shape[0] ])
			stride = shape[1]           # distance jump between two elements in the same series
			idist  = 1                  # distance jump between two consecutive batches

			inembed = np.array( [shape[0],stride] ) 
			onembed = np.array( [shape[0],stride] ) 

			rank = 1
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              inembed.ctypes.data, stride , idist ,  onembed.ctypes.data, stride, idist,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform dimension')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass


class Plan_Z2Z_2D_Axis1:
	"""
	fft in axis 1 for a 2D array
	Parameters
	----------
	shape : ntuple of four elements
	"""
	def __init__(self, shape):  
		self.batch = shape[0]
		self.fft_type = CUFFT_Z2Z

		if len(shape) == 2:
			n = np.array([ shape[1] ])
			stride = 1                  # distance jump between two elements in the same series
			idist  = shape[1]	    # distance jump between two consecutive batches

			inembed = np.array( shape ) 
			onembed = np.array( shape ) 

			rank = 1
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              inembed.ctypes.data, stride , idist ,  onembed.ctypes.data, stride, idist,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform dimension')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass




#=========================== single precision ======================================
#........................... 1D,2D,3D fft...........................................

class Plan_C2C:
	"""
	CUFFT plan class.

	This class represents an FFT  plan for CUFFT for complex single precission Z2Z

	Parameters
	----------
	shape : ntuple 
	batch : int
        Number of FFTs to configure in parallel (default is 1).
	"""
	def __init__(self, shape, batch=1):
		self.shape = shape  
		self.batch = batch
		self.fft_type = CUFFT_C2C

		if len(self.shape) > 0:
			n = np.asarray(shape, np.int32)
			rank = len(shape)
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              None, 1, 0, None, 1, 0,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform size')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass

			
#............................... 4D fft	..........................................
class Plan_C2C_4D_Axis123:
	"""
	4D fft in axis 1,2,3
	Parameters
	----------
	shape : ntuple of four elements
	"""
	def __init__(self, shape):
		self.shape3D = (shape[1],shape[2],shape[3])  
		self.batch = shape[0]
		self.fft_type = CUFFT_C2C

		if len(shape) == 4:
			n = np.asarray(self.shape3D, np.int32)
			rank = 3
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              None, 1, 0, None, 1, 0,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform size')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass

class Plan_C2C_4D_Axis0:
	"""
	4D fft in axis 0
	Parameters
	----------
	shape : ntuple of four elements
	"""
	def __init__(self, shape):  
		self.batch = shape[1]*shape[2]*shape[3]
		self.fft_type = CUFFT_C2C

		if len(shape) == 4:
			n = np.array([ shape[0] ])
			stride = shape[1]*shape[2]*shape[3]     # distance jump between two elements in the same series
			idist  = 1                              # distance jump between two consecutive batches

			inembed = np.array( [shape[0],stride] ) 
			onembed = np.array( [shape[0],stride] ) 

			rank = 1
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              inembed.ctypes.data, stride , idist ,  onembed.ctypes.data, stride, idist,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform dimension')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass

class Plan_C2C_4D_Axis3:
	"""
	4D fft in axis 3
	Parameters
	----------
	shape : ntuple of four elements
	"""
	def __init__(self, shape):  
		self.batch = shape[0]*shape[1]*shape[2]
		self.fft_type = CUFFT_C2C

		if len(shape) == 4:
			n = np.array([ shape[3] ])
			stride = 1                  # distance jump between two elements in the same series
			idist  = shape[3]	    # distance jump between two consecutive batches

			inembed = np.array( shape ) 
			onembed = np.array( shape ) 

			rank = 1
			self.handle = cufftPlanMany(rank, n.ctypes.data,
                                              inembed.ctypes.data, stride , idist ,  onembed.ctypes.data, stride, idist,
                                              self.fft_type, self.batch)
		else:
			raise ValueError('invalid transform dimension')

	def __del__(self):
        # Don't complain if handle destruction fails because the plan
        # may have already been cleaned up:
		try:
			cufft.cufftDestroy(self.handle)
		except:
			pass


#===========================================================================

#....................... Double precision ..................................

def fft_Z2Z(x_input_gpu, y_output_gpu, plan ):
		cufftExecZ2Z(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_FORWARD)

def ifft_Z2Z(x_input_gpu, y_output_gpu, plan ):
		cufftExecZ2Z(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_INVERSE)
		#scale = x_input_gpu.size/plan.batch
		#y_output_gpu.gpudata = (y_output_gpu/np.cast[y_output_gpu.dtype](scale)).gpudata


#..................... Single precision ....................................

def fft_C2C(x_input_gpu, y_output_gpu, plan ):
		cufftExecC2C(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_FORWARD)

def ifft_C2C(x_input_gpu, y_output_gpu, plan ):
		cufftExecC2C(plan.handle, int(x_input_gpu.gpudata), int(y_output_gpu.gpudata), CUFFT_INVERSE)
#		scale = x_input_gpu.size/plan.batch
#		y_output_gpu.gpudata = (y_output_gpu/np.cast[y_output_gpu.dtype](scale)).gpudata

#------------------------------------------------------------------------------------



