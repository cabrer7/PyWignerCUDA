#
#	Propagator of the density matrix with MUBs
#

import numpy as np
import scipy.fftpack as fftpack
from scipy.special import hyp1f1
from scipy.special import laguerre
from scipy.special import hermite
from scipy.special import legendre

import sympy as sympy
from string import Template


import math
import h5py

import pylab as plt
import scipy.linalg as linalg
import time
import ctypes
import cv2


from pycuda.elementwise import ElementwiseKernel

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import scikits.cuda.linalg as cu_linalg
cu_linalg.init()

#--------------------------------------------------------------------------

class double2(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double)
        ]

class float2(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float)
        ]

cuDoubleComplex = double2
cuFloatComplex = float2

_libcublas = ctypes.cdll.LoadLibrary('libcublas.so')

_libcublas.cublasZgemm.restype = None
_libcublas.cublasZgemm.argtypes = [ctypes.c_char,
                                   ctypes.c_char,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   cuDoubleComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   cuDoubleComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int]

_libcublas.cublasCgemm.restype = None
_libcublas.cublasCgemm.argtypes = [ctypes.c_char,
                                   ctypes.c_char,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   ctypes.c_int,
                                   cuDoubleComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   ctypes.c_void_p,
                                   ctypes.c_int,
                                   cuDoubleComplex,
                                   ctypes.c_void_p,
                                   ctypes.c_int]



def gpu_dot(out_gpu, x_gpu, y_gpu):
    
    cublas_func = _libcublas.cublasZgemm  
    
    alpha = cuDoubleComplex(1, 0)
    beta = cuDoubleComplex(0, 0)
    
    k, m = y_gpu.shape
    n, l = x_gpu.shape
    
    if l != k:
            raise ValueError('objects are not aligned')
    
    lda = max(1, m)
    ldb = max(1, k)
    ldc = max(1, m)
    
    transa = 'n'
    transb = 'n'
    
    cublas_func(transb, transa, m, n, k, alpha, int(y_gpu.gpudata),
                    lda, int(x_gpu.gpudata), ldb, beta, int(out_gpu.gpudata), ldc)
    
    return 1

def gpu_dot_dagger(out_gpu, x_gpu, y_gpu):
    
    cublas_func = _libcublas.cublasZgemm  
    
    alpha = cuDoubleComplex(1, 0)
    beta = cuDoubleComplex(0, 0)
    
    n, l = x_gpu.shape
    m, k = y_gpu.shape
    
    if l != k:
            raise ValueError('objects are not aligned')
    
    #lda = max(1, m)
    
    lda = max(1, k)
    ldb = max(1, k)
    ldc = max(1, m)
    
    transa = 'n'
    transb = 'c'
    
    cublas_func(transb, transa, m, n, k, alpha, int(y_gpu.gpudata),
                    lda, int(x_gpu.gpudata), ldb, beta, int(out_gpu.gpudata), ldc)
    
    return 1


def gpu_dot_complex64(out_gpu, x_gpu, y_gpu):
     
    alpha = cuDoubleComplex(1, 0)
    beta = cuDoubleComplex(0, 0)
    
    k, m = y_gpu.shape
    n, l = x_gpu.shape
    
    if l != k:
            raise ValueError('objects are not aligned')
    
    lda = max(1, m)
    ldb = max(1, k)
    ldc = max(1, m)
    
    transa = 'n'
    transb = 'n'
    
    _libcublas.cublasCgemm(transb, transa, m, n, k, alpha, int(y_gpu.gpudata),
                    lda, int(x_gpu.gpudata), ldb, beta, int(out_gpu.gpudata), ldc)
    
    return 1


# trace


#--------------------------------------------------------------------------


round_source = """
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( double *B )
{
    
   int X1_DIM = blockDim.x;
 
   int index1 = 2*threadIdx.x + blockDim.x * X1_DIM    ;
   int index2 = 2*threadIdx.x + blockDim.x * X1_DIM + 1;
   
   if(  sqrt(abs(B[ index1 ]*B[ index1 ]))  < 0.0001  )  B[ index1 ] = 0.0;
   if(  sqrt(abs(B[ index2 ]*B[ index2 ]))  < 0.0001  )  B[ index2 ] = 0.0;

}
"""

gpuarray_copy_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *B_out , pycuda::complex<double> *B_in )
{
    int X_DIM = blockDim.x;
    int P_DIM = gridDim.x;
 
    const int indexTotal = threadIdx.x + X_DIM*blockIdx.x;

    B_out[ indexTotal ] =  B_in[ indexTotal ];
}

"""

gpuarray_complex64_copy_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<float> *B_out , pycuda::complex<float> *B_in )
{
    int X_DIM = blockDim.x;
    int P_DIM = gridDim.x;
 
    const int indexTotal = threadIdx.x + X_DIM*blockIdx.x;

    B_out[ indexTotal ] =  B_in[ indexTotal ];
}

"""

#-----------------------------------------------------------------------------

W_PhaseSpace_GPUFunction_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

$CONSTANTS

__device__ double smoothAbs( double p, double epsilon ){

return sqrt( p*p + epsilon );
}

__device__ double smoothSign( double p, double epsilon ){

return p/sqrt( p*p + epsilon );
}

__device__  pycuda::complex<double> iExp( double phase ){

return pycuda::complex<double>(  cos(phase)  , sin(phase)  );
}

__device__ pycuda::complex<double> PhaseSpaceFunction(double x, double p)
{
return $FUNCTION ;
}


__global__ void Kernel( pycuda::complex<double> *W )
{
    int gridDIM = blockDim.x;
 
    int k = threadIdx.x;
    int j = blockIdx.x;
    int i = blockIdx.y; 	

    const int indexTotal = i*gridDIM*gridDIM + j*gridDIM + k;
	
    const double p     =     dp*( k - gridDIM/2) + dp/2.     ;
    const double x     =     dx*( j - gridDIM/2) + dx/2.     ;
    const double theta = dtheta*( i - gridDIM/2) + dtheta/2. ;

    pycuda::complex<double> f( cos(theta*p), -sin(theta*p)  );

    f *= PhaseSpaceFunction( x + theta/2. , p ) ;
		  	
    W[ indexTotal ] = f;

}

"""


SumAxis0_GPUFunction_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *W_p1_p2, pycuda::complex<double> *W_theta_p1_p2 )
{
    int gridDIM = blockDim.x;
 
    int j = threadIdx.x;
    int k = blockIdx.x;

    pycuda::complex<double> sum(0.,0.);
 
    for( int i=0; i< gridDIM ; i++ ) 	
	sum += W_theta_p1_p2[ i*gridDIM*gridDIM + j*gridDIM + k ];
	
    W_p1_p2[ j*gridDIM + k  ] = sum;

}

"""

SumAxis2_GPUFunction_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *W_theta_x, pycuda::complex<double> *W_theta_x_p )
{
    int gridDIM = blockDim.x;
 
    int i = blockIdx.x;
    int j = threadIdx.x;

    pycuda::complex<double> sum(0.,0.);
 
    for( int k=0; k< gridDIM ; k++) 	
	sum += W_theta_x_p[ i*gridDIM*gridDIM + j*gridDIM + k ];
	
    W_theta_x[ i*gridDIM + j  ] = sum;

}

"""

Lift_To_X1X2Theta_GPU_Function_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *W, pycuda::complex<double> *B )
{
    int gridDIM = blockDim.x;

    int i = blockIdx.x;
    int j = threadIdx.x;

    W[ i*gridDIM*gridDIM + j*gridDIM + j ] = B[ i*gridDIM + j];
}
"""

GetAxis0_Slice_GPU_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *B, pycuda::complex<double> *W , int i)
{
    int gridDIM = blockDim.x;

    int j = blockIdx.x;
    int k = threadIdx.x;

    B[ j*gridDIM + k ] = W[ i*gridDIM*gridDIM + j*gridDIM + k ];	

}
"""

PutAxis0_Slice_GPU_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *W, pycuda::complex<double> *B , int i)
{
    int gridDIM = blockDim.x;

    int j = blockIdx.x;
    int k = threadIdx.x;

    W[ i*gridDIM*gridDIM + j*gridDIM + k ] = B[ j*gridDIM + k ];	

}
"""
#

Product_Exp_iThetaP_GPU_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

$CONSTANTS

__global__ void Kernel( pycuda::complex<double> *W){


    int i = blockIdx.y ;
    int j = blockIdx.x ;
    int k = threadIdx.x;	

    int gridDIM = blockDim.x;
   
    const double p     =     dp*( k - gridDIM/2 + 0.5 );
    const double theta =     dx*( i - gridDIM/2 + 0.5 );

    double phase = theta*p;

    W[ i*gridDIM*gridDIM + j*gridDIM + k ] *= pycuda::complex<double>(  cos(phase)  , sin(phase)  );  ;	

}
"""



#-----------------------------------------------------------------------------



def AbsorvingBoundariesMatrix(gridDIM,s):
	"""
	Absorbing aboundary conditions mask 

	gridDIM: the dimension of the density matrix
	s:       absorving boundary frame 
	"""	
	A = np.zeros( (gridDIM,gridDIM) )
	indexRange = range(gridDIM)
	J = (gridDIM-1.)/2
	s=64
	for n in indexRange:
		for m in indexRange:
			jn = (2*n - gridDIM + 1.)/2. 
			jm = (2*m - gridDIM + 1.)/2. 
			A[n,m] = (1 - np.exp( -(jn-J)**2/s**2))*(1 - np.exp( -(jn+J)**2/s**2))*\
				 (1-np.exp( -(jm-J)**2/s**2))*(1- np.exp(-(jm+J)**2/s**2  )) 
	return A

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class Rho2D_Frame:
	def __init__(self, gridDIM, amplitude):
		"""
		Parameters:
			gridDIM:      grid dimension
			amplitude:    amplitude in X
		"""
		
		self.gridDIM = gridDIM

		self.dX = 2.*amplitude/float(self.gridDIM)

		self.X_amplitude = amplitude 

		# The correct dP is not:
		self.dP = 2.*np.pi/(2.*self.X_amplitude)
	
		self.F  = self.SymmetricInverseFourierMatrix()
		self.iF = self.F.conj().T

		self.F_GPU  = gpuarray.to_gpu( np.ascontiguousarray( self.F ))
		self.F_temp_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.F ))

		self.P_amplitude = self.dP*gridDIM/2. 

		#self.X_FFT = fftpack.fftshift(self.X_Range_FFT())[np.newaxis,:]

		# Grid

		self.X_grid =  self.X_Range() [np.newaxis,:]
		self.P_grid =  self.P_Range() [:,np.newaxis]

		self.copyGPU_Function = SourceModule( gpuarray_copy_source ).get_function( "Kernel" )
		self.copy_complex64_GPU_Function = SourceModule( gpuarray_complex64_copy_source ).get_function( "Kernel" )

		"""self.X2_FFT    = fftpack.fftshift( self.X_Range_FFT()     )[np.newaxis,np.newaxis,:]

		self.X1_FFT    = fftpack.fftshift( self.X_Range_FFT()     )[np.newaxis,:,np.newaxis]		
		self.P_FFT     = fftpack.fftshift( self.P_Range_FFT()     )[:,np.newaxis,np.newaxis]

		self.Theta_FFT = fftpack.fftshift( self.Theta_Range_FFT() )[:,np.newaxis,np.newaxis]"""

		#self.absorbGrid_Axis0 =  (1 - np.exp( -self.P_Range_FFT()**2/36 ))[:,np.newaxis] + self.X_FFT*0
		

	def SymmetricInverseFourierMatrix(self):
		F = np.zeros( (self.gridDIM,self.gridDIM) ) +0j
		omega = np.exp(-1j*2*np.pi/float(self.gridDIM))
		indexRange = range(self.gridDIM)
		for n in indexRange:
			for m in indexRange:
				jn = (2*n - self.gridDIM + 1.)/2. 
				jm = (2*m - self.gridDIM + 1.)/2. 
				F[n,m] = omega**(jn*jm)
		return F/np.sqrt(self.gridDIM) 

	def DisplacementOperator(self):
		"""
		Displacement operator T:
			T = log(  j dX P )
			with P as the momentum in the X Basis
		"""
		idM = np.identity( self.gridDIM  )
		d = np.array( map( lambda x: np.roll(x,-1),  idM ) )
		d[0, self.gridDIM -1 ] = -1.
		return d

	#

	def X_To_P_Basis_GPU(self, A_GPU ):
		gpu_dot( self.F_temp_GPU   , self.F_GPU      ,  A_GPU       )
		gpu_dot_dagger( A_GPU      , self.F_temp_GPU ,  self.F_GPU  )	

	def P_To_X_Basis_GPU(self, A_GPU ):
		gpu_dot_dagger( self.F_temp_GPU , self.F_GPU ,       A_GPU       )
		gpu_dot(        A_GPU ,           self.F_temp_GPU ,  self.F_GPU  )
	#

	def X_To_P_Basis(self, A):
		return np.dot(  self.F ,  A.dot(self.iF)  )

	def P_To_X_Basis(self, A):
		return np.dot(  self.iF ,  A.dot(self.F)  )	

	def X_Range(self):
		return np.linspace( -self.X_amplitude + self.dX/2., self.X_amplitude - self.dX/2., self.gridDIM)

	def P_Range(self):
		return np.linspace( -self.P_amplitude + self.dP/2., self.P_amplitude - self.dP/2., self.gridDIM)

	# FFT....................................................................................

	def X_Range_FFT(self):
		return np.linspace( -self.X_amplitude, self.X_amplitude - self.dX, self.gridDIM)

	def P_Range_FFT(self):
		return np.linspace( -self.P_amplitude, self.P_amplitude - self.dP, self.gridDIM)

	def Theta_Range_FFT(self):
		return np.linspace( -self.X_amplitude, self.X_amplitude - self.dX, self.gridDIM)
	#

	def MomentumAuxFun(self,n,m,gridDIM):
		pi = np.pi
		if(n==m):
			return 0. #np.sum( self.P_Range()/self.gridDIM )
		else:
			return (np.sin( pi*(n-m)  )*np.cos( pi*(n-m)/gridDIM) - gridDIM*np.cos( pi*(n-m) )*\
			np.sin( pi*(n-m)/gridDIM ))/( np.sin( pi*(n-m)/gridDIM)**2)
   

	#.......................................................................................
	def OperatorP_XBasis(self):
		"""
		Operator P in the X basis
		gridDIM: grid dimension
		dX:      discretization step in X
		"""
		P = np.zeros( (self.gridDIM,self.gridDIM) )
		indexRange = range(self.gridDIM)
		for n in indexRange:
			for m in indexRange:
				jn = (2*n - self.gridDIM + 1.)/2. 
				jm = (2*m - self.gridDIM + 1.)/2. 
				P[n,m] = self.MomentumAuxFun(jn,jm,self.gridDIM)
		return np.pi*1j/(self.dX*self.gridDIM**2)*P


	def OperatorP_PBasis(self):
		"""
		Operator P in the P basis
		gridDIM: grid dimension
		dx:      discretization step in X
		"""
		return np.diag( self.P_Range() ).astype(np.complex128)


	def OperatorX_XBasis(self):
		"""
		Operator X in the X basis
		gridDIM: grid dimension
		dX:      discretization step in X
		"""	
		return np.diag( self.X_Range()  ).astype(np.complex128)

	def OperatorX_PBasis(self):
		"""
		Operator X in the P basis
		gridDIM: grid dimension
		dX:      discretization step in X
		"""
		return self.X_To_P_Basis( self.OperatorX_XBasis() )
		#return self.Operator_PBasis_from_XBasis( self.OperatorX_XBasis() )
		
	# ........................ Fast Symmetric Fourier Transforms .................

	def _PhaseShift_FFT_(self, x):
		gridDIM = x.shape[0]
		J      = (gridDIM - 1.)/2.
		omega  = np.exp(1j *2* np.pi/float(gridDIM) )
		return x*omega**( -J*np.arange(gridDIM) ) 

	def _PhaseShift_InverseFFT_(self,x):
		gridDIM = x.shape[0]
		J      = (gridDIM - 1.)/2.
		omega  = np.exp(-1j *2* np.pi/float(gridDIM) )
		return x*omega**( -J*np.arange(gridDIM) ) 

	def PhaseShift_FFT(self,x, axis=0):
		return np.apply_along_axis(  self._PhaseShift_FFT_, axis, x )

	def PhaseShift_InverseFFT(self,x, axis=0):
		return np.apply_along_axis(  self._PhaseShift_InverseFFT_, axis, x )

	def Fast_SymmetricFFT(self,x, a):
		#  Equivalent to the application of self.iF 
		#  Transforms a vector x from basis  p -> x
		gridDIM = x.shape[0]
		J  = ( gridDIM - 1.)/2.
		omega  = np.exp(1j *2* np.pi/float( gridDIM) )
		return np.sqrt(gridDIM)*omega**(J*J)*\
			self.PhaseShift_FFT(  fftpack.ifft( self.PhaseShift_FFT(x,axis=a), axis=a  ), axis=a  )

	def Fast_SymmetricInverseFFT(self,x, a):
		#  Equivalent to the application of self.F 
		#   Transforms a vector x from basis x -> p
		gridDIM = x.shape[0]
		J  = (gridDIM - 1.)/2.
		omega  = np.exp(-1j *2* np.pi/float(gridDIM) )
		return 1./np.sqrt(gridDIM)*\
			omega**(J*J)*self.PhaseShift_InverseFFT(fftpack.fft(self.PhaseShift_InverseFFT(x,axis=a),axis=a),axis=a)

	
	#.........................Potential...........................................
	def Potential(self,t,x):
		"""
		Potential used to draw the energy level sets
		"""	
		pow = np.power
		atan = np.arctan
		M_E = np.e
		return eval ( self.potentialString, np.__dict__, locals() )	
		#return 200* np.tanh(  (  0.025*x**4 +  self.omega**2*x**2 )/200.    )

	def dPotential(self,t,x):
		"""
		derivative of Potential 
		"""	
		pow = np.power
		atan = np.arctan
		M_E = np.e
		return eval ( self.dPotentialString, np.__dict__, locals() )
		#return x * ( 0.1*x**2 + 2*self.omega**2 )*np.cosh( ( 0.025*x**4 + x**2*self.omega**2  )/200.   )**(-2)

	def _XdPotential(self,t,x):
		"""
		derivative of Potential 
		"""	
		pow = np.power
		atan = np.arctan
		M_E = np.e
		return x*eval ( self.dPotentialString, np.__dict__, locals() )

	#................................states..............................
	def Psi_Gaussian (self,x0,p0,X):
		return np.exp(1j*p0*X - (X-x0)**2/2.)

        def Psi_Morse(self, n , r , morse_Depth, morse_a, d ):
		"""
		Eigenstate wavefunction of the Morse potential
		Parameters:
			n: quantum number
			r: coordinate
			d: coordinate displacement
			morse_Depth, morse_a
		"""
                x = (r-d)*morse_a
                LAMBDA = np.sqrt( 2*self.mass*morse_Depth )/morse_a
                z = 2.*LAMBDA*np.exp(-x)
                return z**( LAMBDA - n - 0.5 )*np.exp( -0.5*z )*hyp1f1( -n , 2*LAMBDA - 2*n  , z )

	def Psi_HarmonicOscillator(self, n, omega, X0, P0, X):

		k = np.sqrt(self.mass*omega/self.hBar ) 
		return np.exp(1j*P0*(X-X0))*np.exp( -0.5*k**2*(X-X0)**2 )*hermite(n)( k*(X-X0) )				



	def Step_Function(x):
	    return 1 * (x > 0)

	def BuresDistance(self, Rho1, Rho2 ):
	    return  2*( 1.-np.trace( linalg.sqrtm( linalg.sqrtm( Rho1 ).dot( Rho2 ).dot( linalg.sqrtm( Rho1 )  )  )  ) )

	
	def Rho_Gaussian (self,x0,p0):
		"""
		Rho initial state in the X representation
		Parameters:
			x0,p0
		Returns:
			Rho gaussian state      
		"""
		#X = self.dX*(2*np.array( range(self.gridDIM)  ) - self.gridDIM + 1.)/2.
		X = self.X_Range()
		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 		
		
		Rho = self.Psi_Gaussian(x0,p0,X_col).dot(self.Psi_Gaussian(x0,p0,X_row).conj())    
		norm = np.trace(Rho)
		return Rho/norm

	def Rho_GaussianCat1 (self,x0, p0_1, p0_2):
		"""
		Rho initial state in the X representation
		Parameters:
			x0,p0
		Returns:
			Rho gaussian state      
		"""
		X = self.X_Range()

		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 		

		psi_col = self.Psi_Gaussian(x0, p0_1,X_col) + self.Psi_Gaussian(x0,-p0_2, X_col)
		psi_row = self.Psi_Gaussian(x0, p0_1,X_row) + self.Psi_Gaussian(x0,-p0_2, X_row)	

		Rho = psi_col.dot( psi_row.conj() )    
		norm = np.trace(Rho)
		return Rho/norm


	def Rho_HarmonicOscillator(self, n, omega, X0,P0):
		"""
		Rho initial state in the X representation
		Parameters:
			n
			X0
			omega
		Returns:
			Rho harmonic oscillator      
		"""
		X = self.dX*(2*np.array( range(self.gridDIM)  ) - self.gridDIM + 1.)/2.
		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 		
		
		Rho=self.Psi_HarmonicOscillator(n,omega,X0,P0,X_col).dot(self.Psi_HarmonicOscillator(n,omega,X0,P0,X_row).conj())    
		norm = np.trace(Rho)
		return Rho/norm

	def Rho_HarmonicOscillatorCat(self, n, omega, X0,P0, X1,P1):
		"""
		Rho initial state in the X representation
		Parameters:
			n
			X0
			omega
		Returns:
			Rho harmonic oscillator      
		"""
		X = self.dX*(2*np.array( range(self.gridDIM)  ) - self.gridDIM + 1.)/2.
		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 	
		psi_col = 0.5*self.Psi_HarmonicOscillator(n,omega,X0,P0,X_col)
		psi_col += 0.5*self.Psi_HarmonicOscillator(n,omega,X1,P1,X_col)

		psi_row = 0.5*self.Psi_HarmonicOscillator(n,omega,X0,P0,X_row)
		psi_row += 0.5*self.Psi_HarmonicOscillator(n,omega,X1,P1,X_row)

		Rho=psi_col.dot(psi_row.conj())    		
		norm = np.trace(Rho)
		return Rho/norm


	def  Wigner_HarmonicOscillator(self,n,omega,x0,p0):
	        """
	        Wigner function of the Harmonic oscillator
		Parameters
		s    : standard deviation in x
		x0,p0  : center of packet
		n    : Quantum number  
	        """

	        r2 = 0.5*(  self.mass*omega**2*((self.X_grid - x0 ))**2 + ((self.P_grid - p0 ))**2/self.mass  )  

	        W =  (-1)**(n)*laguerre(n)( 4*r2/(omega*self.hBar)  )*np.exp( -2*r2/(omega*self.hBar)  )
	        norm = np.sum( W )*self.dX*self.dP
	        return W/norm
	

	def Rho_Half_HarmonicOscillator(self, n, omega, X0,P0):
		"""
		Rho initial state in the X representation
		Parameters:
			n
			X0
			omega
		Returns:
			Rho harmonic oscillator      
		"""
		X = self.dX*(2*np.array( range(self.gridDIM)  ) - self.gridDIM + 1.)/2.
		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 		
		
		psiCol = self.Psi_HarmonicOscillator(n,omega,X0,P0,X_col)
		psiRow = self.Psi_HarmonicOscillator(n,omega,X0,P0,X_row)

		Rho = psiCol.dot( psiRow ).conj()    
		norm = np.trace(Rho)
		return Rho/norm

	def Rho_Arbitrary(self):
		"""
		Arbitrary state
		"""
		X = self.dX*(2*np.array( range(self.gridDIM)  ) - self.gridDIM + 1.)/2.
		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 		
		
		Rho = self.Psi_Arbitrary(X_col).dot(self.Psi_Arbitrary(X_row).conj())    
		norm = np.trace(Rho)
		return Rho/norm

	def Rho_Morse (self, n , morse_Depth, morse_a , d ):
		"""
		Density eigenstate of the Morse potential
		Parameters:
			n: quantum number
			d: coordinate displacement
			morse_Depth, morse_a
		"""
		X = self.dX*(2*np.array( range(self.gridDIM)  ) - self.gridDIM + 1.)/2.
		X_row = X.reshape( (1,self.gridDIM) )
		X_col = X.reshape( (self.gridDIM,1) ) 		
		
		Rho=self.Psi_Morse(n , X_col , morse_Depth, morse_a, d ).dot( \
				self.Psi_Morse(n, X_row, morse_Depth, morse_a,d ).conj() ) 
   
		norm = np.trace(Rho)
		return Rho/norm	

	#..........................................

	def symplecticExpK(self,mass, dt, c , Z ):
	    x = Z[0]
	    p = Z[1]
	    return Z + np.array([ dt*c*p/mass, 0 ])
    
	def symplecticExpV(self,mass, dt, d , dVdx, Z ):
	    x = Z[0]
	    p = Z[1]
	    return Z + np.array([ 0, -dt*d*dVdx(x) ])

	def symplecticExpKExpV2(self, mass, dt , dVdx, Z0 ):
	    """
	    Second order CLASSICAL symplectic propagator step
	    Parameters:
	    	mass
	    	dt
	    	dVdx function
	    	Z0 : initial state in phase space				     
	    """
	    c1 = 0.5
	    c2 = 0.5
	    d1 = 1.0
	    d2 = 0.0
	    Z = self.symplecticExpK(mass,dt,c1,self.symplecticExpV(mass, dt, d1 , dVdx , Z0 ))
	    return self.symplecticExpK(mass,dt,c2, self.symplecticExpV(mass, dt, d2 , dVdx, Z ) )

	def _SymplecticPropagator(self, mass, dt, n_iterations, dVdx, Z0):
		"""
		Second order classical symplectic propagator for constant gamma damping
		Parameters:
		    	mass
		    	dt
		    	dVdx function
		    	Z0 : initial state in phase space				     
		"""
		Z = Z0
		trajectory = []
		for i in range(n_iterations):
 			trajectory.append( np.append(Z,dt*(i+1) ))
			Z = self.symplecticExpKExpV2( mass, dt , dVdx ,Z )
			Z[1] = np.exp(-2.*self.gammaDamping*dt)*Z[1]
		return np.array(trajectory)    

	def _SymplecticPropagator(self, mass, gammaDamping, dt, n_iterations, dVdx, Z0):
		"""
		Second order classical symplectic propagator 
		Parameters:
		    	mass
			gamma: array with the values of gamma as a function of time
		    	dt
		    	dVdx function
		    	Z0 : initial state in phase space				     
		"""
		Z = Z0
		trajectory = []
		for i in range(n_iterations):
 			trajectory.append( np.append(Z,dt*(i+1) ))
			Z = self.symplecticExpKExpV2( mass, dt , dVdx ,Z )
			Z[1] = np.exp(-2.*gammaDamping[i]*dt)*Z[1]
		return np.array(trajectory)    


	def SymplecticPropagator(self, mass, gammaDamping, dt, n_iterations, dVdx, Z0):
		"""
		Second order classical symplectic propagator 
		Parameters:
		    	mass
			gamma: array with the values of gamma as a function of time
		    	dt
		    	dVdx function
		    	Z0 : initial state in phase space				     
		"""
		Z = Z0
		trajectory = []
		for i in range(n_iterations):
 			trajectory.append( np.append(Z,dt*(i+1) ))
			Z = self.symplecticExpKExpV2( mass, dt , dVdx ,Z )
			Z[1] = np.exp(-2.*gammaDamping*dt)*Z[1]
		return np.array(trajectory)    


	def WignerTransform(self, Psi ):
	    """ For the one dimensional Psi  """    
	    PSI    = Psi.copy()
	    PSI[:] = fftpack.fft(PSI, overwrite_x = True)
	    
	    X1_gridDIM   = Psi.shape[0]
	    
	    X1_amplitude = 1.
	    dK1 = 2.*np.pi/(2.*X1_amplitude)

	    K1_amplitude = dK1*X1_gridDIM/2.

	    dX1          = 2. * X1_amplitude / float(X1_gridDIM)
	 
	    X1_range      =  np.linspace(-X1_amplitude      , X1_amplitude  -dX1 , X1_gridDIM )
	    K1_range      =  np.linspace(-K1_amplitude      , K1_amplitude  -dK1 , X1_gridDIM )
	    
	    x1     = fftpack.fftshift(X1_range)[ np.newaxis , : ]
	    k1     = fftpack.fftshift(K1_range)[ np.newaxis , : ]
	    Theta1 = fftpack.fftshift(X1_range)[ : , np.newaxis ]
	    
	    PsiPlus   = PSI[np.newaxis , : ]
	    PsiMinus  = PSI[np.newaxis , : ]
	    
	    PsiPlus   = PsiPlus  + 0*Theta1 
	    PsiMinus  = PsiMinus + 0*Theta1 
	    
	    PsiPlus   *= np.exp(+1j*k1*Theta1/2.)
	    PsiPlus[:] = fftpack.ifft( PsiPlus, axis=1, overwrite_x = True )
	    
	    PsiMinus   *= np.exp(-1j*k1*Theta1/2. )
	    PsiMinus[:] = fftpack.ifft( PsiMinus ,axis=1, overwrite_x = True )
	    
	    W = PsiMinus
	    W *= np.conj(PsiPlus)
	    
	    W[:] = fftpack.ifft(W,axis=0)
	    W[:]/= np.sum(W)*dX1*dK1
	    
	    return W.real

	def _DensityMatrix_To_Wigner(self, Rho):
		Rho2 = Rho.copy()
		n = 32
		phi = (np.pi/4.)/float(n)
		    
		X_grid  =  self.X_Range()[np.newaxis,:]
		P_grid  =  self.X_Range()[:,np.newaxis]

		Lambda_grid = self.P_Range()[np.newaxis,:]
		Theta_grid  = self.P_Range()[:,np.newaxis]
		    
		exp_xtheta  = np.exp( -1j*phi * X_grid * Theta_grid  )
		exp_plambda = np.exp(  1j*phi * P_grid * Lambda_grid )
	    
		for i in range(n):
			Rho2 = self.Fast_SymmetricInverseFFT( Rho2, 0 )
			Rho2 *= exp_xtheta
			Rho2 = self.Fast_SymmetricFFT( Rho2, 0 )
		
			Rho2 = self.Fast_SymmetricInverseFFT( Rho2, 1 )
			Rho2 *= exp_plambda
			Rho2 = self.Fast_SymmetricFFT( Rho2, 1 )
	    
		W = self.Fast_SymmetricInverseFFT( Rho2 , 1)
		dx = self.X_Range()[1] - self.X_Range()[0];
		dp = self.P_Range()[1] - self.P_Range()[0];
		norm = np.sum(W)*dx*dp;
		W /= norm
        
		return W.T[::-1,:]*2

	def _DensityMatrix_To_Wigner(self,Rho):
	    Rho2 = Rho.copy()
	    rows,cols = Rho.shape
	    M = cv2.getRotationMatrix2D((cols/2,rows/2),-45,1)
		
	    Rho2_real = cv2.warpAffine( Rho2.real ,M,(cols,rows))
	    Rho2_imag = cv2.warpAffine( Rho2.imag ,M,(cols,rows))
		    
	    Rho2 = Rho2_real + 1j*Rho2_imag
		
	    W = self.Fast_SymmetricInverseFFT( Rho2 , 1)
	    dx = self.X_Range()[1] - self.X_Range()[0];
	    dp = self.P_Range()[1] - self.P_Range()[0];
	    norm = np.sum(W)*dx*dp;
	    W /= norm
	       
	    return W.T[::-1,:]*2

	def RotateRho(self,Rho):
		# with padding
		rows,cols = Rho.shape
		#pix_extra = int(round(max(Rho.shape)*(np.sqrt(2.)-1)/2))
		pix_extra = rows/2
		#
		Rho2 = np.lib.pad(Rho, [(pix_extra,pix_extra), (pix_extra,pix_extra)], 'constant', constant_values=(0,0))
		#Rho2 = np.lib.pad(Rho, [(pix_extra,pix_extra), (pix_extra,pix_extra)], 'linear_ramp', end_values=(0, 0))
	       
		rows,cols = Rho2.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-45,1)
		
		Rho2_real = cv2.warpAffine( Rho2.real ,M,(cols,rows))
		Rho2_imag = cv2.warpAffine( Rho2.imag ,M,(cols,rows))
			    
		return Rho2_real + 1j*Rho2_imag

	def DensityMatrix_To_Wigner(self,Rho):
	   		
	    W = self.Fast_SymmetricInverseFFT( self.RotateRho(Rho) , 1)
	    dx = self.X_Range()[1] - self.X_Range()[0];
	    dp = self.P_Range()[1] - self.P_Range()[0];
	    norm = np.sum(W)*dx*dp;
	    W /= norm
	       
	    return W.T[::-1,:]*4


	#==========================================================================================================
	#.......................................................................................................

	def Fourier_P_to_Theta(self, W):
		return fftpack.fft( W, axis = 0 )

	def Fourier_Theta_to_P(self, W):
		return fftpack.ifft( W, axis = 0 )

	def ThetaP(self, W ):	
		LW = W.copy()
		LW *= self.P_FFT
		
		LW = self.Fourier_P_to_Theta( LW )
		LW *= self.Theta_FFT	
		return self.Fourier_Theta_to_P( LW )

	def ScaleAxis0_Order2(self, W, gamma):
		
		LW  = self.ThetaP( W )
		LW  = W + 1./2. * 1j * gamma * LW			
		
		LW  = self.ThetaP( LW )
		LW  = W + 1./1. * 1j * gamma * LW

		return LW


	def AbsorvingBoundary(self,W):
		W *= self.absorbGrid_Axis0
		norm = np.sum( W )*self.dX*self.dP
		return W/norm


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

	def smoothAbs(self,x,epsilon):
		return   np.sqrt( x**2 + epsilon )

	def smoothSign(self,x,epsilon):
		return x/np.sqrt( x**2 + epsilon ) 


	def saveFrame(self, t, Rho_GPU ):
		print 'progress ', 100*t/(self.timeSteps+1), '%'
		Rho = Rho_GPU.get()
		#self.f.create_dataset(str(t), data = np.frombuffer( Rho ) )	
		#self.f.create_dataset('Rho_real/'+str(t), Rho.real )
		#self.f.create_dataset('Rho_imag/'+str(t), Rho.imag )
		self.f['Rho_imag/'+str(t)] = Rho.imag
		self.f['Rho_real/'+str(t)] = Rho.real

	def SaveLindbladianOperatorA(self,fileName,A):
		ff = h5py.File(fileName,'w')
		ff['A_real'] =  np.real(A)
		ff['A_imag'] =  np.imag(A)
		#ff['epsilon'] = epsilon
		#ff['L'] = L
		ff.close()

	def LoadLindbladianOperatorA(self,fileName):
		ff      = h5py.File(fileName,'r')
		#L       = ff['/L'].value
		#epsilon = ff['/epsilon'].value
		A  = ff['/A_real'][...] + 1j*ff['/A_imag'][...]
		ff.close()

		return A 

	def Save_SignP_PP_X(self, fileName, SignP_PP_X ):
		ff = h5py.File(fileName,'w')
		ff['SignP_PP_X_real'] =  np.real( SignP_PP_X )
		ff['SignP_PP_X_imag'] =  np.imag( SignP_PP_X )
		self.SignP_PP_X = SignP_PP_X
		ff.close()

	def Load_SignP_PP_X(self, fileName ):
		ff = h5py.File(fileName,'r')
		self.SignP_PP_X = ff['/SignP_PP_X_real'][...] + 1j*ff['/SignP_PP_X_imag'][...]
		ff.close()
		
		return self.SignP_PP_X

	def Psi_To_Rho(self,psi):
		n = psi.shape[0] 
		psi       = psi.reshape(n,1)
		psiDagger = psi.reshape(1,n).conj()
		return psi.dot(psiDagger)

	def LoadRho(self,fileName,n):
		FILE = h5py.File( fileName ,'r')
		#W = FILE['/'+str(n)][...]
		W  =     FILE['/Rho_real/'+str(n)][...] + 1j*FILE['/Rho_imag/'+str(n)][...]
		FILE.close() 

		return W  

	def Load_Ehrenfest(self):
		FILE = h5py.File( self.fileName ,'r')

		self.X_average =  FILE['/Ehrenfest/X_average'][...]
		self.X2_average = FILE['/Ehrenfest/X2_average'][...]   

		self.P_average           = FILE['/Ehrenfest/P_average'][...] 
		self.P2_average          = FILE['/Ehrenfest/P2_average'][...]   

		self.XP_average = FILE['/Ehrenfest/XP_average'][...]   

		self.Potential_average      = FILE['/Ehrenfest/Potential_average'][...] 
		self.dPotentialdX_average   = FILE['/Ehrenfest/dPotentialdX_average'][...]   
		self.XdPotentialdX_average  = FILE['/Ehrenfest/XdPotentialdX_average'][...]   	
		self.PdPotentialdX_average  = FILE['/Ehrenfest/PdPotentialdX_average'][...]  

		self.alpha_average  = FILE['/Ehrenfest/alpha_average'][...] 
		self.beta_average   = FILE['/Ehrenfest/beta_average'][...]   
		self.delta_average  = FILE['/Ehrenfest/delta_average'][...] 

		self.f_Damping_average = FILE['Ehrenfest/f_Damping_average'][...] 

		self.Rho_end  = FILE['/Rho_end'][...] 
		self.Rho_init = FILE['/Rho_init'][...] 

		self.H_grid = FILE['/H_grid'][...]


		FILE.close() 
  


#============================================================================================================

class Rho2D_Dissipation(Rho2D_Frame):
	def __init__(self, gridDIM, amplitude, dt, timeSteps, skipFrames):
		"""
		Generates operators in the P basis required to propagate a dissipative quantum system

		Parameters:
			 gridDIM:        Discretization number
			 amplitude:      Amplitude in X used for momentum resolution.  

			 amplitudeBound: The maximum effective amplitude to be achived. 
					 The physical system must be defined well below this limit 	
			 dt, timeSteps, skipFrames, mass, gammaDamping
			 
		"""
		Rho2D_Frame.__init__(self,gridDIM, amplitude)
		
		self.hBar = 1
		#self.fileName = fileName

		#self.X_truncationProportion = X_truncationProportion
		#self.P_truncationProportion = P_truncationProportion

		self.dt         = dt
		self.timeSteps  = timeSteps	
		self.skipFrames = skipFrames
		#self.mass       = mass
		#self.gammaDamping = gammaDamping
		#self.D_Theta = D_Theta

		# Operators are in the X basis

		self.X = self.OperatorX_XBasis()

		self.P = self.OperatorP_XBasis()

		self.X2 = self.X.dot(self.X)
		self.P2 = self.P.dot(self.P)
		
		# Making Hamiltonian
		print 'V(x)     = ', self.potentialString
		print '            '
		print 'dV(x)/dx = ', self.dPotentialString

		X = self.X_Range()
		#self.V_PBasis  =  np.diag( self.Potential(0., X ) ) 
		self.H        = (self.P.dot(self.P)/(2.*self.mass) +  np.diag( self.Potential(0., X ) )).astype(np.complex128)

		self.H_grid = (self.P_grid**2/(2.*self.mass) + 	self.Potential(0., self.X_grid )).real

		self.dVdX_XBasis  = np.diag( self.dPotential(0., X ) ).astype(np.complex128)
		self.dPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.dVdX_XBasis )  )
		
		#
		self.timeRange = self.dt*np.arange(0, self.timeSteps+1)


	def VectorP_To_X_Representation(self, w ):
		return self.P_To_X_Basis( np.diag(w) )		
		#return self.Operator_XBasis_from_PBasis( np.diag(w) )


	def SmoothAbs(self,P):
		return np.sqrt( 0.0001 + P**2  )



	# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	def WeylTranform(self, W_PhaseSpace_Function  ):

		print '	Weyl transform '
		X_Op = self.OperatorX_XBasis()
		P_Op = self.OperatorP_XBasis()
		lambdaRange = self.P_Range()
		thetaRange  = self.X_Range()
		
		Ambiguity = self.Fast_SymmetricInverseFFT( 
				 	self.Fast_SymmetricInverseFFT( W_PhaseSpace_Function( self.X_grid , self.P_grid ), 0 )
							, 1)
		Ambiguity *= self.gridDIM*self.dX*self.dP/(2*np.pi)

		sumA = np.zeros( (self.gridDIM,self.gridDIM) , dtype = np.complex )

		for i in range( self.gridDIM ):
			for j in range(self.gridDIM):
				Theta  =  thetaRange[i]
				Lambda = lambdaRange[j]	
				sumA  += Ambiguity[i,j]*linalg.expm( 1j*Lambda*X_Op + 1j*Theta*P_Op )

		return sumA * self.dX*self.dP/(2*np.pi) 

#......................................................................

	def MatrixExp_GPU(self, expA_GPU, A_GPU, temp_GPU):

		q = 7

		DIM = A_GPU.shape[0]
		nA =  float( gpuarray.max( A_GPU.__abs__() ).get() )
		#print 'norm A =', nA

		if nA == 0:
			expA_GPU *= 0
			expA_GPU += gpuarray.to_gpu(  np.ascontiguousarray( np.identity(DIM) , dtype=np.complex128 ) )
		
		val = np.log2(nA)
		e = int( np.floor(val) )
		j = max(0,e+1)

		#print 'j=',j

		A_GPU /= (2.0**j)

		X_GPU = A_GPU.copy()
		c = 1.0/2

		cX_GPU  = A_GPU.copy()
		cX_GPU *= c

		N_GPU  = gpuarray.to_gpu( np.ascontiguousarray( np.identity(DIM) , dtype=np.complex128 ) )
		N_GPU += cX_GPU		

		D_GPU  = gpuarray.to_gpu( np.ascontiguousarray( np.identity(DIM) , dtype=np.complex128 ) )
		D_GPU -= cX_GPU

		for k in range(2,q+1):
			
			c = c * (q-k+1) / (k*(2*q-k+1))

			gpu_dot( temp_GPU, A_GPU, X_GPU )
			self.copyGPU_Function( X_GPU,  temp_GPU, block=(self.gridDIM,1,1), grid=(self.gridDIM,1) ) 		

			self.copyGPU_Function( cX_GPU,  X_GPU, block=(self.gridDIM,1,1), grid=(self.gridDIM,1) )
			cX_GPU *= c
	
			N_GPU += cX_GPU

			if not k%2:
				D_GPU += cX_GPU
			else:
				D_GPU -= cX_GPU

		cu_linalg.inv( D_GPU, overwrite=True )
		gpu_dot( expA_GPU, D_GPU , N_GPU )

		for k in range(1,j+1):
			gpu_dot( temp_GPU, expA_GPU , expA_GPU )
			self.copyGPU_Function( expA_GPU,  temp_GPU, block=(self.gridDIM,1,1), grid=(self.gridDIM,1) )

		return expA_GPU

	#

	def WeylTranform_GPU(self, W_PhaseSpace_Function  ):

		print '	Weyl transform '
		X_Op_GPU = gpuarray.to_gpu(  np.ascontiguousarray( self.OperatorX_XBasis(), dtype=np.complex128 )   )
		P_Op_GPU = gpuarray.to_gpu(  np.ascontiguousarray( self.OperatorP_XBasis(), dtype=np.complex128 )   )

		lambdaRange = self.P_Range()
		thetaRange  = self.X_Range()
		
		Ambiguity = self.Fast_SymmetricInverseFFT( 
				 	self.Fast_SymmetricInverseFFT( W_PhaseSpace_Function( self.X_grid , self.P_grid ), 0 )
							, 1)
		Ambiguity *= self.gridDIM*self.dX*self.dP/(2*np.pi)

		sumA = np.zeros( (self.gridDIM,self.gridDIM) , dtype = np.complex )

		temp_GPU = gpuarray.zeros_like( X_Op_GPU ) 
		expD_GPU = gpuarray.zeros_like( X_Op_GPU ) 

		lambdaX_GPU = gpuarray.zeros_like( X_Op_GPU ) 
		thetaP_GPU  = gpuarray.zeros_like( X_Op_GPU ) 

		for i in range( self.gridDIM ):
			for j in range(self.gridDIM):
				Theta  =  thetaRange[i]
				Lambda = lambdaRange[j]	

				lambdaX_GPU *= 0.
				lambdaX_GPU += X_Op_GPU
				lambdaX_GPU *= 1j*Lambda

				thetaP_GPU *= 0.
				thetaP_GPU += P_Op_GPU
				thetaP_GPU *= 1j*Theta				
				
				thetaP_GPU += lambdaX_GPU

				self.MatrixExp_GPU( expD_GPU, thetaP_GPU, temp_GPU )

				sumA += Ambiguity[i,j]*expD_GPU.get()

				#print '  i , j  = ', i,'    ',j
				
				#sumA  += Ambiguity[i,j]*linalg.expm( 1j*Lambda*X_Op + 1j*Theta*P_Op )

		return sumA * self.dX*self.dP/(2*np.pi) 

	# 

	


	def A_Damping(self, f=None , L=None ):
		
		if f==None and L==None :
			f = self.f_Damping
			L = self.L_material

		def Axp(X,P):
			return np.sqrt( 2.*L * self.f_Damping( np.abs(P) + self.hBar/(2.*L)  )  )* np.exp(  -1j*np.sign(P)*X/L )
		
		
		return self.WeylTranform_GPU( Axp )


	def SignP_PP_X(self,epsilon ):

		def W(X,P):
			return -2*self.smoothSign(P,epsilon)*P*P*X

		return self.WeylTranform_GPU( W )

	def SignP_X(self ):

		def W(X,P):
			return -2*self.smoothSign(P,epsilon)*X

		return self.WeylTranform_GPU( W )



	def AntiCommutator(self,A,B):
		return A.dot(B) + B.dot(A)


	def LindbladianOperator_Damping(self, Rho1_GPU, Rho0_GPU, Rho_Half_GPU , RhoTemp_GPU,
				   chi, A_GPU, A_daggerA_GPU, gammaRenormalization, gammaDamping):
		
		gpu_dot_dagger( RhoTemp_GPU, Rho_Half_GPU,  A_GPU )
		gpu_dot( Rho1_GPU,     A_GPU,  RhoTemp_GPU    )
			
		gpu_dot( RhoTemp_GPU, Rho_Half_GPU,  A_daggerA_GPU )
		RhoTemp_GPU *= 0.5
		Rho1_GPU -= RhoTemp_GPU
	
		gpu_dot( RhoTemp_GPU, A_daggerA_GPU, Rho_Half_GPU  )
		RhoTemp_GPU *= 0.5
		Rho1_GPU -= RhoTemp_GPU

		Rho1_GPU *= gammaDamping*self.dt*chi*gammaRenormalization
		Rho1_GPU += Rho0_GPU

	def LindbladianOperator_Decoherence(self, Rho1_GPU, Rho0_GPU, Rho_Half_GPU , RhoTemp_GPU, chi):
				
		gpu_dot_dagger( RhoTemp_GPU, Rho_Half_GPU,  self.X_GPU )
		gpu_dot( Rho1_GPU,     self.X_GPU,  RhoTemp_GPU    )
			
		gpu_dot( RhoTemp_GPU, Rho_Half_GPU,  self.X2_GPU )
		RhoTemp_GPU *= 0.5
		Rho1_GPU -= RhoTemp_GPU
	
		gpu_dot( RhoTemp_GPU, self.X2_GPU, Rho_Half_GPU  )
		RhoTemp_GPU *= 0.5
		Rho1_GPU -= RhoTemp_GPU

		Rho1_GPU *= 2.*self.D_Theta*self.dt*chi
		Rho1_GPU += Rho0_GPU

	def alpha_Ehrenfest(self):
		# p square
		L     = self.L_material
		gamma = self.gammaDamping
 
		P = self.P_Range()
		return self.VectorP_To_X_Representation(  -2.*gamma*self.f_Damping(np.abs(P))*( 2.*np.abs(P) - self.hBar/L  )   )
		#return self.VectorP_To_X_Representation(  -2.*gamma*self.f_Damping(np.abs(P))*( 2.*P - self.hBar/L  )   )

	def beta_Ehrenfest(self):

		L     = self.L_material
		gamma = self.gammaDamping

		P = self.P_Range()
		temp =  self.VectorP_To_X_Representation(  -2.*gamma*self.f_Damping(np.abs(P))*np.sign(P) )
		return self.X.dot(temp) - 1j/2.*self.VectorP_To_X_Representation( self.df_Damping(P)*np.sign(P) )

	def delta_Ehrenfest(self):
		
		L     = self.L_material
		gamma = self.gammaDamping
		P = self.P_Range()

		return self.VectorP_To_X_Representation(  self.Ehrenfest_X2_QuantumCorrection(P)   )

	def f_Damping_Ehrenfest(self):
		
		L     = self.L_material
		gamma = self.gammaDamping
		P = self.P_Range()

		return self.VectorP_To_X_Representation(  -2*gamma*np.sign(P)*self.f_Damping(P)   )
		
		
	def SetXP_GPUVariables(self ):

		self.X_GPU   =  gpuarray.to_gpu(  np.ascontiguousarray(self.X.astype(np.complex128))  )
		self.X2_GPU  =  gpuarray.to_gpu(  np.ascontiguousarray( self.X.dot(self.X).astype(np.complex128))  )

		self.P_GPU   =  gpuarray.to_gpu(  np.ascontiguousarray(self.P.astype(np.complex128))  )
		self.P2_GPU  =  gpuarray.to_gpu(  np.ascontiguousarray(self.P.dot(self.P).astype(np.complex128))  )

		XP = ( self.P.dot(self.X) + self.X.dot(self.P) )/2. 
		self.XP_GPU  =  gpuarray.to_gpu(  np.ascontiguousarray(XP.astype(np.complex128))  )

		dPotentialdX       = np.diag( self.dPotential(0., self.X_Range() ) ).astype(np.complex128)
		PdPotentialdX      = ( self.P.dot(dPotentialdX) + dPotentialdX.dot(self.P) )/2.
		self.PdPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray(PdPotentialdX,dtype=np.complex128) )		
		
		self.XdPotential = np.diag( self.X_Range()*self.dPotential(0., self.X_Range() ) ).astype(np.complex128)
		self.XdPotentialdX_GPU = gpuarray.to_gpu(  np.ascontiguousarray(self.XdPotential,dtype=np.complex128)  )

		#
		Potential     = np.diag( self.Potential(0., self.X_Range() )  ).astype(np.complex128)
		self.Potential_GPU = gpuarray.to_gpu(  np.ascontiguousarray( Potential , dtype=np.complex128)  )

		self.alpha_GPU = gpuarray.to_gpu(  np.ascontiguousarray( self.alpha_Ehrenfest()  , dtype=np.complex128)  )

		self.beta_GPU  = gpuarray.to_gpu(  np.ascontiguousarray( self.beta_Ehrenfest()  , dtype=np.complex128)  )

		self.delta_GPU = gpuarray.to_gpu(  np.ascontiguousarray( self.delta_Ehrenfest()  , dtype=np.complex128)  )

		self.f_Damping_Ehrenfest_GPU = gpuarray.to_gpu(  
			np.ascontiguousarray( self.f_Damping_Ehrenfest()  , dtype=np.complex128)  )


#-----------------------------------------------------------------------------------------------------------------


	#def Run_SingleA(self, F, epsilon, gammaRenormalization, gammaDamping ,Rho0=0 ):

	def Run(self, A, gammaRenormalization, Rho0=0 ):
		"""
		Parameters:
			A
			f
			df
			gamma
			gammaRenormalization
			Rho_init
		Returns:
			
		"""
		gN = gammaRenormalization	

		#self.A = A


		try :
			import os
			os.remove (self.fileName)
		except OSError:
			pass

		self.f = h5py.File(self.fileName)

		#self.roundFunction = SourceModule(round_source).get_function( "Kernel" )
		blockCUDA = (self.gridDIM ,1,1)
		gridCUDA  = (self.gridDIM ,1)

		# GPU variables
		
		try:
			Rho0.shape
			Rho0_GPU = gpuarray.to_gpu( np.ascontiguousarray( Rho0 ))
 		except AttributeError:
			Rho0_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Rho_init.astype(np.complex128) )  )
			

		Rho1_GPU      = gpuarray.empty_like(Rho0_GPU)
		Rho2_GPU      = gpuarray.empty_like(Rho0_GPU)
		RhoTemp_GPU   = gpuarray.empty_like(Rho0_GPU)

		self.SetXP_GPUVariables()

		initial_time = time.time()

		self.A_1_GPU           = gpuarray.to_gpu( np.ascontiguousarray( A                  )  )
		self.A_1_daggerA_1_GPU = gpuarray.to_gpu( np.ascontiguousarray( A.conj().T.dot(A)  )  )


		expH     = linalg.expm( - 1j*self.dt*self.H  ) 
		expH_GPU = gpuarray.to_gpu(  np.ascontiguousarray( expH )  )

		#self.timeRange = self.dt*np.arange(0, self.timeSteps+1)  

		X_average  = []
		X2_average  = []
		#X4_average  = []

		P_average          = []
		#Smooth_P_average = []

		P2_average  = []

		XP_average  = []

		Potential_average = []

		dPotentialdX_average = []

		XdPotentialdX_average = []

		PdPotentialdX_average = []

		#SignP_PP_X_average = []

		f_Damping_average = []

		alpha_average = []
		beta_average  = []
		delta_average = []
		

		for tIndex in range(0, self.timeSteps+1):
			if tIndex%self.skipFrames==0: 
				#print 'iteration = ', tIndex
				self.saveFrame(tIndex,Rho0_GPU )

			X_average.append (  gpuarray.dot(  Rho0_GPU, self.X_GPU.conj()   ).get()  )
			X2_average.append(  gpuarray.dot(  Rho0_GPU, self.X2_GPU.conj()  ).get()  )

			P_average.append(        gpuarray.dot(  Rho0_GPU, self.P_GPU.conj()        ).get()  )
			
			P2_average.append(       gpuarray.dot(  Rho0_GPU, self.P2_GPU.conj()       ).get()  )
			XP_average.append(       gpuarray.dot(  Rho0_GPU, self.XP_GPU.conj()       ).get()  )

			Potential_average.append(  gpuarray.dot(  Rho0_GPU, self.Potential_GPU.conj()  ).get()   )

			dPotentialdX_average.append(  gpuarray.dot(  Rho0_GPU, self.dPotentialdX_GPU.conj()  ).get()    )
			XdPotentialdX_average.append(  gpuarray.dot(  Rho0_GPU, self.XdPotentialdX_GPU.conj()  ).get()    )

			PdPotentialdX_average.append(  gpuarray.dot(  Rho0_GPU, self.PdPotentialdX_GPU.conj()  ).get()    )
			
			alpha_average.append(  gpuarray.dot(  Rho0_GPU, self.alpha_GPU.conj()  ).get()   )
			beta_average.append(   gpuarray.dot(  Rho0_GPU, self.beta_GPU.conj()   ).get()   )
			delta_average.append(  gpuarray.dot(  Rho0_GPU, self.delta_GPU.conj()  ).get()   )
	
			f_Damping_average.append( gpuarray.dot(  Rho0_GPU, self.f_Damping_Ehrenfest_GPU.conj()  ).get()  )


			# Unitary evolution
			gpu_dot(RhoTemp_GPU, expH_GPU, Rho0_GPU)    
			gpu_dot_dagger(Rho1_GPU,    RhoTemp_GPU, expH_GPU) 
			
			_RhoTemp_GPU = Rho0_GPU
			Rho0_GPU = Rho1_GPU
			Rho1_GPU = _RhoTemp_GPU

        		# Lindbladian evolution for damping			

			self.LindbladianOperator_Damping( Rho1_GPU,Rho0_GPU,Rho0_GPU,RhoTemp_GPU,
			 1./3.,self.A_1_GPU,self.A_1_daggerA_1_GPU,gN, self.gammaDamping )

			self.LindbladianOperator_Damping( Rho2_GPU,Rho0_GPU,Rho1_GPU,RhoTemp_GPU,
			 1./2.,self.A_1_GPU,self.A_1_daggerA_1_GPU,gN, self.gammaDamping )
			
			self.LindbladianOperator_Damping( Rho1_GPU,Rho0_GPU,Rho2_GPU,RhoTemp_GPU,
			 1.   ,self.A_1_GPU,self.A_1_daggerA_1_GPU,gN, self.gammaDamping )
    
			_RhoTemp_GPU = Rho0_GPU
			Rho0_GPU = Rho1_GPU
			Rho1_GPU = _RhoTemp_GPU

			#  Lindbladian evolution for decoherence

			if self.D_Theta != 0:

				self.LindbladianOperator_Decoherence( Rho1_GPU,Rho0_GPU,Rho0_GPU,RhoTemp_GPU, 1./3. )

				self.LindbladianOperator_Decoherence( Rho2_GPU,Rho0_GPU,Rho1_GPU,RhoTemp_GPU, 1./2. )
			
				self.LindbladianOperator_Decoherence( Rho1_GPU,Rho0_GPU,Rho2_GPU,RhoTemp_GPU, 1. )
	    
				_RhoTemp_GPU = Rho0_GPU
				Rho0_GPU = Rho1_GPU
				Rho1_GPU = _RhoTemp_GPU	

			#norm = cu_linalg.trace( Rho0_GPU  )
			#Rho0_GPU /= norm
				
			



		print '                     '

		print 'blockCUDA = ', blockCUDA
		print 'gridCUDA = ' , gridCUDA		


		self.Rho_end = Rho0_GPU.get()

		self.X_average   = np.array(X_average).real
		self.X2_average  = np.array(X2_average).real

		self.P_average          = np.array(P_average).real
		
		self.P2_average         = np.array(P2_average).real

		self.XP_average  = np.array(XP_average).real

		self.Potential_average      = np.array( Potential_average ).real
		self.dPotentialdX_average   = np.array(  dPotentialdX_average).real
		self.XdPotentialdX_average  = np.array( XdPotentialdX_average).real
		self.PdPotentialdX_average  = np.array( PdPotentialdX_average).real

		
		self.alpha_average = np.array( alpha_average ).real
		self.beta_average  = np.array(  beta_average ).real
		self.delta_average = np.array( delta_average ).real	 
		self.f_Damping_average = np.array(f_Damping_average).real


		self.f['X_amplitude']  = self.X_amplitude
		self.f['X_gridDIM']    = self.gridDIM
		self.f['dt']           = self.dt         
		self.f['timeSteps']    = self.timeSteps  
		self.f['skipFrames']   = self.skipFrames 

		self.f['Ehrenfest/X_average']  = self.X_average
		self.f['Ehrenfest/X2_average'] = self.X2_average

		self.f['Ehrenfest/P_average']         = self.P_average
		self.f['Ehrenfest/P2_average'] = self.P2_average
		

		self.f['Ehrenfest/XP_average'] = self.XP_average

		self.f['Ehrenfest/Potential_average']     = self.Potential_average
		self.f['Ehrenfest/dPotentialdX_average']  = self.dPotentialdX_average
		self.f['Ehrenfest/XdPotentialdX_average'] = self.XdPotentialdX_average		
		self.f['Ehrenfest/PdPotentialdX_average'] = self.PdPotentialdX_average

		self.f['Ehrenfest/alpha_average'] = self.alpha_average
		self.f['Ehrenfest/beta_average' ] = self.beta_average
		self.f['Ehrenfest/delta_average'] = self.delta_average

		self.f['Ehrenfest/f_Damping_average'] = self.f_Damping_average

		self.f['Rho_end']   = self.Rho_end
		self.f['Rho_init']  = self.Rho_init
		self.f['H_grid']    = self.H_grid

		self.f.close()

		return self.Rho_end

