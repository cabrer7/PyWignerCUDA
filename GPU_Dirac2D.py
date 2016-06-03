#!/usr/local/epd/bin/python
#------------------------------------------------------------------------------------------------------
#  Dirac propagator based on:
#  Fillion-Gourdeau, Francois, Lorin, Emmanuel, Bandrauk, Andre D.
#  Numerical Solution of the Time-Dependent Dirac Equation in Coordinate Space without Fermion-Doubling			
#------------------------------------------------------------------------------------------------------

import numpy as np
import scipy.fftpack as fftpack
import h5py
import time
import sys
from scipy.special import laguerre
from scipy.special import genlaguerre
from scipy.special import legendre

#from pyfft.cuda import Plan
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import cufft_wrapper as cuda_fft
#-------------------------------------------------------------------------------


Potential_0_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double Potential0(double t, double x, double y)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel( pycuda::complex<double>* preExpectationValue, 
pycuda::complex<double>* Psi1,  pycuda::complex<double>* Psi2,  pycuda::complex<double>* Psi3,  pycuda::complex<double>* Psi4,
double t)
{

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;

  int j  =       (threadIdx.x + DIM_X/2)%%DIM_X;
  int i  =       (blockIdx.x  + DIM_Y/2)%%DIM_Y;

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x ; 

  double out;

  out =  Potential0( t, x, y)* pow( abs( Psi1[indexTotal] ) , 2 );
  out += Potential0( t, x, y)* pow( abs( Psi2[indexTotal] ) , 2 );
  out += Potential0( t, x, y)* pow( abs( Psi3[indexTotal] ) , 2 );
  out += Potential0( t, x, y)* pow( abs( Psi4[indexTotal] ) , 2 );

  preExpectationValue[indexTotal] = out;

}

"""



#--------------------------------------------------------------------------------

BaseCUDAsource_K = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__global__ void Kernel(
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{
  pycuda::complex<double> I;

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;

  int j  =  (threadIdx.x + DIM_X/2)%%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%%DIM_Y;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x; 

  double px = dPx*(j - DIM_X/2);
  double py = dPy*(i - DIM_Y/2);

  double pp  = sqrt( px*px + py*py + pow(10.,-12));
  double sdt = sin( c*dt*pp )/pp;
  double cosdt = cos(c*dt*pp);

  pycuda::complex<double> p_plus = pycuda::complex<double>(py,px);
  pycuda::complex<double> p_mius = pycuda::complex<double>(py,-px);

 _Psi1= cos(c*dt*pp)*Psi1[indexTotal]                                                             -   p_plus*sdt*Psi4[indexTotal];
 _Psi2=                              cos(c*dt*pp)*Psi2[indexTotal] +    p_mius*sdt*Psi3[indexTotal]                		   ;
 _Psi3=    			    -  p_plus*sdt*Psi2[indexTotal] +  cos(c*dt*pp)*Psi3[indexTotal]                              ;
 _Psi4= p_mius*sdt*Psi1[indexTotal]            			                               + cos(c*dt*pp)*Psi4[indexTotal];

  Psi1[indexTotal] = _Psi1;
  Psi2[indexTotal] = _Psi2;
  Psi3[indexTotal] = _Psi3;
  Psi4[indexTotal] = _Psi4;

}
"""





DiracPropagatorA_source = """
//
//   source code for the Dirac propagator with scalar-vector potential interaction
//   and smooth time dependence
//

#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s // Define the essential constants 

// The vector potential must be supplied with UP indices and: eA

__device__  double A0(double t, double x, double y)
{
    return %s ;
}
__device__  double A1(double t, double x, double y)
{
   return %s ;
}

__device__  double A2(double t, double x, double y)
{
   return %s ;
}

__device__  double A3(double t, double x, double y)
{
   return %s ;
}

__device__ double VectorPotentialSquareSum(double t, double x, double y)
{
 return pow( A1(t,x,y), 2.) + pow( A2(t,x,y), 2.) + pow( A3(t,x,y), 2.);
}

//-------------------------------------------------------------------------------------------------------------

__global__ void DiracPropagatorA_Kernel(
 pycuda::complex<double>  *Psi1, pycuda::complex<double>   *Psi2, pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4, double t )
{
  pycuda::complex<double> I = pycuda::complex<double>(0.,1.);

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;

  int j  =       (threadIdx.x + DIM_X/2)%%DIM_X;
  int i  =       (blockIdx.x  + DIM_Y/2)%%DIM_Y;


  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x ; 

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  double F;
  F = sqrt( pow(mass*c*c,2.) + c*VectorPotentialSquareSum(t,x,y)  );	
 
  pycuda::complex<double> expV0 = exp( -I*dt*A0(t,x,y) );	


  pycuda::complex<double> U11 = pycuda::complex<double>( cos(dt*F) ,  -mass*c*c*sin(dt*F)/F );
  pycuda::complex<double> U22 = U11;


  pycuda::complex<double> U33 = pycuda::complex<double>( cos(dt*F) ,  mass*c*c*sin(dt*F)/F );	
  pycuda::complex<double> U44 = U33;

  pycuda::complex<double> U31 = pycuda::complex<double>( 0., A3(t,x,y)*sin(dt*F)/F );
  pycuda::complex<double> U41 = pycuda::complex<double>( -A2(t,x,y)*sin(dt*F)/F , A1(t,x,y)*sin(dt*F)/F );
  
  pycuda::complex<double> U32 = pycuda::complex<double>( A2(t,x,y)*sin(dt*F)/F , A1(t,x,y)*sin(dt*F)/F );
  pycuda::complex<double> U42 = pycuda::complex<double>( 0., -A3(t,x,y)*sin(dt*F)/F );

  pycuda::complex<double> U13 = U31;
  pycuda::complex<double> U14 = U32;
  pycuda::complex<double> U23 = U41;
  pycuda::complex<double> U24 = U42;
 

  _Psi1 = expV0*( U11*Psi1[indexTotal] + U13*Psi3[indexTotal] + U14*Psi4[indexTotal] );

  _Psi2 = expV0*( U22*Psi2[indexTotal] + U23*Psi3[indexTotal] + U24*Psi4[indexTotal] );	

  _Psi3 = expV0*( U31*Psi1[indexTotal] + U32*Psi2[indexTotal] + U33*Psi3[indexTotal] );

  _Psi4 = expV0*( U41*Psi1[indexTotal] + U42*Psi2[indexTotal] + U44*Psi4[indexTotal] );

  Psi1[indexTotal] = _Psi1;
  Psi2[indexTotal] = _Psi2;
  Psi3[indexTotal] = _Psi3;
  Psi4[indexTotal] = _Psi4;

}

"""


BaseCUDAsource_AbsorbBoundary_xy  =  """
//............................................................................................
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void AbsorbBoundary_Kernel(
		pycuda::complex<double>  *Psi1,  pycuda::complex<double>  *Psi2, 
                pycuda::complex<double>  *Psi3    , pycuda::complex<double>  *Psi4 )
{

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;

  const int j  =       (threadIdx.x + DIM_X/2)%DIM_X;
  const int i  =       (blockIdx.x  + DIM_Y/2)%DIM_Y;

  const int indexTotal = threadIdx.x +  DIM_X*blockIdx.x + DIM_X*DIM_Y*blockIdx.y; 

  double wx = pow(3.*double(DIM_X)/100.,2); 
  double wy = pow(3.*double(DIM_Y)/100.,2); 


//--------------------------- boundary in x --------------------------------------


	double expB = 	1. - exp( -double(j*j)/wx );

  	Psi1[indexTotal] *=   expB;
	Psi2[indexTotal] *=   expB;
	Psi3[indexTotal] *=   expB;
	Psi4[indexTotal] *=   expB;	

	expB = 1.- exp(  -(j - DIM_X+1. )*(j - DIM_X+1.)/wx );

        Psi1[indexTotal] *=   expB;
	Psi2[indexTotal] *=   expB;
	Psi3[indexTotal] *=   expB;
	Psi4[indexTotal] *=   expB;	 

//-------------- boundary in y

	expB = 1.- exp(  -double(i*i)/wy  );

	Psi1[indexTotal] *=   expB;
	Psi2[indexTotal] *=   expB;
	Psi3[indexTotal] *=   expB;
	Psi4[indexTotal] *=   expB;	

	expB = 1. - exp( -double( (i - DIM_Y + 1)*(i - DIM_Y + 1) )/wy  );

	Psi1[indexTotal] *=   expB;
	Psi2[indexTotal] *=   expB;
	Psi3[indexTotal] *=   expB;
	Psi4[indexTotal] *=   expB;

}

"""


#-----------------------------------------------------------------------------------------------

class GPU_Dirac2D:
	"""
	Propagator 2D for the Dirac equation
	Parameters:
		gridDIM_X
		gridDIM_Y

		min_X
		min_Y

		timeSteps

		skipFrames: Number of frames to be saved

		frameSaveMode = 'Density' saves only the density 
		frameSaveMode = 'Spinor' saves the whole spinor
	"""

	def __init__(self, gridDIM, amplitude, dt, timeSteps, skipFrames = 1,frameSaveMode='Density'):

		X_amplitude,Y_amplitude = amplitude
		X_gridDIM, Y_gridDIM    = gridDIM
		

		self.dX = 2.*X_amplitude/np.float(X_gridDIM)
		self.dY = 2.*Y_amplitude/np.float(Y_gridDIM)
	
		self.X_amplitude = X_amplitude
		self.Y_amplitude = Y_amplitude

		self.X_gridDIM = X_gridDIM
		self.Y_gridDIM = Y_gridDIM
		
		self.min_X = -X_amplitude
		self.min_Y = -Y_amplitude
		
		self.timeSteps = timeSteps
		self.skipFrames = skipFrames
		self.frameSaveMode = frameSaveMode

		rangeX  = np.linspace(-X_amplitude, X_amplitude - self.dX,  X_gridDIM )
		rangeY  = np.linspace(-Y_amplitude, Y_amplitude - self.dY,  Y_gridDIM )

		self.X = fftpack.fftshift(rangeX)[np.newaxis, :  ]
		self.Y = fftpack.fftshift(rangeY)[:, np.newaxis  ]

		self.X_GPU     = gpuarray.to_gpu( np.ascontiguousarray( self.X + 0.*self.Y, dtype = np.complex128)     )
		self.Y_GPU     = gpuarray.to_gpu( np.ascontiguousarray( self.Y + 0.*self.X, dtype = np.complex128)     )

		# min_Px = np.pi*self.X_gridDIM/(2*self.min_X)

		Px_amplitude = np.pi/self.dX
		self.dPx     = 2*Px_amplitude/self.X_gridDIM
		Px_range     = np.linspace( -Px_amplitude, Px_amplitude - self.dPx, self.X_gridDIM )

		Py_amplitude = np.pi/self.dY
		self.dPy     = 2*Py_amplitude/self.Y_gridDIM
		Py_range     = np.linspace( -Py_amplitude, Py_amplitude - self.dPy, self.Y_gridDIM )

		self.Px = fftpack.fftshift(Px_range)[np.newaxis,:]
		self.Py = fftpack.fftshift(Py_range)[:,np.newaxis]

		self.Px_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Px + 0.*self.Py, dtype = np.complex128)     )
		self.Py_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Py + 0.*self.Px, dtype = np.complex128)     )

		self.dt = dt
		

		#................ Strings: mass,c,dt must be defined in children class.................... 
	
		self.CUDA_constants_essential  = '__constant__ double mass=%f; '%self.mass
		self.CUDA_constants_essential += '__constant__ double c=%f;    '%self.c
		self.CUDA_constants_essential += '__constant__ double dt=%f;   '%self.dt
		self.CUDA_constants_essential += '__constant__ double dX=%f;   '%self.dX
		self.CUDA_constants_essential += '__constant__ double dY=%f;   '%self.dY
 
		self.CUDA_constants_essential += '__constant__ double dPx=%f;   '%self.dPx
		self.CUDA_constants_essential += '__constant__ double dPy=%f;   '%self.dPy

		self.CUDA_constants = 	self.CUDA_constants_essential #+ self.CUDA_constants_additional	

		#................ CUDA Kernels ...........................................................
				

		self.DiracPropagatorK = SourceModule(BaseCUDAsource_K%self.CUDA_constants,arch="sm_20").get_function( "Kernel" )

		self.DiracPropagatorA  =  \
		SourceModule( DiracPropagatorA_source%(
					self.CUDA_constants,
					self.Potential_0_String, 
 					self.Potential_1_String, 
					self.Potential_2_String, 
					self.Potential_3_String),arch="sm_20").get_function( "DiracPropagatorA_Kernel" )

		self.Potential_0_Average_Function = \
			SourceModule( Potential_0_Average_source%(
			self.CUDA_constants,self.Potential_0_String) ).get_function("Kernel" )


		self.DiracAbsorbBoundary_xy  =  \
		SourceModule(BaseCUDAsource_AbsorbBoundary_xy,arch="sm_20").get_function( "AbsorbBoundary_Kernel" )

		#...........................FFT PLAN.................................................

		self.plan_Z2Z_2D = cuda_fft.Plan_Z2Z(  (self.X_gridDIM,self.Y_gridDIM)  )



	def Fourier_X_To_P_GPU(self,W_out_GPU):
		cuda_fft.fft_Z2Z(  W_out_GPU, W_out_GPU , self.plan_Z2Z_2D )

	def Fourier_P_To_X_GPU(self,W_out_GPU):
		cuda_fft.ifft_Z2Z( W_out_GPU, W_out_GPU , self.plan_Z2Z_2D )
		W_out_GPU *= 1./float(self.X_gridDIM*self.Y_gridDIM)

	def Fourier_4_X_To_P_GPU(self, Psi1, Psi2, Psi3, Psi4):
		self.Fourier_X_To_P_GPU(Psi1)
		self.Fourier_X_To_P_GPU(Psi2)
		self.Fourier_X_To_P_GPU(Psi3)
		self.Fourier_X_To_P_GPU(Psi4)

	def Fourier_4_P_To_X_GPU(self, Psi1, Psi2, Psi3, Psi4):
		self.Fourier_P_To_X_GPU(Psi1)
		self.Fourier_P_To_X_GPU(Psi2)
		self.Fourier_P_To_X_GPU(Psi3)
		self.Fourier_P_To_X_GPU(Psi4)

#-------------------------------------------------------------------------------------------------------------------
#           Gaussian PARTICLE spinors 


	def Spinor_Particle_SpinUp(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py = p_init	

		rho  = np.exp(1j*self.X*px + 1j*self.Y*py )*modulation_Function( self.X , self.Y ) 

		p0  = np.sqrt( px*px + py*py + (self.mass*self.c)**2 )
		
		Psi1 =  rho*( p0  + self.mass*self.c ) 
		Psi2 =  rho*0.
		Psi3 =  rho*0.
		Psi4 =  rho*( px + 1j*py )	
		
		return np.array([Psi1, Psi2, Psi3, Psi4 ])

	def Spinor_Particle_SpinDown(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py = p_init	

		rho  = np.exp(1j*self.X*px + 1j*self.Y*py )*modulation_Function( self.X , self.Y ) 

		p0  = np.sqrt( px*px + py*py + (self.mass*self.c)**2 )
		
		Psi1 =  rho*0.
		Psi2 =  rho*( p0  + self.mass*self.c )  
		Psi3 =  rho*( px - 1j*py )
		Psi4 =  rho*0.
		
		return np.array([Psi1, Psi2, Psi3, Psi4 ])


	def Spinor_AntiParticle_SpinDown(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py = p_init	

		rho  = np.exp(1j*self.X*px + 1j*self.Y*py )*modulation_Function( self.X , self.Y ) 

		p0  = -np.sqrt( px*px + py*py + (self.mass*self.c)**2 )
		
		Psi1 =  rho*0.
		Psi2 =  rho*( px + 1j*py )	 
		Psi3 =  rho*( p0  - self.mass*self.c ) 
		Psi4 =  rho*0.	
		
		return -1j*np.array([Psi1, Psi2, Psi3, Psi4 ])


	def Spinor_AntiParticle_SpinUp(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py = p_init	

		rho  = np.exp(1j*self.X*px + 1j*self.Y*py )*modulation_Function( self.X , self.Y ) 

		p0  = -np.sqrt( px*px + py*py + (self.mass*self.c)**2 )
		
		Psi1 =  rho*( px - 1j*py )	
		Psi2 =  rho*0.
		Psi3 =  rho*0.
		Psi4 =  rho*( p0  - self.mass*self.c ) 
		
		return -1j*np.array([Psi1, Psi2, Psi3, Psi4 ])

	#.......................................................................

	def Boost(self, p1,p2):
		#  Boost matrix in Dirac gamma matrices
		p0 = np.sqrt( (self.mass*self.c)**2  +  p1**2 + p2**2 )
		K = np.sqrt( 2*self.mass*self.c*(self.mass*self.c + p0) )
		
		B00     = self.mass*self.c + p0
		p_Plus  = p1 + 1j*p2 
		p_Minus = p1 - 1j*p2 
		
		return np.array([ [B00, 0., 0., p_Minus], [0., B00, p_Plus,0.], [0.,p_Minus,B00,0.], [p_Plus,0.,0.,B00]  ])
		

	def LandauLevelSpinor(self, B , n , x , y ,type=1):
		# Symmetric Gauge 
		def energy(n):
			return np.sqrt( (self.mass*self.c**2)**2 + 2*B*self.c*self.hBar*n  )

		K = B*( (self.X-x)**2 + (self.Y-y)**2)/( 4.*self.c*self.hBar )
		
		psi1 = np.exp(-K)*( energy(n) + self.mass*self.c**2  )*laguerre(n)( 2*K ) 

		psi3 = np.exp(-K)*( energy(n) - self.mass*self.c**2  )*laguerre(n)( 2*K ) 

		if n>0:
			psi2 = 1j*np.exp(-K)*( self.X-x + 1j*(self.Y-y) )*genlaguerre(n-1,1)( 2*K )

		else: 
			psi2 = 0.*K		

		psi4 = psi2	

		if type==1:
			spinor = np.array([ psi1 , 0*psi2 , 0*psi3 , psi4  ])

		elif type ==2:
			spinor = np.array([ 0*psi1 , psi2 , psi3 , 0*psi4  ])

		else :
			print 'Error: type spinor must be 1 or 2'

		norm = self.Norm(spinor)

       		spinor /= norm 

		return spinor

	def LandauLevelSpinor_Boosted(self, B , n , x , y , py ):

		K = B*( (self.X-x)**2 + (self.Y-y)**2)/( 4.*self.c*self.hBar )

		p0 = np.sqrt( (self.mass*self.c)**2 + py**2 )

		psi1 = 0j*K
		psi2 =  1j*self.c* np.exp(-K) * py   *(  p0 - self.mass*self.c  )
		psi3 =     self.c* np.exp(-K) * py*py*(  p0 - self.mass*self.c  ) + 0j
		psi4 = 0j*K

		spinor = np.array([ psi1 , psi2 , psi3 , psi4  ])

		norm = self.Norm(spinor)

       		spinor /= norm 

		return spinor
 



	def LandaoLevelSpinor_GaugeX(self, B , n ,  Py ):
		def energy(n):
			return np.sqrt( (self.mass*self.c**2)**2 + 2*B*self.c*self.hBar*n  )

		K = B*(self.X - self.c*Py/B)**2/( 2.*self.c*self.hBar )
		
		psi1 = np.exp(-K)*(  self.mass*self.c**2 + energy(n) )* legendre(n)( K/np.sqrt(B*self.c*self.hBar) ) 

		psi3 = np.exp(-K)*(  self.mass*self.c**2 + energy(n) )* legendre(n)( K/np.sqrt(B*self.c*self.hBar) ) 

		if n>0:
			psi2 = np.exp(-K)*(  self.mass*self.c**2 + energy(n) )* legendre(n-1)( K/np.sqrt(B*self.c*self.hBar) ) 	
			psi2 = 2*1j*n*np.sqrt(B*self.c*self.hBar)

			psi4 = -psi2

		else: 
			psi2 = 0.*K
			psi4 = 0.*K	


		spinor = np.array([psi1 , psi2 , psi3 , psi2  ])

		norm = self.Norm(spinor)
       		spinor /= norm 

		return spinor

#.............................................................................................

	def _FilterElectrons(self,sign):
 		'''
		Routine that uses the Fourier transform to filter positrons/electrons
		Options:
			sign=1   Leaves electrons
			sign=-1	 Leaves positrons
		'''
		print '  '
		print '  	Filter Electron routine '
		print '  '
                min_Px = np.pi*self.X_gridDIM/(2*self.min_X)
		dPx = 2*np.abs(min_Px)/self.X_gridDIM
		px_Vector  = fftpack.fftshift ( np.linspace(min_Px, np.abs(min_Px) - dPx, self.X_gridDIM ))

		min_Py = np.pi*self.Y_gridDIM/(2*self.min_Y)
		dPy = 2*np.abs(min_Py)/self.Y_gridDIM
		py_Vector  = fftpack.fftshift ( np.linspace(min_Py, np.abs(min_Py) - dPy, self.Y_gridDIM ))


		px = px_Vector[np.newaxis,:]
		py = py_Vector[:,np.newaxis]


		sqrtp = sign*2*np.sqrt( self.mass*self.mass*self.c**4 + self.c*self.c*px*px + self.c*self.c*py*py )
		aa = sign*self.mass*self.c*self.c/sqrtp
		bb = sign*(px/sqrtp - 1j*py/sqrtp)
		cc = sign*(px/sqrtp + 1j*py/sqrtp)
	        
		ElectronProjector = np.matrix([ [0.5+aa , 0.  , 0.  , bb  ],
						[0. , 0.5+aa  , cc  , 0.  ],
						[0. , bb  , 0.5-aa  , 0.  ],
						[cc , 0.  , 0.  , 0.5-aa] ])

		psi1_fft = fftpack.fft2( self.Psi1_init   ) 
		psi2_fft = fftpack.fft2( self.Psi2_init   ) 
		psi3_fft = fftpack.fft2( self.Psi3_init   ) 
		psi4_fft = fftpack.fft2( self.Psi4_init   ) 		
		
		psi1_fft_electron = ElectronProjector[0,0]*psi1_fft + ElectronProjector[0,1]*psi2_fft +\
		ElectronProjector[0,2]*psi3_fft + ElectronProjector[0,3]*psi4_fft	

		psi2_fft_electron = ElectronProjector[1,0]*psi1_fft + ElectronProjector[1,1]*psi2_fft +\
		ElectronProjector[1,2]*psi3_fft + ElectronProjector[1,3]*psi4_fft

                psi3_fft_electron = ElectronProjector[2,0]*psi1_fft + ElectronProjector[2,1]*psi2_fft +\
		ElectronProjector[2,2]*psi3_fft + ElectronProjector[2,3]*psi4_fft	

                psi4_fft_electron = ElectronProjector[3,0]*psi1_fft + ElectronProjector[3,1]*psi2_fft +\
		ElectronProjector[3,2]*psi3_fft + ElectronProjector[3,3]*psi4_fft

                self.Psi1_init  = fftpack.ifft2( psi1_fft_electron   ) 
		self.Psi2_init  = fftpack.ifft2( psi2_fft_electron   ) 
		self.Psi3_init  = fftpack.ifft2( psi3_fft_electron   ) 
		self.Psi4_init  = fftpack.ifft2( psi4_fft_electron   ) 					



	def FilterElectrons(self,sign, Psi):
 		'''
		Routine that uses the Fourier transform to filter positrons/electrons
		Options:
			sign=1   Leaves electrons
			sign=-1	 Leaves positrons
		'''
		print '  '
		print '  	Filter Electron routine '
		print '  '

		px = self.c*self.Px
		py = self.c*self.Py
		
		m = self.mass
		c= self.c

		energy = np.sqrt(  (m*c**2)**2 + px**2 + py**2 )

		EP_11 = 1. + sign*m*c**2/energy
		EP_12 = 0.
		EP_13 = 0.
		EP_14 = sign*(px - 1j*py)/energy
		
		EP_21 = 0.
		EP_22 = 1. + sign*m*c**2/energy
		EP_23 = sign*(px + 1j*py)/energy
		EP_24  = 0.

		EP_31 = 0.
		EP_32 = sign*(px - 1j*py)/energy
		EP_33 = 1. - sign*m*c**2/energy
		EP_34 = 0.

		EP_41 = sign*(px + 1j*py)/energy
		EP_42 = 0.
		EP_43 = 0.
		EP_44 = 1. - sign*m*c**2/energy	
		
		#Psi1, Psi2, Psi3, Psi4 = Psi

		psi1_fft = fftpack.fft2( Psi[0]  ) 
		psi2_fft = fftpack.fft2( Psi[1]  ) 
		psi3_fft = fftpack.fft2( Psi[2]  ) 
		psi4_fft = fftpack.fft2( Psi[3]  ) 		
		
		psi1_fft_electron = EP_11*psi1_fft + EP_12*psi2_fft + EP_13*psi3_fft + EP_14*psi4_fft
		psi2_fft_electron = EP_21*psi1_fft + EP_22*psi2_fft + EP_23*psi3_fft + EP_24*psi4_fft		
		psi3_fft_electron = EP_31*psi1_fft + EP_32*psi2_fft + EP_33*psi3_fft + EP_34*psi4_fft
		psi4_fft_electron = EP_41*psi1_fft + EP_42*psi2_fft + EP_43*psi3_fft + EP_44*psi4_fft
						
		return np.array([ fftpack.ifft2( psi1_fft_electron   ),
				  fftpack.ifft2( psi2_fft_electron   ),
	   			  fftpack.ifft2( psi3_fft_electron   ),
				  fftpack.ifft2( psi4_fft_electron   )  ])


	def save_Spinor(self,f1, t, Psi1_GPU,Psi2_GPU,Psi3_GPU,Psi4_GPU):
		print ' progress ', 100*t/(self.timeSteps+1), '%'

		PsiTemp = Psi1_GPU.get()
		f1['1/real/'+str(t)] = PsiTemp.real
		f1['1/imag/'+str(t)] = PsiTemp.imag

		PsiTemp = Psi2_GPU.get()
		f1['2/real/'+str(t)] = PsiTemp.real
		f1['2/imag/'+str(t)] = PsiTemp.imag

		PsiTemp = Psi3_GPU.get()
		f1['3/real/'+str(t)] = PsiTemp.real
		f1['3/imag/'+str(t)] = PsiTemp.imag

		PsiTemp = Psi4_GPU.get()
		f1['4/real/'+str(t)] = PsiTemp.real
		f1['4/imag/'+str(t)] = PsiTemp.imag

	def save_Density(self,f1,t,Psi1_GPU,Psi2_GPU,Psi3_GPU,Psi4_GPU):
		print ' progress ', 100*t/(self.timeSteps+1), '%'
		PsiTemp1 = Psi1_GPU.get()
		PsiTemp2 = Psi2_GPU.get()
		PsiTemp3 = Psi3_GPU.get()
		PsiTemp4 = Psi4_GPU.get()
		
		rho  =  np.abs(PsiTemp1)**2 
		rho +=  np.abs(PsiTemp2)**2 
		rho +=  np.abs(PsiTemp3)**2 
		rho +=  np.abs(PsiTemp4)**2 

		#print ' Save norm = ', np.sum(rho)*self.dX*self.dY

		f1[str(t)] = np.ascontiguousarray(fftpack.fftshift(rho).astype(np.float32))
  

	def load_Density(self, n, fileName=None ):

		if fileName==None:
			FILE = h5py.File(self.fileName)
		else :
			FILE = h5py.File(fileName)

		probability = FILE['/'+str(n)][...]
		FILE.close()

		return probability     

	def load_Spinor(self, n, fileName=None ):

		if fileName==None:
			FILE = h5py.File(self.fileName)
		else :
			FILE = h5py.File(fileName)

		psi1 = FILE['1/real/'+str(n)][...] + 1j*FILE['1/imag/'+str(n)][...]
		psi2 = FILE['2/real/'+str(n)][...] + 1j*FILE['2/imag/'+str(n)][...]
		psi3 = FILE['3/real/'+str(n)][...] + 1j*FILE['3/imag/'+str(n)][...]
		psi4 = FILE['4/real/'+str(n)][...] + 1j*FILE['4/imag/'+str(n)][...]

		FILE.close()

		return np.array([ psi1, psi2, psi3, psi4 ])     


	def Density_From_Spinor(self,Psi):
		rho  = np.abs(Psi[0])**2
		rho += np.abs(Psi[1])**2
		rho += np.abs(Psi[2])**2
		rho += np.abs(Psi[3])**2
		return rho


	def Norm( self, Psi):
		norm  = np.sum(np.abs(Psi[0])**2)
		norm += np.sum(np.abs(Psi[1])**2)
		norm += np.sum(np.abs(Psi[2])**2)
		norm += np.sum(np.abs(Psi[3])**2)
		norm *= self.dX*self.dY
		norm = np.sqrt(norm)		

		return norm


	def Norm_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm  = gpuarray.sum( Psi1.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi2.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi3.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi4.__abs__()**2  ).get()

		norm = np.sqrt(norm*self.dX * self.dY )

		#print '               norm GPU = ', norm		
		
		return norm

	def Normalize_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm = self.Norm_GPU(Psi1, Psi2, Psi3, Psi4)
		Psi1 /= norm
		Psi2 /= norm
		Psi3 /= norm
		Psi4 /= norm

	#........................................................................

	def Average_X( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.X_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.X_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.X_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.X_GPU).get()

		average *= self.dX*self.dY

		return average	

	def Average_Y( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.Y_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.Y_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.Y_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.Y_GPU).get()

		average *= self.dX*self.dY

		return average		

	def Average_Px( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.Px_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.Px_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.Px_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.Px_GPU).get()

		average *= self.dX*self.dY

		return average	

	def Average_Py( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.Py_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.Py_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.Py_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.Py_GPU).get()

		average *= self.dX*self.dY

		return average		

	#........................................................................

	def _Average_Alpha1( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average  =  gpuarray.dot(Psi4_GPU,Psi1_GPU.conj()).get()
		average +=  gpuarray.dot(Psi3_GPU,Psi2_GPU.conj()).get()
		average +=  gpuarray.dot(Psi2_GPU,Psi3_GPU.conj()).get()
		average +=  gpuarray.dot(Psi1_GPU,Psi4_GPU.conj()).get()

		average *= self.dX*self.dY

		return average

	def Average_Alpha1( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average  =  gpuarray.dot( Psi2_GPU, Psi3_GPU.conj() ).get().real
		average +=  gpuarray.dot( Psi1_GPU, Psi4_GPU.conj() ).get().real

		average *= 2.*self.dX*self.dY

		return average

	def Average_Alpha2( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average  = - gpuarray.dot(Psi4_GPU,Psi1_GPU.conj()).get()
		average +=   gpuarray.dot(Psi3_GPU,Psi2_GPU.conj()).get()
		average += - gpuarray.dot(Psi2_GPU,Psi3_GPU.conj()).get()
		average +=   gpuarray.dot(Psi1_GPU,Psi4_GPU.conj()).get()

		average *= 1j*self.dX*self.dY

		return average

	def Average_Beta( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average =     gpuarray.dot(Psi1_GPU,Psi1_GPU.conj()).get()		
		average +=    gpuarray.dot(Psi2_GPU,Psi2_GPU.conj()).get()
		average +=  - gpuarray.dot(Psi3_GPU,Psi3_GPU.conj()).get()
		average +=  - gpuarray.dot(Psi4_GPU,Psi4_GPU.conj()).get()
		
		average *= self.dX*self.dY

		return average

	def Average_KEnergy( self, temp_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
				
		energy  = gpuarray.sum( Psi1_GPU*Psi1_GPU.conj() ).get()
		energy += gpuarray.sum( Psi2_GPU*Psi2_GPU.conj() ).get()
		energy -= gpuarray.sum( Psi3_GPU*Psi3_GPU.conj() ).get()
		energy -= gpuarray.sum( Psi4_GPU*Psi4_GPU.conj() ).get()

		energy *= self.mass*self.c*self.c*self.dPx*self.dPy	
		
		#
		temp_GPU *= 0.

		temp_GPU += Psi4_GPU * Psi1_GPU.conj()
		temp_GPU += Psi1_GPU * Psi4_GPU.conj()
		temp_GPU += Psi3_GPU * Psi2_GPU.conj()
		temp_GPU += Psi2_GPU * Psi3_GPU.conj()
	
		temp_GPU *= self.Px_GPU
		#temp_GPU *= self.c

		energy += gpuarray.sum( temp_GPU ).get()*self.dPx*self.dPy*self.c
		#
		temp_GPU *= 0.

		temp_GPU += Psi4_GPU * Psi1_GPU.conj()
		temp_GPU -= Psi1_GPU * Psi4_GPU.conj()
		temp_GPU -= Psi3_GPU * Psi2_GPU.conj()
		temp_GPU += Psi2_GPU * Psi3_GPU.conj()

		temp_GPU *= self.Py_GPU
		#temp_GPU *= -1j

		energy += gpuarray.sum( temp_GPU ).get()*self.dPx*self.dPy*self.c*(-1j)

		return energy


	def Potential_0_Average(self, temp_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU ,t):

		self.Potential_0_Average_Function( temp_GPU, 
			 Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU, t , block=self.blockCUDA, grid=self.gridCUDA  )

		return   self.dX*self.dY * gpuarray.sum(temp_GPU).get()	

	def Norm_X_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm  = gpuarray.sum( Psi1.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi2.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi3.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi4.__abs__()**2  ).get()

		norm = np.sqrt(norm*self.dX * self.dY )	
		
		return norm

	def Norm_P_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm  = gpuarray.sum( Psi1.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi2.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi3.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi4.__abs__()**2  ).get()

		norm = np.sqrt(norm*self.dPx * self.dPy )	
		
		return norm


	def Normalize_X_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm = self.Norm_X_GPU(Psi1, Psi2, Psi3, Psi4)
		Psi1 /= norm
		Psi2 /= norm
		Psi3 /= norm
		Psi4 /= norm

	def Normalize_P_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm = self.Norm_P_GPU(Psi1, Psi2, Psi3, Psi4)
		Psi1 /= norm
		Psi2 /= norm
		Psi3 /= norm
		Psi4 /= norm

	#.....................................................................

	def Run(self):
		try :
			import os
			os.remove (self.fileName)
				
		except OSError:
			pass
		
		f1 = h5py.File(self.fileName)


		print '--------------------------------------------'
		print '              Dirac Propagator 2D           '
		print '--------------------------------------------'
		print '  save Mode  =  ',	self.frameSaveMode			

		f1['x_gridDIM'] = self.X_gridDIM
		f1['y_gridDIM'] = self.Y_gridDIM

		#f1['x_min'] = self.min_X
		#f1['y_min'] = self.min_Y
		f1['x_amplitude'] = self.X_amplitude
		f1['y_amplitude'] = self.Y_amplitude

		#  Redundant information on dx dy dz       
		f1['dx'] = self.dX
		f1['dy'] = self.dY

		

		f1['Potential_0_String'] = self.Potential_0_String
		f1['Potential_1_String'] = self.Potential_1_String
		f1['Potential_2_String'] = self.Potential_2_String
		f1['Potential_3_String'] = self.Potential_3_String

		self.Psi1_init, self.Psi2_init, self.Psi3_init, self.Psi4_init = self.Psi_init		

		Psi1_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi1_init, dtype=np.complex128) )
		Psi2_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi2_init, dtype=np.complex128) )
		Psi3_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi3_init, dtype=np.complex128) )
		Psi4_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi4_init, dtype=np.complex128) )

		_Psi1_GPU = gpuarray.zeros_like(Psi1_GPU)
		_Psi2_GPU = gpuarray.zeros_like(Psi1_GPU)
		_Psi3_GPU = gpuarray.zeros_like(Psi1_GPU)
		_Psi4_GPU = gpuarray.zeros_like(Psi1_GPU)

		#

		print '                                                               '
		print 'number of steps  =  ', self.timeSteps, ' dt = ',self.dt
		print 'dX = ', self.dX, 'dY = ', self.dY
		print 'dPx = ', self.dPx, 'dPy = ', self.dPy
		print '                                                               '
		print '  '


		if self.frameSaveMode=='Spinor':
			self.save_Spinor(f1, 0 , Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)
		if self.frameSaveMode=='Density':
			self.save_Density(f1, 0, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

		#  ............................... Main LOOP .....................................
 
		self.blockCUDA = (self.X_gridDIM,1,1)
		self.gridCUDA  = (self.Y_gridDIM,1)

		timeRange = range(1, self.timeSteps+1)

		initial_time = time.time()

		X_average       = []
		Y_average       = []
		Alpha1_average  = []
		Alpha2_average  = []
		Beta_average    = []
		
		KEnergy_average = []
		Potential_0_average = [] 	

		self.Normalize_X_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

		for t_index in timeRange:

				t_GPU = np.float64(self.dt * t_index )
							

				X_average.append( self.Average_X( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU) )
				Y_average.append( self.Average_Y( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU) )

				Alpha1_average.append( self.Average_Alpha1( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)  )
				Alpha2_average.append( self.Average_Alpha2( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)  )
				Beta_average.append(   self.Average_Beta(   Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)  )

				Potential_0_average.append( 
					self.Potential_0_Average(  _Psi1_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU ,t_GPU)  )	


				self.Fourier_4_X_To_P_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

				#..................................................
				#          Kinetic 
				#..................................................
				self.Normalize_P_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

				KEnergy_average.append(
					self.Average_KEnergy( _Psi1_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU) )

				self.DiracPropagatorK(  Psi1_GPU,  Psi2_GPU,  Psi3_GPU,  Psi4_GPU,
					 		block=self.blockCUDA, grid=self.gridCUDA )

				self.Fourier_4_P_To_X_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)


				#..............................................
				#             Mass potential
				#..............................................

				self.DiracPropagatorA( Psi1_GPU,  Psi2_GPU,  Psi3_GPU,  Psi4_GPU,
							   t_GPU, block=self.blockCUDA, grid=self.gridCUDA )
				
				#        Absorbing boundary
				
				self.DiracAbsorbBoundary_xy(
				Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
				block=self.blockCUDA, grid=self.gridCUDA )
				
				#
				#	Normalization
				#
				self.Normalize_X_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)
				
				#   Saving files

				if t_index % self.skipFrames == 0:
					if self.frameSaveMode=='Spinor':
						self.save_Spinor( f1,t_index,Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)
					if self.frameSaveMode=='Density':
						self.save_Density(f1,t_index,Psi1_GPU,Psi2_GPU,Psi3_GPU,Psi4_GPU)

		final_time = time.time()		
			
		print ' computational time  = ', final_time - initial_time 
	
		f1.close()

		self.Psi_end = np.array( [  Psi1_GPU.get(), Psi2_GPU.get(), Psi3_GPU.get(), Psi4_GPU.get()  ]  )

		self.Psi_init = np.array( [ self.Psi1_init,  self.Psi2_init,  self.Psi3_init,  self.Psi4_init ] )

		self.timeRange      = np.array(timeRange)
		self.X_average      = np.array(X_average).real
		self.Y_average      = np.array(Y_average).real
		self.Alpha1_average = np.array(Alpha1_average).real
		self.Alpha2_average = np.array(Alpha2_average).real
		self.Beta_average   = np.array(Beta_average).real 

		self.KEnergy_average     = np.array( KEnergy_average ).real
		self.Potential_0_average = np.array( Potential_0_average ).real 

		return 0


		
