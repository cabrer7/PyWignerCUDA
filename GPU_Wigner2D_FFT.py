#!/usr/local/epd/bin/python
#-----------------------------------------------------------------------------
#                Time independent Quantum propagator by FFT Split operator
#			
#-----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot 

import scipy.fftpack as fftpack
import h5py
from scipy.special import laguerre

from scipy.special import hyp1f1
from scipy.special import legendre

import time

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import cufft_wrapper as cuda_fft

# =====================================================================================================

exp_PotentialKvN_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__device__ double dPotential(double t, double x)
{
return %s;
}

__device__ double Potential_XTheta(double t, double x, double theta)
{

return -dPotential(t,x)*theta;

}


__global__ void Kernel(  double t_GPU, pycuda::complex<double> *B )
{
//  x runs on thread-blocks and p runs on the grid

    //pycuda::complex<double> I(0,1.);
    double t = t_GPU;
	
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    const int i =  (blockIdx.y                           + P_DIM/2) %% P_DIM ;	
    const int j =  (threadIdx.x + blockDim.x*blockIdx.x  + X_DIM/2) %% X_DIM ;
    	
    const double x     =     dx*( j - 0.5*X_DIM  );
    const double theta = dtheta*( i - 0.5*P_DIM  );

    double phase = dt*Potential_XTheta(t, x, theta);
    double r = exp( - D_Theta*theta*theta*dt );
	  	
    B[ indexTotal ] *= pycuda::complex<double>(  r*cos(phase)  ,  -r*sin(phase)  );

    double x_max = dx*(X_DIM-1.)/2.;
    B[indexTotal] *= 1. - exp( - pow(x-x_max,2)/pow(10.*dx,2)   );
    B[indexTotal] *= 1. - exp( - pow(x+x_max,2)/pow(10.*dx,2)   );		
}

"""


# =====================================================================================================

expPotential_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__device__ double Potential(double t, double x)
{
return %s;
}

__device__ double Potential_XTheta(double t, double x, double theta)
{
return (Potential(t, x - hBar*theta/2.) - Potential(t, x + hBar*theta/2.))/hBar;
}


__global__ void Kernel(  double t_GPU, pycuda::complex<double> *B )
{
//  x runs on thread-blocks and p runs on the grid

    //pycuda::complex<double> I(0,1.);
    double t = t_GPU;
	
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    const int i =  (blockIdx.y                           + P_DIM/2) %% P_DIM ;	
    const int j =  (threadIdx.x + blockDim.x*blockIdx.x  + X_DIM/2) %% X_DIM ;
    	
    const double x     =     dx*( j - 0.5*X_DIM  );
    const double theta = dtheta*( i - 0.5*P_DIM  );

    double phase = dt*Potential_XTheta(t, x, theta);
    double r = exp( - D_Theta*theta*theta*dt );
	  	
    B[ indexTotal ] *= pycuda::complex<double>(  r*cos(phase)  ,  -r*sin(phase)  );

    double x_max = dx*(X_DIM-1.)/2.;
    B[indexTotal] *= 1. - exp( - pow(x-x_max,2)/pow(10.*dx,2)   );
    B[indexTotal] *= 1. - exp( - pow(x+x_max,2)/pow(10.*dx,2)   );		
}

"""


#------------------------------------------------------------

expPLambda_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s
__global__ void Kernel( pycuda::complex<double> *B )
{
    pycuda::complex<double> I(0,1.);
	
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    const int i =  (blockIdx.y                          + P_DIM/2) %% P_DIM ;	
    const int j =  (threadIdx.x + blockDim.x*blockIdx.x + X_DIM/2) %% X_DIM ;
 
    double lambda     =  dlambda*(j - 0.5*X_DIM  );
    double p          =       dp*(i - 0.5*P_DIM  );	

    double phase = dt*lambda*p/mass;
    double r = exp( - D_Lambda*lambda*lambda );	

    B[ indexTotal ] *= pycuda::complex<double>( r*cos(phase), -r*sin(phase) );  
}

"""

#------------------------------------------------------------

zero_negative_source = """
#include <pycuda-complex.hpp>
#include<math.h>

__global__ void Kernel( pycuda::complex<double> *Bout, pycuda::complex<double> *B )
{
    
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;
  
    if( B[indexTotal].real()  < 0.  )	
 	   Bout[ indexTotal  ] = B[ indexTotal ];	
    else 
	Bout[ indexTotal ] = 0.;		
}

"""

fft_shift_source = """
#include <pycuda-complex.hpp>
#include<math.h>

__global__ void Kernel( pycuda::complex<double> *Bout, pycuda::complex<double> *B )
{
    
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    const int i =  (blockIdx.y                           + P_DIM/2) %% P_DIM ;	
    const int j =  (threadIdx.x + blockDim.x*blockIdx.x  + X_DIM/2) %% X_DIM ;

    const int jj = blockIdx.y +  gridDim.x*blockIdx.x;	
    const int ii = threadIdx.x;
    
    Bout[ jj + ii*X_DIM  ] = B[ j + i*X_DIM ];	
}

"""


#---------------------------------------------------------------

dampingLaxWendorf_source = """
#include <pycuda-complex.hpp>
#include <math.h>

%s

//
//  Caldeira Legget damping by finite differences
//  

__global__ void Kernel( pycuda::complex<double> *B0 )
//
//  Caldeira Leggett damping
{
    pycuda::complex<double> I(0., 1.);
    pycuda::complex<double> B_plus;
    pycuda::complex<double> B_minus;
    pycuda::complex<double> B_;
    pycuda::complex<double> B_plus_half;
    pycuda::complex<double> B_minus_half;	   

    int X_DIM = blockDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockIdx.y*X_DIM;

    const int i =  (blockIdx.y  + P_DIM/2) %% P_DIM ;	
    const int j =  (threadIdx.x + X_DIM/2) %% X_DIM ;

    int ip=i+1, im=i-1;
    int jp=j+1, jm=j-1;

    const double theta  = dtheta*(j - 0.5*P_DIM );
    const double x      = dx*(i - 0.5*X_DIM );

    const double theta_plus   = dtheta*(jp - 0.5*P_DIM );
    const double theta_minus  = dtheta*(jm - 0.5*P_DIM );	 	
    const double x_plus       = dx*(ip - 0.5*X_DIM );
    const double x_minus      = dx*(im - 0.5*X_DIM );		

    //    -\gamma \theta \partial_{ \theta} 

    if( j>0 && j < X_DIM-1 ){
       B_plus  = B0[jp + i*X_DIM];
       B_minus = B0[jm + i*X_DIM];
       B_      = B0[j  + i*X_DIM];    

       B_plus_half  = 0.5*(B_plus  + B_) - 0.5*gammaDamping*(dt/2.)*(theta_plus +theta)/2.*(B_plus  - B_)/dtheta;
       B_minus_half = 0.5*(B_minus + B_) - 0.5*gammaDamping*(dt/2.)*(theta_minus+theta)/2.*(B_ - B_minus)/dtheta;

       B0[ j + i*X_DIM ] = B_ - 0.5*gammaDamping*dt*theta*( B_plus_half - B_minus_half  )/dtheta;
    }
  
   	
   //B1[0        +  i*X_DIM  ]  = pycuda::complex<double>(0.,0.);
   //B1[P_DIM-1  +  i*X_DIM  ]  = pycuda::complex<double>(0.,0.);
	

}
"""

copy_gpuarray_source = """
#include <pycuda-complex.hpp>
#include<math.h>

__global__ void Kernel(pycuda::complex<double> *W_new , pycuda::complex<double> *W)
{
    
    int X_DIM = blockDim.x*gridDim.x;
    //int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    W_new[indexTotal] = W[indexTotal];
}

"""

sum_stride_source = """
#include <pycuda-complex.hpp>
#include<math.h>

__global__ void Kernel(pycuda::complex<double> *W, pycuda::complex<double> *W_sum, int m)
{
    
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int k = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM; // indexTotal

    int q = int( pow(2., m) );	

    if( k%(2*q) == 0  ){
	W[ k ] += W[ k + q  ];
    
        W_sum[0] = W[0];	
    }	
}

"""

gpuarray_copy_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( pycuda::complex<double> *B_out , pycuda::complex<double> *B_in )
{
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    B_out[ indexTotal ] =  B_in[ indexTotal ];
}

"""

smoothed_LinearDamping_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s
__global__ void Kernel( pycuda::complex<double> *B )
{
	
    int X_DIM = blockDim.x*gridDim.x;
    int P_DIM = gridDim.y;
 
    const int indexTotal = threadIdx.x + blockDim.x*blockIdx.x  + blockIdx.y*X_DIM;

    const int i =  (blockIdx.y                          + P_DIM/2) %% P_DIM ;	
    const int j =  (threadIdx.x + blockDim.x*blockIdx.x + X_DIM/2) %% X_DIM ;
 
    double p    =  dp*(i - 0.5*P_DIM  );	

    double epsilon2 = pow(0.5,2);

    if( p >= 0.  )
    	B[ indexTotal ] *=  p*p/sqrt( p*p + epsilon2  ); 
    else
        B[ indexTotal ] *= -p*p/sqrt( p*p + epsilon2  ); 

}

"""

#=====================================================================================================

class Propagator_Base :
	def SetTimeTrack(self, dt, timeSteps=128, skipFrames=1, fileName='', compression=None):
		self.runTime  = float(dt)*timeSteps
		self.timeSteps = timeSteps
		self.dt = dt 
		self.skipFrames = skipFrames
		self.fileName = fileName
		self.compression = compression
		self.timeRange = range(1, self.timeSteps+1)
		self.__x_p_representation__ 		= 'x_p'

		self.zero_negative_Function = SourceModule( zero_negative_source,arch="sm_20").get_function("Kernel")

		self.sum_stride_Function = SourceModule(sum_stride_source).get_function( "Kernel" )

		self.copy_gpuarray_Function = SourceModule(copy_gpuarray_source).get_function( "Kernel" )

		


	def SetPhaseSpaceBox2D(self, X_gridDIM, P_gridDIM, X_amplitude, P_amplitude ):
		"""
		X_gridDIM: discretization of the x grid. 
		P_gridDIM: discretization of the p grid. This number is restricted to be always less than 1024
		"""	
		self.X_gridDIM   = X_gridDIM
		self.P_gridDIM   = P_gridDIM
		self.X_amplitude = X_amplitude
		self.P_amplitude = P_amplitude
		self.dX =  2.*X_amplitude/float(X_gridDIM)
		self.dP =  2.*P_amplitude/float(P_gridDIM)
		
		self.dTheta  = 2.*np.pi/(2.*P_amplitude)
		self.Theta_amplitude = self.dTheta*P_gridDIM/2.

		self.dLambda = 2.*np.pi/(2.*X_amplitude)
		self.Lambda_amplitude = self.dLambda*X_gridDIM/2.

		self.SetPhaseSpaceGrid2D()

	def SetPhaseSpaceGrid2D(self):
		
		self.X_range      =  np.linspace(-self.X_amplitude      , self.X_amplitude  -self.dX , self.X_gridDIM )
		self.Lambda_range =  np.linspace(-self.Lambda_amplitude , self.Lambda_amplitude-self.dLambda    ,self.X_gridDIM)

		self.Theta_range  = np.linspace(-self.Theta_amplitude  , self.Theta_amplitude - self.dTheta , self.P_gridDIM)
		self.P_range      = np.linspace(-self.P_amplitude      , self.P_amplitude-self.dP           , self.P_gridDIM)	

		self.X      = fftpack.fftshift(self.X_range)[np.newaxis,:]
		self.Theta  = fftpack.fftshift(self.Theta_range)[:,np.newaxis]

		self.Lambda = fftpack.fftshift(self.Lambda_range)[np.newaxis,:]
		self.P      = fftpack.fftshift(self.P_range)[:,np.newaxis]

		# The operators in GPU are fft shifted
		self.X_GPU     = gpuarray.to_gpu( np.ascontiguousarray( self.X + 0.*self.P, dtype = np.complex128) )
		self.P_GPU     = gpuarray.to_gpu( np.ascontiguousarray( self.P + 0.*self.X, dtype = np.complex128) )

		self.Theta_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Theta + 0.*self.X, dtype = np.complex128) )

		self.X2_GPU    = gpuarray.to_gpu( np.ascontiguousarray( self.X**2 + 0.*self.P, dtype=np.complex128))
		self.P2_GPU    = gpuarray.to_gpu( np.ascontiguousarray( self.P**2 + 0.*self.X, dtype=np.complex128))

		self.XP_GPU    = gpuarray.to_gpu( np.ascontiguousarray( self.P*self.X  ,dtype=np.complex128))
		
		
	def symplecticExpK(self,mass, dt, c , Z ):
	    x = Z[0]
	    p = Z[1]
	    return Z + np.array([ dt*c*p/mass, 0 ])
    
	def symplecticExpV(self,mass, dt, d , dVdx, Z ):
	    x = Z[0]
	    p = Z[1]
	    return Z + np.array([ 0, -dt*d*dVdx(0.,x) ])

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

	def SymplecticPropagator(self, dt, n_iterations, Z0, gammaDamping ):
		"""
		Second order classical symplectic propagator 
		Parameters:
		    	mass
		    	dt
		    	dVdx function
		    	Z0 : initial state in phase space
			gammaDamping: damping coefficient				     
		"""
		Z = Z0
		trajectory = []
		for i in range(n_iterations):
 			trajectory.append( np.append(Z,dt*(i+1) ) )  #Add time as  third element
			Z = self.symplecticExpKExpV2( self.mass, dt , self.dPotential ,Z )
			Z[1] = np.exp(-2.*gammaDamping*dt)*Z[1]
		return np.array(trajectory)    

	def SymplecticPropagator_SmoothedP(self, dt, n_iterations, Z0, gammaDamping ):
		"""
		Second order classical symplectic propagator 
		Parameters:
		    	mass
		    	dt
		    	dVdx function
		    	Z0 : initial state in phase space
			gammaDamping: damping coefficient				     
		"""
		Z = Z0
		trajectory = []
		for i in range(n_iterations):
 			trajectory.append( np.append(Z,dt*(i+1) ) )  #Add time as  third element
			Z = self.symplecticExpKExpV2( self.mass, dt , self.dPotential ,Z )
			Z[1] = Z[1] - 2.*gammaDamping*dt*np.sign(Z[1])*self.fDamping( Z[1] )
		return np.array(trajectory)    

	def save_Frame(self,t,Psi0_GPU):
		print ' progress ', 100*t/(self.timeSteps+1), '%'
		PsiTemp = Psi0_GPU.get()
		self.file.create_dataset(str(t), data = PsiTemp )


	def gpuarray_copy(self, dest_GPU, source_GPU):
		"""
		copy gpuarray
		"""	
		floatSize = dest_GPU.dtype.itemsize
		cuda.memcpy_dtod(dest_GPU.ptr, source_GPU.ptr + floatSize*self.gridDIM_X*self.gridDIM_P, 0)
		

	def Potential(self,t,x):
		"""
		Potential used to draw the energy level sets
		"""	
		pow = np.power
		atan = np.arctan
		M_E = np.e
		sqrt = np.sqrt
		exp = np.exp
		return eval ( self.potentialString, np.__dict__, locals() )	

	def dPotential(self,t,x):
		"""
		derivative of Potential 
		"""	
		pow = np.power
		atan = np.arctan
		M_E = np.e
		return eval ( self.dPotentialString, np.__dict__, locals() )


	def Fourier_XTheta_to_XP(self,M):
		return fftpack.ifft( M ,axis=0 )

	def Morse_EnergyLevels(self,n):
	    nu = self.morse_a/(2*np.pi)*np.sqrt(2*self.morse_Depth/self.mass)
	    return nu*(n + 0.5) - (nu*(n + 0.5))**2/(4*self.morse_Depth) - self.morse_Depth

	def MorsePsi(self, n , morse_a, morse_Depth, mass, r0, r):
		x = (r-r0)*morse_a
		LAMBDA = np.sqrt(2*mass*morse_Depth )/morse_a
		z = 2.*LAMBDA*np.exp(-x)
		return z**( LAMBDA - n - 0.5 )*np.exp( -0.5*z )*hyp1f1( -n , 2*LAMBDA - 2*n  , z )

	def fft_shift2D(self, X ):
		"""
		double fft shift in 2D arrays in both axis	
		""" 
		return fftpack.fftshift(fftpack.fftshift(X,axes=1),axes=0)


	def  Wigner_HarmonicOscillator(self,n,omega,x,p):
	        """
	        Wigner function of the Harmonic oscillator
		Parameters
		s    : standard deviation in x
		x,p  : center of packet
		n    : Quantum number  
	        """
	        self.representation = 'x_p'
	        r2 = self.mass*omega**2*((self.X - x))**2 + ((self.P - p ))**2/self.mass  

	        W =  (-1)**(n)*laguerre(n)( 2*r2  )*np.exp(-r2 )
	        norm = np.sum( W )*self.dX*self.dP
	        return W/norm	


	def Heaviside(self,x):
    		"""
		Heavisite step function
		"""
        	return 0.5*(np.sign(x) + 1.)


	def Psi_Half_HarmonicOscillator(self,n,omega,x0,p0,X):
		#return np.exp( - 0.5*X**2 )
		k = np.sqrt( self.mass*omega/self.hBar ) 
		return np.exp( 1j*p0 )*np.exp( -0.5*k**2*(X-x0)**2 )*legendre(n)( k*(X-x0) )*self.Heaviside( -(X-x0) )		

	def  Wigner_Half_HarmonicOscillator(self,n,omega,x0,p0):
	        """
	        Wigner function of the Harmonic oscillator
		Parameters:
		x0,p0  : center of packet
		n      : Quantum number that must be odd  
	        """
		ncols = self.X_range.shape[0]

		X_minus = self.X_range[np.newaxis,:] - 0.5*self.hBar*self.Theta_range[ :, np.newaxis ]
		       
		X_plus  = self.X_range[np.newaxis,:] + 0.5*self.hBar*self.Theta_range[ :, np.newaxis ]
		
		psi_minus = self.Psi_Half_HarmonicOscillator( n,omega,x0,p0, X_minus )
		psi_plus  = self.Psi_Half_HarmonicOscillator( n,omega,x0,p0, X_plus  )
		W = psi_minus * psi_plus.conj() 

		#W = fftpack.fftshift(  np.exp( -X_minus**2  )*np.exp( -X_plus**2  )  )+0j

		W = self.Fourier_Theta_to_P_CPU( fftpack.fftshift(W) )

	        norm = np.sum( W )*self.dX*self.dP

	        W = W/norm

		print ' norm  W = ', np.sum( W )*self.dX*self.dP
		print '           '

		return W
	

	def Fourier_Theta_to_P_CPU(self, W ):
		return fftpack.ifft( W , axis = 0) 

	def Fourier_P_to_Theta_CPU(self, W ):
		return fftpack.fft( W , axis = 0) 

	def Fourier_X_to_Lambda_CPU(self, W ):
		return fftpack.fft( W , axis = 1) 

	def Fourier_Lambda_to_X_CPU(self, W ):
		return fftpack.ifft( W , axis = 1) 
		
	# GPU 

	def Fourier_X_to_Lambda_GPU(self, W_GPU):
		cuda_fft.fft_Z2Z( W_GPU , W_GPU , self.plan_Z2Z_2D_Axes1 )

	def Fourier_Lambda_to_X_GPU(self, W_GPU):
		cuda_fft.ifft_Z2Z( W_GPU, W_GPU, self.plan_Z2Z_2D_Axes1 )		
		W_GPU *= 1./float(self.X_gridDIM)

	def Fourier_P_to_Theta_GPU(self, W_GPU):
		cuda_fft.fft_Z2Z( W_GPU, W_GPU, self.plan_Z2Z_2D_Axes0 )


	def Fourier_Theta_to_P_GPU(self, W_GPU ):
		cuda_fft.ifft_Z2Z( W_GPU, W_GPU, self.plan_Z2Z_2D_Axes0 )
		W_GPU *= 1./float(self.P_gridDIM)   

	def Fourier_XTheta_to_LambdaP_GPU(self, W_GPU ):
		self.Fourier_X_to_Lambda_GPU( W_GPU)
		self.Fourier_Theta_to_P_GPU( W_GPU )

	def Fourier_LambdaP_to_XTheta_GPU(self, W_GPU):
		self.Fourier_Lambda_to_X_GPU( W_GPU)
		self.Fourier_P_to_Theta_GPU( W_GPU)	

        def sum_gpu_array(self, W_GPU, W_sum_GPU ):
		"""
		Calculates the sum of GPU array W_GPU.
		The size must be power of two	
		"""
		N = W_GPU.size
		#print ' N = ', N
		n = int( np.log2(N) )
		#print ' n = ', n		

		m = np.int32(0)

		#blockCUDA = (512,1,1)
		#gridCUDA  = (self.X_gridDIM/512, self.P_gridDIM)

		for s in range(n):
		    self.sum_stride_Function( W_GPU, W_sum_GPU, m ,  block=self.blockCUDA, grid=self.gridCUDA)
		    m = np.int32( m + 1 )
		    #print ' m = ', m , '  sum = ', np.real( W_sum_GPU.get()[0] )
   
		return np.real(  W_sum_GPU.get()[0]  )

	def SetCUDA_Constants(self):
		self.CUDA_constants =    '__constant__ double dt=%f;'%(self.dt)
		self.CUDA_constants +=   '__constant__ double dx=%f;'%(self.dX)
		self.CUDA_constants +=   '__constant__ double dp=%f;'%(self.dP)
		self.CUDA_constants +=   '__constant__ double dtheta=%f;'%(self.dTheta)
		self.CUDA_constants +=   '__constant__ double dlambda=%f;'%(self.dLambda)
		self.CUDA_constants +=   '__constant__ double mass=%f;'%(self.mass)
		self.CUDA_constants +=   '__constant__ double hBar=%f;'%(self.hBar)
		self.CUDA_constants +=   '__constant__ double D_Theta   =%f;    '%(self.D_Theta)   
		self.CUDA_constants +=   '__constant__ double D_Lambda  =%f;    '%(self.D_Lambda) 

	def WriteHDF5_variables(self):
		self.file['dx'] = self.dX
		self.file['dtheta'] = self.dTheta
		self.file['dp'] = self.dP
		self.file['dlambda'] = self.dLambda

		self.file['x_gridDIM'] = self.X_gridDIM
		self.file['p_gridDIM'] = self.P_gridDIM
		self.file['x_min'] = -self.X_amplitude;	self.file['x_max'] = self.X_amplitude - self.dX
		self.file['p_min'] = -self.P_amplitude;	self.file['p_max'] = self.P_amplitude - self.dP
		self.file['lambda_min'] = self.Lambda_range.min(); self.file['lambda_max'] = self.Lambda_range.max()
		self.file['dt'] = self.dt;
		self.file['timeSteps'] = self.timeSteps
		self.file['skipFrames'] = self.skipFrames

	def SetFFT_Plans(self):
		self.plan_Z2Z_2D = cuda_fft.Plan_Z2Z( (self.P_gridDIM, self.X_gridDIM)   ,  batch=1 )
		#self.plan_Z2Z_1D_Axes0 = cuda_fft.Plan_Z2Z_Axes0( (self.P_gridDIM,) ,  batch = self.X_gridDIM )
		#self.plan_Z2Z_1D_Axes1 = cuda_fft.Plan_Z2Z(       (self.X_gridDIM,) ,  batch = self.P_gridDIM ) 

		self.plan_Z2Z_2D_Axes0 = cuda_fft.Plan_Z2Z_2D_Axis0(  (self.P_gridDIM,self.X_gridDIM)  )
		self.plan_Z2Z_2D_Axes1 = cuda_fft.Plan_Z2Z_2D_Axis1(  (self.P_gridDIM,self.X_gridDIM)  ) 

	def SetCUDA_Functions(self):
		self.expPotentialFunction = SourceModule(\
					expPotential_source%(self.CUDA_constants,self.potentialString),
					arch="sm_20").get_function( "Kernel" )

		self.expPotentialKvNFunction = SourceModule(\
					exp_PotentialKvN_source%(self.CUDA_constants,self.dPotentialString),
					arch="sm_20").get_function( "Kernel" )

		self.expPLambdaFunction = SourceModule(\
					expPLambda_source%(self.CUDA_constants),arch="sm_20").get_function("Kernel")

		self.smoothed_LinearDamping_Function = SourceModule(\
					smoothed_LinearDamping_source%(self.CUDA_constants), arch="sm_20").get_function("Kernel")

		self.gpu_array_copy_Function = SourceModule(									     						gpuarray_copy_source, arch="sm_20").get_function( "Kernel" )

		#self.dampingLaxWendorf_Function = SourceModule(dampingLaxWendorf_source%self.CUDA_constants).get_function("Kernel")

	def WignerFunctionFromFile(self,n):
		FILE = h5py.File(self.fileName)
		W = FILE['/'+str(n)][...]
		FILE.close()
		return W                                

	def WignerMarginal_Probability_x(self,W):
		return np.sum( W , axis=0 )*self.dP  

	def WignerMarginal_Probability_p(self,W):
		return np.sum( W , axis=1 )*self.dX 

	def PlotWignerFrame(self, W_input , plotRange , global_color , energy_Levels ,aspectRatio=1):

	    x_plotRange,p_plotRange = plotRange
	    global_min, global_max  = global_color 
	    
	    W = W_input.copy()
	    W = fftpack.fftshift(W.real)    
	    
	    dp    = self.dP
	    p_min = -self.P_amplitude
	    p_max =  self.P_amplitude - dp    
	    
	    x_min = -self.X_amplitude
	    x_max =  self.X_amplitude - self.dX
		
	    print 'min = ', np.min( W ), ' max = ', np.max( W )
	    #print 'final time =', self.timeRange[-1] ,'a.u.  =',\
	    print 'normalization = ', np.sum( W )*self.dX*dp

	    zero_position =  abs( global_min +1e-3) / (abs( global_max) + abs(global_min)) 
	    wigner_cdict = {'red' 	: 	((0., 0., 0.),
								(zero_position, 1., 1.), 
								(1., 1., 1.)),
						'green' :	((0., 0., 0.),
								(zero_position, 1., 1.),
								(1., 0., 0.)),
						'blue'	:	((0., 1., 1.),
								(zero_position, 1., 1.),
								(1., 0., 0.)) }
	    wigner_cmap = matplotlib.colors.LinearSegmentedColormap('wigner_colormap', wigner_cdict, 512)

	    fig, ax = matplotlib.pyplot.subplots(figsize=(12, 10))

	    cax = ax.imshow( W ,origin='lower',interpolation='none',\
	    extent=[ x_min , x_max, p_min, p_max], vmin= global_min, vmax=global_max, cmap=wigner_cmap)

	    min_energy, max_energy, delta_energy  = energy_Levels
	    
	    ax.contour(self.Hamiltonian ,
		        np.arange( min_energy, max_energy, delta_energy ),
			origin='lower',extent=[x_min,x_max,p_min,p_max],
		       linewidths=0.25,colors='k')
	    
	    axis_font = {'size':'24'}
	    
	    ax.set_xlabel(r'$x$',**axis_font)
	    ax.set_ylabel(r'$p$',**axis_font)
	    
	    ax.set_xlim( (x_plotRange[0] , x_plotRange[1] ) )
	    ax.set_ylim( (p_plotRange[0] , p_plotRange[1] ) )
	    ax.set_aspect(aspectRatio)
	    #ax.grid('on')
	    cbar = fig.colorbar(cax, ticks=[-0.3, -0.2,-0.1, 0, 0.1, 0.2 , 0.3])
	    matplotlib.rcParams.update({'font.size': 18})
	    return fig

	def ThetaP(self, LW_GPU, W_GPU):
		"""
		Caldeira Legget dissipator
		"""
		self.gpu_array_copy_Function( LW_GPU , W_GPU , block=self.blockCUDA , grid=self.gridCUDA )
		LW_GPU *= self.P_GPU
		# x p  ->  theta p
		self.Fourier_P_to_Theta_GPU( LW_GPU )
		LW_GPU *= self.Theta_GPU	
		self.Fourier_Theta_to_P_GPU( LW_GPU )

	def CaldeiraDissipatorOrder2(self, LW_GPU, LW_temp_GPU, W_GPU):
		LW_GPU  *= 0j
		LW_GPU  +=   W_GPU
		self.ThetaP( LW_temp_GPU , W_GPU )
		LW_GPU  +=  1j * self.dt *self.gammaDamping * LW_temp_GPU			
		
		self.ThetaP( LW_temp_GPU , LW_GPU )
		W_GPU  +=  2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU

	def _CaldeiraDissipatorOrder3(self, LW_GPU, LW_temp_GPU, W_GPU ):
		LW_GPU  *= 0j
		LW_GPU  +=   W_GPU
		self.ThetaP( LW_temp_GPU , W_GPU )
		LW_GPU  += 2./3. * 1j * self.dt *self.gammaDamping * LW_temp_GPU			
		
		self.ThetaP( LW_temp_GPU , LW_GPU )
		LW_GPU  +=  2./2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU

		self.ThetaP( LW_temp_GPU , LW_GPU )
		W_GPU  +=  2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU


	def CaldeiraDissipatorOrder3(self, LW_GPU, LW_temp_GPU, W_GPU, dampingFunction):
		LW_GPU  *= 0j
		LW_GPU  +=   W_GPU

		dampingFunction( LW_temp_GPU , W_GPU )
		LW_GPU  += 2./3. * 1j * self.dt *self.gammaDamping * LW_temp_GPU			
		
		dampingFunction( LW_temp_GPU , LW_GPU )
		LW_GPU  +=  2./2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU

		dampingFunction( LW_temp_GPU , LW_GPU )
		W_GPU  +=  2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU

	def LoadEhrenfestFromFile(self):

		FILE = h5py.File( self.fileName ,'r')		

		timeRange           = FILE['timeRange'][...]  
		timeRangeIndexSaved = FILE['timeRangeIndexSaved'][...]   	

		self.X_average   = FILE['/Ehrenfest/X_Ehrenfest'][...]   
		self.X2_average  = FILE['/Ehrenfest/X2_Ehrenfest'][...]

		self.P_average   = FILE['/Ehrenfest/P_Ehrenfest'][...] 
		self.P2_average  = FILE['/Ehrenfest/P2_Ehrenfest'][...]

		self.XP_average = FILE['/Ehrenfest/XP_Ehrenfest'][...]  

		self.dPotentialdX_average  = FILE['/Ehrenfest/dPotentialdX_Ehrenfest'][...] 
		self.PdPotentialdX_average = FILE['/Ehrenfest/PdPotentialdX_average'][...]			
		self.XdPotentialdX_average = FILE['/Ehrenfest/XdPotentialdX_average'][...]
		self.Hamiltonian_average   = FILE['/Ehrenfest/Hamiltonian_average'][...] 

		self.W_init = FILE['W_init'][...] +0j
		self.W_end  = FILE['W_end'][...]  +0j

		FILE.close() 



#=====================================================================================================
#
#        Propagation Wigner
#
#=====================================================================================================

class GPU_Wigner2D_FFT(Propagator_Base):
	"""
	Wigner Propagator in 2D phase space with diffusion and amplituse damping
	"""

	def __init__(self,X_gridDIM,P_gridDIM,X_amplitude,P_amplitude, hBar ,mass,
			D_Theta, D_Lambda, gammaDamping, potentialString, dPotentialString, normalization = 'Wigner'):
		"""
		
		"""
		self.normalization = normalization
		self.D_Theta  = D_Theta
		self.D_Lambda = D_Lambda
		self.gammaDamping      =  gammaDamping

		self.potentialString   =  potentialString
		self.dPotentialString  =  dPotentialString

		self.SetPhaseSpaceBox2D(X_gridDIM, P_gridDIM, X_amplitude, P_amplitude)		
		self.hBar = hBar
		self.mass = mass

		self.SetCUDA_Constants() 

		
		##################
		
		self.SetFFT_Plans()

		self.SetCUDA_Functions()

		self.Hamiltonian =  self.P**2 / (2.*self.mass) + self.Potential(0,self.X) 
		self.Hamiltonian_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Hamiltonian.astype(np.complex128) )  )

		#self.f.create_dataset('Hamiltonian', data = self.Hamiltonian.real )
		self.Hamiltonian   = self.fft_shift2D( self.Hamiltonian )


	def Run(self, dampingFunction = 'CaldeiraLeggett' ):

		try :
			import os
			os.remove (self.fileName)
		except OSError:
			pass

		self.file = h5py.File(self.fileName)

		self.WriteHDF5_variables()
		self.file.create_dataset('Hamiltonian', data = self.Hamiltonian.real )

		print " X_gridDIM = ", self.X_gridDIM, "   P_gridDIM = ", self.P_gridDIM
		print " dx = ", self.dX, " dp = ", self.dP
		print " dLambda = ", self.dLambda, " dTheta = ", self.dTheta
		#print ' 1/gamma = ', 1./self.gammaDamping
		print '  '

		print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		#timeRangeIndex = range(0, self.timeSteps+1)
		timeRangeIndex = range(0, self.timeSteps)

		W_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.W_init , dtype = np.complex128) )	
		norm = gpuarray.sum( W_GPU  ).get()*self.dX*self.dP
		W_GPU /= norm
		

		dPotentialdX = self.dPotential(0. , self.X) + 0.*self.P 
		self.dPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( dPotentialdX.astype(np.complex128) )  )

		PdV = self.P*self.dPotential(0. , self.X)  
		self.PdPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( PdV.astype(np.complex128) )  )

		XdV = self.dPotential(0. , self.X)*self.X + 0.*self.P 
		self.XdPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( XdV.astype(np.complex128) )  )

		LW_GPU      = gpuarray.to_gpu( np.ascontiguousarray( self.W_init , dtype = np.complex128) )
		LW_temp_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.W_init , dtype = np.complex128) )

		# Filter to calculate tunneling time
		Filter=np.array(self.P**2/2. + self.Potential(0,self.X + 0.*self.P )>-0.1,dtype=np.complex128)
		self.Filter_GPU = gpuarray.to_gpu(np.ascontiguousarray(Filter, dtype = np.complex128) )
		#Psi = np.zeros((self.field.size,self.X_gridDIM),dtype = np.complex128)

		print '         GPU memory Free  post gpu loading ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'
		print ' ------------------------------------------------------------------------------- '
		print '     Split Operator Propagator  GPU with damping                                 '
		print ' ------------------------------------------------------------------------------- '

		timeRange           = []
		timeRangeIndexSaved = []

		X_average   = []
		X2_average  = []

		dPotentialdX_average   = []
		PdPotentialdX_average  = []
		XdPotentialdX_average  = []

		P_average              = []
		P2_average  = []

		XP_average  = []
		Overlap     = []

		Hamiltonian_average = []

		negativeArea = []

		dXdP = self.dX * self.dP 

		self.blockCUDA = (512,1,1)
		self.gridCUDA  = (self.X_gridDIM/512, self.P_gridDIM)
		
		for tIndex in timeRangeIndex:
			t = (tIndex)*self.dt
			t_GPU = np.float64(t)


			timeRange.append(t)

			X_average.append(    dXdP*gpuarray.dot(W_GPU,self.X_GPU ).get()  )
			X2_average.append(   dXdP*gpuarray.dot(W_GPU,self.X2_GPU ).get() )

			P_average.append(    dXdP*gpuarray.dot(W_GPU,self.P_GPU  ).get() )	
			P2_average.append(   dXdP*gpuarray.dot(W_GPU,self.P2_GPU ).get() )	
		
			XP_average.append(   dXdP*gpuarray.dot(W_GPU,self.XP_GPU ).get() )	

			dPotentialdX_average.append(
						dXdP*gpuarray.dot(W_GPU,self.dPotentialdX_GPU).get() )			

			PdPotentialdX_average.append(
				dXdP*gpuarray.dot(W_GPU,self.PdPotentialdX_GPU).get() )	

			XdPotentialdX_average.append(
				dXdP*gpuarray.dot(W_GPU,self.XdPotentialdX_GPU).get() )	

			Hamiltonian_average.append(
				dXdP*gpuarray.dot(W_GPU,self.Hamiltonian_GPU).get() )	

			self.zero_negative_Function( LW_temp_GPU , W_GPU ,  block=self.blockCUDA, grid=self.gridCUDA)
			negativeArea.append( gpuarray.sum(LW_temp_GPU).get()*dXdP )			

			if tIndex%self.skipFrames == 0:
				timeRangeIndexSaved.append(tIndex)
				self.save_Frame(tIndex,W_GPU)	   

			# p x  ->  p lambda 
			self.Fourier_X_to_Lambda_GPU( W_GPU )
			self.expPLambdaFunction( W_GPU, block=self.blockCUDA, grid=self.gridCUDA )			
			# p lambda  ->  p x
			self.Fourier_Lambda_to_X_GPU( W_GPU)
			
			# p x -> theta x
			self.Fourier_P_to_Theta_GPU( W_GPU )	
			self.expPotentialFunction( t_GPU, W_GPU, block=self.blockCUDA, grid=self.gridCUDA )
			self.Fourier_Theta_to_P_GPU( W_GPU )	
			
			if self.gammaDamping != 0:
				if dampingFunction == 'CaldeiraLeggett':
					self.CaldeiraDissipatorOrder3( LW_GPU, LW_temp_GPU, W_GPU, self.ThetaP )
				else :
					self.CaldeiraDissipatorOrder3( LW_GPU, LW_temp_GPU, W_GPU, dampingFunction ) 

				if self.normalization == 'Wigner':	
					norm = gpuarray.sum( W_GPU  ).get()*(self.dX*self.dP)
					W_GPU /= norm	


		self.timeRange             = np.array(timeRange)
		#self.Overlap               = np.array(Overlap)

		self.X_average             = np.array(X_average).real
		self.X2_average            = np.array(X2_average).real

		self.P_average             = np.array(P_average).real
		self.P2_average            = np.array(P2_average).real

		self.XP_average            = np.array(XP_average).real

		self.dPotentialdX_average   = np.array( dPotentialdX_average  ).real
		self.PdPotentialdX_average  = np.array( PdPotentialdX_average ).real
		self.XdPotentialdX_average  = np.array( XdPotentialdX_average ).real
		self.Hamiltonian_average    = np.array( Hamiltonian_average   ).real

		self.negativeArea = np.array(negativeArea).real

		self.file['timeRange']             = timeRange
		self.file['timeRangeIndexSaved']   = timeRangeIndexSaved	

		self.file['/Ehrenfest/X_Ehrenfest']  = self.X_average 
		self.file['/Ehrenfest/X2_Ehrenfest'] = self.X2_average 

		self.file['/Ehrenfest/P_Ehrenfest']  = self.P_average 
		self.file['/Ehrenfest/P2_Ehrenfest'] = self.P2_average 

		self.file['/Ehrenfest/XP_Ehrenfest'] = self.XP_average 

		self.file['/Ehrenfest/dPotentialdX_Ehrenfest'] = self.dPotentialdX_average
		self.file['/Ehrenfest/PdPotentialdX_average'] = self.PdPotentialdX_average			
		self.file['/Ehrenfest/XdPotentialdX_average'] = self.XdPotentialdX_average
		self.file['/Ehrenfest/Hamiltonian_average'] = self.Hamiltonian_average

		self.file['W_init'] = self.W_init.real
		self.file['W_end']  = self.W_end.real

		self.file['negativeArea'] = self.negativeArea

		self.file.close()
		cuda_fft.cufftDestroy( self.plan_Z2Z_2D.handle )
		cuda_fft.cufftDestroy( self.plan_Z2Z_2D_Axes0.handle )
		cuda_fft.cufftDestroy( self.plan_Z2Z_2D_Axes1.handle )

		self.W_end =  W_GPU.get().real

		return  self.W_end


#=====================================================================================================
#
#        Propagation Fokker-Planck
#
#=====================================================================================================

class GPU_FokkerPlank2D_FFT(Propagator_Base):
	"""
	Wigner Propagator in 2D phase space with diffusion and amplituse damping
	"""

	def __init__(self,X_gridDIM,P_gridDIM,X_amplitude,P_amplitude, hBar ,mass,
			D_Theta, D_Lambda, gammaDamping, potentialString, dPotentialString, normalization = 'Wigner'):
		"""
		
		"""
		self.normalization = normalization
		self.D_Theta  = D_Theta
		self.D_Lambda = D_Lambda
		self.gammaDamping      =  gammaDamping

		self.potentialString   =  potentialString
		self.dPotentialString  =  dPotentialString

		self.SetPhaseSpaceBox2D(X_gridDIM, P_gridDIM, X_amplitude, P_amplitude)		
		self.hBar = hBar
		self.mass = mass

		self.SetCUDA_Constants() 

		
		##################
		
		self.SetFFT_Plans()

		self.SetCUDA_Functions()

		self.Hamiltonian =  self.P**2 / (2.*self.mass) + self.Potential(0,self.X) 
		self.Hamiltonian_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Hamiltonian.astype(np.complex128) )  )

		#self.f.create_dataset('Hamiltonian', data = self.Hamiltonian.real )
		self.Hamiltonian   = self.fft_shift2D( self.Hamiltonian )


	def Run(self, dampingFunction = 'CaldeiraLeggett' ):

		try :
			import os
			os.remove (self.fileName)
		except OSError:
			pass

		self.file = h5py.File(self.fileName)

		self.WriteHDF5_variables()
		self.file.create_dataset('Hamiltonian', data = self.Hamiltonian.real )

		print " X_gridDIM = ", self.X_gridDIM, "   P_gridDIM = ", self.P_gridDIM
		print " dx = ", self.dX, " dp = ", self.dP
		print " dLambda = ", self.dLambda, " dTheta = ", self.dTheta
		#print ' 1/gamma = ', 1./self.gammaDamping
		print '  '

		print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		#timeRangeIndex = range(0, self.timeSteps+1)
		timeRangeIndex = range(0, self.timeSteps)

		W_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.W_init , dtype = np.complex128) )	
		norm = gpuarray.sum( W_GPU  ).get()*self.dX*self.dP
		W_GPU /= norm
		

		dPotentialdX = self.dPotential(0. , self.X) + 0.*self.P 
		self.dPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( dPotentialdX.astype(np.complex128) )  )

		PdV = self.P*self.dPotential(0. , self.X)  
		self.PdPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( PdV.astype(np.complex128) )  )

		XdV = self.dPotential(0. , self.X)*self.X + 0.*self.P 
		self.XdPotentialdX_GPU = gpuarray.to_gpu( np.ascontiguousarray( XdV.astype(np.complex128) )  )

		LW_GPU      = gpuarray.to_gpu( np.ascontiguousarray( self.W_init , dtype = np.complex128) )
		LW_temp_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.W_init , dtype = np.complex128) )

		# Filter to calculate tunneling time
		Filter=np.array(self.P**2/2. + self.Potential(0,self.X + 0.*self.P )>-0.1,dtype=np.complex128)
		self.Filter_GPU = gpuarray.to_gpu(np.ascontiguousarray(Filter, dtype = np.complex128) )
		#Psi = np.zeros((self.field.size,self.X_gridDIM),dtype = np.complex128)

		print '         GPU memory Free  post gpu loading ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'
		print ' ------------------------------------------------------------------------------- '
		print '     Split Operator Propagator  GPU with damping                                 '
		print ' ------------------------------------------------------------------------------- '

		timeRange           = []
		timeRangeIndexSaved = []

		X_average   = []
		X2_average  = []

		dPotentialdX_average   = []
		PdPotentialdX_average  = []
		XdPotentialdX_average  = []

		P_average              = []
		P2_average  = []

		XP_average  = []
		Overlap     = []

		Hamiltonian_average = []

		negativeArea = []

		dXdP = self.dX * self.dP 

		self.blockCUDA = (512,1,1)
		self.gridCUDA  = (self.X_gridDIM/512, self.P_gridDIM)
		
		for tIndex in timeRangeIndex:
			t = (tIndex)*self.dt
			t_GPU = np.float64(t)


			timeRange.append(t)

			X_average.append(    dXdP*gpuarray.dot(W_GPU,self.X_GPU ).get()  )
			X2_average.append(   dXdP*gpuarray.dot(W_GPU,self.X2_GPU ).get() )

			P_average.append(    dXdP*gpuarray.dot(W_GPU,self.P_GPU  ).get() )	
			P2_average.append(   dXdP*gpuarray.dot(W_GPU,self.P2_GPU ).get() )	
		
			XP_average.append(   dXdP*gpuarray.dot(W_GPU,self.XP_GPU ).get() )	

			dPotentialdX_average.append(
						dXdP*gpuarray.dot(W_GPU,self.dPotentialdX_GPU).get() )			

			PdPotentialdX_average.append(
				dXdP*gpuarray.dot(W_GPU,self.PdPotentialdX_GPU).get() )	

			XdPotentialdX_average.append(
				dXdP*gpuarray.dot(W_GPU,self.XdPotentialdX_GPU).get() )	

			Hamiltonian_average.append(
				dXdP*gpuarray.dot(W_GPU,self.Hamiltonian_GPU).get() )	

			self.zero_negative_Function( LW_temp_GPU , W_GPU ,  block=self.blockCUDA, grid=self.gridCUDA)
			negativeArea.append( gpuarray.sum(LW_temp_GPU).get()*dXdP )			

			if tIndex%self.skipFrames == 0:
				timeRangeIndexSaved.append(tIndex)
				self.save_Frame(tIndex,W_GPU)	   

			# p x  ->  p lambda 
			self.Fourier_X_to_Lambda_GPU( W_GPU )
			self.expPLambdaFunction( W_GPU, block=self.blockCUDA, grid=self.gridCUDA )			
			# p lambda  ->  p x
			self.Fourier_Lambda_to_X_GPU( W_GPU)
			
			# p x -> theta x
			self.Fourier_P_to_Theta_GPU( W_GPU )	
			self.expPotentialKvNFunction( t_GPU, W_GPU, block=self.blockCUDA, grid=self.gridCUDA )
			self.Fourier_Theta_to_P_GPU( W_GPU )	
			
			if self.gammaDamping != 0:
				if dampingFunction == 'CaldeiraLeggett':
					self.CaldeiraDissipatorOrder3( LW_GPU, LW_temp_GPU, W_GPU, self.ThetaP )
				else :
					self.CaldeiraDissipatorOrder3( LW_GPU, LW_temp_GPU, W_GPU, dampingFunction ) 

				if self.normalization == 'Wigner':	
					norm = gpuarray.sum( W_GPU  ).get()*(self.dX*self.dP)
					W_GPU /= norm	


		self.timeRange             = np.array(timeRange)

		self.X_average             = np.array(X_average).real
		self.X2_average            = np.array(X2_average).real

		self.P_average             = np.array(P_average).real
		self.P2_average            = np.array(P2_average).real

		self.XP_average            = np.array(XP_average).real

		self.dPotentialdX_average   = np.array( dPotentialdX_average  ).real
		self.PdPotentialdX_average  = np.array( PdPotentialdX_average ).real
		self.XdPotentialdX_average  = np.array( XdPotentialdX_average ).real
		self.Hamiltonian_average    = np.array( Hamiltonian_average   ).real

		self.negativeArea = np.array(negativeArea).real

		self.file['timeRange']             = timeRange
		self.file['timeRangeIndexSaved']   = timeRangeIndexSaved	

		self.file['/Ehrenfest/X_Ehrenfest']  = self.X_average 
		self.file['/Ehrenfest/X2_Ehrenfest'] = self.X2_average 

		self.file['/Ehrenfest/P_Ehrenfest']  = self.P_average 
		self.file['/Ehrenfest/P2_Ehrenfest'] = self.P2_average 

		self.file['/Ehrenfest/XP_Ehrenfest'] = self.XP_average 

		self.file['/Ehrenfest/dPotentialdX_Ehrenfest'] = self.dPotentialdX_average
		self.file['/Ehrenfest/PdPotentialdX_average'] = self.PdPotentialdX_average			
		self.file['/Ehrenfest/XdPotentialdX_average'] = self.XdPotentialdX_average
		self.file['/Ehrenfest/Hamiltonian_average'] = self.Hamiltonian_average
		

		self.file['negativeArea'] = self.negativeArea

		cuda_fft.cufftDestroy( self.plan_Z2Z_2D.handle )
		cuda_fft.cufftDestroy( self.plan_Z2Z_2D_Axes0.handle )
		cuda_fft.cufftDestroy( self.plan_Z2Z_2D_Axes1.handle )

		self.W_end =  W_GPU.get().real

		self.file['W_init'] = self.W_init.real
		self.file['W_end']  = self.W_end.real


		self.file.close()
		return  0



