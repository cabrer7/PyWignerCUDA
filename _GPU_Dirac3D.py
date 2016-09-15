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

K_Energy_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

{CUDAconstants} 

__global__ void Kernel( double *Weight , 
pycuda::complex<double>* Psi1,  pycuda::complex<double>* Psi2,  pycuda::complex<double>* Psi3,  pycuda::complex<double>* Psi4)
{{

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  double  weight1 , weight2 ;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double px = dPx*(j - DIM_X/2);
  double py = dPy*(i - DIM_Y/2);
  double pz = dPz*(k - DIM_Z/2);

  weight1   =  pycuda::real<double>( Psi1[indexTotal] * pycuda::conj( Psi1[indexTotal] )  );
  weight1  +=  pycuda::real<double>( Psi2[indexTotal] * pycuda::conj( Psi2[indexTotal] )  );
  weight1  -=  pycuda::real<double>( Psi3[indexTotal] * pycuda::conj( Psi3[indexTotal] )  );
  weight1  -=  pycuda::real<double>( Psi4[indexTotal] * pycuda::conj( Psi4[indexTotal] )  );

  weight1 *= mass*c*c;

  weight2 = weight1;
  
  //...............................

  weight1   =  pycuda::real<double>( Psi4[indexTotal] * pycuda::conj( Psi1[indexTotal] )  );
  weight1  +=  pycuda::real<double>( Psi3[indexTotal] * pycuda::conj( Psi2[indexTotal] )  );
   
  weight1 *= 2*c*px;
  weight2 += weight1;
  //...............................
 
  weight1   =  pycuda::imag<double>( Psi4[indexTotal] * pycuda::conj( Psi1[indexTotal] )  );
  weight1  +=  pycuda::imag<double>( Psi2[indexTotal] * pycuda::conj( Psi3[indexTotal] )  );
   
  weight1 *= 2*c*py;
  weight2 += weight1;
  //...............................

  weight1   =  pycuda::real<double>( Psi3[indexTotal] * pycuda::conj( Psi1[indexTotal] )  );
  weight1  -=  pycuda::real<double>( Psi4[indexTotal] * pycuda::conj( Psi2[indexTotal] )  );
   
  weight1 *= 2*c*pz;
  weight2 += weight1;
  
  Weight[indexTotal] = weight2;

}}
"""

#

Potential_0_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

{CUDAconstants} 

__device__  double Potential0(double t, double x, double y, double z)
{{
    return {A0};
}}

//............................................................................................................

__global__ void Kernel( double *Weight, 
pycuda::complex<double>* Psi1,  pycuda::complex<double>* Psi2,  pycuda::complex<double>* Psi3,  pycuda::complex<double>* Psi4,
double t)
{{

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  //pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);
  double z = dZ*(k - DIM_Z/2);

  double out;

  out =  Potential0( t, x, y, z)* pow( abs( Psi1[indexTotal] ) , 2 );
  out += Potential0( t, x, y, z)* pow( abs( Psi2[indexTotal] ) , 2 );
  out += Potential0( t, x, y, z)* pow( abs( Psi3[indexTotal] ) , 2 );
  out += Potential0( t, x, y, z)* pow( abs( Psi4[indexTotal] ) , 2 );

  Weight[indexTotal] = out;

}}

"""

#-------------------------------------------------------------------------------

Average_Px_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 double *weighted,
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double px = dPx*( j - DIM_X/2 );
  //double py = dY*( i - DIM_Y/2 );
  //double pz = dZ*( k - DIM_Z/2 );

  double out;
  out  =  px * pow( abs(Psi1[indexTotal]) ,2);
  out +=  px * pow( abs(Psi2[indexTotal]) ,2);
  out +=  px * pow( abs(Psi3[indexTotal]) ,2);
  out +=  px * pow( abs(Psi4[indexTotal]) ,2);

  weighted[ indexTotal ] = out;
	
}}
"""

Average_Py_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 double *weighted,
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  //double px = dPx*( j - DIM_X/2 );
    double py = dPy*( i - DIM_Y/2 );
  //double pz = dPz*( k - DIM_Z/2 );

  double out;
  out  =  py * pow( abs(Psi1[indexTotal]) ,2);
  out +=  py * pow( abs(Psi2[indexTotal]) ,2);
  out +=  py * pow( abs(Psi3[indexTotal]) ,2);
  out +=  py * pow( abs(Psi4[indexTotal]) ,2);

  weighted[ indexTotal ] = out;
	
}}
"""

Average_Pz_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 double *weighted,
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  //double px = dPx*( j - DIM_X/2 );
  //double py = dPy*( i - DIM_Y/2 );
  double pz = dPz*( k - DIM_Z/2 );

  double out;
  out  =  pz * pow( abs(Psi1[indexTotal]) ,2);
  out +=  pz * pow( abs(Psi2[indexTotal]) ,2);
  out +=  pz * pow( abs(Psi3[indexTotal]) ,2);
  out +=  pz * pow( abs(Psi4[indexTotal]) ,2);

  weighted[ indexTotal ] = out;
	
}}
"""

#-------------------------------------------------------------------------------

Average_X_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 double *weighted,
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);
  double z = dZ*(k - DIM_Z/2);

  double out;
  out  =  x * pow( abs(Psi1[indexTotal]) ,2);
  out +=  x * pow( abs(Psi2[indexTotal]) ,2);
  out +=  x * pow( abs(Psi3[indexTotal]) ,2);
  out +=  x * pow( abs(Psi4[indexTotal]) ,2);

  weighted[ indexTotal ] = out;
	
}}
"""

Average_Y_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 double *weighted,
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  //pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);
  double z = dZ*(k - DIM_Z/2);

  double out;
  out  =  y * pow( abs(Psi1[indexTotal]) ,2);
  out +=  y * pow( abs(Psi2[indexTotal]) ,2);
  out +=  y * pow( abs(Psi3[indexTotal]) ,2);
  out +=  y * pow( abs(Psi4[indexTotal]) ,2);

  weighted[ indexTotal ] = out;
	
}}
"""

Average_Z_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 double *weighted,
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);
  double z = dZ*(k - DIM_Z/2);

  double out;
  out  =  z * pow( abs(Psi1[indexTotal]) ,2);
  out +=  z * pow( abs(Psi2[indexTotal]) ,2);
  out +=  z * pow( abs(Psi3[indexTotal]) ,2);
  out +=  z * pow( abs(Psi4[indexTotal]) ,2);

  weighted[ indexTotal ] = out;
	
}}
"""

#-------------------------------------------------------------------------------

DiracPropagatorK_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel(
 pycuda::complex<double>  *Psi1, pycuda::complex<double>  *Psi2, 
 pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4 )
{{

  {CUDAconstants}

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double px = dPx*(j - DIM_X/2);
  double py = dPy*(i - DIM_Y/2);
  double pz = dPz*(k - DIM_Z/2);

  double mass_ = 0.5*mass;

  double Energy  =  c*sqrt( px*px + py*py + pz*pz + pow(mass_*c,2) );
  double phi     =  Energy*dt/hBar; 

  pycuda::complex<double> p_plus  = pycuda::complex<double>(  py ,  px );
  pycuda::complex<double> p_minus = pycuda::complex<double>(  py , -px );

  pycuda::complex<double> U11 =  pycuda::complex<double>( cos(phi)  ,  - c*c*mass_*sin( phi )/Energy );
  pycuda::complex<double> U13 =  pycuda::complex<double>( 0.        ,  - c*pz*sin( phi )/Energy     );
  pycuda::complex<double> U14 = -p_plus*c* sin(phi)/Energy;
  

  pycuda::complex<double> U22 = U11; 
  pycuda::complex<double> U23 = (c*sin( phi )/Energy) * p_minus;
  pycuda::complex<double> U24 = pycuda::complex<double>( 0.         ,    c*pz*sin( phi )/Energy );


  pycuda::complex<double> U31 =  pycuda::complex<double>( 0.  ,  -c*pz*sin(phi)/Energy   );
  pycuda::complex<double> U32 = -(c*sin(phi)/Energy) * p_plus; 
  pycuda::complex<double> U33 =  pycuda::complex<double>( cos(phi)   ,  c*c*mass_*sin( phi )/Energy );

  pycuda::complex<double> U41 = p_minus * (c*sin(phi)/Energy);
  pycuda::complex<double> U42 = pycuda::complex<double>( 0.  ,  c*pz*sin(phi)/Energy   );
  pycuda::complex<double> U44 = U33;


  _Psi1 = U11*Psi1[indexTotal]                          +  U13*Psi3[indexTotal]  +  U14*Psi4[indexTotal];
  _Psi2 =                         U22*Psi2[indexTotal]  +  U23*Psi3[indexTotal]  +  U24*Psi4[indexTotal];
  _Psi3 = U31*Psi1[indexTotal]  + U32*Psi2[indexTotal]  +  U33*Psi3[indexTotal]                         ;
  _Psi4 = U41*Psi1[indexTotal]  + U42*Psi2[indexTotal]                           +  U44*Psi4[indexTotal];

  Psi1[indexTotal] = _Psi1;
  Psi2[indexTotal] = _Psi2;
  Psi3[indexTotal] = _Psi3;
  Psi4[indexTotal] = _Psi4;

}}
"""


DiracPropagatorA_source = """
//
//   source code for the Dirac propagator with scalar-vector potential interaction
//   and smooth time dependence
//

#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

{CUDAconstants}

__device__  double A0(double t, double x, double y, double z)
{{
    return {A0};
}}
__device__  double A1(double t, double x, double y, double z)
{{
   return {A1};
}}

__device__  double A2(double t, double x, double y, double z)
{{
   return {A2};
}}

__device__  double A3(double t, double x, double y, double z)
{{
   return {A3};
}}

__device__ double VectorPotentialSquareSum(double t, double x, double y, double z)
{{
 return pow( A1(t,x,y,z), 2.) + pow( A2(t,x,y,z), 2.) + pow( A3(t,x,y,z), 2.);
}}

//-------------------------------------------------------------------------------------------------------------

__global__ void DiracPropagatorA_Kernel(
 pycuda::complex<double>  *Psi1, pycuda::complex<double>   *Psi2, pycuda::complex<double>  *Psi3, pycuda::complex<double>  *Psi4, double t )
{{
  

  const int DIM_X = blockDim.x;
  const int DIM_Y = gridDim.x;
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  const int indexTotal = threadIdx.x +  DIM_X * blockIdx.x + DIM_X*DIM_Y * blockIdx.y; 

  double x = dX*(j - DIM_X/2);
  double y = dY*(i - DIM_Y/2);
  double z = dZ*(k - DIM_Z/2);

  pycuda::complex<double>  _Psi1, _Psi2, _Psi3, _Psi4;

  double F,omega;
  double mass_ = 0.5*mass;
	
  F = sqrt( pow(mass_*c*c,2.) + VectorPotentialSquareSum(t,x,y,z)  );	
  omega = F/hBar; 
  pycuda::complex<double> iA_plus  = pycuda::complex<double>( -A2(t,x,y,z) , A1(t,x,y,z) );
  pycuda::complex<double> iA_minus = pycuda::complex<double>(  A2(t,x,y,z) , A1(t,x,y,z) );

  pycuda::complex<double> expV0 = exp(  pycuda::complex<double>(0. ,-dt*A0(t,x,y,z)) );	

  pycuda::complex<double> U11 = pycuda::complex<double>( cos(dt*omega) ,  -mass_*c*c*sin(dt*omega)/F );
  pycuda::complex<double> U22 = U11;


  pycuda::complex<double> U33 = pycuda::complex<double>( cos(dt*omega) ,  mass_*c*c*sin(dt*omega)/F );	
  pycuda::complex<double> U44 = U33;

  pycuda::complex<double> U13 = pycuda::complex<double>( 0. , A3(t,x,y,z)*sin(dt*omega)/F );
  pycuda::complex<double> U14 = iA_minus*sin(dt*omega)/F;
  
  pycuda::complex<double> U23 =  iA_plus*sin(dt*omega)/F;
  pycuda::complex<double> U24 =  pycuda::complex<double>( 0. , -A3(t,x,y,z)*sin(dt*omega)/F );

  pycuda::complex<double> U31 = pycuda::complex<double>( 0. , A3(t,x,y,z)*sin(dt*omega)/F );
  pycuda::complex<double> U32 = iA_minus*sin(dt*omega)/F;

  pycuda::complex<double> U41 = iA_plus*sin(dt*omega)/F;
  pycuda::complex<double> U42 = pycuda::complex<double>( 0. , -A3(t,x,y,z)*sin(dt*omega)/F );
  

  _Psi1 = expV0*( U11*Psi1[indexTotal]                        + U13*Psi3[indexTotal] + U14*Psi4[indexTotal] );

  _Psi2 = expV0*(                        U22*Psi2[indexTotal] + U23*Psi3[indexTotal] + U24*Psi4[indexTotal] );	

  _Psi3 = expV0*( U31*Psi1[indexTotal] + U32*Psi2[indexTotal] + U33*Psi3[indexTotal]                        );

  _Psi4 = expV0*( U41*Psi1[indexTotal] + U42*Psi2[indexTotal]                        + U44*Psi4[indexTotal] );

  Psi1[indexTotal] = _Psi1;
  Psi2[indexTotal] = _Psi2;
  Psi3[indexTotal] = _Psi3;
  Psi4[indexTotal] = _Psi4;

}}

"""


DiracAbsorbBoundary_source  =  """
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
  const int DIM_Z = gridDim.y;

  int j  =  (threadIdx.x + DIM_X/2)%DIM_X;
  int i  =  (blockIdx.x  + DIM_Y/2)%DIM_Y;
  int k  =  (blockIdx.y  + DIM_Z/2)%DIM_Z;

  const int indexTotal = threadIdx.x +  DIM_X*blockIdx.x + DIM_X*DIM_Y*blockIdx.y; 

  double wx = pow(3.*double(DIM_X)/100.,2); 
  double wy = pow(3.*double(DIM_Y)/100.,2); 
  double wz = pow(3.*double(DIM_Z)/100.,2); 


//--------------------------- boundary in x --------------------------------------


	double expB = 	(1. - exp( -double(j*j)/wx ));
	expB  *= 1. - exp(  -(j - DIM_X+1. )*(j - DIM_X+1.)/wx );

//-------------- boundary in y

	expB  *= 1. - exp(  -double(i*i)/wy  );
	expB  *= 1. - exp( -double( (i - DIM_Y + 1)*(i - DIM_Y + 1) )/wy  );

//-------------- boundary in z

	expB  *= 1. - exp(  -double(k*k)/wz  );
        expB  *= 1. - exp( -double( (k - DIM_Z + 1)*(k - DIM_Z + 1) )/wz  );

	Psi1[indexTotal] *=   expB;
	Psi2[indexTotal] *=   expB;
	Psi3[indexTotal] *=   expB;
	Psi4[indexTotal] *=   expB;	

}

"""


#-----------------------------------------------------------------------------------------------

class GPU_Dirac3D:
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

		X_amplitude,Y_amplitude,Z_amplitude = amplitude
		X_gridDIM, Y_gridDIM, Z_gridDIM    = gridDIM
		

		self.dX = 2.*X_amplitude/np.float(X_gridDIM)
		self.dY = 2.*Y_amplitude/np.float(Y_gridDIM)
		self.dZ = 2.*Z_amplitude/np.float(Z_gridDIM)	

		self.X_amplitude = X_amplitude
		self.Y_amplitude = Y_amplitude
		self.Z_amplitude = Z_amplitude

		self.X_gridDIM = X_gridDIM
		self.Y_gridDIM = Y_gridDIM
		self.Z_gridDIM = Z_gridDIM
		
		self.min_X = -X_amplitude
		self.min_Y = -Y_amplitude
		
		self.timeSteps     = timeSteps
		self.skipFrames    = skipFrames
		self.frameSaveMode = frameSaveMode

		rangeX  = np.linspace( -X_amplitude, X_amplitude - self.dX,  X_gridDIM )
		rangeY  = np.linspace( -Y_amplitude, Y_amplitude - self.dY,  Y_gridDIM )
		rangeZ  = np.linspace( -Z_amplitude, Z_amplitude - self.dZ,  Z_gridDIM )

		self.X = fftpack.fftshift(rangeX)[ np.newaxis, np.newaxis ,     :       ]
 		self.Y = fftpack.fftshift(rangeY)[ np.newaxis,      :     , np.newaxis  ]
     		self.Z = fftpack.fftshift(rangeZ)[      :    , np.newaxis , np.newaxis  ]

		#self.X_GPU  = gpuarray.to_gpu( np.ascontiguousarray(   self.X + 0.*self.Y + 0.*self.Z, dtype = np.complex128)   )
		#self.Y_GPU  = gpuarray.to_gpu( np.ascontiguousarray( 0*self.X +    self.Y + 0.*self.Z, dtype = np.complex128)   )
		#self.Z_GPU  = gpuarray.to_gpu( np.ascontiguousarray( 0*self.X + 0.*self.Y +    self.Z, dtype = np.complex128)   )

		Px_amplitude = np.pi/self.dX
		self.dPx     = 2*Px_amplitude/self.X_gridDIM
		Px_range     = np.linspace( Px_amplitude, Px_amplitude - self.dPx, self.X_gridDIM )

		Py_amplitude = np.pi/self.dY
		self.dPy     = 2*Py_amplitude/self.Y_gridDIM
		Py_range     = np.linspace( Py_amplitude, Py_amplitude - self.dPy, self.Y_gridDIM )

		Pz_amplitude = np.pi/self.dZ
		self.dPz     = 2*Pz_amplitude/self.Z_gridDIM
		Pz_range     = np.linspace( Pz_amplitude, Pz_amplitude - self.dPz, self.Z_gridDIM )

		self.Px = fftpack.fftshift(Px_range)[ np.newaxis, np.newaxis ,     :       ]
		self.Py = fftpack.fftshift(Py_range)[ np.newaxis,      :     , np.newaxis  ]
		self.Pz = fftpack.fftshift(Pz_range)[      :    , np.newaxis , np.newaxis  ]

		#self.Px_GPU  = gpuarray.to_gpu( np.ascontiguousarray(   self.Px + 0.*self.Py + 0.*self.Pz, dtype = np.complex128) )
		#self.Py_GPU  = gpuarray.to_gpu( np.ascontiguousarray( 0*self.Px +    self.Py + 0.*self.Pz, dtype = np.complex128) )
		#self.Pz_GPU  = gpuarray.to_gpu( np.ascontiguousarray( 0*self.Px + 0.*self.Py +    self.Pz, dtype = np.complex128) )

		self.dt = dt
		
		#................ Strings: mass,c,dt must be defined in children class.................... 
	
		self.CUDA_constants_essential  = 'const double hBar=%f; '%self.hBar
		self.CUDA_constants_essential += 'const double mass=%f; '%self.mass
		self.CUDA_constants_essential += 'const double c=%f;    '%self.c
		self.CUDA_constants_essential += 'const double dt=%f;\n   '%self.dt
		self.CUDA_constants_essential += 'const double dX=%f;   '%self.dX
		self.CUDA_constants_essential += 'const double dY=%f;   '%self.dY
		self.CUDA_constants_essential += 'const double dZ=%f;   '%self.dZ
 
		self.CUDA_constants_essential += 'const double dPx=%f;\n   '%self.dPx
		self.CUDA_constants_essential += 'const double dPy=%f;   '%self.dPy
		self.CUDA_constants_essential += 'const double dPz=%f;   '%self.dPz

		self.CUDA_constants = 	self.CUDA_constants_essential #+ self.CUDA_constants_additional	

		#................ CUDA Kernels ...........................................................
		
		DiracPropagatorK_source_final = DiracPropagatorK_source.format(CUDAconstants=self.CUDA_constants)
		self.DiracPropagatorK = SourceModule( DiracPropagatorK_source_final , arch="sm_20" ).get_function("Kernel")

		self.K_Energy_Average_GPU = SourceModule( 
			K_Energy_Average_source.format(CUDAconstants=self.CUDA_constants) ).get_function("Kernel")


		self.Average_X_GPU = SourceModule( 
			Average_X_source.format(CUDAconstants=self.CUDA_constants)  ).get_function("Kernel")

		self.Average_Y_GPU = SourceModule( 
			Average_Y_source.format(CUDAconstants=self.CUDA_constants)  ).get_function("Kernel")

		self.Average_Z_GPU = SourceModule( 
			Average_Z_source.format(CUDAconstants=self.CUDA_constants)  ).get_function("Kernel")

		self.Average_Px_GPU = SourceModule( 
			Average_Px_source.format(CUDAconstants=self.CUDA_constants)  ).get_function("Kernel")

		self.Average_Py_GPU = SourceModule( 
			Average_Py_source.format(CUDAconstants=self.CUDA_constants)  ).get_function("Kernel")

		self.Average_Pz_GPU = SourceModule( 
			Average_Pz_source.format(CUDAconstants=self.CUDA_constants)  ).get_function("Kernel")

		self.Potential_0_Average_GPU = SourceModule( 
			Potential_0_Average_source.format(CUDAconstants=self.CUDA_constants,A0=self.Potential_0_String)  
			).get_function("Kernel")

		"""print DiracPropagatorA_source.format(
					CUDAconstants=self.CUDA_constants,
					A0=self.Potential_0_String, 
 					A1=self.Potential_1_String, 
					A2=self.Potential_2_String, 
					A3=self.Potential_3_String)"""

		self.DiracPropagatorA  =  \
		SourceModule( DiracPropagatorA_source.format(
					CUDAconstants=self.CUDA_constants,
					A0=self.Potential_0_String, 
 					A1=self.Potential_1_String, 
					A2=self.Potential_2_String, 
					A3=self.Potential_3_String) ).get_function( "DiracPropagatorA_Kernel" )

		self.DiracAbsorbBoundary  =  \
		SourceModule(DiracAbsorbBoundary_source,arch="sm_20").get_function( "AbsorbBoundary_Kernel" )

		#self.plan_Z2Z_2D = cuda_fft.Plan_Z2Z(  (self.X_gridDIM,self.Y_gridDIM)  )
		self.plan_Z2Z_3D = cuda_fft.Plan_Z2Z(  (self.X_gridDIM, self.Y_gridDIM, self.Z_gridDIM)  )



	def Fourier_X_To_P_GPU(self,W_out_GPU):
		cuda_fft.fft_Z2Z(  W_out_GPU, W_out_GPU , self.plan_Z2Z_3D )


	def Fourier_P_To_X_GPU(self,W_out_GPU):
		cuda_fft.ifft_Z2Z( W_out_GPU, W_out_GPU , self.plan_Z2Z_3D )
		W_out_GPU *= 1./float(self.X_gridDIM*self.Y_gridDIM*self.Z_gridDIM)

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
		px, py, pz = p_init	

		rho  = np.exp( 1j*px*self.X + 1j*py*self.Y + 1j*pz*self.Z )*modulation_Function( self.X , self.Y , self.Z) 

		p0  = np.sqrt( px*px + py*py + pz*pz + (self.mass*self.c)**2 )
		
		Psi1 =  rho*( p0  + self.mass*self.c ) 
		Psi2 =  rho*0.
		Psi3 =  rho*pz
		Psi4 =  rho*( px + 1j*py )	
		
		return np.array([Psi1, Psi2, Psi3, Psi4 ])

	def Spinor_Particle_SpinDown(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py, pz = p_init	


		rho  = np.exp( 1j*px*self.X + 1j*py*self.Y + 1j*pz*self.Z )*modulation_Function( self.X , self.Y , self.Z) 

		p0  = np.sqrt( px*px + py*py + pz*pz + (self.mass*self.c)**2 )
		
		Psi1 =  rho*0.
		Psi2 =  rho*( p0  + self.mass*self.c )  
		Psi3 =  rho*( px - 1j*py )
		Psi4 =  rho*( -pz )
		
		return np.array([Psi1, Psi2, Psi3, Psi4 ])


	def Spinor_AntiParticle_SpinUp(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py, pz = p_init	

		rho  = np.exp( -1j*px*self.X - 1j*py*self.Y - 1j*pz*self.Z )*modulation_Function( self.X , self.Y , self.Z) 

		p0  = np.sqrt( px*px + py*py + pz*pz + (self.mass*self.c)**2 )
		
		Psi1 =   rho*( px - 1j*py )	
		Psi2 =  -rho*pz
		Psi3 =   rho*0.
		Psi4 =   rho*( p0  - self.mass*self.c ) 
		
		return -1j*np.array([Psi1, Psi2, Psi3, Psi4 ])

	def Spinor_AntiParticle_SpinDown(self, p_init, modulation_Function ):
		"""
		
		"""
		px, py, pz = p_init	

		rho  = np.exp( -1j*px*self.X - 1j*py*self.Y - 1j*pz*self.Z )*modulation_Function( self.X , self.Y , self.Z) 

		p0  = np.sqrt( px*px + py*py + pz*pz + (self.mass*self.c)**2 )
		
		Psi1 =   rho*pz	
		Psi2 =   rho*( px + 1j*py ) 
		Psi3 =   rho*( p0  - self.mass*self.c )
		Psi4 =   rho*0. 
		
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

	def FilterElectrons(self,sign):
 		'''
		Routine that uses the Fourier transform to filter positrons/electrons
		Options:
			sign=1   Leaves electrons
			sign=-1	 Leaves positrons
		'''
		print '  '
		print '  	Filter Electron routine '
		print '  '

                min_Px = np.pi*self.X_gridDIM/(-2*self.X_amplitude)
		dPx = 2*np.abs(min_Px)/self.X_gridDIM
		px_Vector  = fftpack.fftshift ( np.linspace(min_Px, np.abs(min_Px) - dPx, self.X_gridDIM ))

		min_Py = np.pi*self.Y_gridDIM/(-2*self.Y_amplitude)
		dPy = 2*np.abs(min_Py)/self.Y_gridDIM
		py_Vector  = fftpack.fftshift ( np.linspace(min_Py, np.abs(min_Py) - dPy, self.Y_gridDIM ))

		min_Pz = np.pi*self.Z_gridDIM/(-2*self.Z_amplitude)
		dPz = 2*np.abs(min_Pz)/self.Z_gridDIM
		pz_Vector  = fftpack.fftshift ( np.linspace(min_Pz, np.abs(min_Pz) - dPz, self.Z_gridDIM ))


		px = px_Vector[np.newaxis,np.newaxis,:]
		py = py_Vector[np.newaxis,:,np.newaxis]
		pz = pz_Vector[:,np.newaxis,np.newaxis]

		sqrtp = sign*2*np.sqrt( 
		self.mass*self.mass*self.c**4 + self.c*self.c*px*px + self.c*self.c*py*py + self.c*self.c*pz*pz  )
		aa = sign*self.mass*self.c*self.c/sqrtp
		bb = sign*(px/sqrtp - 1j*py/sqrtp)
		cc = sign*(px/sqrtp + 1j*py/sqrtp)
		dd = sign*pz/sqrtp
		#dd = 0.5      - sign*self.mass*self.c*self.c/sqrtp
	        
		ElectronProjector = np.matrix([ [0.5+aa , 0.  , dd  , bb  ],
						[0. , 0.5+aa  , cc  , -dd  ],
						[dd , bb  , 0.5-aa  , 0.  ],
						[cc , -dd  , 0.  , 0.5-aa] ])

		psi1_fft = fftpack.fftn( self.Psi_init[0]  ) 
		psi2_fft = fftpack.fftn( self.Psi_init[1]  ) 
		psi3_fft = fftpack.fftn( self.Psi_init[2]  ) 
		psi4_fft = fftpack.fftn( self.Psi_init[3]  ) 		
		
		psi1_fft_electron = ElectronProjector[0,0]*psi1_fft + ElectronProjector[0,1]*psi2_fft +\
		ElectronProjector[0,2]*psi3_fft + ElectronProjector[0,3]*psi4_fft	

		psi2_fft_electron = ElectronProjector[1,0]*psi1_fft + ElectronProjector[1,1]*psi2_fft +\
		ElectronProjector[1,2]*psi3_fft + ElectronProjector[1,3]*psi4_fft

                psi3_fft_electron = ElectronProjector[2,0]*psi1_fft + ElectronProjector[2,1]*psi2_fft +\
		ElectronProjector[2,2]*psi3_fft + ElectronProjector[2,3]*psi4_fft	

                psi4_fft_electron = ElectronProjector[3,0]*psi1_fft + ElectronProjector[3,1]*psi2_fft +\
		ElectronProjector[3,2]*psi3_fft + ElectronProjector[3,3]*psi4_fft

                self.Psi1_init  = fftpack.ifftn( psi1_fft_electron   ) 
		self.Psi2_init  = fftpack.ifftn( psi2_fft_electron   ) 
		self.Psi3_init  = fftpack.ifftn( psi3_fft_electron   ) 
		self.Psi4_init  = fftpack.ifftn( psi4_fft_electron   )

		self.Psi_init = np.array([ self.Psi1_init, self.Psi2_init, self.Psi3_init, self.Psi4_init  ]) 					

	def save_Spinor(self,f1, t, Psi1_GPU,Psi2_GPU,Psi3_GPU,Psi4_GPU):
		print ' progress ', 100*t/(self.timeSteps+1), '%'

		PsiTemp = Psi1_GPU.get()
		f1['1/real/'+str(t)] = np.real( PsiTemp )
		f1['1/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi2_GPU.get()
		f1['2/real/'+str(t)] = np.real( PsiTemp )
		f1['2/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi3_GPU.get()
		f1['3/real/'+str(t)] = np.real( PsiTemp )
		f1['3/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi4_GPU.get()
		f1['4/real/'+str(t)] = np.real( PsiTemp )
		f1['4/imag/'+str(t)] = np.imag( PsiTemp )


	def save_Spinor_SliceYZ(self,f1, t, Psi1_GPU,Psi2_GPU,Psi3_GPU,Psi4_GPU, sliceY, sliceZ ):
		print ' progress ', 100*t/(self.timeSteps+1), '%'

		PsiTemp = Psi1_GPU.get()[sliceZ,:,:]
		f1['xy/1/real/'+str(t)] = np.real( PsiTemp )
		f1['xy/1/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi2_GPU.get()[sliceZ,:,:]
		f1['xy/2/real/'+str(t)] = np.real( PsiTemp )
		f1['xy/2/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi3_GPU.get()[sliceZ,:,:]
		f1['xy/3/real/'+str(t)] = np.real( PsiTemp )
		f1['xy/3/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi4_GPU.get()[sliceZ,:,:]
		f1['xy/4/real/'+str(t)] = np.real( PsiTemp )
		f1['xy/4/imag/'+str(t)] = np.imag( PsiTemp )

		#		

		PsiTemp = Psi1_GPU.get()[:,sliceY,:]
		f1['xz/1/real/'+str(t)] = np.real( PsiTemp )
		f1['xz/1/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi2_GPU.get()[:,sliceY,:]
		f1['xz/2/real/'+str(t)] = np.real( PsiTemp )
		f1['xz/2/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi3_GPU.get()[:,sliceY,:]
		f1['xz/3/real/'+str(t)] = np.real( PsiTemp )
		f1['xz/3/imag/'+str(t)] = np.imag( PsiTemp )

		PsiTemp = Psi4_GPU.get()[:,sliceY,:]
		f1['xz/4/real/'+str(t)] = np.real( PsiTemp )
		f1['xz/4/imag/'+str(t)] = np.imag( PsiTemp )


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

		psi1 = FILE['1/'+str(n)][...]
		psi2 = FILE['2/'+str(n)][...]
		psi3 = FILE['3/'+str(n)][...]
		psi4 = FILE['4/'+str(n)][...]

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
		norm *= self.dX*self.dY*self.dZ
		norm = np.sqrt(norm)		

		return norm


	def Norm_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm  = gpuarray.sum( Psi1.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi2.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi3.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi4.__abs__()**2  ).get()

		norm = np.sqrt(norm*self.dX * self.dY * self.dZ )

		#print '               norm GPU = ', norm		
		
		return norm

	def Normalize_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm = self.Norm_GPU(Psi1, Psi2, Psi3, Psi4)
		Psi1 /= norm
		Psi2 /= norm
		Psi3 /= norm
		Psi4 /= norm

	def Norm_P_GPU( self, Psi1, Psi2, Psi3, Psi4):
		norm  = gpuarray.sum( Psi1.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi2.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi3.__abs__()**2  ).get()
		norm += gpuarray.sum( Psi4.__abs__()**2  ).get()

		norm = np.sqrt( norm*self.dPx * self.dPy * self.dPz )
		
		return norm

	#........................................................................

	def _Average_X( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.X_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.X_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.X_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.X_GPU).get()

		average *= self.dX*self.dY*self.dZ

		return average	

	def _Average_Y( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.Y_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.Y_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.Y_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.Y_GPU).get()

		average *= self.dX*self.dY*self.dZ

		return average		

	def _Average_Px( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.Px_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.Px_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.Px_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.Px_GPU).get()

		average *= self.dX*self.dY*self.dZ

		return average	

	def _Average_Py( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):

		average  = gpuarray.dot(Psi1_GPU.__abs__()**2,self.Py_GPU).get()
		average += gpuarray.dot(Psi2_GPU.__abs__()**2,self.Py_GPU).get()
		average += gpuarray.dot(Psi3_GPU.__abs__()**2,self.Py_GPU).get()
		average += gpuarray.dot(Psi4_GPU.__abs__()**2,self.Py_GPU).get()

		average *= self.dX*self.dY*self.dZ

		return average		

	#........................................................................

	def Average_Alpha1( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average  =  gpuarray.dot(Psi4_GPU, Psi1_GPU.conj()).get()
		average +=  gpuarray.dot(Psi3_GPU, Psi2_GPU.conj()).get()
		average +=  gpuarray.dot(Psi2_GPU, Psi3_GPU.conj()).get()
		average +=  gpuarray.dot(Psi1_GPU, Psi4_GPU.conj()).get()

		average *= self.dX*self.dY*self.dZ

		return average

	def Average_Alpha2( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average  = - gpuarray.dot(Psi4_GPU,Psi1_GPU.conj()).get()
		average +=   gpuarray.dot(Psi3_GPU,Psi2_GPU.conj()).get()
		average += - gpuarray.dot(Psi2_GPU,Psi3_GPU.conj()).get()
		average +=   gpuarray.dot(Psi1_GPU,Psi4_GPU.conj()).get()

		average *= 1j*self.dX*self.dY*self.dZ

		return average

	def Average_Alpha3( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average  =  gpuarray.dot(Psi3_GPU, Psi1_GPU.conj()).get()
		average -=  gpuarray.dot(Psi4_GPU, Psi2_GPU.conj()).get()
		average +=  gpuarray.dot(Psi1_GPU, Psi3_GPU.conj()).get()
		average -=  gpuarray.dot(Psi2_GPU, Psi4_GPU.conj()).get()

		average *= self.dX*self.dY*self.dZ

		return average

	def Average_Beta( self, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU):
		average =     gpuarray.dot(Psi1_GPU,Psi1_GPU.conj()).get()		
		average +=    gpuarray.dot(Psi2_GPU,Psi2_GPU.conj()).get()
		average +=  - gpuarray.dot(Psi3_GPU,Psi3_GPU.conj()).get()
		average +=  - gpuarray.dot(Psi4_GPU,Psi4_GPU.conj()).get()
		
		average *= self.dX*self.dY*self.dZ

		return average



	#.....................................................................

	def Run(self):
		try :
			import os
			os.remove (self.fileName)
				
		except OSError:
			pass
		
		f1 = h5py.File(self.fileName)


		print '--------------------------------------------'
		print '              Dirac Propagator 3D           '
		print '--------------------------------------------'
		print '  save Mode  =  ',	self.frameSaveMode			

		f1['x_gridDIM'] = self.X_gridDIM
		f1['y_gridDIM'] = self.Y_gridDIM
		f1['z_gridDIM'] = self.Y_gridDIM

		f1['x_amplitude'] = self.X_amplitude
		f1['y_amplitude'] = self.Y_amplitude
		f1['z_amplitude'] = self.Y_amplitude

		#  Redundant information on dx dy dz       
		f1['dx'] = self.dX
		f1['dy'] = self.dY
		f1['dz'] = self.dY

		f1['Potential_0_String'] = self.Potential_0_String
		f1['Potential_1_String'] = self.Potential_1_String
		f1['Potential_2_String'] = self.Potential_2_String
		f1['Potential_3_String'] = self.Potential_3_String

		self.Psi1_init, self.Psi2_init, self.Psi3_init, self.Psi4_init = self.Psi_init		

		Weight_GPU = gpuarray.empty( self.Psi1_init.shape , dtype = np.float64 )

		Psi1_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi1_init, dtype=np.complex128) )
		Psi2_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi2_init, dtype=np.complex128) )
		Psi3_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi3_init, dtype=np.complex128) )
		Psi4_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Psi4_init, dtype=np.complex128) )

		_Psi1_GPU = gpuarray.zeros_like(Psi1_GPU)
		_Psi2_GPU = gpuarray.zeros_like(Psi1_GPU)
		_Psi3_GPU = gpuarray.zeros_like(Psi1_GPU)
		_Psi4_GPU = gpuarray.zeros_like(Psi1_GPU)

		#

		print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'
		print '                                                               '
		print 'number of steps  =  ', self.timeSteps, ' dt = ',self.dt
		print 'dX = ', self.dX, 'dY = ', self.dY,  'dZ = ', self.dZ
		print 'dPx = ', self.dPx, 'dPy = ', self.dPy,  'dPz = ', self.dPz
		print '                                                               '
		print '  '

		if self.frameSaveMode=='Spinor':
			#self.save_Spinor(f1, 0 , Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)
			self.save_Spinor_SliceYZ(
					 f1,0,Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU, 0 , 0)

		if self.frameSaveMode=='Density':
			self.save_Density(f1, 0, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

		#  ............................... Main LOOP .....................................
 
		self.blockCUDA = (self.X_gridDIM,1,1)
		self.gridCUDA  = (self.Y_gridDIM,self.Z_gridDIM)

		timeRange = range(1, self.timeSteps+1)

		initial_time = time.time()

		X_average       = []
		Y_average       = []
		Z_average       = []

		Px_average       = []
		Py_average       = []
		Pz_average       = []

		Alpha1_average  = []
		Alpha2_average  = []
		Alpha3_average  = []
		Beta_average    = []

		Potential_0_average = []
		K_Energy_average = []
		
		self.Normalize_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

		for t_index in timeRange:

				t_GPU = np.float64(self.dt * t_index )

				# 	Averages				
				self.Average_X_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
				block=self.blockCUDA, grid=self.gridCUDA ) 
				X_average.append(  gpuarray.sum(Weight_GPU).get()*self.dX*self.dY*self.dZ   )
				#
				self.Average_Y_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
				block=self.blockCUDA, grid=self.gridCUDA ) 
				Y_average.append(  gpuarray.sum(Weight_GPU).get()*self.dX*self.dY*self.dZ   )
				#
				self.Average_Z_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
				block=self.blockCUDA, grid=self.gridCUDA ) 
				Z_average.append(  gpuarray.sum(Weight_GPU).get()*self.dX*self.dY*self.dZ   )
 
				#
				Alpha1_average.append( self.Average_Alpha1( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)  )
				Alpha2_average.append( self.Average_Alpha2( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)  )
				Alpha3_average.append( self.Average_Alpha3( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)  )
				#

				self.Potential_0_Average_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU, t_GPU,
										block=self.blockCUDA, grid=self.gridCUDA ) 

				Potential_0_average.append(  gpuarray.sum(Weight_GPU).get()*self.dX*self.dY*self.dZ  )

				#======================================================================================
				self.Fourier_4_X_To_P_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)

				#
				if self.Compute_Ehrenfest_P == True:	
					if t_index==1:
						norm = self.Norm_P_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU )
						print ' norm step 1 = ', norm
					Psi1_GPU /= norm; Psi2_GPU /= norm;
					Psi3_GPU /= norm; Psi4_GPU /= norm

					self.Average_Px_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
										block=self.blockCUDA, grid=self.gridCUDA ) 
					Px_average.append(  gpuarray.sum(Weight_GPU).get()*self.dPx*self.dPy*self.dPz   )

					#
					self.Average_Py_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
										block=self.blockCUDA, grid=self.gridCUDA ) 
					Py_average.append(  gpuarray.sum(Weight_GPU).get()*self.dPx*self.dPy*self.dPz   )

					#
					self.Average_Pz_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
										block=self.blockCUDA, grid=self.gridCUDA ) 
					Pz_average.append(  gpuarray.sum(Weight_GPU).get()*self.dPx*self.dPy*self.dPz   )

					#			
					self.K_Energy_Average_GPU( Weight_GPU, Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU,
										block=self.blockCUDA, grid=self.gridCUDA ) 

					K_Energy_average.append(  gpuarray.sum(Weight_GPU).get()*self.dPx*self.dPy*self.dPz  )

					Psi1_GPU *= norm;  Psi2_GPU *= norm
					Psi3_GPU *= norm;  Psi4_GPU *= norm


				#..................................................
				#          Kinetic Energy 
				#..................................................

				self.DiracPropagatorK(  Psi1_GPU,  Psi2_GPU,  Psi3_GPU,  Psi4_GPU,
					 		block=self.blockCUDA, grid=self.gridCUDA )

				#print gpuarray.sum(Psi1_GPU).get()

				self.Fourier_4_P_To_X_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)


				#..............................................
				#             Mass potential
				#..............................................

				self.DiracPropagatorA( Psi1_GPU,  Psi2_GPU,  Psi3_GPU,  Psi4_GPU,
							   t_GPU, block=self.blockCUDA, grid=self.gridCUDA )
				
				#        Absorbing boundary
				
				#self.DiracAbsorbBoundary( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU, block=blockCUDA, grid=gridCUDA )
				
				#
				#	Normalization
				#
				#self.Normalize_GPU( Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)
				
				#   Saving files

				if t_index % self.skipFrames == 0:
					if self.frameSaveMode=='Spinor':
						#self.save_Spinor( f1,t_index,Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU)
						self.save_Spinor_SliceYZ(f1,t_index,
						Psi1_GPU, Psi2_GPU, Psi3_GPU, Psi4_GPU, 0 , 0)
						
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
		self.Z_average      = np.array(Z_average).real

		self.Px_average     = np.array( Px_average ).real
		self.Py_average     = np.array( Py_average ).real
		self.Pz_average     = np.array( Pz_average ).real

		self.Alpha1_average= np.array(Alpha1_average).real
		self.Alpha2_average= np.array(Alpha2_average).real
		self.Alpha3_average= np.array(Alpha3_average).real

		self.Potential_0_average = np.array( Potential_0_average ).real
		self.K_Energy_average = np.array(K_Energy_average)	
		
		"""f1['dt'] = self.dt
		f1['dX'] = self.dX
		f1['dY'] = self.dY
		f1['dZ'] = self.dZ
		f1['X_gridDIM'] = self.X_gridDIM
		f1['Y_gridDIM'] = self.Y_gridDIM
		f1['Z_gridDIM'] = self.Z_gridDIM"""


		return 0


		
