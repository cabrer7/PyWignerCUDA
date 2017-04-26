#!/usr/local/epd/bin/python
#------------------------------------------------------------------------------------------------------
#
#  Dirac propagator in the phase space x-px
#
#------------------------------------------------------------------------------------------------------

import numpy as np
import scipy.fftpack as fftpack
import h5py
import time
import sympy as sympy

#from pyfft.cuda import Plan
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pycuda.reduction as reduction

import cufft_wrapper as cuda_fft

#-----------------------------------------------------------------------------------------

gpu_array_copy_source = """
#include <pycuda-complex.hpp>
#include<math.h>

__global__ void Kernel(pycuda::complex<double> *W_new , pycuda::complex<double> *W)
{
    
    const int X_gridDIM = blockDim.x * gridDim.z;
    const int P_gridDIM = gridDim.x;
  
    const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

    W_new[indexTotal] = W[indexTotal];
}

"""

pickup_negatives_source =  """
//............................................................................................
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void pickup_negatives_Kernel( pycuda::complex<double>  *W_neg,
   pycuda::complex<double>  *W11,  pycuda::complex<double>  *W22, pycuda::complex<double>  *W33, pycuda::complex<double>  *W44 )
{
 
  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;
  
  pycuda::complex<double> value = W11[indexTotal];
  value += W22[indexTotal];
  value += W33[indexTotal];
  value += W44[indexTotal];   

  double value_re = pycuda::real<double>( value );
 
  if( value_re < 0. ) W_neg[indexTotal] = pycuda::complex<double>(value_re,0.);

  else W_neg[indexTotal] = pycuda::complex<double>(0. , 0.);
  
}
"""

transmission_source =  """
//............................................................................................
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__global__ void transmission_Kernel( pycuda::complex<double>  *W_transmission,
   pycuda::complex<double>  *W11,  pycuda::complex<double>  *W22, pycuda::complex<double>  *W33, pycuda::complex<double>  *W44 )
{
 
  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  //const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x     =     dx*( j - 0.5*X_gridDIM  );
  //double theta = dtheta*( i - 0.5*P_gridDIM  );
  
  pycuda::complex<double> value = W11[indexTotal];
  value += W22[indexTotal];
  value += W33[indexTotal];
  value += W44[indexTotal];   

  //double value_re = pycuda::real<double>( value );
 
  if( x > 10. ) W_transmission[indexTotal] = value;

  else W_transmission[indexTotal] = pycuda::complex<double>(0. , 0.);
  
}
"""

#------------------------------------------------------------------------------------------

CUDAsource_AbsorbBoundary_x  =  """
//............................................................................................
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

__global__ void Kernel( double width, 
   pycuda::complex<double>  *W11,  pycuda::complex<double>  *W12, pycuda::complex<double>  *W13, pycuda::complex<double>  *W14,
   pycuda::complex<double>  *W21,  pycuda::complex<double>  *W22, pycuda::complex<double>  *W23, pycuda::complex<double>  *W24,
   pycuda::complex<double>  *W31,  pycuda::complex<double>  *W32, pycuda::complex<double>  *W33, pycuda::complex<double>  *W34,
   pycuda::complex<double>  *W41,  pycuda::complex<double>  *W42, pycuda::complex<double>  *W43, pycuda::complex<double>  *W44 )
{
 
  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;
  int j = threadIdx.x + blockIdx.z*blockDim.x;

  double j2 = pow( double(j-X_gridDIM/2)/width , 2);

    	W11[indexTotal] *=   1. - exp( -j2  );
	W12[indexTotal] *=   1. - exp( -j2  );
	W13[indexTotal] *=   1. - exp( -j2  );
	W14[indexTotal] *=   1. - exp( -j2  );

	W21[indexTotal] *=   1. - exp( -j2  );
	W22[indexTotal] *=   1. - exp( -j2  );
	W23[indexTotal] *=   1. - exp( -j2  );
	W24[indexTotal] *=   1. - exp( -j2  );

        W31[indexTotal] *=   1. - exp( -j2  );
	W32[indexTotal] *=   1. - exp( -j2  );
	W33[indexTotal] *=   1. - exp( -j2  );
	W34[indexTotal] *=   1. - exp( -j2  );
	
	W41[indexTotal] *=   1. - exp( -j2  );
	W42[indexTotal] *=   1. - exp( -j2  );
	W43[indexTotal] *=   1. - exp( -j2  );
	W44[indexTotal] *=   1. - exp( -j2  );


}

"""

#------------------------------------------------------------------------------------------

CUDAsource_P_plus_Lambda = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__global__ void Kernel(
   pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14,
   pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
   pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
   pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44 )
{
    const int X_gridDIM = blockDim.x * gridDim.z;
    const int P_gridDIM = gridDim.x;
  
    const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

    const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
    const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

    double p1          =       dp*( i - 0.5*P_gridDIM  );
    double lambda1     =  dlambda*( j - 0.5*X_gridDIM  );

    double lambdap1 = p1 + lambda1/2. ;

    double shell = sqrt( pow( mass_half*c , 2 ) + pow(lambdap1 , 2)  );

    pycuda::complex<double> U11 = pycuda::complex<double>( cos( dt*c*shell ), - mass_half*c*sin(c*dt*shell)/shell );
    pycuda::complex<double> U44 = pycuda::complex<double>( cos( dt*c*shell ),   mass_half*c*sin(c*dt*shell)/shell );
    pycuda::complex<double> U14 = pycuda::complex<double>( 0. , -lambdap1 * sin( dt*c*shell )/shell );
    pycuda::complex<double> U41 = pycuda::complex<double>( 0. , -lambdap1 * sin( dt*c*shell )/shell );

    pycuda::complex<double> U22 = U11;
    pycuda::complex<double> U23 = U41;
    pycuda::complex<double> U32 = U14;
    pycuda::complex<double> U33 = U44; 

    //..............................................................................................................

    pycuda::complex<double> W_Plus11, W_Plus12, W_Plus13, W_Plus14;
    pycuda::complex<double> W_Plus21, W_Plus22, W_Plus23, W_Plus24;
    pycuda::complex<double> W_Plus31, W_Plus32, W_Plus33, W_Plus34;
    pycuda::complex<double> W_Plus41, W_Plus42, W_Plus43, W_Plus44;

    //PsiPlus1 = U11 *Psi1[indexTotal]                                                +  U14 *Psi4[indexTotal];
    //PsiPlus2 =                      U22 *Psi2[indexTotal]  +  U23 * Psi3[indexTotal]                         ;
    //PsiPlus3 =                      U32 *Psi2[indexTotal]  +  U33 * Psi3[indexTotal]                         ;
    //PsiPlus4 = U41 *Psi1[indexTotal]                                                +  U44 *Psi4[indexTotal];
   
    W_Plus11 = U11 *W11[indexTotal]                                                +  U14 *W41[indexTotal];
    W_Plus21 =                      U22 *W21[indexTotal]  +  U23 * W31[indexTotal]                         ;
    W_Plus31 =                      U32 *W21[indexTotal]  +  U33 * W31[indexTotal]                         ;
    W_Plus41 = U41 *W11[indexTotal]                                                +  U44 *W41[indexTotal];

    W11[indexTotal] = W_Plus11;
    W21[indexTotal] = W_Plus21;
    W31[indexTotal] = W_Plus31;
    W41[indexTotal] = W_Plus41;

   //.........................

    W_Plus12 = U11 *W12[indexTotal]                                                +  U14 *W42[indexTotal];
    W_Plus22 =                      U22 *W22[indexTotal]  +  U23 * W32[indexTotal]                         ;
    W_Plus32 =                      U32 *W22[indexTotal]  +  U33 * W32[indexTotal]                         ;
    W_Plus42 = U41 *W12[indexTotal]                                                +  U44 *W42[indexTotal];

    W12[indexTotal] = W_Plus12;
    W22[indexTotal] = W_Plus22;
    W32[indexTotal] = W_Plus32;
    W42[indexTotal] = W_Plus42;

   //........................

    W_Plus13 = U11 *W13[indexTotal]                                                +  U14 *W43[indexTotal];
    W_Plus23 =                      U22 *W23[indexTotal]  +  U23 * W33[indexTotal]                         ;
    W_Plus33 =                      U32 *W23[indexTotal]  +  U33 * W33[indexTotal]                         ;
    W_Plus43 = U41 *W13[indexTotal]                                                +  U44 *W43[indexTotal];

    W13[indexTotal] = W_Plus13;
    W23[indexTotal] = W_Plus23;
    W33[indexTotal] = W_Plus33;
    W43[indexTotal] = W_Plus43;

   //........................

    W_Plus14 = U11 *W14[indexTotal]                                                +  U14 *W44[indexTotal];
    W_Plus24 =                      U22 *W24[indexTotal]  +  U23 *W34[indexTotal]                         ;
    W_Plus34 =                      U32 *W24[indexTotal]  +  U33 *W34[indexTotal]                         ;
    W_Plus44 = U41 *W14[indexTotal]                                                +  U44 *W44[indexTotal];

    W14[indexTotal] = W_Plus14;
    W24[indexTotal] = W_Plus24;
    W34[indexTotal] = W_Plus34;
    W44[indexTotal] = W_Plus44;

}

"""

CUDAsource_P_minus_Lambda = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES
%s;
__global__ void Kernel(
   pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14,
   pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
   pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
   pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44 )
{

    const int X_gridDIM = blockDim.x * gridDim.z;
    const int P_gridDIM = gridDim.x;
  
    const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

    const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
    const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

    double p1          =       dp*( i - 0.5*P_gridDIM  );
    double lambda1     =  dlambda*( j - 0.5*X_gridDIM  );

    double lambdap1 = p1 - lambda1/2. ;

    double shell = sqrt( pow( mass_half*c , 2 ) + pow(lambdap1 , 2)  );

    pycuda::complex<double> U11 = pycuda::complex<double>( cos( dt*c*shell ), - mass_half*c*sin(-c*dt*shell)/shell );
    pycuda::complex<double> U44 = pycuda::complex<double>( cos( dt*c*shell ),   mass_half*c*sin(-c*dt*shell)/shell );
    pycuda::complex<double> U14 = pycuda::complex<double>( 0. , -lambdap1 * sin( -dt*c*shell )/shell );
    pycuda::complex<double> U41 = pycuda::complex<double>( 0. , -lambdap1 * sin( -dt*c*shell )/shell );

    pycuda::complex<double> U22 = U11;
    pycuda::complex<double> U23 = U41;
    pycuda::complex<double> U32 = U14;
    pycuda::complex<double> U33 = U44; 

    //..............................................................................................................

    pycuda::complex<double> W_Plus11, W_Plus12, W_Plus13, W_Plus14;
    pycuda::complex<double> W_Plus21, W_Plus22, W_Plus23, W_Plus24;
    pycuda::complex<double> W_Plus31, W_Plus32, W_Plus33, W_Plus34;
    pycuda::complex<double> W_Plus41, W_Plus42, W_Plus43, W_Plus44;
   
    W_Plus11 = U11 *W11[indexTotal]                                                +  U41 *W14[indexTotal];
    W_Plus21 =                      U22 *W12[indexTotal]  +  U32 * W13[indexTotal]                         ;
    W_Plus31 =                      U23 *W12[indexTotal]  +  U33 * W13[indexTotal]                         ;
    W_Plus41 = U14 *W11[indexTotal]                                                +  U44 *W14[indexTotal];

    W11[indexTotal] = W_Plus11;
    W12[indexTotal] = W_Plus21;
    W13[indexTotal] = W_Plus31;
    W14[indexTotal] = W_Plus41;

   //.........................

    W_Plus12 = U11 *W21[indexTotal]                                                +  U41 *W24[indexTotal];
    W_Plus22 =                      U22 *W22[indexTotal]  +  U32 * W23[indexTotal]                         ;
    W_Plus32 =                      U23 *W22[indexTotal]  +  U33 * W23[indexTotal]                         ;
    W_Plus42 = U14 *W21[indexTotal]                                                +  U44 *W24[indexTotal];

    W21[indexTotal] = W_Plus12;
    W22[indexTotal] = W_Plus22;
    W23[indexTotal] = W_Plus32;
    W24[indexTotal] = W_Plus42;

   //........................

    W_Plus13 = U11 *W31[indexTotal]                                                +  U41 *W34[indexTotal];
    W_Plus23 =                      U22 *W32[indexTotal]  +  U32 * W33[indexTotal]                         ;
    W_Plus33 =                      U23 *W32[indexTotal]  +  U33 * W33[indexTotal]                         ;
    W_Plus43 = U14 *W31[indexTotal]                                                +  U44 *W34[indexTotal];

    W31[indexTotal] = W_Plus13;
    W32[indexTotal] = W_Plus23;
    W33[indexTotal] = W_Plus33;
    W34[indexTotal] = W_Plus43;

   //........................

    W_Plus14 = U11 *W41[indexTotal]                                                +  U41 *W44[indexTotal];
    W_Plus24 =                      U22 *W42[indexTotal]  +  U32 *W43[indexTotal]                         ;
    W_Plus34 =                      U23 *W42[indexTotal]  +  U33 *W43[indexTotal]                         ;
    W_Plus44 = U14 *W41[indexTotal]                                                +  U44 *W44[indexTotal];

    W41[indexTotal] = W_Plus14;
    W42[indexTotal] = W_Plus24;
    W43[indexTotal] = W_Plus34;
    W44[indexTotal] = W_Plus44;

}

"""



#------------------------------------------------------------------------------------

DiracPropagator_X_minus_Theta_source_Base = """
//
//   source code for the right Dirac propagator without vector potential interaction 
//   

#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s // Constants 


__device__  double Potential0(double t, double x)
{
    return %s ;
}
__device__  double Potential1(double t, double x)
{
   return %s ;
}

__device__  double Potential2(double t, double x)
{
   return %s ;
}
__device__  double Potential3(double t, double x)
{
  return %s ;
}

__device__ double VectorPotentialSquareSum(double t, double x)
{
 return pow(Potential1(t,x), 2.) + pow(Potential2(t,x), 2.) + pow(Potential3(t,x), 2.);
}


//............................................................................................................

__global__ void DiracPropagation4_Kernel(
   pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14,
   pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
   pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
   pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44, 
   double t , pycuda::complex<double> *B_GP_minus_GPU, pycuda::complex<double> *B_GP_plus_GPU, double aGPitaevskii )
{
  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x     =     dx*( j - 0.5*X_gridDIM  );
  double theta = dtheta*( i - 0.5*P_gridDIM  );

  double xtheta = x - 0.5*theta ;

  double F;
  
  F = sqrt( pow( mass_half*c*c*dt ,2.) + VectorPotentialSquareSum(t,xtheta)*dt*dt  );

  pycuda::complex<double> I = pycuda::complex<double>(0.,1.);
  pycuda::complex<double> U11 = pycuda::complex<double>( cos(F) , -mass_half*c*c*dt*sin(F)/F );
  pycuda::complex<double> U33 = pycuda::complex<double>( cos(F) ,  mass_half*c*c*dt*sin(F)/F );
	
  pycuda::complex<double>         U13,U14;
  pycuda::complex<double>     U22,U23,U24;	
  pycuda::complex<double> U31,U32        ;
  pycuda::complex<double> U41,U42    ,U44;

  double phaseGP = aGPitaevskii * pycuda::real<double>( B_GP_minus_GPU[indexTotal]  ); 
 
  pycuda::complex<double> expV = exp( -dt*D_Theta*theta*theta/2. - I*dt*( Potential0(t,xtheta)  + phaseGP  ) );		
	

  U22 = U11;    U44 = U33;
  U13 = I*dt*Potential3(t,xtheta)*sin(F)/F;   
  U14 = dt*(I*Potential1(t,xtheta) + Potential2(t,xtheta) )*sin(F)/F;
  
  U23 = dt*(I*Potential1(t,xtheta) - Potential2(t,xtheta) )*sin(F)/F;
  U24 = -U13;  

  U31 = U13; U32 = U14;

  U41 = U23;
  U42 = U24; 

  pycuda::complex<double> PsiPlus1, PsiPlus2, PsiPlus3, PsiPlus4;

  //..........................................................................................................
  
  PsiPlus1=expV*( U11*W11[indexTotal]                        + U13*W31[indexTotal]    + U14*W41[indexTotal] );

  PsiPlus2=expV*(                        U22*W21[indexTotal] + U23*W31[indexTotal]    + U24*W41[indexTotal] );	

  PsiPlus3=expV*( U31*W11[indexTotal] + U32*W21[indexTotal] + U33*W31[indexTotal]                           );

  PsiPlus4=expV*( U41*W11[indexTotal] + U42*W21[indexTotal]                           + U44*W41[indexTotal] );	

   W11[indexTotal] = PsiPlus1;
   W21[indexTotal] = PsiPlus2;
   W31[indexTotal] = PsiPlus3;
   W41[indexTotal] = PsiPlus4;

  //..........................................................................................................
  
   PsiPlus1=expV*( U11*W12[indexTotal]                         + U13*W32[indexTotal]    + U14*W42[indexTotal]  );

   PsiPlus2=expV*(                        U22*W22[indexTotal]  + U23*W32[indexTotal]    + U24*W42[indexTotal]  );	

   PsiPlus3=expV*( U31*W12[indexTotal] +  U32*W22[indexTotal]  + U33*W32[indexTotal]                           );

   PsiPlus4=expV*( U41*W12[indexTotal] +  U42*W22[indexTotal]                           + U44*W42[indexTotal]  );	

   W12[indexTotal] = PsiPlus1;
   W22[indexTotal] = PsiPlus2;
   W32[indexTotal] = PsiPlus3;
   W42[indexTotal] = PsiPlus4;

  //..........................................................................................................
  
   PsiPlus1=expV*( U11*W13[indexTotal]                         + U13*W33[indexTotal]    + U14*W43[indexTotal]  );

   PsiPlus2=expV*(                        U22*W23[indexTotal]  + U23*W33[indexTotal]    + U24*W43[indexTotal]  );	

   PsiPlus3=expV*( U31*W13[indexTotal] +  U32*W23[indexTotal]  + U33*W33[indexTotal]                           );

   PsiPlus4=expV*( U41*W13[indexTotal] +  U42*W23[indexTotal]                           + U44*W43[indexTotal]  );	

   W13[indexTotal] = PsiPlus1;
   W23[indexTotal] = PsiPlus2;
   W33[indexTotal] = PsiPlus3;
   W43[indexTotal] = PsiPlus4;

  //..........................................................................................................
  
   PsiPlus1=expV*( U11*W14[indexTotal]                         + U13*W34[indexTotal]    + U14*W44[indexTotal]  );

   PsiPlus2=expV*(                        U22*W24[indexTotal]  + U23*W34[indexTotal]    + U24*W44[indexTotal]  );	

   PsiPlus3=expV*( U31*W14[indexTotal] +  U32*W24[indexTotal]  + U33*W34[indexTotal]                           );

   PsiPlus4=expV*( U41*W14[indexTotal] +  U42*W24[indexTotal]                           + U44*W44[indexTotal]  );	

   W14[indexTotal] = PsiPlus1;
   W24[indexTotal] = PsiPlus2;
   W34[indexTotal] = PsiPlus3;
   W44[indexTotal] = PsiPlus4;


}

"""



#------------------------------------------------------------------------------------

DiracPropagator_X_plus_Theta_source_Base  = """
//
//   source code for the right Dirac propagator without vector potential interaction 
//   

#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s // Constants 

__device__  double Potential0(double t, double x)
{
    return %s ;
}
__device__  double Potential1(double t, double x)
{
   return %s ;
}

__device__  double Potential2(double t, double x)
{
   return %s ;
}
__device__  double Potential3(double t, double x)
{
  return %s ;
}

__device__ double VectorPotentialSquareSum(double t, double x)
{
 return pow(Potential1(t,x), 2.) + pow(Potential2(t,x), 2.) + pow(Potential3(t,x), 2.);
}


//............................................................................................................

__global__ void DiracPropagation4_Kernel(
   pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14,
   pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
   pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
   pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44, 
   double t , pycuda::complex<double> *B_GP_minus_GPU, pycuda::complex<double> *B_GP_plus_GPU , double aGPitaevskii)
{
  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x     =     dx*( j - 0.5*X_gridDIM  );
  double theta = dtheta*( i - 0.5*P_gridDIM  );

  double xtheta = x + 0.5*theta ;

  double F;
  
  F = sqrt( pow( mass_half*c*c*dt , 2.) + VectorPotentialSquareSum(t,xtheta)*dt*dt  );

  pycuda::complex<double> I = pycuda::complex<double>(0.,1.);
  pycuda::complex<double> U11 = pycuda::complex<double>( cos(F) ,   mass_half*c*c*dt*sin(F)/F );
  pycuda::complex<double> U33 = pycuda::complex<double>( cos(F) ,  -mass_half*c*c*dt*sin(F)/F );
	
  pycuda::complex<double>         U13,U14;
  pycuda::complex<double>     U22,U23,U24;	
  pycuda::complex<double> U31,U32        ;
  pycuda::complex<double> U41,U42    ,U44;

  double phaseGP = aGPitaevskii * pycuda::real<double>( B_GP_plus_GPU[indexTotal]  ); 

  pycuda::complex<double> expV = exp( -dt*D_Theta*theta*theta/2. + I*dt*( Potential0(t,xtheta) + phaseGP ) );		
	

  U22 = U11;    U44 = U33;
  U13 = -I*dt*Potential3(t,xtheta)*sin(F)/F;   
  U14 = -dt*(I*Potential1(t,xtheta) + Potential2(t,xtheta) )*sin(F)/F;
  
  U23 = -dt*(I*Potential1(t,xtheta) - Potential2(t,xtheta) )*sin(F)/F;
  U24 = -U13;  

  U31 = U13; U32 = U14;

  U41 = U23;
  U42 = U24; 

  pycuda::complex<double> W_Plus11, W_Plus12, W_Plus13, W_Plus14;
  pycuda::complex<double> W_Plus21, W_Plus22, W_Plus23, W_Plus24;
  pycuda::complex<double> W_Plus31, W_Plus32, W_Plus33, W_Plus34;
  pycuda::complex<double> W_Plus41, W_Plus42, W_Plus43, W_Plus44;

  //..........................................................................................................
  
   W_Plus11 = expV*( W11[indexTotal]*U11                       + W13[indexTotal]*U31  + W14[indexTotal]*U41 );

   W_Plus12 = expV*(                       W12[indexTotal]*U22 + W13[indexTotal]*U32  + W14[indexTotal]*U42 );	

   W_Plus13 = expV*( W11[indexTotal]*U13 + W12[indexTotal]*U23 + W13[indexTotal]*U33                        );

   W_Plus14 = expV*( W11[indexTotal]*U14 + W12[indexTotal]*U24                        + W14[indexTotal]*U44 );	

   W11[indexTotal] = W_Plus11 ;
   W12[indexTotal] = W_Plus12 ;
   W13[indexTotal] = W_Plus13 ;
   W14[indexTotal] = W_Plus14 ;

  //..........................................................................................................
  
   W_Plus21 = expV*( W21[indexTotal]*U11                       + W23[indexTotal]*U31  + W24[indexTotal]*U41 );

   W_Plus22 = expV*(                       W22[indexTotal]*U22 + W23[indexTotal]*U32  + W24[indexTotal]*U42 );	

   W_Plus23 = expV*( W21[indexTotal]*U13 + W22[indexTotal]*U23 + W23[indexTotal]*U33                        );

   W_Plus24 = expV*( W21[indexTotal]*U14 + W22[indexTotal]*U24                        + W24[indexTotal]*U44 );	

   W21[indexTotal] = W_Plus21 ;
   W22[indexTotal] = W_Plus22 ;
   W23[indexTotal] = W_Plus23 ;
   W24[indexTotal] = W_Plus24 ;

  //..........................................................................................................
  
   W_Plus31 = expV*( W31[indexTotal]*U11                       + W33[indexTotal]*U31  + W34[indexTotal]*U41 );

   W_Plus32 = expV*(                       W32[indexTotal]*U22 + W33[indexTotal]*U32  + W34[indexTotal]*U42 );	

   W_Plus33 = expV*( W31[indexTotal]*U13 + W32[indexTotal]*U23 + W33[indexTotal]*U33                        );

   W_Plus34 = expV*( W31[indexTotal]*U14 + W32[indexTotal]*U24                        + W34[indexTotal]*U44 );	

   W31[indexTotal] = W_Plus31 ;
   W32[indexTotal] = W_Plus32 ;
   W33[indexTotal] = W_Plus33 ;
   W34[indexTotal] = W_Plus34 ;

//..........................................................................................................
  
   W_Plus41 = expV*( W41[indexTotal]*U11                       + W43[indexTotal]*U31  + W44[indexTotal]*U41 );

   W_Plus42 = expV*(                       W42[indexTotal]*U22 + W43[indexTotal]*U32  + W44[indexTotal]*U42 );	

   W_Plus43 = expV*( W41[indexTotal]*U13 + W42[indexTotal]*U23 + W43[indexTotal]*U33                        );

   W_Plus44 = expV*( W41[indexTotal]*U14 + W42[indexTotal]*U24                        + W44[indexTotal]*U44 );	

   W41[indexTotal] = W_Plus41 ;
   W42[indexTotal] = W_Plus42 ;
   W43[indexTotal] = W_Plus43 ;
   W44[indexTotal] = W_Plus44 ;

}

"""

#.........................................................................................................

BaseCUDAsource_FilterGPU = """
//
//   source code for filtering particles/antiparticles 
//   

#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

//............................................................................................................

__global__ void Filter_Kernel(
		pycuda::complex<double>  *_Psi11, pycuda::complex<double>  *_Psi12, pycuda::complex<double>  *_Psi13,
                pycuda::complex<double>  *_Psi14, pycuda::complex<double>  *_Psi21,          
                pycuda::complex<double>   *_Psi22, pycuda::complex<double>  *_Psi23, pycuda::complex<double> *_Psi24,
                pycuda::complex<double>  *_Psi31, pycuda::complex<double>  *_Psi32, pycuda::complex<double> *_Psi33, 
                pycuda::complex<double>  *_Psi34, pycuda::complex<double>  *_Psi41, pycuda::complex<double> *_Psi42,        
                pycuda::complex<double>  *_Psi43, pycuda::complex<double>  *_Psi44, pycuda::complex<double> *psi11,       
                pycuda::complex<double>  *psi12, pycuda::complex<double>  *psi13, pycuda::complex<double>  *psi14,                
                pycuda::complex<double>  *psi21, pycuda::complex<double>  *psi22, pycuda::complex<double>  *psi23,   
                pycuda::complex<double>  *psi24, pycuda::complex<double>  *psi31, pycuda::complex<double>  *psi32,             
                pycuda::complex<double>  *psi33, pycuda::complex<double>  *psi34, pycuda::complex<double>  *psi41,
                pycuda::complex<double>  *psi42, pycuda::complex<double>  *psi43, pycuda::complex<double>  *psi44,    
                int sign )
{
	
    const int X_gridDIM = blockDim.x * gridDim.z;
    const int P_gridDIM = gridDim.x;
  
    const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

    const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
    const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

    double p          =       dp*( i - 0.5*P_gridDIM  );
    double Lambda     =  dlambda*( j - 0.5*X_gridDIM  );


  double     bb,sqrtp,aa;
  double     cc,dd,sqrtpL;	
  double          aaL,bbL;
  double          ccL,ddL;


  sqrtp = 2.*sqrt( mass*mass*c*c*c*c + c*c*(p+Lambda/2.)*(p+Lambda/2.)  );
  aa = 0.5      + sign*(mass*c*c/sqrtp);
  bb = c*(sign*(p+Lambda/2.)/sqrtp);
  cc = c*(sign*(p+Lambda/2.)/sqrtp);
  dd = 0.5      - sign*(mass*c*c/sqrtp);

  sqrtpL = 2.*sqrt( mass*mass*c*c*c*c + c*c*(p-Lambda/2.)*(p-Lambda/2.)  );
  aaL = 0.5      + sign*(mass*c*c/sqrtpL);
  bbL = c*(sign*(p-Lambda/2.)/sqrtpL);
  ccL = c*(sign*(p-Lambda/2.)/sqrtpL);
  ddL = 0.5      - (sign*mass*c*c/sqrtpL);

   pycuda::complex<double> Projector00, ProjectorL00;
   pycuda::complex<double> Projector03, ProjectorL03;
   pycuda::complex<double> Projector11, ProjectorL11;
   pycuda::complex<double> Projector21, ProjectorL21;
   pycuda::complex<double> Projector30, ProjectorL30;
   pycuda::complex<double> Projector12, ProjectorL12;
   pycuda::complex<double> Projector22, ProjectorL22;
   pycuda::complex<double> Projector33, ProjectorL33;
   pycuda::complex<double> Projector01, ProjectorL01;
   pycuda::complex<double> Projector02, ProjectorL02;
   pycuda::complex<double> Projector10, ProjectorL10;
   pycuda::complex<double> Projector13, ProjectorL13;
   pycuda::complex<double> Projector20, ProjectorL20;
   pycuda::complex<double> Projector23, ProjectorL23;
   pycuda::complex<double> Projector31, ProjectorL31;
   pycuda::complex<double> Projector32, ProjectorL32;
  
   Projector00 = pycuda::complex<double>(aa,0.);
   ProjectorL00 = pycuda::complex<double>(aaL,0.);

  Projector03 = pycuda::complex<double>(bb,0.);
  ProjectorL03 = pycuda::complex<double>(bbL,0.);
	
  Projector11 = pycuda::complex<double>(aa,0.);
  ProjectorL11 = pycuda::complex<double>(aaL,0.);

  Projector21 = pycuda::complex<double>(bb,0.);
  ProjectorL21 = pycuda::complex<double>(bbL,0.);

  Projector30 = pycuda::complex<double>(cc,0.);
  ProjectorL30 = pycuda::complex<double>(ccL,0.);

  Projector12 = pycuda::complex<double>(cc,0.);
  ProjectorL12 = pycuda::complex<double>(ccL,0.);

  Projector22 = pycuda::complex<double>(dd,0.);
  ProjectorL22 = pycuda::complex<double>(ddL,0.);

  Projector33 = pycuda::complex<double>(dd,0.);
  ProjectorL33 = pycuda::complex<double>(ddL,0.);

  Projector01 = pycuda::complex<double>(0.,0.);
  ProjectorL01 = pycuda::complex<double>(0.,0.);

  Projector02 = pycuda::complex<double>(0.,0.);
  ProjectorL02 = pycuda::complex<double>(0.,0.);
 
  Projector10 = pycuda::complex<double>(0.,0.);
  ProjectorL10 = pycuda::complex<double>(0.,0.);
  
  Projector13 = pycuda::complex<double>(0.,0.);
  ProjectorL13 = pycuda::complex<double>(0.,0.);

  Projector20 = pycuda::complex<double>(0.,0.);
  ProjectorL20 = pycuda::complex<double>(0.,0.);

  Projector23 = pycuda::complex<double>(0.,0.);
  ProjectorL23 = pycuda::complex<double>(0.,0.);

  Projector31 = pycuda::complex<double>(0.,0.);
  ProjectorL31 = pycuda::complex<double>(0.,0.);

  Projector32 = pycuda::complex<double>(0.,0.);
  ProjectorL32 = pycuda::complex<double>(0.,0.);
	
  


  _Psi11[indexTotal] = (Projector00*psi11[indexTotal] + Projector01*psi21[indexTotal] +\
                Projector02*psi31[indexTotal] + Projector03*psi41[indexTotal])*ProjectorL00 +\
		(Projector00*psi12[indexTotal] + Projector01*psi22[indexTotal] +\
                Projector02*psi32[indexTotal] + Projector03*psi42[indexTotal])*ProjectorL10 +\
		(Projector00*psi13[indexTotal] + Projector01*psi23[indexTotal] +\
                Projector02*psi33[indexTotal] + Projector03*psi43[indexTotal])*ProjectorL20 +\
		(Projector00*psi14[indexTotal] + Projector01*psi24[indexTotal] +\
                Projector02*psi34[indexTotal] + Projector03*psi44[indexTotal])*ProjectorL30;

		
  _Psi12[indexTotal] = (Projector00*psi11[indexTotal] + Projector01*psi21[indexTotal] +\
                Projector02*psi31[indexTotal] + Projector03*psi41[indexTotal])*ProjectorL01 +\
		(Projector00*psi12[indexTotal] + Projector01*psi22[indexTotal] +\
                Projector02*psi32[indexTotal] + Projector03*psi42[indexTotal])*ProjectorL11 +\
		(Projector00*psi13[indexTotal] + Projector01*psi23[indexTotal] +\
                Projector02*psi33[indexTotal] + Projector03*psi43[indexTotal])*ProjectorL21 +\
		(Projector00*psi14[indexTotal] + Projector01*psi24[indexTotal] +\
                Projector02*psi34[indexTotal] + Projector03*psi44[indexTotal])*ProjectorL31;

  _Psi13[indexTotal] = (Projector00*psi11[indexTotal] + Projector01*psi21[indexTotal] +\
                Projector02*psi31[indexTotal] + Projector03*psi41[indexTotal])*ProjectorL02 +\
		(Projector00*psi12[indexTotal] + Projector01*psi22[indexTotal] +\
                Projector02*psi32[indexTotal] + Projector03*psi42[indexTotal])*ProjectorL12 +\
		(Projector00*psi13[indexTotal] + Projector01*psi23[indexTotal] +\
                Projector02*psi33[indexTotal] + Projector03*psi43[indexTotal])*ProjectorL22 +\
		(Projector00*psi14[indexTotal] + Projector01*psi24[indexTotal] +\
                Projector02*psi34[indexTotal] + Projector03*psi44[indexTotal])*ProjectorL32;

  _Psi14[indexTotal] = (Projector00*psi11[indexTotal] + Projector01*psi21[indexTotal] +\
                Projector02*psi31[indexTotal] + Projector03*psi41[indexTotal])*ProjectorL03 +\
		(Projector00*psi12[indexTotal] + Projector01*psi22[indexTotal] +\
                Projector02*psi32[indexTotal] + Projector03*psi42[indexTotal])*ProjectorL13 +\
		(Projector00*psi13[indexTotal] + Projector01*psi23[indexTotal] +\
                Projector02*psi33[indexTotal] + Projector03*psi43[indexTotal])*ProjectorL23 +\
		(Projector00*psi14[indexTotal] + Projector01*psi24[indexTotal] +\
                Projector02*psi34[indexTotal] + Projector03*psi44[indexTotal])*ProjectorL33;


  _Psi21[indexTotal] = (Projector10*psi11[indexTotal] + Projector11*psi21[indexTotal] +\
                Projector12*psi31[indexTotal] + Projector13*psi41[indexTotal])*ProjectorL00 +\
		(Projector10*psi12[indexTotal] + Projector11*psi22[indexTotal] +\
                Projector12*psi32[indexTotal] + Projector13*psi42[indexTotal])*ProjectorL10 +\
		(Projector10*psi13[indexTotal] + Projector11*psi23[indexTotal] +\
                Projector12*psi33[indexTotal] + Projector13*psi43[indexTotal])*ProjectorL20 +\
		(Projector10*psi14[indexTotal] + Projector11*psi24[indexTotal] +\
                Projector12*psi34[indexTotal] + Projector13*psi44[indexTotal])*ProjectorL30;

  _Psi22[indexTotal] = (Projector10*psi11[indexTotal] + Projector11*psi21[indexTotal] +\
                Projector12*psi31[indexTotal] + Projector13*psi41[indexTotal])*ProjectorL01 +\
		(Projector10*psi12[indexTotal] + Projector11*psi22[indexTotal] +\
                Projector12*psi32[indexTotal] + Projector13*psi42[indexTotal])*ProjectorL11 +\
		(Projector10*psi13[indexTotal] + Projector11*psi23[indexTotal] +\
                Projector12*psi33[indexTotal] + Projector13*psi43[indexTotal])*ProjectorL21 +\
		(Projector10*psi14[indexTotal] + Projector11*psi24[indexTotal] +\
                Projector12*psi34[indexTotal] + Projector13*psi44[indexTotal])*ProjectorL31;

  _Psi23[indexTotal] = (Projector10*psi11[indexTotal] + Projector11*psi21[indexTotal] +\
                Projector12*psi31[indexTotal] + Projector13*psi41[indexTotal])*ProjectorL02 +\
		(Projector10*psi12[indexTotal] + Projector11*psi22[indexTotal] +\
                Projector12*psi32[indexTotal] + Projector13*psi42[indexTotal])*ProjectorL12 +\
		(Projector10*psi13[indexTotal] + Projector11*psi23[indexTotal] +\
                Projector12*psi33[indexTotal] + Projector13*psi43[indexTotal])*ProjectorL22 +\
		(Projector10*psi14[indexTotal] + Projector11*psi24[indexTotal] +\
                Projector12*psi34[indexTotal] + Projector13*psi44[indexTotal])*ProjectorL32;

  _Psi24[indexTotal] = (Projector10*psi11[indexTotal] + Projector11*psi21[indexTotal] +\
                Projector12*psi31[indexTotal] + Projector13*psi41[indexTotal])*ProjectorL03 +\
		(Projector10*psi12[indexTotal] + Projector11*psi22[indexTotal] +\
                Projector12*psi32[indexTotal] + Projector13*psi42[indexTotal])*ProjectorL13 +\
		(Projector10*psi13[indexTotal] + Projector11*psi23[indexTotal] +\
                Projector12*psi33[indexTotal] + Projector13*psi43[indexTotal])*ProjectorL23 +\
		(Projector10*psi14[indexTotal] + Projector11*psi24[indexTotal] +\
                Projector12*psi34[indexTotal] + Projector13*psi44[indexTotal])*ProjectorL33;

  _Psi31[indexTotal] = (Projector20*psi11[indexTotal] + Projector21*psi21[indexTotal] +\
                Projector22*psi31[indexTotal] + Projector23*psi41[indexTotal])*ProjectorL00 +\
		(Projector20*psi12[indexTotal] + Projector21*psi22[indexTotal] +\
                Projector22*psi32[indexTotal] + Projector23*psi42[indexTotal])*ProjectorL10 +\
		(Projector20*psi13[indexTotal] + Projector21*psi23[indexTotal] +\
                Projector22*psi33[indexTotal] + Projector23*psi43[indexTotal])*ProjectorL20 +\
		(Projector20*psi14[indexTotal] + Projector21*psi24[indexTotal] +\
                Projector22*psi34[indexTotal] + Projector23*psi44[indexTotal])*ProjectorL30;
		
  _Psi32[indexTotal] = (Projector20*psi11[indexTotal] + Projector21*psi21[indexTotal] +\
                Projector22*psi31[indexTotal] + Projector23*psi41[indexTotal])*ProjectorL01 +\
		(Projector20*psi12[indexTotal] + Projector21*psi22[indexTotal] +\
                Projector22*psi32[indexTotal] + Projector23*psi42[indexTotal])*ProjectorL11 +\
		(Projector20*psi13[indexTotal] + Projector21*psi23[indexTotal] +\
                Projector22*psi33[indexTotal] + Projector23*psi43[indexTotal])*ProjectorL21 +\
		(Projector20*psi14[indexTotal] + Projector21*psi24[indexTotal] +\
                Projector22*psi34[indexTotal] + Projector23*psi44[indexTotal])*ProjectorL31;

  _Psi33[indexTotal] = (Projector20*psi11[indexTotal] + Projector21*psi21[indexTotal] +\
                Projector22*psi31[indexTotal] + Projector23*psi41[indexTotal])*ProjectorL02 +\
		(Projector20*psi12[indexTotal] + Projector21*psi22[indexTotal] +\
                Projector22*psi32[indexTotal] + Projector23*psi42[indexTotal])*ProjectorL12 +\
		(Projector20*psi13[indexTotal] + Projector21*psi23[indexTotal] +\
                Projector22*psi33[indexTotal] + Projector23*psi43[indexTotal])*ProjectorL22 +\
		(Projector20*psi14[indexTotal] + Projector21*psi24[indexTotal] +\
                Projector22*psi34[indexTotal] + Projector23*psi44[indexTotal])*ProjectorL32;

  _Psi34[indexTotal] = (Projector20*psi11[indexTotal] + Projector21*psi21[indexTotal] +\
                Projector22*psi31[indexTotal] + Projector23*psi41[indexTotal])*ProjectorL03 +\
		(Projector20*psi12[indexTotal] + Projector21*psi22[indexTotal] +\
                Projector22*psi32[indexTotal] + Projector23*psi42[indexTotal])*ProjectorL13 +\
		(Projector20*psi13[indexTotal] + Projector21*psi23[indexTotal] +\
                Projector22*psi33[indexTotal] + Projector23*psi43[indexTotal])*ProjectorL23 +\
		(Projector20*psi14[indexTotal] + Projector21*psi24[indexTotal] +\
                Projector22*psi34[indexTotal] + Projector23*psi44[indexTotal])*ProjectorL33;
                
  _Psi41[indexTotal] = (Projector30*psi11[indexTotal] + Projector31*psi21[indexTotal] +\
                Projector32*psi31[indexTotal] + Projector33*psi41[indexTotal])*ProjectorL00 +\
		(Projector30*psi12[indexTotal] + Projector31*psi22[indexTotal] +\
                Projector32*psi32[indexTotal] + Projector33*psi42[indexTotal])*ProjectorL10 +\
		(Projector30*psi13[indexTotal] + Projector31*psi23[indexTotal] +\
                Projector32*psi33[indexTotal] + Projector33*psi43[indexTotal])*ProjectorL20 +\
		(Projector30*psi14[indexTotal] + Projector31*psi24[indexTotal] +\
                Projector32*psi34[indexTotal] + Projector33*psi44[indexTotal])*ProjectorL30;

  _Psi42[indexTotal] = (Projector30*psi11[indexTotal] + Projector31*psi21[indexTotal] +\
                Projector32*psi31[indexTotal] + Projector33*psi41[indexTotal])*ProjectorL01 +\
		(Projector30*psi12[indexTotal] + Projector31*psi22[indexTotal] +\
                Projector32*psi32[indexTotal] + Projector33*psi42[indexTotal])*ProjectorL11 +\
		(Projector30*psi13[indexTotal] + Projector31*psi23[indexTotal] +\
                Projector32*psi33[indexTotal] + Projector33*psi43[indexTotal])*ProjectorL21 +\
		(Projector30*psi14[indexTotal] + Projector31*psi24[indexTotal] +\
                Projector32*psi34[indexTotal] + Projector33*psi44[indexTotal])*ProjectorL31;


  _Psi43[indexTotal] = (Projector30*psi11[indexTotal] + Projector31*psi21[indexTotal] +\
                Projector32*psi31[indexTotal] + Projector33*psi41[indexTotal])*ProjectorL02 +\
		(Projector30*psi12[indexTotal] + Projector31*psi22[indexTotal] +\
                Projector32*psi32[indexTotal] + Projector33*psi42[indexTotal])*ProjectorL12 +\
		(Projector30*psi13[indexTotal] + Projector31*psi23[indexTotal] +\
                Projector32*psi33[indexTotal] + Projector33*psi43[indexTotal])*ProjectorL22 +\
		(Projector30*psi14[indexTotal] + Projector31*psi24[indexTotal] +\
                Projector32*psi34[indexTotal] + Projector33*psi44[indexTotal])*ProjectorL32;

  _Psi44[indexTotal] = (Projector30*psi11[indexTotal] + Projector31*psi21[indexTotal] +\
                Projector32*psi31[indexTotal] + Projector33*psi41[indexTotal])*ProjectorL03 +\
		(Projector30*psi12[indexTotal] + Projector31*psi22[indexTotal] +\
                Projector32*psi32[indexTotal] + Projector33*psi42[indexTotal])*ProjectorL13 +\
		(Projector30*psi13[indexTotal] + Projector31*psi23[indexTotal] +\
                Projector32*psi33[indexTotal] + Projector33*psi43[indexTotal])*ProjectorL23 +\
		(Projector30*psi14[indexTotal] + Projector31*psi24[indexTotal] +\
                Projector32*psi34[indexTotal] + Projector33*psi44[indexTotal])*ProjectorL33;


}
"""


DiracPropagator_DampingODM_source = """
//
//   source code for damping ODM 
//   

#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

//............................................................................................................

__global__ void DampingODM_Kernel(
pycuda::complex<double>*psi11,pycuda::complex<double>*psi12,pycuda::complex<double>*psi13,pycuda::complex<double>*psi14,             
pycuda::complex<double>*psi21,pycuda::complex<double>*psi22,pycuda::complex<double>*psi23,pycuda::complex<double>*psi24, 
pycuda::complex<double>*psi31,pycuda::complex<double>*psi32,pycuda::complex<double>*psi33,pycuda::complex<double>*psi34, pycuda::complex<double>*psi41,pycuda::complex<double>*psi42,pycuda::complex<double>*psi43,pycuda::complex<double>*psi44, 
double gammaDamping)
{


  double t = dt;
  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x     =     dx*( j - 0.5*X_gridDIM  );
  double theta = dtheta*( i - 0.5*P_gridDIM  );

  double mass_c = mass*c;
  pycuda::complex<double> I = pycuda::complex<double>(0.,1.);		
  
  double F = sqrt(lambdaBar);

   pycuda::complex<double> PsiPlus11, PsiPlus12, PsiPlus13, PsiPlus14;
   pycuda::complex<double> PsiPlus21, PsiPlus22, PsiPlus23, PsiPlus24;
   pycuda::complex<double> PsiPlus31, PsiPlus32, PsiPlus33, PsiPlus34;
   pycuda::complex<double> PsiPlus41, PsiPlus42, PsiPlus43, PsiPlus44;

   PsiPlus11 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi11[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi44[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi11[indexTotal] * cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi44[indexTotal] * cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi14[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi41[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi14[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi41[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus11 *= 0.5;

   PsiPlus14 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi14[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi41[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi14[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi41[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi11[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi44[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi11[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi44[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
  
   PsiPlus14 *= 0.5;
   
   PsiPlus41 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi14[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi41[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi14[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi41[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi11[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi44[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi11[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi44[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus41 *= 0.5;

   PsiPlus44 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi11[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi44[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi11[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi44[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi14[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi41[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi14[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi41[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus44 *= 0.5;

   PsiPlus12 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi12[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi43[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi12[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi43[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi13[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi42[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi13[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi42[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus12 *= 0.5;
 
   PsiPlus13 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi13[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi42[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi13[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi42[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi12[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi43[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi12[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi43[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus13 *= 0.5;

   PsiPlus42 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi13[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi42[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi13[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi42[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi12[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi43[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi12[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi43[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus42 *= 0.5;

   PsiPlus43 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi12[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi43[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi12[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi43[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi13[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi42[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi13[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi42[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus43 *= 0.5;

   PsiPlus21 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi21[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi34[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi21[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi34[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi24[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi31[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi24[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi31[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus21 *= 0.5;

   PsiPlus24 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi24[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi31[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi24[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi31[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi21[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi34[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi21[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi34[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus24 *= 0.5;

   PsiPlus31 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi24[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi31[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi24[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi31[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi21[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi34[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi21[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi34[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus31 *= 0.5;

   PsiPlus34 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi21[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi34[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi21[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi34[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi24[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi31[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi24[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi31[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus34 *= 0.5;

   PsiPlus22 = exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi22[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi33[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi22[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi33[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi23[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi32[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi23[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi32[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus22 *= 0.5;

   PsiPlus23 =exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi23[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		-exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi32[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi23[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi32[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi22[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi33[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi22[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi33[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus23 *= 0.5;

   PsiPlus32 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi23[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi32[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi23[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi32[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi22[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi33[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi22[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi33[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus32 *= 0.5;

   PsiPlus33 = -exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi22[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi33[indexTotal] *cos(4.*mass_c*t*x*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi22[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping)
		+exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi33[indexTotal] *cos(2.*mass_c*t*theta*gammaDamping) 
		-I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi23[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-8.*mass_c*t*gammaDamping*x*x/(F*F)) *psi32[indexTotal] *sin(4.*mass_c*t*x*gammaDamping)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi23[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c)
		+I*exp(-2.*mass_c*t*gammaDamping*theta*theta/(F*F)) *psi32[indexTotal]* sin(2.*t*gammaDamping*theta*mass_c);
   
   PsiPlus33 *= 0.5;

  psi11[indexTotal] = PsiPlus11;
  psi12[indexTotal] = PsiPlus12;
  psi13[indexTotal] = PsiPlus13;
  psi14[indexTotal] = PsiPlus14; 

  psi21[indexTotal] = PsiPlus21;
  psi22[indexTotal] = PsiPlus22;
  psi23[indexTotal] = PsiPlus23;
  psi24[indexTotal] = PsiPlus24; 

  psi31[indexTotal] = PsiPlus31;
  psi32[indexTotal] = PsiPlus32;
  psi33[indexTotal] = PsiPlus33;
  psi34[indexTotal] = PsiPlus34; 

  psi41[indexTotal] = PsiPlus41;
  psi42[indexTotal] = PsiPlus42;
  psi43[indexTotal] = PsiPlus43;
  psi44[indexTotal] = PsiPlus44; 

}
"""

#..........................................................................................................


TakabayashiAngle_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

//............................................................................................................

__global__ void Kernel( double * TakabayashiAngle,
pycuda::complex<double>* W11,  pycuda::complex<double>* W12,  pycuda::complex<double>* W13,  pycuda::complex<double>* W14,
pycuda::complex<double>* W21,  pycuda::complex<double>* W22,  pycuda::complex<double>* W23,  pycuda::complex<double>* W24,
pycuda::complex<double>* W31,  pycuda::complex<double>* W32,  pycuda::complex<double>* W33,  pycuda::complex<double>* W34,
pycuda::complex<double>* W41,  pycuda::complex<double>* W42,  pycuda::complex<double>* W43,  pycuda::complex<double>* W44 )
{

   const int X_gridDIM = blockDim.x * gridDim.z;
   const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  double s = 2.*pycuda::imag<double>( W13[indexTotal] ) + 2.*pycuda::imag<double>( W24[indexTotal] );

  double c =  pycuda::real<double>(W11[indexTotal]);
         c += pycuda::real<double>(W22[indexTotal]);
  	 c -= pycuda::real<double>(W33[indexTotal]);
	 c -= pycuda::real<double>(W44[indexTotal]);

  TakabayashiAngle[indexTotal] = atan2(s,c);

}

"""

#..........................................................................................................

Potential_0_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double Potential0(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel( pycuda::complex<double>* preExpectationValue, 
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44 ,
double t)
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  double p = dp*( i - 0.5*P_gridDIM  );

  preExpectationValue[indexTotal] = Potential0( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

Potential_1_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double Potential1(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel( pycuda::complex<double>* preExpectationValue,
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44 ,
double t )
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  double p = dp*( i - 0.5*P_gridDIM  );

  preExpectationValue[indexTotal] = Potential1( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

Potential_2_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double Potential2(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel( pycuda::complex<double>* preExpectationValue,  
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44 ,
double t)
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  double p = dp*( i - 0.5*P_gridDIM  );

  preExpectationValue[indexTotal] = Potential2( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

Potential_3_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double Potential3(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel( pycuda::complex<double>* preExpectationValue, 
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44 ,
double t)
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  double p = dp*( i - 0.5*P_gridDIM  );

  preExpectationValue[indexTotal] = Potential3( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

#..........................................................................

D_1_Potential_0_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 


__device__  double D_1_Potential_0(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel(  pycuda::complex<double>* out,
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44,
double t)
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  //const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  //const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  //double p = dp*( i - 0.5*P_gridDIM  );

  out[indexTotal] = D_1_Potential_0( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

X1_D_1_Potential_0_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double D_1_Potential_0(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel(  pycuda::complex<double>* out,
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44,
double t)
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  //const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  //const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  //double p = dp*( i - 0.5*P_gridDIM  );

  out[indexTotal] = x*D_1_Potential_0( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

P1_D_1_Potential_0_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 

__device__  double D_1_Potential_0(double t, double x)
{
    return %s ;
}

//............................................................................................................

__global__ void Kernel(  pycuda::complex<double>* out,
pycuda::complex<double>* W11,  pycuda::complex<double>* W22,  pycuda::complex<double>* W33,  pycuda::complex<double>* W44,
double t)
{

  const int X_gridDIM = blockDim.x * gridDim.z;
  const int P_gridDIM = gridDim.x;
  
  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

  double x = dx*( j - 0.5*X_gridDIM  );
  double p = dp*( i - 0.5*P_gridDIM  );

  out[indexTotal] = p*D_1_Potential_0( t, x)*(W11[indexTotal] + W22[indexTotal] + W33[indexTotal] + W44[indexTotal]); 

}

"""

X_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W11, pycuda::complex<double> *W22, pycuda::complex<double> *W33, pycuda::complex<double> *W44)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  double x     =     dx*( j - 0.5*X_gridDIM  );
	  //double theta = dtheta*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W11[indexTotal]*x;
	  _out +=  W22[indexTotal]*x;
	  _out +=  W33[indexTotal]*x;
	  _out +=  W44[indexTotal]*x;	

	  out[indexTotal] = _out;
	  
	}
	"""

P_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W11, pycuda::complex<double> *W22, pycuda::complex<double> *W33, pycuda::complex<double> *W44)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  //double x =  dx*( j - 0.5*X_gridDIM  );
	  double p =  dp*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W11[indexTotal]*p;
	  _out +=  W22[indexTotal]*p;
	  _out +=  W33[indexTotal]*p;
	  _out +=  W44[indexTotal]*p;	

	  out[indexTotal] = _out;
	  
	}
	"""

XP_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W11, pycuda::complex<double> *W22, pycuda::complex<double> *W33, pycuda::complex<double> *W44)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  double x =  dx*( j - 0.5*X_gridDIM  );
	  double p =  dp*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W11[indexTotal]*x*p;
	  _out +=  W22[indexTotal]*x*p;
	  _out +=  W33[indexTotal]*x*p;
	  _out +=  W44[indexTotal]*x*p;	

	  out[indexTotal] = _out;
	  
	}
	"""

XX_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W11, pycuda::complex<double> *W22, pycuda::complex<double> *W33, pycuda::complex<double> *W44)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  //const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  double x     =     dx*( j - 0.5*X_gridDIM  );
	  //double theta = dtheta*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W11[indexTotal]*x*x;
	  _out +=  W22[indexTotal]*x*x;
	  _out +=  W33[indexTotal]*x*x;
	  _out +=  W44[indexTotal]*x*x;	

	  out[indexTotal] = _out;
	  
	}
	"""

PP_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W11, pycuda::complex<double> *W22, pycuda::complex<double> *W33, pycuda::complex<double> *W44)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  //const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  //double x     =     dx*( j - 0.5*X_gridDIM  );
	  double p = dp*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W11[indexTotal]*p*p;
	  _out +=  W22[indexTotal]*p*p;
	  _out +=  W33[indexTotal]*p*p;
	  _out +=  W44[indexTotal]*p*p;	

	  out[indexTotal] = _out;
	  
	}
	"""

Alpha_1_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W14, pycuda::complex<double> *W23, pycuda::complex<double> *W32, pycuda::complex<double> *W41)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  //const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  //const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  //double x     =     dx*( j - 0.5*X_gridDIM  );
	  //double theta = dtheta*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W14[indexTotal];
	  _out +=  W23[indexTotal];
	  _out +=  W32[indexTotal];
	  _out +=  W41[indexTotal];	

	  out[indexTotal] = _out;
	  
	}
	"""


Alpha_2_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W14, pycuda::complex<double> *W23, pycuda::complex<double> *W32, pycuda::complex<double> *W41)
	  {

          pycuda::complex<double> I = pycuda::complex<double>(0.,1.);

	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  //const int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  //const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  //double x     =     dx*( j - 0.5*X_gridDIM  );
	  //double theta = dtheta*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =   -I*W14[indexTotal];
	  _out +=    I*W23[indexTotal];
	  _out +=   -I*W32[indexTotal];
	  _out +=    I*W41[indexTotal];	

	  out[indexTotal] = _out;
	  
	}
	"""



P1_Alpha_1_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W14, pycuda::complex<double> *W23, pycuda::complex<double> *W32, pycuda::complex<double> *W41)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  const   int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  //const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  //double x = dx*( j - 0.5*X_gridDIM  );
	  double   p = dp*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W14[indexTotal];
	  _out +=  W23[indexTotal];
	  _out +=  W32[indexTotal];
	  _out +=  W41[indexTotal];	

	  out[indexTotal] = _out*p;
	  
	}
	"""

X1_Alpha_1_Average_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s; // Constants 
	
__global__ void Kernel( pycuda::complex<double> *out, 
	   pycuda::complex<double> *W14, pycuda::complex<double> *W23, pycuda::complex<double> *W32, pycuda::complex<double> *W41)
	  {
	  const int X_gridDIM = blockDim.x * gridDim.z;
	  const int P_gridDIM = gridDim.x;
	  
	  const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;	

	  //const   int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
	  const int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	

	  double x = dx*( j - 0.5*X_gridDIM  );
	  //double   p = dp*( i - 0.5*P_gridDIM  );

	  pycuda::complex<double> _out; 
	  
	  _out  =  W14[indexTotal];
	  _out +=  W23[indexTotal];
	  _out +=  W32[indexTotal];
	  _out +=  W41[indexTotal];	

	  out[indexTotal] = _out*x;
	  
	}
	"""

#..........................................................................................................

# Non-linear phase space

gpu_sum_axis0_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__global__ void Kernel( pycuda::complex<double> *Probability_x ,
  pycuda::complex<double> *W11, pycuda::complex<double> *W22, pycuda::complex<double> *W33, pycuda::complex<double> *W44,
  int P_DIM)
{
   int X_DIM = blockDim.x*gridDim.x;
 
   const int index_x = threadIdx.x + blockDim.x*blockIdx.x ;

   pycuda::complex<double> sum=0.;
   for(int i=0; i<P_DIM; i++ ){
      sum += W11[ index_x + i*X_DIM ];
      sum += W22[ index_x + i*X_DIM ];
      sum += W33[ index_x + i*X_DIM ];
      sum += W44[ index_x + i*X_DIM ];
		}	
	
   Probability_x[ index_x ] = pycuda::real(sum);
}

"""

roll_FirstRowCopy_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__global__ void Kernel( pycuda::complex<double> *W, pycuda::complex<double> *Probability_X  , int P_DIM)
{
   int X_DIM = blockDim.x*gridDim.x;
 
   const int index_x = threadIdx.x + blockDim.x*blockIdx.x ;

   pycuda::complex<double> firstRow = Probability_X[index_x];

   for(int i=0; i<P_DIM; i++ )  W[ index_x + i*X_DIM ] = firstRow;

}

"""

#.................................................................

theta_fp_source = """
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES

%s

__device__ double f( double p)
{
return %s;
}

__global__ void Kernel( pycuda::complex<double> *B )
{
	
   const int X_gridDIM = blockDim.x * gridDim.z;
   const int P_gridDIM = gridDim.x;
	  
   const int indexTotal = threadIdx.x + blockIdx.z*blockDim.x + X_gridDIM * blockIdx.x   ;

   const   int i =  (blockIdx.x                           +  P_gridDIM/2) %% P_gridDIM ;	
   const   int j =  (threadIdx.x + blockIdx.z*blockDim.x  +  X_gridDIM/2) %% X_gridDIM ;	
 
   double   p = dp*( i - 0.5*P_gridDIM  );

    if( p >= 0.  )
    	B[ indexTotal ] *=  f(p); 
    else
        B[ indexTotal ] *= -f(p); 

}

"""


#................................................................


class GPU_WignerDirac2D_4x4:
	"""
	Propagator in the X-Theta representation
	This version propagates the 16 components of the Wigner function components
	This algorithm advances in time at double the rate of the ordinary Dirac propagator dt = 2 dx/c 	  
	"""
	def __init__(self, X_gridDIM, P_gridDIM, X_amplitude, P_amplitude,
			mass, c,  dt,timeSteps, skipFrames = 1,
			frameSaveMode='Density', antiParticleNorm = True, antiParticleStepFiltering=False,
			computeEnergy = 'False'):

		self.mass = mass
		self.c = c
		self.dt = dt
		self.timeSteps     = timeSteps
		self.frameSaveMode = frameSaveMode
		self.skipFrames    = skipFrames
		self.antiParticleNorm = antiParticleNorm
		self.antiParticleStepFiltering = antiParticleStepFiltering
		self.computeEnergy = computeEnergy
		#self.dampingModel = dampingModel

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

		self.X_range      =  np.linspace(-self.X_amplitude      , self.X_amplitude  -self.dX , self.X_gridDIM )
		self.Lambda_range =  np.linspace(-self.Lambda_amplitude , self.Lambda_amplitude-self.dLambda    ,self.X_gridDIM)

		self.Theta_range  = np.linspace(-self.Theta_amplitude  , self.Theta_amplitude - self.dTheta , self.P_gridDIM)
		self.P_range      = np.linspace(-self.P_amplitude      , self.P_amplitude-self.dP           , self.P_gridDIM)	

		self.X      = fftpack.fftshift(self.X_range)[np.newaxis,:]
		self.Theta  = fftpack.fftshift(self.Theta_range)[:,np.newaxis]

		self.Lambda = fftpack.fftshift(self.Lambda_range)[np.newaxis,:]
		self.P      = fftpack.fftshift(self.P_range)[:,np.newaxis]
		
		#........................... Strings .........................................

		self.CUDA_constants  =  '__constant__ double mass=%f;     '%(self.mass)
		self.CUDA_constants +=  '__constant__ double mass_half=%f;'%(self.mass/2.)
		self.CUDA_constants +=  '__constant__ double c=%f;        '%self.c
		self.CUDA_constants +=  '__constant__ double dt=%f;       '%self.dt
		self.CUDA_constants +=  '__constant__ double dx=%f;       '%self.dX
		self.CUDA_constants +=  '__constant__ double dlambda=%f;  '%self.dLambda
		self.CUDA_constants +=  '__constant__ double dp=%f;       '%self.dP
		self.CUDA_constants +=  '__constant__ double dtheta=%f;   '%self.dTheta

		try: 
			self.CUDA_constants += '__constant__ double D_Theta  = %f;'%(self.D_Theta )
			self.CUDA_constants += '__constant__ double D_Lambda = %f;'%(self.D_Lambda)
		except:
			pass



		try: 
			self.CUDA_constants += '__constant__ double lambdaBar    = %f;'%(self.lambdaBar)
		except:
			pass


		#............................... Initializing CUDA ...........................		

		self.D_1_Potential_0_String = sympy.ccode( sympy.N( sympy.diff(self.Potential_0_String,'x') ) ) 
		self.D_1_Potential_0_String = str( self.D_1_Potential_0_String  ) + ' + 0.*x'
				
		self.D_1_Potential_1_String = sympy.ccode( sympy.N( sympy.diff(self.Potential_1_String,'x') ) ) 
		self.D_1_Potential_1_String = str( self.D_1_Potential_1_String  ) + ' + 0.*x'

		self.D_1_Potential_2_String = sympy.ccode( sympy.N( sympy.diff(self.Potential_2_String,'x') ) ) 
		self.D_1_Potential_2_String = str( self.D_1_Potential_2_String  ) + ' + 0.*x'

		self.D_1_Potential_3_String = sympy.ccode( sympy.N( sympy.diff(self.Potential_3_String,'x') ) ) 
		self.D_1_Potential_3_String = str( self.D_1_Potential_3_String  ) + ' + 0.*x'
		

		print '  D_1_Potential_0 = ', self.D_1_Potential_0_String

		self.Make_CUFFTPlan()

		self.Allocate_GPUVariables()

		self.Compiling_CUDA_functions()

		self.phase_LambdaTheta_GPU = gpuarray.to_gpu( 
				 np.ascontiguousarray( np.exp( 0.5*1j*self.Lambda*self.Theta ) , dtype=np.complex128) )


		

#-------------------------------------------------------------------------------------------------------------------
#           Gaussian PARTICLE spinors 
#-------------------------------------------------------------------------------------------------------------------

	def Make_CUFFTPlan(self):

		self.plan_Z2Z_1D_Axes0 = cuda_fft.Plan_Z2Z_2D_Axis0(  (self.P_gridDIM,self.X_gridDIM)  )
		self.plan_Z2Z_1D_Axes1 = cuda_fft.Plan_Z2Z_2D_Axis1(  (self.P_gridDIM,self.X_gridDIM)  ) 

		self.plan_Z2Z_1D = cuda_fft.Plan_Z2Z(  (self.X_gridDIM,)  ,  batch=1 )

	def Allocate_GPUVariables(self):

		self.X_GPU     = gpuarray.to_gpu( np.ascontiguousarray( self.X + 0.*self.P, dtype = np.complex128)     )
		self.P_GPU     = gpuarray.to_gpu( np.ascontiguousarray( self.P + 0.*self.X, dtype = np.complex128)     )

		self.Theta_GPU = gpuarray.to_gpu( np.ascontiguousarray( self.Theta + 0.*self.X, dtype = np.complex128) )

		self.XX_GPU    = gpuarray.to_gpu( np.ascontiguousarray( self.X**2 + 0.*self.P, dtype=np.complex128)    )
		self.PP_GPU    = gpuarray.to_gpu( np.ascontiguousarray( self.P**2 + 0.*self.X, dtype=np.complex128)    )

		self.XP_GPU    = gpuarray.to_gpu( np.ascontiguousarray( self.P*self.X  ,dtype=np.complex128)           )


		self.Potential0 = self.PotentialFunction_0( 0., self.X + 0j*self.P )
		self.Potential0_GPU = gpuarray.to_gpu( self.Potential0 ) 


		self.D_1_Potential0_GPU = gpuarray.to_gpu( self.D_1_PotentialFunction_0( 0., self.X + 0j*self.P)  )

		#self.D_1_Potential1_GPU = gpuarray.to_gpu( self.D_1_PotentialFunction_1( se']lf.X + 0j*self.P)  )
		#self.D_1_Potential2_GPU = gpuarray.to_gpu( self.D_1_PotentialFunction_2( self.X + 0j*self.P)  )
		#self.D_1_Potential3_GPU = gpuarray.to_gpu( self.D_1_PotentialFunction_3( self.X + 0j*self.P)  )
	
		self.X1_D_1_Potential0_GPU = gpuarray.to_gpu( self.X*self.D_1_PotentialFunction_0( 0., self.X + 0j*self.P)  )
		self.P1_D_1_Potential0_GPU = gpuarray.to_gpu( self.P*self.D_1_PotentialFunction_0( 0., self.X + 0j*self.P)  )


	def Compiling_CUDA_functions(self):
		
		#print CUDAsource_P_plus_Lambda%(self.CUDA_constants)
	
		self.DiracPropagator_P_plus_Lambda   =  \
		SourceModule(CUDAsource_P_plus_Lambda%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )


		self.DiracPropagator_P_minus_Lambda  =  \
		SourceModule(CUDAsource_P_minus_Lambda%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		
		DiracPropagator_X_minus_Theta_source = DiracPropagator_X_minus_Theta_source_Base%(
				self.CUDA_constants,self.Potential_0_String,self.Potential_1_String,
				self.Potential_2_String,self.Potential_3_String)

		self.DiracPropagator_X_minus_Theta  =  \
			SourceModule( DiracPropagator_X_minus_Theta_source ,arch="sm_20").get_function( "DiracPropagation4_Kernel" )

		
		DiracPropagator_X_plus_Theta_source = DiracPropagator_X_plus_Theta_source_Base%(
				self.CUDA_constants,self.Potential_0_String,self.Potential_1_String,
				self.Potential_2_String,self.Potential_3_String)

		self.DiracPropagator_X_plus_Theta  =  \
			SourceModule( DiracPropagator_X_plus_Theta_source ,arch="sm_20").get_function( "DiracPropagation4_Kernel" )

		self.gpu_array_copy_Function = SourceModule(gpu_array_copy_source).get_function( "Kernel" )

		self.FilterElectrons_Function = \
		SourceModule(BaseCUDAsource_FilterGPU%(self.CUDA_constants),arch="sm_20").get_function("Filter_Kernel" )

		self.AbsorbBoundary_x_Function = SourceModule(CUDAsource_AbsorbBoundary_x).get_function("Kernel") 


		self.pickup_negatives_Function = SourceModule(pickup_negatives_source).get_function("pickup_negatives_Kernel") 


		self.transmission_Function = SourceModule(
			transmission_source%(self.CUDA_constants)).get_function("transmission_Kernel") 

		self.TakabayashiAngle_Function = SourceModule( TakabayashiAngle_source ).get_function("Kernel" )

		try :
			self.DiracPropagator_DampingODM = \
			SourceModule( DiracPropagator_DampingODM_source%(self.CUDA_constants),
			arch="sm_20").get_function("DampingODM_Kernel")
		except:
			pass		



		self.Potential_0_Average_Function = \
		SourceModule( Potential_0_Average_source%(
			self.CUDA_constants,self.Potential_0_String),arch="sm_20").get_function("Kernel" )

		self.Potential_1_Average_Function = \
		SourceModule( Potential_1_Average_source%(
			self.CUDA_constants,self.Potential_1_String),arch="sm_20").get_function("Kernel" )

		self.Potential_2_Average_Function = \
		SourceModule( Potential_2_Average_source%(
			self.CUDA_constants,self.Potential_2_String),arch="sm_20").get_function("Kernel" )

		self.Potential_3_Average_Function = \
		SourceModule( Potential_3_Average_source%(
			self.CUDA_constants,self.Potential_3_String),arch="sm_20").get_function("Kernel" )

		#self.D_1_Potential_0_ExpectationValue_Function = \
		#SourceModule( D_1_Potential_0_Expectation_source%(
		#	self.CUDA_constants),arch="sm_20").get_function("Kernel" )

		self.X_Average_Function  =  \
		SourceModule( X_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.P_Average_Function  =  \
		SourceModule( P_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.XP_Average_Function  =  \
		SourceModule( XP_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.XX_Average_Function  =  \
		SourceModule( XX_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.Alpha_1_Average_Function  =  \
		SourceModule( Alpha_1_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.P1_Alpha_1_Average_Function  =  \
		SourceModule( P1_Alpha_1_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.X1_Alpha_1_Average_Function  =  \
		SourceModule( X1_Alpha_1_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )

		self.PP_Average_Function  =  \
		SourceModule( PP_Average_source%(self.CUDA_constants),arch="sm_20").get_function( "Kernel" )
		
		self.D_1_Potential_0_Average_Function = \
		SourceModule( D_1_Potential_0_Average_source%(
				self.CUDA_constants,self.D_1_Potential_0_String),arch="sm_20").get_function( "Kernel" )

		self.X1_D_1_Potential_0_Average_Function = \
		SourceModule( X1_D_1_Potential_0_Average_source%(
				self.CUDA_constants,self.D_1_Potential_0_String),arch="sm_20").get_function( "Kernel" )

		self.P1_D_1_Potential_0_Average_Function = \
		SourceModule( P1_D_1_Potential_0_Average_source%(
				self.CUDA_constants,self.D_1_Potential_0_String),arch="sm_20").get_function( "Kernel" )
		
		#

		self.roll_FirstRowCopy_Function = SourceModule(									     						roll_FirstRowCopy_source%self.CUDA_constants, arch="sm_20").get_function( "Kernel" )

		self.gpu_sum_axis0_Function = SourceModule(									     						gpu_sum_axis0_source%self.CUDA_constants, arch="sm_20").get_function( "Kernel" )

		try:
			self.theta_fp_Damping_Function = SourceModule(\
					theta_fp_source%(self.CUDA_constants,self.fp_Damping_String), 
					arch="sm_20").get_function("Kernel")
		except AttributeError: 
			pass	

	#......................................................................................	

	def Potential_0_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t):
		self.Potential_0_Average_Function( temp_GPU, 
			 W11_GPU, W22_GPU, W33_GPU, W44_GPU, t , block=self.blockCUDA, grid=self.gridCUDA  )
		return self.dX*self.dP * gpuarray.sum(temp_GPU).get()	

	def Potential_1_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU , t):
		self.Potential_1_Average_Function( temp_GPU,
			 W11_GPU, W22_GPU, W33_GPU, W44_GPU, t, block=self.blockCUDA, grid=self.gridCUDA  )
		return self.dX*self.dP * gpuarray.sum(temp_GPU).get()	
	
	def Potential_2_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t):
		self.Potential_2_Average_Function( temp_GPU, 
			 W11_GPU, W22_GPU, W33_GPU, W44_GPU, t, block=self.blockCUDA, grid=self.gridCUDA  )
		return self.dX*self.dP * gpuarray.sum(temp_GPU).get()	

	def Potential_3_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t):
		self.Potential_3_Average_Function( temp_GPU, 
			 W11_GPU, W22_GPU, W33_GPU, W44_GPU, t, block=self.blockCUDA, grid=self.gridCUDA  )
		return self.dX*self.dP * gpuarray.sum(temp_GPU).get()	


	#def D_1_Potential_0_Average(self, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t, temp_GPU):
	#	self.D_1_Potential_0_Average_Function(
	#		 W11_GPU, W22_GPU, W33_GPU, W44_GPU, t, temp_GPU, block=self.blockCUDA, grid=self.gridCUDA  )
	#	return gpuarray.sum(temp_GPU).get()	

	#......................................................................................

	def PotentialFunction_0(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.Potential_0_String, np.__dict__, locals() )

	def PotentialFunction_1(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.Potential_1_String, np.__dict__, locals() )

	def PotentialFunction_2(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.Potential_2_String, np.__dict__, locals() )

	def PotentialFunction_3(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.Potential_3_String, np.__dict__, locals() )

	#-------

	def D_1_PotentialFunction_0(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		
		return eval ( self.D_1_Potential_0_String , np.__dict__, locals() )

	def D_1_PotentialFunction_1(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.D_1_Potential_1_String  , np.__dict__, locals() )

	def D_1_PotentialFunction_2(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.D_1_Potential_2_String  , np.__dict__, locals() )

	def D_1_PotentialFunction_3(self, t, x ):
		pow = np.power
		atan = np.arctan
		sqrt = np.sqrt
		cosh = np.cosh
		return eval ( self.D_1_Potential_3_String  , np.__dict__, locals() )

	#............................

	def GaussianSpinor_ParticleUp(self,x,px,s,X) :
		"""
		x,p: Gaussian center
		s:   Gaussian standard deviation in x
		X:   variable
		"""
		
		rho = np.exp(1j*X*px) * np.exp(  -0.5*( (X - x)/s )**2  ) + 0j

		p0  = np.sqrt( px*px + self.mass*self.mass*self.c*self.c )
		
		Psi1 =  rho*( p0  + self.mass*self.c )
		Psi2 =  X*0j 
		Psi3 =  X*0j  
		Psi4 =  rho*px	
		
		return np.array([Psi1, Psi2, Psi3, Psi4 ])

	def GaussianSpinor_ParticleDown(self,x,px,s,X):
		"""
		x,p: Gaussian center
		s:   Gaussian standard deviation in x
		X:   variable
		"""
		rho = np.exp(1j*X*px) * np.exp( -0.5*( (X - x)/s )**2  ) + 0j

		p0  = np.sqrt( px*px + self.mass*self.mass*self.c*self.c )
		
		Psi1 =  1j*X*0j  
		Psi2 =  1j*rho*( p0 + self.mass*self.c ) 
		Psi3 =  1j*rho*px	  
		Psi4 =  1j*X*0j	
		
		return np.array([Psi1, Psi2, Psi3, Psi4 ])

	#
	def GaussianSpinor_AntiParticleDown(self,x,px,s,X) :
		"""
		x,p: Gaussian center
		s:   Gaussian standard deviation in x
		X:   variable
		"""
		rho = np.exp(1j*X*px) * np.exp(  -0.5*( (X - x)/s )**2  ) + 0j

		p0  = np.sqrt( px*px + self.mass*self.mass*self.c*self.c )
		
		Psi1 =  X*0j 
		Psi2 =  rho*px	
		Psi3 =  rho*( -p0  - self.mass*self.c ) 
		Psi4 =  X*0j 	
		
		return -1j*np.array([Psi1, Psi2, Psi3, Psi4 ])


	def GaussianSpinor_AntiParticleUp(self,x,px,s,X) :
		"""
		x,p: Gaussian center
		s:   Gaussian standard deviation in x
		X:   variable
		"""
		rho = np.exp(1j*X*px) * np.exp(  -0.5*( (X - x)/s )**2  ) + 0j

		p0  = np.sqrt( px*px + self.mass*self.mass*self.c*self.c )
		
		Psi1 =  rho*px 
		Psi2 =  X*0j 	
		Psi3 =  X*0j 	
		Psi4 =  rho*( -p0  - self.mass*self.c ) 
		
		return -1j*np.array([Psi1, Psi2, Psi3, Psi4 ])		

	# 
	def SpinorNorm(self,Psi):
		norm =  np.sum( np.abs(Psi[0])**2  )*self.dX*self.dP
		norm +=	np.sum( np.abs(Psi[1])**2  )*self.dX*self.dP
		norm +=	np.sum( np.abs(Psi[2])**2  )*self.dX*self.dP
		norm +=	np.sum( np.abs(Psi[3])**2  )*self.dX*self.dP

		return norm

	def TakabayashiAngle_CPU(self,W):
	        WW = W.copy()
	        s = 2.*np.imag( WW[0,2] ) + 2.*np.imag( WW[1,3] );
	        c =  np.real( WW[0,0] ) + np.real( WW[1,1] ) - np.real( WW[2,2] ) - np.real( WW[3,3] );
	
	        return  np.arctan2( s , np.abs(c) );

	#...................................................................................

	def _ConstructMajoranaSpinor(self, Psi_real ):
		"""
		returns a spinor in the stanbdard representation from a spinor in the Majorana representation
		Depreciated
		"""
		PsiMajorana    =  np.empty_like( Psi_real + 0j ) 

		PsiMajorana[0] = (   -Psi_real[0]                               + 1j*Psi_real[3] )/np.sqrt(2)
		PsiMajorana[1] = (   		   -Psi_real[1] - 1j*Psi_real[2]                 )/np.sqrt(2)
		PsiMajorana[2] = (              -1j*Psi_real[1] -    Psi_real[2]                 )/np.sqrt(2)
		PsiMajorana[3] = ( 1j*Psi_real[0]                                   -Psi_real[3] )/np.sqrt(2)
	
		return PsiMajorana


	#...................................................................................

	def MajoranaSpinorPlus(self, Psi ):
		PsiMajorana    =  np.empty_like( Psi + 0j ) 

		PsiMajorana[0] = 0.5*( Psi[0] - Psi[3].conj() )
		PsiMajorana[1] = 0.5*( Psi[1] + Psi[2].conj() )
		PsiMajorana[2] = 0.5*( Psi[2] + Psi[1].conj() )
		PsiMajorana[3] = 0.5*( Psi[3] - Psi[0].conj() )

		return PsiMajorana

	def MajoranaSpinorMinus(self, Psi ):
		PsiMajorana    =  np.empty_like( Psi + 0j ) 

		PsiMajorana[0] = 0.5*( Psi[0] + Psi[3].conj() )
		PsiMajorana[1] = 0.5*( Psi[1] - Psi[2].conj() )
		PsiMajorana[2] = 0.5*( Psi[2] - Psi[1].conj() )
		PsiMajorana[3] = 0.5*( Psi[3] + Psi[0].conj() )

		return PsiMajorana

	#...................................................................................

	def WignerDirac_MatrixProduct(self, outC , inA , inB):
		sumM = np.zeros([ self.P_gridDIM , self.X_gridDIM], dtype = np.complex128 )

		for i in range(4) : 
			for j in range(4):		
				sumM *=0
				for k in range(4):
					sumM += inA[i,k] * inB[k,j]
				outC[i,j][:,:] = sumM

	def DiracElectronProjector(self,p,sign):

		mass = self.mass
		c   =  self.c

                sqrtp = 2*np.sqrt( (mass*c**2)**2 + (c*p)**2  )

                aa = 0.5      + sign*mass*c*c/sqrtp
                bb = c*sign*p/sqrtp
                cc = c*sign*p/sqrtp
                dd = 0.5      - sign*mass*c*c/sqrtp

		z = np.zeros([ self.P_gridDIM , self.X_gridDIM])

		return np.array( [ [aa , z   , z   , bb  ],
                                   [z  , aa  , cc  , z   ],
                                   [z  , bb  , dd  , z   ],
                                   [cc , z   , z   , dd] ])	


        def FilterElectrons(self, W ,sign):
		
		ElectronProjectorL = self.DiracElectronProjector( self.P + 0.5*self.Lambda , sign)
		ElectronProjectorR = self.DiracElectronProjector( self.P - 0.5*self.Lambda , sign)

		for i in range(4):
			for j in range(4):
				W[i,j][:,:] = self.Fourier_X_To_Lambda(W[i,j])   #fftpack.fft( W[i,j],  axis=1  )

		W_ = np.empty_like(W)

		self.WignerDirac_MatrixProduct( W_  , ElectronProjectorL , W                   )		
		self.WignerDirac_MatrixProduct( W   , W_                , ElectronProjectorR  )

		for i in range(4):
			for j in range(4):
				W[i,j][:,:] = self.Fourier_Lambda_To_X(W[i,j])  #fftpack.ifft( W[i,j],  axis=1  )

	
		

	#................
	
	def FilterElectrons_GPU(self,
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
		 		 sign):

		self.FilterElectrons_Function( 
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
		 		 sign, block=self.blockCUDA, grid=self.gridCUDA )

	#..............

	def AntiParticlePopulation(self,
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU):

		sign = np.int32(-1)		

		self.FilterElectrons_GPU(
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
		 		 sign)

		self.Fourier_Lambda_To_X_GPU( _W11_GPU )
		self.Fourier_Lambda_To_X_GPU( _W22_GPU )
		self.Fourier_Lambda_To_X_GPU( _W33_GPU )
		self.Fourier_Lambda_To_X_GPU( _W44_GPU )
		
		return self.Wigner_4x4_Norm_GPU( _W11_GPU , _W22_GPU , _W33_GPU, _W44_GPU)

	#................

	def ParticlePopulation(self,
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU):

		sign = np.int32(1)		

		self.FilterElectrons_GPU(
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
		 		 sign)

		self.Fourier_Lambda_To_X_GPU( _W11_GPU )
		self.Fourier_Lambda_To_X_GPU( _W22_GPU )
		self.Fourier_Lambda_To_X_GPU( _W33_GPU )
		self.Fourier_Lambda_To_X_GPU( _W44_GPU )
		
		return self.Wigner_4x4_Norm_GPU( _W11_GPU , _W22_GPU , _W33_GPU, _W44_GPU)

	#................

	def FilterAntiParticles(self,
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU):
		"""

		"""
		sign_positive = np.int32(1)

		"""self.Fourier_4X4_X_To_Lambda_GPU(
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					W41_GPU, W42_GPU, W43_GPU, W44_GPU )"""

		self.FilterElectrons_GPU(
				_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
				_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
				_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
				_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
				 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
				 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
				 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
				 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
		 		 sign_positive)

		

	#..........................................

	def Fourier_X_To_Lambda(self,W):
		return fftpack.fft( W  ,  axis=1  )
	def Fourier_Lambda_To_X(self,W):
		return fftpack.ifft( W ,  axis=1  )

	def Fourier_Theta_To_P(self,W):
		return fftpack.ifft( W ,  axis=0  )
	def Fourier_P_To_Theta(self,W):
		return fftpack.fft( W  ,  axis=0  )

	def Fourier_4X4_Theta_To_P(self,W):
		for i in range(4):
			for j in range(4):
				W[i,j][:,:] = self.Fourier_Theta_To_P( W[i,j] )

	def Fourier_4X4_P_To_Theta(self,W):
		for i in range(4):
			for j in range(4):
				W[i,j][:,:] = self.Fourier_P_To_Theta( W[i,j] )

	#............................................................................

	def Fourier_P_To_Theta_GPU(self, W_out_GPU ):
		cuda_fft.fft_Z2Z(  W_out_GPU , W_out_GPU , self.plan_Z2Z_1D_Axes0 )

	def Fourier_Theta_To_P_GPU(self, W_out_GPU ):
		cuda_fft.ifft_Z2Z( W_out_GPU , W_out_GPU , self.plan_Z2Z_1D_Axes0 )
		W_out_GPU *= 1./float(self.P_gridDIM)   # NEGATIVE SIGN

	def Fourier_X_To_Lambda_GPU(self,W_out_GPU):
		cuda_fft.fft_Z2Z( W_out_GPU, W_out_GPU , self.plan_Z2Z_1D_Axes1 )

	def Fourier_Lambda_To_X_GPU(self,W_out_GPU):
		cuda_fft.ifft_Z2Z( W_out_GPU, W_out_GPU , self.plan_Z2Z_1D_Axes1 )
		W_out_GPU *= 1./float(self.X_gridDIM)

	def Fourier_4X4_P_To_Theta_GPU(self, W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					 W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		self.Fourier_P_To_Theta_GPU( W11_GPU )
		self.Fourier_P_To_Theta_GPU( W12_GPU )
		self.Fourier_P_To_Theta_GPU( W13_GPU )	
		self.Fourier_P_To_Theta_GPU( W14_GPU )

		self.Fourier_P_To_Theta_GPU( W21_GPU )
		self.Fourier_P_To_Theta_GPU( W22_GPU )
		self.Fourier_P_To_Theta_GPU( W23_GPU )	
		self.Fourier_P_To_Theta_GPU( W24_GPU )	

		self.Fourier_P_To_Theta_GPU( W31_GPU )
		self.Fourier_P_To_Theta_GPU( W32_GPU )
		self.Fourier_P_To_Theta_GPU( W33_GPU )	
		self.Fourier_P_To_Theta_GPU( W34_GPU )

		self.Fourier_P_To_Theta_GPU( W41_GPU )
		self.Fourier_P_To_Theta_GPU( W42_GPU )
		self.Fourier_P_To_Theta_GPU( W43_GPU )	
		self.Fourier_P_To_Theta_GPU( W44_GPU )


	def Fourier_4X4_Theta_To_P_GPU(self, W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					 W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		self.Fourier_Theta_To_P_GPU( W11_GPU )
		self.Fourier_Theta_To_P_GPU( W12_GPU )
		self.Fourier_Theta_To_P_GPU( W13_GPU )	
		self.Fourier_Theta_To_P_GPU( W14_GPU )

		self.Fourier_Theta_To_P_GPU( W21_GPU )
		self.Fourier_Theta_To_P_GPU( W22_GPU )
		self.Fourier_Theta_To_P_GPU( W23_GPU )	
		self.Fourier_Theta_To_P_GPU( W24_GPU )	

		self.Fourier_Theta_To_P_GPU( W31_GPU )
		self.Fourier_Theta_To_P_GPU( W32_GPU )
		self.Fourier_Theta_To_P_GPU( W33_GPU )	
		self.Fourier_Theta_To_P_GPU( W34_GPU )

		self.Fourier_Theta_To_P_GPU( W41_GPU )
		self.Fourier_Theta_To_P_GPU( W42_GPU )
		self.Fourier_Theta_To_P_GPU( W43_GPU )	
		self.Fourier_Theta_To_P_GPU( W44_GPU )

	def Fourier_4X4_X_To_Lambda_GPU(self,
					 W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					 W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		self.Fourier_X_To_Lambda_GPU( W11_GPU )
		self.Fourier_X_To_Lambda_GPU( W12_GPU )
		self.Fourier_X_To_Lambda_GPU( W13_GPU )	
		self.Fourier_X_To_Lambda_GPU( W14_GPU )

		self.Fourier_X_To_Lambda_GPU( W21_GPU )
		self.Fourier_X_To_Lambda_GPU( W22_GPU )
		self.Fourier_X_To_Lambda_GPU( W23_GPU )	
		self.Fourier_X_To_Lambda_GPU( W24_GPU )	

		self.Fourier_X_To_Lambda_GPU( W31_GPU )
		self.Fourier_X_To_Lambda_GPU( W32_GPU )
		self.Fourier_X_To_Lambda_GPU( W33_GPU )	
		self.Fourier_X_To_Lambda_GPU( W34_GPU )

		self.Fourier_X_To_Lambda_GPU( W41_GPU )
		self.Fourier_X_To_Lambda_GPU( W42_GPU )
		self.Fourier_X_To_Lambda_GPU( W43_GPU )	
		self.Fourier_X_To_Lambda_GPU( W44_GPU )


	def Fourier_4X4_Lambda_To_X_GPU(self,
					 W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					 W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		self.Fourier_Lambda_To_X_GPU( W11_GPU )
		self.Fourier_Lambda_To_X_GPU( W12_GPU )
		self.Fourier_Lambda_To_X_GPU( W13_GPU )	
		self.Fourier_Lambda_To_X_GPU( W14_GPU )

		self.Fourier_Lambda_To_X_GPU( W21_GPU )
		self.Fourier_Lambda_To_X_GPU( W22_GPU )
		self.Fourier_Lambda_To_X_GPU( W23_GPU )	
		self.Fourier_Lambda_To_X_GPU( W24_GPU )	

		self.Fourier_Lambda_To_X_GPU( W31_GPU )
		self.Fourier_Lambda_To_X_GPU( W32_GPU )
		self.Fourier_Lambda_To_X_GPU( W33_GPU )	
		self.Fourier_Lambda_To_X_GPU( W34_GPU )

		self.Fourier_Lambda_To_X_GPU( W41_GPU )
		self.Fourier_Lambda_To_X_GPU( W42_GPU )
		self.Fourier_Lambda_To_X_GPU( W43_GPU )	
		self.Fourier_Lambda_To_X_GPU( W44_GPU )

	
		
	def Wigner_4x4_Norm(self,W):
	        W0  = np.sum(  W[0,0]  )
    		W0 += np.sum(  W[1,1]  )
    		W0 += np.sum(  W[2,2]  )
    		W0 += np.sum(  W[3,3]  )
    
    	        return W0.real * self.dX * self.dP
		
	def Wigner_4X4__SpinTrace(self,W):
		W0  = W[0,0].copy() 
		W0 += W[1,1]
		W0 += W[2,2]
		W0 += W[3,3]
		
		return W0

	#...................................................................................

	def MakeGrossPitaevskiiTerms(self, B_minus_GPU, B_plus_GPU, Prob_X_GPU ):
		"""
		Makes the non-linear terms that characterize the Gross-Pitaevskii equation
		"""
		P_gridDIM_32 = np.int32(self.P_gridDIM)

		cuda_fft.fft_Z2Z( Prob_X_GPU, Prob_X_GPU, self.plan_Z2Z_1D )

		self.roll_FirstRowCopy_Function( B_minus_GPU, Prob_X_GPU, P_gridDIM_32,
				 block=self.blockCUDA, grid=(self.X_gridDIM/512,1)  )	

		self.roll_FirstRowCopy_Function( B_plus_GPU,  Prob_X_GPU, P_gridDIM_32,
				 block=self.blockCUDA, grid=(self.X_gridDIM/512,1)  )	

		B_minus_GPU /= self.phase_LambdaTheta_GPU 
		B_plus_GPU  *= self.phase_LambdaTheta_GPU 

		self.Fourier_Lambda_To_X_GPU( B_minus_GPU )
		self.Fourier_Lambda_To_X_GPU( B_plus_GPU  )

	#...................................................................................

	def WignerDiracPurity(self, W_4x4):
		sum2  = np.sum( W_4x4[0,0]*W_4x4[0,0] )
		sum2 += np.sum( W_4x4[0,1]*W_4x4[1,0] ) 
		sum2 += np.sum( W_4x4[0,2]*W_4x4[2,0] )
		sum2 += np.sum( W_4x4[0,3]*W_4x4[3,0] )   

		sum2 += np.sum( W_4x4[1,0]*W_4x4[0,1] ) 
		sum2 += np.sum( W_4x4[1,1]*W_4x4[1,1] ) 
		sum2 += np.sum( W_4x4[1,2]*W_4x4[2,1] )
		sum2 += np.sum( W_4x4[1,3]*W_4x4[3,1] )   
		
		sum2 += np.sum( W_4x4[2,0]*W_4x4[0,2] ) 
		sum2 += np.sum( W_4x4[2,1]*W_4x4[1,2] ) 
		sum2 += np.sum( W_4x4[2,2]*W_4x4[2,2] )
		sum2 += np.sum( W_4x4[2,3]*W_4x4[3,2] )   

		sum2 =  np.sum( W_4x4[3,0]*W_4x4[0,3] )
		sum2 += np.sum( W_4x4[3,1]*W_4x4[1,3] )
		sum2 += np.sum( W_4x4[3,2]*W_4x4[2,3] )
		sum2 += np.sum( W_4x4[3,3]*W_4x4[3,3] )

		return 2*np.pi*sum2.real*self.dX*self.dP

	def Wigner_4x4_Norm_GPU(self,W11,W22,W33,W44):
	        W0  = gpuarray.sum(  W11  ).get()
    		W0 += gpuarray.sum(  W22  ).get()
    		W0 += gpuarray.sum(  W33  ).get()
    		W0 += gpuarray.sum(  W44  ).get()
    
    	        return W0.real * self.dX * self.dP


	def Wigner_4X4_Normalize(self, 
				W11_GPU, W12_GPU, W13_GPU, W14_GPU,
				W21_GPU, W22_GPU, W23_GPU, W24_GPU,
				W31_GPU, W32_GPU, W33_GPU, W34_GPU,
				W41_GPU, W42_GPU, W43_GPU, W44_GPU ):
		norm = self.Wigner_4x4_Norm_GPU(W11_GPU, W22_GPU, W33_GPU, W44_GPU)

		W11_GPU /= norm
		W12_GPU /= norm
		W13_GPU /= norm
		W14_GPU /= norm

		W21_GPU /= norm
		W22_GPU /= norm
		W23_GPU /= norm
		W24_GPU /= norm

		W31_GPU /= norm
		W32_GPU /= norm
		W33_GPU /= norm
		W34_GPU /= norm

		W41_GPU /= norm
		W42_GPU /= norm
		W43_GPU /= norm
		W44_GPU /= norm


	def DiracEnergy(self,   temp_GPU,
				W11_GPU, W12_GPU, W13_GPU, W14_GPU,
				W21_GPU, W22_GPU, W23_GPU, W24_GPU,
				W31_GPU, W32_GPU, W33_GPU, W34_GPU,
				W41_GPU, W42_GPU, W43_GPU, W44_GPU, t):

		dXdP = self.dX*self.dP	

		c= self.c
		mass = self.mass

		energy  = c*dXdP*gpuarray.dot(W14_GPU,self.P_GPU ).get()
		energy += c*dXdP*gpuarray.dot(W23_GPU,self.P_GPU ).get() 
		energy += c*dXdP*gpuarray.dot(W32_GPU,self.P_GPU ).get() 
		energy += c*dXdP*gpuarray.dot(W41_GPU,self.P_GPU ).get() 

		energy += c**2*mass*dXdP*gpuarray.sum(W11_GPU).get() 
		energy += c**2*mass*dXdP*gpuarray.sum(W22_GPU).get() 
		energy -= c**2*mass*dXdP*gpuarray.sum(W33_GPU).get()
		energy -= c**2*mass*dXdP*gpuarray.sum(W44_GPU).get()

		energy += self.Potential_0_Average( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t )
		energy += self.Potential_1_Average( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t )
		energy += self.Potential_2_Average( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t )
		energy += self.Potential_3_Average( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ,t )

		return energy

	#...................................................................

	def Copy_gpuarray_row(self, matrix_GPU , n):
		'''
		Return the row n of a gpuarray matrix as a gpu array itself 
		'''
		ncols = matrix_GPU.shape[1]
		floatSize = matrix_GPU.dtype.itemsize
		matrix_row = gpuarray.empty( ncols , matrix_GPU.dtype)
		cuda.memcpy_dtod(matrix_row.ptr ,  matrix_GPU.ptr + floatSize*ncols*n  , floatSize*ncols)
		return matrix_row
	

	def save_FrameItem(self,f,Psi_GPU,t):
		PsiTemp = Psi_GPU.get()
		f.create_dataset(str(t), data = np.frombuffer(PsiTemp))			
	
	def save_Density(self,f11,t_index,W11_GPU,W22_GPU,W33_GPU,W44_GPU):
		print ' progress ', 100*t_index / (self.timeSteps+1), '%'
		
		W0  =  W11_GPU.get() 
		W0 +=  W22_GPU.get() 
		W0 +=  W33_GPU.get() 
		W0 +=  W44_GPU.get() 
		W0 = W0.real.astype(np.float64)

		#print ' normalization = ', np.sum( W0 )*self.dX * self.dP

		f11.create_dataset( str(t_index), data = W0 )


	def save_WignerFunction(self,f11, t,
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
		    			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
		    			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
		    			W41_GPU, W42_GPU, W43_GPU, W44_GPU):

		W_= self.fftshift( W11_GPU.get() );
		f11.create_dataset('W_real_11/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_11/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W12_GPU.get() );
		f11.create_dataset('W_real_12/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_12/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W13_GPU.get() );
		f11.create_dataset('W_real_13/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_13/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W14_GPU.get() );	
		f11.create_dataset('W_real_14/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_14/'+str(t), data = np.imag(W_) )

		#
	
		W_= self.fftshift( W21_GPU.get() );
		f11.create_dataset('W_real_21/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_21/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W21_GPU.get() );
		f11.create_dataset('W_real_22/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_22/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W21_GPU.get() );
		f11.create_dataset('W_real_23/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_23/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W21_GPU.get() );
		f11.create_dataset('W_real_24/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_24/'+str(t), data = np.imag(W_) )

		#
		
		W_= self.fftshift( W31_GPU.get() );
		f11.create_dataset('W_real_31/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_31/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W32_GPU.get() );
		f11.create_dataset('W_real_32/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_32/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W33_GPU.get() );
		f11.create_dataset('W_real_33/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_33/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W34_GPU.get() );
		f11.create_dataset('W_real_34/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_34/'+str(t), data = np.imag(W_) )

		#

		W_= self.fftshift( W41_GPU.get() );
		f11.create_dataset('W_real_41/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_41/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W42_GPU.get() );
		f11.create_dataset('W_real_42/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_42/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W43_GPU.get() );
		f11.create_dataset('W_real_43/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_43/'+str(t), data = np.imag(W_) )

		W_= self.fftshift( W44_GPU.get() );
		f11.create_dataset('W_real_44/'+str(t), data = np.real(W_) )
		f11.create_dataset('W_imag_44/'+str(t), data = np.imag(W_) )


	#--------------------------------------------------------------------

	def Load_Density(self,fileName,n):
		f11 = h5py.File( fileName ,'r')

		rho = f11[str(n)][...]

		f11.close() 

		return rho


	def Load_WignerFunction(self,fileName,n):
		f11 = h5py.File( fileName ,'r')
		#W = FILE['/'+str(n)][...]
		
		W11  =  f11['W_real_11/'+str(n)][...] + 1j*f11['W_imag_11/'+str(n)][...]
		W12  =  f11['W_real_12/'+str(n)][...] + 1j*f11['W_imag_12/'+str(n)][...]
		W13  =  f11['W_real_13/'+str(n)][...] + 1j*f11['W_imag_13/'+str(n)][...]
		W14  =  f11['W_real_14/'+str(n)][...] + 1j*f11['W_imag_14/'+str(n)][...]

		W21  =  f11['W_real_21/'+str(n)][...] + 1j*f11['W_imag_21/'+str(n)][...]
		W22  =  f11['W_real_22/'+str(n)][...] + 1j*f11['W_imag_22/'+str(n)][...]
		W23  =  f11['W_real_23/'+str(n)][...] + 1j*f11['W_imag_23/'+str(n)][...]
		W24  =  f11['W_real_24/'+str(n)][...] + 1j*f11['W_imag_24/'+str(n)][...]

		W31  =  f11['W_real_31/'+str(n)][...] + 1j*f11['W_imag_31/'+str(n)][...]
		W32  =  f11['W_real_32/'+str(n)][...] + 1j*f11['W_imag_32/'+str(n)][...]
		W33  =  f11['W_real_33/'+str(n)][...] + 1j*f11['W_imag_33/'+str(n)][...]
		W34  =  f11['W_real_34/'+str(n)][...] + 1j*f11['W_imag_34/'+str(n)][...]

		W41  =  f11['W_real_41/'+str(n)][...] + 1j*f11['W_imag_41/'+str(n)][...]
		W42  =  f11['W_real_42/'+str(n)][...] + 1j*f11['W_imag_42/'+str(n)][...]
		W43  =  f11['W_real_43/'+str(n)][...] + 1j*f11['W_imag_43/'+str(n)][...]
		W44  =  f11['W_real_44/'+str(n)][...] + 1j*f11['W_imag_44/'+str(n)][...]

		f11.close() 

		return np.array( [[W11,W12,W13,W14],[W21,W22,W23,W24],[W31,W32,W33,W34],[W41,W42,W43,W44]] )  

	#....................................................................
	#           Caldeira Legget Damping
	#....................................................................


	def Theta_fp_Damping(self, LW_GPU, W_GPU):
        	self.gpu_array_copy_Function( LW_GPU , W_GPU , block=self.blockCUDA , grid=self.gridCUDA )

		self.theta_fp_Damping_Function( LW_GPU , block=self.blockCUDA , grid=self.gridCUDA )

		# x p  ->  theta p
        	self.Fourier_P_To_Theta_GPU( LW_GPU )
        	LW_GPU *= self.Theta_GPU	
        	self.Fourier_Theta_To_P_GPU( LW_GPU )


	def CaldeiraDissipatorOrder3(self, LW_GPU, LW_temp_GPU, W_GPU, dampingFunction):
		# dampingFunction is a function of momentum
		LW_GPU  *= 0j
		LW_GPU  +=   W_GPU

		dampingFunction( LW_temp_GPU , W_GPU )
		LW_GPU  += 2./3. * 1j * self.dt *self.gammaDamping * LW_temp_GPU			
		
		dampingFunction( LW_temp_GPU , LW_GPU )
		LW_GPU  +=  2./2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU

		dampingFunction( LW_temp_GPU , LW_GPU )
		W_GPU  +=  2. * 1j * self.dt *self.gammaDamping * LW_temp_GPU

	def CaldeiraDissipatorOrder3_4x4(self, LW1_GPU, LW2_GPU, dampingFunction,
					 W11_GPU, W12_GPU, W13_GPU, W14_GPU,
		    			 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
		    			 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
		    			 W41_GPU, W42_GPU, W43_GPU, W44_GPU):

		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W11_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W12_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W13_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W14_GPU ,dampingFunction)

		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W21_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W22_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W23_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W24_GPU ,dampingFunction)

		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W31_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W32_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W33_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W34_GPU ,dampingFunction)

		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W41_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W42_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W43_GPU ,dampingFunction)
		self.CaldeiraDissipatorOrder3( LW1_GPU, LW2_GPU, W44_GPU ,dampingFunction)



	def Product_ThetaP_GPU(self, W_GPU):
		W_GPU *= self.P_GPU

		self.Fourier_P_To_Theta_GPU(W_GPU)
		W_GPU *= self.Theta_GPU	
		self.Fourier_Theta_To_P_GPU(W_GPU)


	def CaldeiraDissipatorOrder2(self, LW1_GPU, LW2_GPU, W_GPU):
		
		self.gpu_array_copy_Function( LW1_GPU , W_GPU , block=self.blockCUDA , grid=self.gridCUDA )
		self.gpu_array_copy_Function( LW2_GPU , W_GPU , block=self.blockCUDA , grid=self.gridCUDA )

		self.Product_ThetaP_GPU( LW1_GPU )
		LW2_GPU += 1j * self.dt *self.gammaDamping * LW1_GPU			
		
		self.Product_ThetaP_GPU( LW2_GPU )
		W_GPU   +=  2. * 1j * self.dt *self.gammaDamping * LW2_GPU

	def CaldeiraDissipatorOrder2_4x4(self, LW1_GPU, LW2_GPU,
					 W11_GPU, W12_GPU, W13_GPU, W14_GPU,
		    			 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
		    			 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
		    			 W41_GPU, W42_GPU, W43_GPU, W44_GPU):

		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W11_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W12_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W13_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W14_GPU )

		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W21_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W22_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W23_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W24_GPU )

		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W31_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W32_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W33_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W34_GPU )

		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W41_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W42_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W43_GPU )
		self.CaldeiraDissipatorOrder2( LW1_GPU, LW2_GPU, W44_GPU )

		
	#....................... Ehrenfest functions ................................

	def X_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ):
		
		self.X_Average_Function(  temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )			

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 
	
	def P_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ):
		
		self.P_Average_Function(  temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )			

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 

	
	def Alpha_1_Average(self, temp_GPU,
			W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			W41_GPU, W42_GPU, W43_GPU, W44_GPU ):
		
		self.Alpha_1_Average_Function(  temp_GPU, W14_GPU, W23_GPU, W32_GPU, W41_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )			
		
		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 		


	def XP_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ):
		
		self.XP_Average_Function(  temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )			

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 


	def XX_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ):
		
		self.XX_Average_Function( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )			

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 


	def PP_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ):
		
		self.PP_Average_Function( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )			

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 

	def D_1_Potential_0_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t ):

		self.D_1_Potential_0_Average_Function( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t,
					block=self.blockCUDA, grid=self.gridCUDA )

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 

	def D_1_Potential_1_Average(self,
			W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		x  = self.dX*self.dP * gpuarray.dot( W14_GPU, self.D_1_Potential1_GPU ).get() 
		x += self.dX*self.dP * gpuarray.dot( W23_GPU, self.D_1_Potential1_GPU ).get()  
		x += self.dX*self.dP * gpuarray.dot( W32_GPU, self.D_1_Potential1_GPU ).get()  
		x += self.dX*self.dP * gpuarray.dot( W41_GPU, self.D_1_Potential1_GPU ).get() 

		return x

	def D_1_Potential_2_Average(self,
			W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		x  =  1j*self.dX*self.dP * gpuarray.dot( W14_GPU, self.D_1_Potential2_GPU ).get() 
		x += -1j*self.dX*self.dP * gpuarray.dot( W23_GPU, self.D_1_Potential2_GPU ).get()  
		x +=  1j*self.dX*self.dP * gpuarray.dot( W32_GPU, self.D_1_Potential2_GPU ).get()  
		x += -1j*self.dX*self.dP * gpuarray.dot( W41_GPU, self.D_1_Potential2_GPU ).get() 

		return x

	def D_1_Potential_3_Average(self,
			W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			W41_GPU, W42_GPU, W43_GPU, W44_GPU ):

		x  =  self.dX*self.dP * gpuarray.dot( W13_GPU, self.D_1_Potential3_GPU ).get() 
		x += -self.dX*self.dP * gpuarray.dot( W24_GPU, self.D_1_Potential3_GPU ).get()  
		x +=  self.dX*self.dP * gpuarray.dot( W31_GPU, self.D_1_Potential3_GPU ).get()  
		x += -self.dX*self.dP * gpuarray.dot( W42_GPU, self.D_1_Potential3_GPU ).get() 

		return x


	def X1_D_1_Potential_0_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t ):

		self.X1_D_1_Potential_0_Average_Function( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t,
					block=self.blockCUDA, grid=self.gridCUDA )

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 


	def P1_D_1_Potential_0_Average(self, temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t ):

		self.P1_D_1_Potential_0_Average_Function( temp_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU, t,
					block=self.blockCUDA, grid=self.gridCUDA )

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 

	def P1_Alpha_1_Average(self, temp_GPU,
			W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			W41_GPU, W42_GPU, W43_GPU, W44_GPU):

		self.P1_Alpha_1_Average_Function( temp_GPU, W14_GPU, W23_GPU, W32_GPU, W41_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 


	def X1_Alpha_1_Average(self, temp_GPU,
			W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			W41_GPU, W42_GPU, W43_GPU, W44_GPU):

		self.X1_Alpha_1_Average_Function( temp_GPU, W14_GPU, W23_GPU, W32_GPU, W41_GPU,
					block=self.blockCUDA, grid=self.gridCUDA )

		return self.dX*self.dP*( gpuarray.sum( temp_GPU ).get() ) 


	def swap(self,A,B):
		temp = A
		A = B
		B = temp
		return A,B

	def PlotWigner(self, W, color_min, color_max, xlim, ylim):
    
	    W0 = self.Wigner_4X4__SpinTrace( W ).real
	    
	    x_min = -self.X_amplitude
	    x_max = self.X_amplitude - self.dX
	    
	    p_min = -self.P_amplitude
	    p_max = self.P_amplitude - self.dP
	    
	    global_max = color_max        #  Maximum value used to select the color range
	    global_min = color_min        # 

	    #print 'min = ', np.min( W0 ), ' max = ', np.max( W0 )
	    #print 'normalization = ', np.sum( W0 )*self.dX*self.dP

	    zero_position =  abs( global_min) / (abs( global_max) + abs(global_min)) 
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

	    fig, ax = plt.subplots(figsize=(20, 7))
	     
	    cax = ax.imshow( W0 ,origin='lower',interpolation='nearest',\
	    extent=[x_min, x_max, p_min, p_max], vmin= global_min, vmax=global_max, cmap=wigner_cmap)

	    ax.set_xlabel('x')
	    ax.set_ylabel('p')
	    ax.set_xlim( xlim )    
	    ax.set_ylim( ylim )    
	    ax.set_aspect(1)
	    ax.grid('on')
	    return fig	

	############################################################################
	#
	#                              Run 
	#
	############################################################################
	
	def Run(self):

		try :
			import os
			os.remove (self.fileName)
				
		except OSError:
			pass

		print '----------------------------------------------'
		print ' Relativistic Wigner-Dirac Propagator:  x-Px  '
		print '----------------------------------------------'

		print ' dt      = ', self.dt
		print ' dx      = ', self.dX
		print ' dp      = ', self.dP
		print ' dLambda = ', self.dLambda
		print '            '

		# the CUDA X_gridDIM is set to be equal to the discretization number up to 512

		f11 = h5py.File(self.fileName)

		timeRangeIndex = range(1, self.timeSteps+1)

		f11['x_gridDIM'] = self.X_gridDIM
		f11['x_amplitude'] = self.X_amplitude
		f11['p_amplitude'] = self.P_amplitude
		f11['dx'] = self.dX
		
		f11['dtheta'] = self.dTheta
		f11['dp'] = self.dP
		f11['p_gridDIM'] = self.P_gridDIM
		
		f11['mass'] = self.mass
		f11['dt'] = self.dt
		f11['c'] = self.c

		f11['x_range'] = self.X_range
		f11['potential_0_String'] = self.Potential_0_String
		f11['potential_1_String'] = self.Potential_1_String
		f11['potential_2_String'] = self.Potential_2_String
		f11['potential_3_String'] = self.Potential_3_String
		

		self.TakabayashiAngle_GPU = gpuarray.zeros( (self.P_gridDIM,self.X_gridDIM) , dtype = np.float64 )	

		W11_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[0,0],dtype = np.complex128) )
		W12_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[0,1],dtype = np.complex128) )
		W13_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[0,2],dtype = np.complex128) )
		W14_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[0,3],dtype = np.complex128) )

		W21_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[1,0],dtype = np.complex128) )
		W22_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[1,1],dtype = np.complex128) )
		W23_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[1,2],dtype = np.complex128) )
		W24_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[1,3],dtype = np.complex128) )

		W31_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[2,0],dtype = np.complex128) )
		W32_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[2,1],dtype = np.complex128) )
		W33_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[2,2],dtype = np.complex128) )
		W34_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[2,3],dtype = np.complex128) )

		W41_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[3,0],dtype = np.complex128) )
		W42_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[3,1],dtype = np.complex128) )
		W43_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[3,2],dtype = np.complex128) )
		W44_GPU = gpuarray.to_gpu( np.ascontiguousarray(self.W_init[3,3],dtype = np.complex128) )

		
		_W11_GPU = gpuarray.zeros_like(W11_GPU)
		_W12_GPU = gpuarray.zeros_like(W11_GPU)
		_W13_GPU = gpuarray.zeros_like(W11_GPU)
		_W14_GPU = gpuarray.zeros_like(W11_GPU)

		_W21_GPU = gpuarray.zeros_like(W11_GPU)
		_W22_GPU = gpuarray.zeros_like(W11_GPU)
		_W23_GPU = gpuarray.zeros_like(W11_GPU)
		_W24_GPU = gpuarray.zeros_like(W11_GPU)

		_W31_GPU = gpuarray.zeros_like(W11_GPU)
		_W32_GPU = gpuarray.zeros_like(W11_GPU)
		_W33_GPU = gpuarray.zeros_like(W11_GPU)
		_W34_GPU = gpuarray.zeros_like(W11_GPU)

		_W41_GPU = gpuarray.zeros_like(W11_GPU)
		_W42_GPU = gpuarray.zeros_like(W11_GPU)
		_W43_GPU = gpuarray.zeros_like(W11_GPU)
		_W44_GPU = gpuarray.zeros_like(W11_GPU)


		#################################################
		
		print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		timeSavedIndexRange = [0]

		if self.frameSaveMode == 'Wigner_4X4':
				self.save_WignerFunction(   f11, 0,
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
	    				W21_GPU, W22_GPU, W23_GPU, W24_GPU,
	    				W31_GPU, W32_GPU, W33_GPU, W34_GPU,
	    				W41_GPU, W42_GPU, W43_GPU, W44_GPU)

		if self.frameSaveMode=='Density':
						self.save_Density(
							f11,0,W11_GPU,W22_GPU,W33_GPU,W44_GPU)		

		#..............................................................................

		################################################################################################################
		#
		#                        MAIN LOOP
		#
		################################################################################################################

		self.blockCUDA = (512,1,1)
		self.gridCUDA  = (self.P_gridDIM, 1 ,self.X_gridDIM/512 )

		initial_time = time.time()

		B_GP_minus_GPU = gpuarray.empty_like( W11_GPU )
		B_GP_plus_GPU  = gpuarray.empty_like( W22_GPU )
		Prob_X_GPU     = gpuarray.empty( (self.X_gridDIM) , dtype = np.complex128 )

		X_Average  = []
		P_Average  = []
		XP_Average = []
		XX_Average = []
		PP_Average = []

		antiParticle_population    = []
		particle_population        = []
		Dirac_energy               = []
		Alpha_1_Average             = []

		D_1_Potential_0_Average     = []
		X1_D_1_Potential_0_Average  = []
		P1_D_1_Potential_0_Average  = []
		P1_Alpha_1_Average          = []
		X1_Alpha_1_Average          = []

		negativity   = []
		transmission = []
		

		timeRange        = np.array([0.])

		print ' cuda grid =   (', self.blockCUDA , ' , ' , self.gridCUDA, ')'

		P_gridDIM_32   = np.int32(self.P_gridDIM)


		aGPitaevskii_GPU = np.float64( self.grossPitaevskiiCoefficient )
		gammaDamping_GPU = np.float64( self.gammaDamping )

		#-------------------------------------------------------------------------
		self.TakabayashiAngle_Function( self.TakabayashiAngle_GPU,  
						W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			    			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			    			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			    			W41_GPU, W42_GPU, W43_GPU, W44_GPU, block=self.blockCUDA, grid=self.gridCUDA)

		self.TakabayashiAngle_init = self.TakabayashiAngle_GPU.get()
		#-------------------------------------------------------------------------	

		for t_index in timeRangeIndex:
				timeRange = np.append( timeRange ,  self.dt * t_index )

				t_GPU = np.float64( self.dt * t_index )
				
				#...............................................................................

				if self.grossPitaevskiiCoefficient != 0. :
					self.gpu_sum_axis0_Function( 
						 Prob_X_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,
						 P_gridDIM_32, block=(512,1,1), grid=(self.X_gridDIM/512,1)   )

					self.MakeGrossPitaevskiiTerms( B_GP_minus_GPU, B_GP_plus_GPU, Prob_X_GPU ) 
				
				#............... Ehrenfest ....................................................

				X_Average.append(  self.X_Average(  _W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ) )
				P_Average.append(  self.P_Average(  _W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ) )
				XP_Average.append( self.XP_Average( _W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ) )
				XX_Average.append( self.XX_Average( _W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ) )
				PP_Average.append( self.PP_Average( _W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU ) )

				Alpha_1_Average.append(
					self.Alpha_1_Average(_W11_GPU, W11_GPU, W12_GPU, W13_GPU, W14_GPU,
 					    			      W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					     		              W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					     		              W41_GPU, W42_GPU, W43_GPU, W44_GPU  ) )

				D_1_Potential_0_Average.append(
					self.D_1_Potential_0_Average(_W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,t_GPU ) )

				#self.D_1_Potential_0_ExpectationValue(
				#	W11_GPU, W22_GPU, W33_GPU, W44_GPU, t_GPU, _W11_GPU)
				
				X1_D_1_Potential_0_Average.append(
					self.X1_D_1_Potential_0_Average(_W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,t_GPU ) )

				P1_D_1_Potential_0_Average.append(
					self.P1_D_1_Potential_0_Average(_W11_GPU, W11_GPU, W22_GPU, W33_GPU, W44_GPU,t_GPU ) )

				P1_Alpha_1_Average.append(
					self.P1_Alpha_1_Average(_W11_GPU,
								W11_GPU, W12_GPU, W13_GPU, W14_GPU,
 						                W21_GPU, W22_GPU, W23_GPU, W24_GPU,
						     	        W31_GPU, W32_GPU, W33_GPU, W34_GPU,
						     	        W41_GPU, W42_GPU, W43_GPU, W44_GPU ) )

				X1_Alpha_1_Average.append(
					self.X1_Alpha_1_Average(_W11_GPU,
								W11_GPU, W12_GPU, W13_GPU, W14_GPU,
 						                W21_GPU, W22_GPU, W23_GPU, W24_GPU,
						     	        W31_GPU, W32_GPU, W33_GPU, W34_GPU,
						     	        W41_GPU, W42_GPU, W43_GPU, W44_GPU  ) )


				self.pickup_negatives_Function(_W11_GPU,
								W11_GPU, W22_GPU, W33_GPU, W44_GPU,
								block=self.blockCUDA, grid=self.gridCUDA)
				negativity.append(
						self.dX*self.dP*np.real(gpuarray.sum(_W11_GPU).get()))				
								

				self.transmission_Function(_W11_GPU,
								W11_GPU, W22_GPU, W33_GPU, W44_GPU,
								block=self.blockCUDA, grid=self.gridCUDA)
				transmission.append(
						self.dX*self.dP*np.real(gpuarray.sum(_W11_GPU).get()))	

				#................ Energy ................................................

				if self.computeEnergy == True:			

					energy = self.DiracEnergy(_W11_GPU, 
						W11_GPU, W12_GPU, W13_GPU, W14_GPU,
						W21_GPU, W22_GPU, W23_GPU, W24_GPU,
						W31_GPU, W32_GPU, W33_GPU, W34_GPU,
						W41_GPU, W42_GPU, W43_GPU, W44_GPU, t_GPU)

					Dirac_energy.append(  energy  )
				

				# x p -> lambda p .........................................

				self.Fourier_4X4_X_To_Lambda_GPU(
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					W41_GPU, W42_GPU, W43_GPU, W44_GPU )

				#

				#............ Antiparticle production .....................

				if self.antiParticleNorm == True:

					antiParticleNorm = self.AntiParticlePopulation(
					_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
					_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
					_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
					_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
					 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
					 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
					 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
					 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU)

					antiParticle_population.append(   antiParticleNorm  )

					"""particleNorm = self.ParticlePopulation(
					_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
					_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
					_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
					_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
					 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
					 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
					 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
					 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU)

					particle_population.append(   particleNorm  )"""

				#############################################################
				#
				#	 	propagation Lambda p
				#
				#############################################################

				self.DiracPropagator_P_plus_Lambda(
								    W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
								    W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
								    W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
								    W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
								    block=self.blockCUDA, grid=self.gridCUDA )
				
				# 
				 
				self.DiracPropagator_P_minus_Lambda( 
								    W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
								    W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
								    W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
								    W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU,
								    block=self.blockCUDA, grid=self.gridCUDA )

				
				###########################################################################
				#
				#	 	propagation potential
				#
				###########################################################################

				
				# lambda p ->  x p
				self.Fourier_4X4_Lambda_To_X_GPU(
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					W41_GPU, W42_GPU, W43_GPU, W44_GPU )


				#  x p -> x theta
				self.Fourier_4X4_P_To_Theta_GPU(
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					W41_GPU, W42_GPU, W43_GPU, W44_GPU )

				self.DiracPropagator_X_minus_Theta( 
								    W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
								    W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
								    W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
								    W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU, t_GPU,
								    B_GP_minus_GPU, B_GP_plus_GPU, aGPitaevskii_GPU,	
								    block=self.blockCUDA, grid=self.gridCUDA )

				
				self.DiracPropagator_X_plus_Theta( 
								    W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
								    W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
								    W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
								    W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU, t_GPU,
								    B_GP_minus_GPU, B_GP_plus_GPU, aGPitaevskii_GPU,
								    block=self.blockCUDA, grid=self.gridCUDA )

				#...........................................................................
				#if t_index == timeRangeIndex[-1]:
					


		#............................................................................
				
				##########################################################
				#
				#              Damping
				#
				##########################################################

				if self.gammaDamping > 0 :
					if self.dampingModel == 'ODM':
						self.DiracPropagator_DampingODM(
								 W11_GPU, W12_GPU, W13_GPU, W14_GPU,
				    				 W21_GPU, W22_GPU, W23_GPU, W24_GPU,
				    			 	 W31_GPU, W32_GPU, W33_GPU, W34_GPU,
				    			 	 W41_GPU, W42_GPU, W43_GPU, W44_GPU,
								 gammaDamping_GPU,
					                	 block=self.blockCUDA, grid=self.gridCUDA )

				#  x theta  ->  x p
				self.Fourier_4X4_Theta_To_P_GPU(
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					W41_GPU, W42_GPU, W43_GPU, W44_GPU )
				
				if self.gammaDamping > 0 :
					if self.dampingModel == 'CaldeiraLegget':
						
						self.CaldeiraDissipatorOrder2_4x4( 
							_W11_GPU, _W12_GPU,
						 	W11_GPU, W12_GPU, W13_GPU, W14_GPU,
		    				 	W21_GPU, W22_GPU, W23_GPU, W24_GPU,
		    			 	 	W31_GPU, W32_GPU, W33_GPU, W34_GPU,
		    			 	 	W41_GPU, W42_GPU, W43_GPU, W44_GPU )

				#.......... Antiparticle filtering.........................

				if self.antiParticleStepFiltering ==True:

					self.Fourier_4X4_X_To_Lambda_GPU(
						W11_GPU, W12_GPU, W13_GPU, W14_GPU,
						W21_GPU, W22_GPU, W23_GPU, W24_GPU,
						W31_GPU, W32_GPU, W33_GPU, W34_GPU,
						W41_GPU, W42_GPU, W43_GPU, W44_GPU )

					self.FilterAntiParticles(
							_W11_GPU, _W12_GPU, _W13_GPU, _W14_GPU,
							_W21_GPU, _W22_GPU, _W23_GPU, _W24_GPU,
							_W31_GPU, _W32_GPU, _W33_GPU, _W34_GPU,
							_W41_GPU, _W42_GPU, _W43_GPU, _W44_GPU,
							 W11_GPU,  W12_GPU,  W13_GPU,  W14_GPU,
							 W21_GPU,  W22_GPU,  W23_GPU,  W24_GPU,
							 W31_GPU,  W32_GPU,  W33_GPU,  W34_GPU,
							 W41_GPU,  W42_GPU,  W43_GPU,  W44_GPU )

					W11_GPU,_W11_GPU = self.swap(W11_GPU,_W11_GPU)
					W12_GPU,_W12_GPU = self.swap(W12_GPU,_W12_GPU)
					W13_GPU,_W13_GPU = self.swap(W13_GPU,_W13_GPU)				
					W14_GPU,_W14_GPU = self.swap(W14_GPU,_W14_GPU)

					W21_GPU,_W21_GPU = self.swap(W21_GPU,_W21_GPU)
					W22_GPU,_W22_GPU = self.swap(W22_GPU,_W22_GPU)
					W23_GPU,_W23_GPU = self.swap(W23_GPU,_W23_GPU)				
					W24_GPU,_W24_GPU = self.swap(W24_GPU,_W24_GPU)

					W31_GPU,_W31_GPU = self.swap(W31_GPU,_W31_GPU)
					W32_GPU,_W32_GPU = self.swap(W32_GPU,_W32_GPU)
					W33_GPU,_W33_GPU = self.swap(W33_GPU,_W33_GPU)				
					W34_GPU,_W34_GPU = self.swap(W34_GPU,_W34_GPU)

					W41_GPU,_W41_GPU = self.swap(W41_GPU,_W41_GPU)
					W42_GPU,_W42_GPU = self.swap(W42_GPU,_W42_GPU)
					W43_GPU,_W43_GPU = self.swap(W43_GPU,_W43_GPU)				
					W44_GPU,_W44_GPU = self.swap(W44_GPU,_W44_GPU)

					self.Fourier_4X4_Lambda_To_X_GPU(
						W11_GPU, W12_GPU, W13_GPU, W14_GPU,
						W21_GPU, W22_GPU, W23_GPU, W24_GPU,
						W31_GPU, W32_GPU, W33_GPU, W34_GPU,
						W41_GPU, W42_GPU, W43_GPU, W44_GPU )


				w_absorb = np.float64(24.)
				self.AbsorbBoundary_x_Function( w_absorb, 
				W11_GPU, W12_GPU, W13_GPU, W14_GPU,
				W21_GPU, W22_GPU, W23_GPU, W24_GPU,
				W31_GPU, W32_GPU, W33_GPU, W34_GPU,
				W41_GPU, W42_GPU, W43_GPU, W44_GPU
				,block=self.blockCUDA, grid=self.gridCUDA )

				#............................................................

				self.Wigner_4X4_Normalize(
					W11_GPU, W12_GPU, W13_GPU, W14_GPU,
					W21_GPU, W22_GPU, W23_GPU, W24_GPU,
					W31_GPU, W32_GPU, W33_GPU, W34_GPU,
					W41_GPU, W42_GPU, W43_GPU, W44_GPU )

				#................ Saving .....................................
					
				if t_index % self.skipFrames == 0:

					timeSavedIndexRange.append(t_index)					
					#print ' Norm = ', self.Wigner_4x4_Norm_GPU(W11_GPU, W22_GPU, W33_GPU, W44_GPU)

					if self.frameSaveMode == 'Wigner_4X4':
						self.save_WignerFunction(   f11, t_index,
						W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			    			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			    			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			    			W41_GPU, W42_GPU, W43_GPU, W44_GPU)

					if self.frameSaveMode=='Density':
						self.save_Density(
							f11,t_index,W11_GPU,W22_GPU,W33_GPU,W44_GPU)
	
				#.............................................................
		
		#self.Fourier_Theta_To_P_GPU( self.TakabayashiAngle_GPU  )
		self.TakabayashiAngle_Function( self.TakabayashiAngle_GPU,  
						W11_GPU, W12_GPU, W13_GPU, W14_GPU,
			    			W21_GPU, W22_GPU, W23_GPU, W24_GPU,
			    			W31_GPU, W32_GPU, W33_GPU, W34_GPU,
			    			W41_GPU, W42_GPU, W43_GPU, W44_GPU, block=self.blockCUDA, grid=self.gridCUDA)

		self.TakabayashiAngle_end = self.TakabayashiAngle_GPU.get()
				

		final_time = time.time()
		print ' computation time = ', final_time - initial_time, ' seconds'



		self.timeRange   = timeRange
		f11['timeRange'] = timeRange		

		self.antiParticle_population   = np.array(antiParticle_population)
		f11['antiParticle_population'] = self.antiParticle_population

		#self.particle_population   = np.array(particle_population)
		#f11['particle_population'] = self.particle_population
		
		self.Dirac_energy = np.array(Dirac_energy).real
		f11['Dirac_energy'] = np.array(Dirac_energy)
					
		self.timeSavedIndexRange = np.array(timeSavedIndexRange)
		f11['timeSavedIndexRange'] = np.array(self.timeSavedIndexRange)
	
		self.X_Average = np.array( X_Average ).real
		f11['X_Average'] = self.X_Average

		self.P_Average = np.array( P_Average ).real
		f11['P_Average'] = self.P_Average

		self.XP_Average = np.array( XP_Average ).real
		f11['XP_Average'] = self.XP_Average

		self.XX_Average = np.array( XX_Average ).real
		f11['XX_Average'] = self.XX_Average

		self.PP_Average = np.array( PP_Average ).real
		f11['PP_Average'] = self.PP_Average

		self.Alpha_1_Average = np.array( Alpha_1_Average ).real
		f11['Alpha_1_Average'] = self.Alpha_1_Average


		self.P1_Alpha_1_Average = np.array( P1_Alpha_1_Average ).real
		f11['P1_Alpha_1_Average'] = self.P1_Alpha_1_Average
		
		self.X1_Alpha_1_Average = np.array( X1_Alpha_1_Average ).real
		f11['X1_Alpha_1_Average'] = self.X1_Alpha_1_Average

		self.D_1_Potential_0_Average   = np.array( D_1_Potential_0_Average ).real
		f11['D_1_Potential_0_Average'] = self.D_1_Potential_0_Average

		self.X1_D_1_Potential_0_Average   = np.array( X1_D_1_Potential_0_Average ).real
		f11['X1_D_1_Potential_0_Average'] = self.X1_D_1_Potential_0_Average

		self.P1_D_1_Potential_0_Average   = np.array( P1_D_1_Potential_0_Average ).real
		f11['P1_D_1_Potential_0_Average'] = self.P1_D_1_Potential_0_Average

		self.negativity = np.array(negativity).real
		f11['Negativity'] = self.negativity

		self.transmission = np.array(transmission).real
		f11['transmission'] = self.transmission

		f11['TakabayashiAngle_end']  = self.TakabayashiAngle_GPU.get()
		f11['TakabayashiAngle_init'] = self.TakabayashiAngle_init
		#.............................................................................

		f11.close()

		W11 = W11_GPU.get()		
		W12 = W12_GPU.get()
		W13 = W13_GPU.get()
		W14 = W14_GPU.get()

		W21 = W21_GPU.get()		
		W22 = W22_GPU.get()
		W23 = W23_GPU.get()
		W24 = W24_GPU.get()

		W31 = W31_GPU.get()		
		W32 = W32_GPU.get()
		W33 = W33_GPU.get()
		W34 = W34_GPU.get()

		W41 = W41_GPU.get()		
		W42 = W42_GPU.get()
		W43 = W43_GPU.get()
		W44 = W44_GPU.get()

		self.W_end = np.array([ [W11,W12,W13,W14], [W21,W22,W23,W24], [W31,W32,W33,W34], [W41,W42,W43,W44] ])


		self.plan_Z2Z_1D_Axes0.__del__()
		self.plan_Z2Z_1D_Axes1.__del__()
		self.plan_Z2Z_1D.__del__()		
		



	
	
