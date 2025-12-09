// Description:
// This program is a CUDA implementation of the temporal second-order and 
// spatial fourth-order conservative compact characteristic finite volume 
// (T2S4-CC-CFV) method for two-dimensional advection-diffusion equation,
// with a Parareal time-parallel algorithm on top.
//
// Usage (Linux):
//   nvcc -O3 -std=c++14 T2S4_Parareal.cu -o T2S4_Parareal.out
//   ./T2S4_Parareal.out config.txt
//
// Version: 2.0 (Parareal)
// 2025.04.28
// Based on original code by Guiyu Wang, Kai Fu, Linjie Zhang
// School of Mathematical Sciences, Oceanic University of China

#include <iterator>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <device_atomic_functions.h>

#define x0 -0.35
#define y0  0.0
#define sigma 0.005
#define X_min -1.0
#define X_max  1.0
#define Y_min -1.0
#define Y_max  1.0

using namespace std;

/*********** Definition of variable types **********/
typedef int INT;
typedef double FLOAT;

/***************************** functions *****************************/

// Velocity functions
__host__ __device__ FLOAT Vx(FLOAT x, FLOAT y) { return -4 * y; }
__host__ __device__ FLOAT Vy(FLOAT x, FLOAT y) { return  4 * x; }

__host__ __device__ FLOAT C0(FLOAT x, FLOAT y) {
  return exp(-(pow((x - x0), 2) + pow((y - y0), 2)) / sigma);
}

__host__ __device__ FLOAT C_Exact(FLOAT x, FLOAT y, FLOAT t, FLOAT K) {
    FLOAT x_star =  x * cos(4 * t) + y * sin(4 * t);
    FLOAT y_star = -x * sin(4 * t) + y * cos(4 * t);
    return sigma / (sigma + 4 * K * t) *
           exp(-(pow((x_star - x0), 2) + pow((y_star - y0), 2)) / (sigma + 4 * K * t));
}

// Source term (here f ≡ 0)
__host__ __device__ inline FLOAT f(FLOAT x, FLOAT y, FLOAT t, FLOAT K) {
    return 0.0; 
}

__host__ __device__ inline FLOAT d_intC_ppm4th(FLOAT vC6, FLOAT vDltc, FLOAT vCL, FLOAT xi) {
  return -1.0 / 3 * vC6 * pow(xi, 3) + 0.5 * (vDltc + vC6) * pow(xi, 2) + vCL * xi;
}

// simple sum on host array
__host__ __device__ FLOAT sum(FLOAT *v_d_C, INT N_grid) {
  FLOAT s = 0.0;
  for (int idx = 0; idx < N_grid; idx++) {
    s += v_d_C[idx];
  }
  return s;
}

__host__ __device__ FLOAT d_x_sum(FLOAT *v_C, INT iL, INT iR, INT j, INT Ny) {
  FLOAT s = 0.0;
  for (int i_cell = iL; i_cell <= iR; i_cell++) {
    s += v_C[i_cell * Ny + j];
  }
  return s;
}

__host__ __device__ FLOAT d_y_sum(FLOAT *v_C, INT jL, INT jR, INT i, INT Ny) {
  FLOAT s = 0.0;
  for (int j_cell = jL; j_cell <= jR; j_cell++) {
    s += v_C[i * Ny + j_cell];
  }
  return s;
}

// Output the result to a file
void output_result(FLOAT *vec, INT t, INT Nx, INT Ny, FLOAT dt) {
  FILE *fp;
  char sfile[256];
  int i_cell, j_cell;
  FLOAT x, y;
  FLOAT dx, dy;

  dx = (X_max - X_min) / Nx;
  dy = (Y_max - Y_min) / Ny;

  sprintf(sfile, "data_T2S4_%06d.txt", t);
  fp = fopen(sfile, "w");
  for (i_cell = 0; i_cell < Nx; i_cell++) {
    x = X_min + dx * (i_cell + 0.5);
    for (j_cell = 0; j_cell < Ny; j_cell++) {
      y = Y_min + dy * (j_cell + 0.5);
      fprintf(fp, "%.6lf %.6lf %.4e\n", x, y, vec[i_cell * Ny + j_cell]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

/***************************** kernel functions *****************************/
// calculate Exact solution (cell average)
__global__ void get_exact_solution_on_gpu(FLOAT *v_d_x, FLOAT *v_d_y, 
                                          FLOAT *v_C_Exact, FLOAT dx, FLOAT dy, FLOAT t, FLOAT K, INT N_grid) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x_i, y_j, x_imhf, x_iphf, y_jmhf, y_jphf;
  while (tid < N_grid) {
    x_i = v_d_x[tid];
    y_j = v_d_y[tid];
    x_imhf = x_i - 0.5 * dx;
    x_iphf = x_i + 0.5 * dx;
    y_jmhf = y_j - 0.5 * dy;
    y_jphf = y_j + 0.5 * dy;

    v_C_Exact[tid] = 1.0/ 36.0 * 
                    (C_Exact(x_imhf, y_jmhf, t, K) + C_Exact(x_iphf, y_jmhf, t, K) + 
                     C_Exact(x_imhf, y_jphf, t, K) + C_Exact(x_iphf, y_jphf, t, K) + 
                     4.0 * (C_Exact(x_imhf, y_j, t, K) + C_Exact(x_iphf, y_j, t, K) + 
                            C_Exact(x_i, y_jmhf, t, K) + C_Exact(x_i, y_jphf, t, K)) + 
                     16.0 * C_Exact(x_i, y_j, t, K));

    tid += gridDim.x * blockDim.x;
  }
}

// calculate source term (cell average)
__global__ void get_source_term_on_gpu(FLOAT *v_d_x, FLOAT *v_d_y, 
                                      FLOAT *v_d_f, FLOAT dx, FLOAT dy, FLOAT t, FLOAT K, INT N_grid) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x_i, y_j, x_imhf, x_iphf, y_jmhf, y_jphf;
  while (tid < N_grid) {
    x_i = v_d_x[tid];
    y_j = v_d_y[tid];
    x_imhf = x_i - 0.5 * dx;
    x_iphf = x_i + 0.5 * dx;
    y_jmhf = y_j - 0.5 * dy;
    y_jphf = y_j + 0.5 * dy;

    v_d_f[tid] = 1.0 / 36.0 * 
                (f(x_imhf, y_jmhf, t, K) + f(x_iphf, y_jmhf, t, K) + 
                  f(x_imhf, y_jphf, t, K) + f(x_iphf, y_jphf, t, K) + 
                  4.0 * (f(x_imhf, y_j, t, K) + f(x_iphf, y_j, t, K) + 
                         f(x_i, y_jmhf, t, K) + f(x_i, y_jphf, t, K)) + 
                  16.0 * f(x_i, y_j, t, K));

    tid += gridDim.x * blockDim.x;
  }
}

// calculate Eulerian points of cell center
__global__ void get_Euler_cell_center_points_on_gpu(FLOAT *v_d_x, FLOAT *v_d_y, 
                                        INT Nx, INT Ny, INT N_grid, 
                                        FLOAT dx, FLOAT dy) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  INT i_cell, j_cell;

  while (tid < N_grid) {
    i_cell = tid / Ny; // 0,...,Nx-1
    j_cell = tid % Ny; // 0,...,Ny-1
    
    v_d_x[tid] = X_min + (i_cell + 0.5) * dx;
    v_d_y[tid] = Y_min + (j_cell + 0.5) * dy;

    tid += gridDim.x * blockDim.x;
  }
}

// calculate Eulerian points of x(i+1/2), y(j+1/2)
__global__ void get_Euler_half_points_on_gpu(FLOAT *v_d_xhf, FLOAT *v_d_yhf, 
                                          INT Nx, INT Ny, INT N_points, 
                                          FLOAT dx, FLOAT dy) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  INT i_point, j_point;

  while (tid < N_points) {
    i_point = tid / (Ny+1); // 0,...,Nx
    j_point = tid % (Ny+1); // 0,...,Ny
    v_d_xhf[tid] = X_min + dx * i_point;
    v_d_yhf[tid] = Y_min + dy * j_point;
    tid += gridDim.x * blockDim.x;
  }
}

__global__ void get_Lagrange_points_on_gpu(FLOAT *v_d_xhf, FLOAT *v_d_yhf, 
                                          FLOAT *v_d_xhf_bar1, 
                                          FLOAT *v_d_yhf_bar2, 
                                          FLOAT *v_d_xhf_bar3, 
                                          INT Nx, INT Ny, INT N_points, 
                                          FLOAT dt) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x_star, y_star;

  while (tid < N_points) {
    // step 1
    x_star = v_d_xhf[tid] - 0.5 * dt * Vx(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_xhf_bar1[tid] = v_d_xhf[tid] 
                        - 0.25 * dt * (Vx(v_d_xhf[tid], v_d_yhf[tid]) + Vx(x_star, v_d_yhf[tid]));
    // step 2
    y_star = v_d_yhf[tid] - dt * Vy(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_yhf_bar2[tid] = v_d_yhf[tid] 
                        -  0.5 * dt * (Vy(v_d_xhf[tid], v_d_yhf[tid]) + Vy(v_d_xhf[tid], y_star));    
    // step 3
    x_star = v_d_xhf[tid] - 0.5 * dt * Vx(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_xhf_bar3[tid] = v_d_xhf[tid] 
                        - 0.25 * dt * (Vx(v_d_xhf[tid], v_d_yhf[tid]) + Vx(x_star, v_d_yhf[tid]));

    tid += gridDim.x * blockDim.x;
  }
}

__global__ void intersection_point_vert_x(FLOAT *v_d_xhf_bar1, 
                                          FLOAT *v_d_x_inter1,
                                          INT Ny, INT N_points_inter1) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N_points_inter1) {
    INT j_cell  = tid % Ny;   // 0,...,Ny-1
    INT i_point = tid / Ny;   // 0,...,Nx
    INT ind = i_point * (Ny + 1) + j_cell;
    v_d_x_inter1[tid] = 0.5 * (v_d_xhf_bar1[ind] + v_d_xhf_bar1[ind + 1]);
    tid += gridDim.x * blockDim.x;
  }
}

__global__ void intersection_point_hori_y(FLOAT *v_d_yhf_bar2,
                                          FLOAT *v_d_y_inter2,
                                          INT Ny, INT N_points_inter2) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < N_points_inter2) {
    INT i_cell  = tid / (Ny + 1);  // 0,...,Nx-1
    INT j_point = tid % (Ny + 1);  // 0,...,Ny
    INT ind      = i_cell * (Ny + 1) + j_point;
    INT ind_next = (i_cell + 1) * (Ny + 1) + j_point;
    v_d_y_inter2[tid] = 0.5 * (v_d_yhf_bar2[ind] + v_d_yhf_bar2[ind_next]);
    tid += gridDim.x * blockDim.x;
  }
}

__global__ void x_step_ppm_coef_on_gpu(FLOAT *v_d_CL, FLOAT *v_d_CR, 
                                     FLOAT *v_d_C6, FLOAT *v_d_DltC, 
                                     FLOAT *v_d_C, INT N_grid, INT Nx, INT Ny) 
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_eleventh    = 1.0 / 11.0;
  FLOAT one_twelfth     = 1.0 / 12.0;
  FLOAT seven_twelfths  = 7.0 / 12.0;
  FLOAT CN, CNp1, Cm1, Cm2, Ci, Cip1, Cip2, Cim1, Cim2;
  
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    CN  = one_eleventh * (-v_d_C[(Nx - 3) * Ny + j_cell] 
                          + 3 * v_d_C[(Nx - 2) * Ny + j_cell] 
                          + 9 * v_d_C[(Nx - 1) * Ny + j_cell]);
    CNp1 = one_eleventh * (-15 * v_d_C[(Nx - 3) * Ny + j_cell] 
                           + 56 * v_d_C[(Nx - 2) * Ny + j_cell] 
                           - 30 * v_d_C[(Nx - 1) * Ny + j_cell]);
    Cm1 = one_eleventh * (- v_d_C[2 * Ny + j_cell] 
                          + 3 * v_d_C[Ny + j_cell] 
                          + 9 * v_d_C[j_cell]);
    Cm2 = one_eleventh * (-15 * v_d_C[2 * Ny + j_cell] 
                          + 56 * v_d_C[Ny + j_cell] 
                          - 30 * v_d_C[j_cell]);

    Ci   = v_d_C[i_cell * Ny + j_cell];
    Cip1 = (i_cell == Nx-1) ? CN  : v_d_C[(i_cell + 1) * Ny + j_cell];
    Cip2 = (i_cell >= Nx-2) ? ((i_cell == Nx-2) ? CN : CNp1)
                            : v_d_C[(i_cell + 2) * Ny + j_cell];
    Cim1 = (i_cell == 0) ? Cm1 : v_d_C[(i_cell - 1) * Ny + j_cell];
    Cim2 = (i_cell <= 1) ? ((i_cell == 1) ? Cm1 : Cm2)
                         : v_d_C[(i_cell - 2) * Ny + j_cell];
    
    v_d_CL[ind] = seven_twelfths * (Ci + Cim1) - one_twelfth * (Cip1 + Cim2);
    v_d_CR[ind] = seven_twelfths * (Cip1 + Ci) - one_twelfth * (Cip2 + Cim1);

    v_d_DltC[ind] = v_d_CR[ind] - v_d_CL[ind];
    v_d_C6[ind]   = 6.0 * (v_d_C[ind] - 0.5 * (v_d_CL[ind] + v_d_CR[ind]));
       
    ind += gridDim.x * blockDim.x;
  }
}

__global__ void x_step_intC_ppm_on_gpu(FLOAT *v_d_CL, FLOAT *v_d_CR, 
                                     FLOAT *v_d_C6, FLOAT *v_d_DltC, 
                                     FLOAT *v_d_C, FLOAT *v_d_int_C_bar,
                                     FLOAT *v_d_xhf, FLOAT *v_d_x_inter1, 
                                     INT N_grid, INT Nx, INT Ny, FLOAT dx) 
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0;

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;
    FLOAT xbL = v_d_x_inter1[i_cell * Ny + j_cell];
    FLOAT xbR = v_d_x_inter1[(i_cell + 1) * Ny + j_cell];

    if (!(xbL > X_max || xbR < X_min || xbL == xbR)) {
      FLOAT intxbL = fmax(xbL, X_min + 1e-15);
      FLOAT intxbR = fmin(xbR, X_max - 1e-15);
      INT iL = (INT)floor((intxbL - X_min) / dx);
      INT iR = (INT)floor((intxbR - X_min) / dx);
      FLOAT xL_xi = (intxbL - v_d_xhf[iL * (Ny + 1) + j_cell]) / dx;
      FLOAT xR_xi = (intxbR - v_d_xhf[iR * (Ny + 1) + j_cell]) / dx;
      
      INT id_R = iR * Ny + j_cell;
      INT id_L = iL * Ny + j_cell;

      if (iR == iL) {
        IntC = d_intC_ppm4th(v_d_C6[id_R], v_d_DltC[id_R], v_d_CL[id_R], xR_xi) -
               d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], xL_xi);
      } else {    
        IntCL = d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], 1.0) -
                d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], xL_xi);
        IntCR = d_intC_ppm4th(v_d_C6[id_R], v_d_DltC[id_R], v_d_CL[id_R], xR_xi);     
        IntCM = d_x_sum(v_d_C, iL + 1, iR - 1, j_cell, Ny);         
        IntC = IntCL + IntCM + IntCR;       
      }
    }
    v_d_int_C_bar[ind] = IntC;
    ind += gridDim.x * blockDim.x;
  }
}

__global__ void y_step_ppm_coef_on_gpu(FLOAT *v_d_CL, FLOAT *v_d_CR, 
                                     FLOAT *v_d_C6, FLOAT *v_d_DltC, 
                                     FLOAT *v_d_C, INT N_grid, INT Nx, INT Ny) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_eleventh    = 1.0 / 11.0;
  FLOAT one_twelfth     = 1.0 / 12.0;
  FLOAT seven_twelfths  = 7.0 / 12.0;
  FLOAT CN, CNp1, Cm1, Cm2, Cj, Cjp1, Cjp2, Cjm1, Cjm2;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    
    CN  = one_eleventh * (-v_d_C[i_cell * Ny + Ny-3] 
                          + 3 * v_d_C[i_cell * Ny + Ny-2]
                          + 9 * v_d_C[i_cell * Ny + Ny-1]);
    CNp1 = one_eleventh * (-15 * v_d_C[i_cell * Ny + Ny-3]
                           + 56 * v_d_C[i_cell * Ny + Ny-2]
                           - 30 * v_d_C[i_cell * Ny + Ny-1]);
    Cm1 = one_eleventh * (-v_d_C[i_cell * Ny + 2]
                          + 3 * v_d_C[i_cell * Ny + 1]
                          + 9 * v_d_C[i_cell * Ny + 0]);
    Cm2 = one_eleventh * (-15 * v_d_C[i_cell * Ny + 2]
                          + 56 * v_d_C[i_cell * Ny + 1]
                          - 30 * v_d_C[i_cell * Ny + 0]);

    Cj   = v_d_C[i_cell * Ny + j_cell];
    Cjp1 = (j_cell > Ny - 2) ? CN 
                             : v_d_C[i_cell * Ny + j_cell + 1];
    Cjp2 = (j_cell > Ny - 3) ? ((j_cell == Ny - 2) ? CN : CNp1)
                             : v_d_C[i_cell * Ny + j_cell + 2];
    Cjm1 = (j_cell < 1) ? Cm1 : v_d_C[i_cell * Ny + j_cell - 1];
    Cjm2 = (j_cell < 2) ? ((j_cell == 1) ? Cm1 : Cm2)
                        : v_d_C[i_cell * Ny + j_cell - 2];

    v_d_CL[ind] = seven_twelfths * (Cj + Cjm1) - one_twelfth * (Cjp1 + Cjm2);
    v_d_CR[ind] = seven_twelfths * (Cjp1 + Cj) - one_twelfth * (Cjp2 + Cjm1);

    v_d_DltC[ind] = v_d_CR[ind] - v_d_CL[ind];
    v_d_C6[ind]   = 6.0 * (v_d_C[ind] - 0.5 * (v_d_CL[ind] + v_d_CR[ind]));

    ind += gridDim.x * blockDim.x;
  } 
}
   
__global__ void y_step_intC_ppm_on_gpu(FLOAT *v_d_CL, FLOAT *v_d_CR, 
                                     FLOAT *v_d_C6, FLOAT *v_d_DltC, 
                                     FLOAT *v_d_C, FLOAT *v_d_int_C_bar,
                                     FLOAT *v_d_yhf, FLOAT *v_d_y_inter2, 
                                     INT N_grid, INT Nx, INT Ny, FLOAT dy)
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0;

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;

    FLOAT ybL = v_d_y_inter2[i_cell * (Ny+1) + j_cell];
    FLOAT ybR = v_d_y_inter2[i_cell * (Ny+1) + j_cell+1];

    if (!(ybL > Y_max || ybR < Y_min || ybL == ybR)) {
      FLOAT intybL = fmax(ybL, Y_min + 1e-16);
      FLOAT intybR = fmin(ybR, Y_max - 1e-16);
      INT jL = (INT)floor((intybL - Y_min) / dy);
      INT jR = (INT)floor((intybR - Y_min) / dy);
      FLOAT yL_xi = (intybL - v_d_yhf[i_cell * (Ny + 1) + jL]) / dy;
      FLOAT yR_xi = (intybR - v_d_yhf[i_cell * (Ny + 1) + jR]) / dy;

      INT id_R = i_cell * Ny + jR;
      INT id_L = i_cell * Ny + jL;

      if (jL == jR) {
        IntC = d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], yR_xi) -
               d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], yL_xi);
      } else {
        IntCL = d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], 1.0) -
                d_intC_ppm4th(v_d_C6[id_L], v_d_DltC[id_L], v_d_CL[id_L], yL_xi);
        IntCR = d_intC_ppm4th(v_d_C6[id_R], v_d_DltC[id_R], v_d_CL[id_R], yR_xi);
        IntCM = d_y_sum(v_d_C, jL + 1, jR - 1, i_cell, Ny);
        IntC = IntCL + IntCM + IntCR;
      }  
    }  
    v_d_int_C_bar[ind] = IntC; 
    ind += gridDim.x * blockDim.x; 
  }
}

__global__ void dcdx_bar_on_gpu(FLOAT *v_d_C,
                              FLOAT *v_d_dCdx_bar, FLOAT *v_d_xhf,
                              FLOAT *v_d_x_inter1,  
                              INT N_points_inter1, 
                              INT Nx, INT Ny, FLOAT dx) {
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_point, j_cell;
  FLOAT one_eleventh = 1.0 / 11.0;
  FLOAT x_bar, xL_xi, xL_xi2, xL_xi3;
  FLOAT CN, CNp1, Cm1, Cm2, Ci, Cip1, Cip2, Cim1, Cim2;

  while (ind < N_points_inter1) {
    i_point = ind / Ny;     // 0..Nx
    j_cell  = ind % Ny;     // 0..Ny-1

    x_bar = v_d_x_inter1[i_point * Ny + j_cell];
    
    if (x_bar >= X_min && x_bar <= X_max)
    {
      FLOAT int_xb = fmin(fmax(x_bar, X_min + 1e-15), X_max - 1e-15);
      INT iL = (INT)floor((int_xb - X_min) / dx); // 0..Nx-1
      xL_xi = (int_xb - v_d_xhf[iL * (Ny + 1) + j_cell]) / dx;

      xL_xi2 = xL_xi * xL_xi;
      xL_xi3 = xL_xi2 * xL_xi;

      CN  = one_eleventh * (-v_d_C[(Nx - 3) * Ny + j_cell] + 3 * v_d_C[(Nx - 2) * Ny + j_cell] 
                          + 9 * v_d_C[(Nx - 1) * Ny + j_cell]);
      CNp1 = one_eleventh * (-15 * v_d_C[(Nx - 3) * Ny + j_cell] + 56 * v_d_C[(Nx - 2) * Ny + j_cell] 
                            - 30 * v_d_C[(Nx - 1) * Ny + j_cell]);
      Cm1 = one_eleventh * (-v_d_C[2 * Ny + j_cell] + 3 * v_d_C[Ny + j_cell] + 9 * v_d_C[j_cell]);
      Cm2 = one_eleventh * (-15 * v_d_C[2 * Ny + j_cell] + 56 * v_d_C[Ny + j_cell] - 30 * v_d_C[j_cell]);

      Ci   = v_d_C[iL * Ny + j_cell];
      Cip1 = (iL > Nx - 2) ? CN 
                           : v_d_C[(iL + 1) * Ny + j_cell];
      Cip2 = (iL > Nx - 3) ? CNp1 
                           : ((iL == Nx - 2) ? CN : v_d_C[(iL + 2) * Ny + j_cell]);
      Cim1 = (iL < 1) ? Cm1 : v_d_C[(iL - 1) * Ny + j_cell];
      Cim2 = (iL < 2) ? Cm2 : ((iL == 1) ? Cm1 : v_d_C[(iL - 2) * Ny + j_cell]);

      v_d_dCdx_bar[ind] = 1.0 / (12.0 * dx) *
                          ((2.0 * xL_xi3 - 6.0 * xL_xi2 + 3.0 * xL_xi + 1.0) * Cim2 +
                          (-8.0 * xL_xi3 + 18.0 * xL_xi2 + 6.0 * xL_xi - 15.0) * Cim1 +
                          (12.0 * xL_xi3 - 18.0 * xL_xi2 - 24.0 * xL_xi + 15.0) * Ci +
                          (-8.0 * xL_xi3 + 6.0 * xL_xi2 + 18.0 * xL_xi - 1.0) * Cip1 +
                          (2.0 * xL_xi3 - 3.0 * xL_xi) * Cip2);
    } 
    else {
      v_d_dCdx_bar[ind] = 0.0;
    }

    ind += gridDim.x * blockDim.x;
  }
}

__global__ void dcdy_bar_on_gpu(FLOAT *v_d_C, 
                              FLOAT *v_d_dCdy_bar, FLOAT *v_d_yhf, 
                              FLOAT *v_d_y_inter2,  
                              INT N_points_inter2, 
                              INT Nx, INT Ny, FLOAT dy) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_eleventh = 1.0 / 11.0;
  FLOAT yL_xi, yL_xi2, yL_xi3;
  FLOAT CN, CNp1, Cm1, Cm2, Ci, Cip1, Cip2, Cim1, Cim2;
  
  while (ind < N_points_inter2) {
    i_cell = ind / (Ny + 1);
    j_cell = ind % (Ny + 1);

    FLOAT y_bar = v_d_y_inter2[i_cell * (Ny + 1) + j_cell];

    if (y_bar >= Y_min && y_bar <= Y_max) {
      FLOAT int_yb = fmin(fmax(y_bar, Y_min + 1e-15), Y_max - 1e-15);
      INT jL = (INT)floor((int_yb - Y_min) / dy);
      yL_xi = (int_yb - v_d_yhf[i_cell * (Ny + 1) + jL]) / dy;

      yL_xi2 = yL_xi * yL_xi;
      yL_xi3 = yL_xi2 * yL_xi;

      CN  = one_eleventh * (-v_d_C[i_cell * Ny + Ny-3] + 3 * v_d_C[i_cell * Ny + Ny-2] + 9 * v_d_C[i_cell * Ny + Ny-1]);
      CNp1 = one_eleventh * (-15 * v_d_C[i_cell * Ny + Ny-3] + 56 * v_d_C[i_cell * Ny + Ny-2] - 30 * v_d_C[i_cell * Ny + Ny-1]);
      Cm1 = one_eleventh * (-v_d_C[i_cell * Ny + 2] + 3 * v_d_C[i_cell * Ny + 1] + 9 * v_d_C[i_cell * Ny + 0]);
      Cm2 = one_eleventh * (-15 * v_d_C[i_cell * Ny + 2] + 56 * v_d_C[i_cell * Ny + 1] - 30 * v_d_C[i_cell * Ny + 0]);

      Ci   = v_d_C[i_cell * Ny + jL];
      Cip1 = (jL > Ny - 2) ? CN 
                           : v_d_C[i_cell * Ny + jL + 1];
      Cip2 = (jL > Ny - 3) ? CNp1 
                           : ((jL == Ny - 2) ? CN : v_d_C[i_cell * Ny + jL + 2]);
      Cim1 = (jL < 1) ? Cm1 : v_d_C[i_cell * Ny + jL - 1];
      Cim2 = (jL < 2) ? Cm2 : ((jL == 1) ? Cm1 : v_d_C[i_cell * Ny + jL - 2]);

      v_d_dCdy_bar[ind] = 1.0 / (12.0 * dy) *
                          ((2.0 * yL_xi3 - 6.0 * yL_xi2 + 3.0 * yL_xi + 1.0) * Cim2 +
                          (-8.0 * yL_xi3 + 18.0 * yL_xi2 + 6.0 * yL_xi - 15.0) * Cim1 +
                          (12.0 * yL_xi3 - 18.0 * yL_xi2 - 24.0 * yL_xi + 15.0) * Ci +
                          (-8.0 * yL_xi3 + 6.0 * yL_xi2 + 18.0 * yL_xi - 1.0) * Cip1 +
                          (2.0 * yL_xi3 - 3.0 * yL_xi) * Cip2);
    } else {
      v_d_dCdy_bar[ind] = 0.0;
    }

    ind += gridDim.x * blockDim.x;
  }
}

// Get LHS matrix for x-direction computation
__global__ void x_A_LHS_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC,
                               INT N_grid, INT Ny, FLOAT K, 
                               FLOAT dt_mul_dx_square_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  while (ind < N_grid) {
    v_d_coA[ind] =  1.0/12 - 0.25 * K * dt_mul_dx_square_inverse;
    v_d_coB[ind] = 10.0/12 +  0.5 * K * dt_mul_dx_square_inverse;
    v_d_coC[ind] =  1.0/12 - 0.25 * K * dt_mul_dx_square_inverse;
    ind += gridDim.x * blockDim.x;
  }
}

// Get b vector for x-direction computation
__global__ void x_b_RHS_on_gpu( FLOAT *v_d_b, 
                                FLOAT *v_d_int_C_bar, FLOAT *v_d_dCdx_bar, FLOAT *v_d_f_p0, 
                                FLOAT *v_d_f_p1, INT N_grid, INT Ny, FLOAT K, FLOAT dt, 
                                FLOAT dt_mul_dx_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    v_d_b[ind] = v_d_int_C_bar[ind] + 0.25 * dt * (v_d_f_p0[ind] + v_d_f_p1[ind])
                + 0.25 * K * dt_mul_dx_inverse * 
                (v_d_dCdx_bar[(i_cell + 1) * Ny + j_cell] - v_d_dCdx_bar[i_cell * Ny + j_cell]);
    ind += gridDim.x * blockDim.x;
  }
}

// Get LHS matrix for y-direction computation
__global__ void y_A_LHS_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC,
                               INT N_grid, INT Ny, FLOAT K,
                               FLOAT dt_mul_dy_square_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  while (ind < N_grid) {
    v_d_coA[ind] =  1.0/12 - 0.5 * K * dt_mul_dy_square_inverse;
    v_d_coB[ind] = 10.0/12  + K * dt_mul_dy_square_inverse;
    v_d_coC[ind] =  1.0/12 - 0.5 * K * dt_mul_dy_square_inverse;
    ind += gridDim.x * blockDim.x;
  }
}

__global__ void y_b_RHS_on_gpu(FLOAT *v_d_b, 
                               FLOAT *v_d_int_C_bar, FLOAT *v_d_dCdy_bar, FLOAT *v_d_f_p0, 
                               FLOAT *v_d_f_p1, INT N_grid, INT Ny, FLOAT K, FLOAT dt, 
                               FLOAT dt_mul_dy_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    v_d_b[ind] = v_d_int_C_bar[ind] + 0.5 * dt * (v_d_f_p0[ind] + v_d_f_p1[ind])
                + 0.5 * K * dt_mul_dy_inverse * 
                (v_d_dCdy_bar[i_cell * (Ny + 1) + j_cell+1] - v_d_dCdy_bar[i_cell * (Ny + 1) + j_cell]);

    ind += gridDim.x * blockDim.x;
  }
}

// Apply compact operator to RHS (x-dir)
__global__ void x_b_rhs_compact_on_gpu(FLOAT *v_d_b, FLOAT *v_d_Cptb, INT N_grid, INT Ny, INT Nx) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell; 

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    if(i_cell == 0){
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell * Ny + j_cell] + v_d_b[(i_cell + 1) * Ny + j_cell]) / 12.0;
    } else if(i_cell == Nx-1) {
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell * Ny + j_cell] + v_d_b[(i_cell - 1) * Ny + j_cell]) / 12.0;
    } else {
        v_d_Cptb[ind] = (v_d_b[(i_cell + 1) * Ny + j_cell] + 10.0 * v_d_b[i_cell * Ny + j_cell] +
                         v_d_b[(i_cell - 1) * Ny + j_cell]) / 12.0;
    }
    ind += gridDim.x * blockDim.x;
  }
}

// Apply compact operator to RHS (y-dir)
__global__ void y_b_rhs_compact_on_gpu(FLOAT *v_d_b, FLOAT *v_d_Cptb, INT N_grid, INT Ny) {
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    if(j_cell == 0){
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell*Ny + j_cell] + v_d_b[i_cell*Ny + j_cell+1]) / 12.0;
    } else if(j_cell == Ny-1) {
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell*Ny + j_cell] + v_d_b[i_cell*Ny + j_cell-1]) / 12.0;
    } else { 
        v_d_Cptb[ind] = (v_d_b[i_cell * Ny + (j_cell + 1)] + 10.0 * v_d_b[i_cell * Ny + j_cell] +
                         v_d_b[i_cell * Ny + (j_cell - 1)]) / 12.0;
    }
    ind += gridDim.x * blockDim.x;
  }
}

// Thomas algorithm in x-dir
__global__ void x_Thomas_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC, FLOAT *v_d_b, 
                                FLOAT *v_d_Cnphf_x, FLOAT *p, FLOAT *q, INT N_grid, INT Nx, INT Ny) {

  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  INT i_cell, j_cell;
  FLOAT denom;
  while (ind < Ny) {
    j_cell = ind;

    p[j_cell] = v_d_coC[j_cell] / v_d_coB[j_cell];
    q[j_cell] = v_d_b[j_cell] / v_d_coB[j_cell];
    
    // Forward substitution
    for (i_cell = 1; i_cell < Nx; i_cell++) {
      denom = 1.0 / (v_d_coB[i_cell * Ny + j_cell] -
                     p[(i_cell - 1) * Ny + j_cell] * v_d_coA[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_d_coC[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_d_b[i_cell * Ny + j_cell] -
                                 q[(i_cell - 1) * Ny + j_cell] * v_d_coA[i_cell * Ny + j_cell]) * denom;
    }

    // Backward substitution
    v_d_Cnphf_x[(Nx - 1) * Ny + j_cell] = q[(Nx - 1) * Ny + j_cell];
    for (i_cell = Nx-2; i_cell >= 0; i_cell--) {
      v_d_Cnphf_x[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] -
                                          p[i_cell * Ny + j_cell] * v_d_Cnphf_x[(i_cell + 1) * Ny + j_cell];
    }    
    ind += gridDim.x * blockDim.x;
  }
}

// Thomas algorithm in y-dir
__global__ void y_Thomas_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC, FLOAT *v_d_b, 
                                FLOAT *v_d_Cnp1_y, FLOAT *p, FLOAT *q, INT N_grid, INT Nx, INT Ny) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT denom;

  while (ind < Nx) {
    i_cell = ind ;

    p[i_cell * Ny] = v_d_coC[i_cell * Ny] / v_d_coB[i_cell * Ny];
    q[i_cell * Ny] = v_d_b[i_cell * Ny] / v_d_coB[i_cell * Ny];

    // Forward substitution
    for (j_cell = 1; j_cell < Ny; j_cell++) {
      denom = 1.0 / (v_d_coB[i_cell * Ny + j_cell] -
                     p[i_cell * Ny + j_cell - 1] * v_d_coA[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_d_coC[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_d_b[i_cell * Ny + j_cell] -
                                 q[i_cell * Ny + j_cell - 1] * v_d_coA[i_cell * Ny + j_cell]) * denom;
    }

    // Backward substitution
    v_d_Cnp1_y[i_cell * Ny + Ny-1] = q[i_cell * Ny + Ny-1];
    for (j_cell = Ny-2; j_cell >= 0; j_cell--) {
      v_d_Cnp1_y[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] -
                                         p[i_cell * Ny + j_cell] * v_d_Cnp1_y[i_cell * Ny + (j_cell + 1)];
    }  
    ind += gridDim.x * blockDim.x;
  }
}

/******************* Parareal-support structures & kernels *******************/

// 把设备指针和网格/物理参数打包，方便在 GpuOneStep/Propagator 里使用
struct GPUContext {
    INT Nx, Ny;
    INT N_grid;
    INT N_points;
    INT N_points_inter1;
    INT N_points_inter2;

    FLOAT dx, dy;
    FLOAT K;

    int numBlocks;
    int threadsPerBlock;

    // geometry
    FLOAT *v_d_x, *v_d_y;
    FLOAT *v_d_xhf, *v_d_yhf;
    FLOAT *v_d_xhf_bar1, *v_d_yhf_bar2, *v_d_xhf_bar3;
    FLOAT *v_d_x_inter1, *v_d_y_inter2, *v_d_x_inter3;

    // solution & intermediates
    FLOAT *v_d_Cn;
    FLOAT *v_d_Cnp1;
    FLOAT *v_d_Cnphf_x;
    FLOAT *v_d_Cn_bar_x;
    FLOAT *v_d_dCndx_bar;
    FLOAT *v_d_Cn_bar_y;
    FLOAT *v_d_Cnp1_y;
    FLOAT *v_d_dCndy_bar;
    FLOAT *v_d_Cnphf_bar_x;
    FLOAT *v_d_dCnphfdx_bar;

    // PPM coefficients
    FLOAT *v_d_CL, *v_d_CR, *v_d_C6, *v_d_DltC;

    // Tridiagonal solver
    FLOAT *v_d_coA_x, *v_d_coB_x, *v_d_coC_x;
    FLOAT *v_d_coA_y, *v_d_coB_y, *v_d_coC_y;
    FLOAT *v_d_Cptb, *v_d_b;
    FLOAT *p, *q;

    // source terms & exact
    FLOAT *v_d_C_Exact;
    FLOAT *v_d_f1_p0, *v_d_f1_p1, *v_d_f1_p0_bar;
    FLOAT *v_d_f2_p0, *v_d_f2_p1, *v_d_f2_p0_bar;
};

// Parareal校正： U_{k+1}^{n+1} = G_{k+1}^{n+1} + (F_{k}^{n+1} - G_{k}^{n+1})
__global__ void parareal_correction_kernel(const FLOAT *G_k1,
                                           const FLOAT *G_k,
                                           const FLOAT *F_np1,
                                           FLOAT *U_next,
                                           INT N_grid)
{
    INT tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < N_grid) {
        U_next[tid] = G_k1[tid] + (F_np1[tid] - G_k[tid]);
        tid += gridDim.x * blockDim.x;
    }
}

// 单步时间推进（完整三步 T2S4-CC-CFV），从 ctx.v_d_Cn 更新到 ctx.v_d_Cnp1，然后拷回 Cn
void GpuOneStep(GPUContext &ctx, FLOAT tn, FLOAT dt_step)
{
    FLOAT tnphf = tn + 0.5 * dt_step;
    FLOAT tnp1  = tn + dt_step;

    INT   Nx = ctx.Nx;
    INT   Ny = ctx.Ny;
    INT   N_grid = ctx.N_grid;
    INT   N_points = ctx.N_points;
    INT   N_points_inter1 = ctx.N_points_inter1;
    INT   N_points_inter2 = ctx.N_points_inter2;

    FLOAT dx = ctx.dx;
    FLOAT dy = ctx.dy;
    FLOAT K  = ctx.K;

    FLOAT dt_mul_dx_inverse        = dt_step / dx;
    FLOAT dt_mul_dy_inverse        = dt_step / dy;
    FLOAT dt_mul_dx_square_inverse = dt_step / (dx * dx);
    FLOAT dt_mul_dy_square_inverse = dt_step / (dy * dy);

    int numBlocks       = ctx.numBlocks;
    int threadsPerBlock = ctx.threadsPerBlock;

    // 对应原始代码：每个时间步都会用当前 dt 重新计算拉格朗日点和交点
    get_Lagrange_points_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_xhf, ctx.v_d_yhf,
        ctx.v_d_xhf_bar1, ctx.v_d_yhf_bar2, ctx.v_d_xhf_bar3,
        Nx, Ny, N_points, dt_step);

    intersection_point_vert_x<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_xhf_bar1, ctx.v_d_x_inter1, Ny, N_points_inter1);

    intersection_point_hori_y<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_yhf_bar2, ctx.v_d_y_inter2, Ny, N_points_inter2);

    intersection_point_vert_x<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_xhf_bar3, ctx.v_d_x_inter3, Ny, N_points_inter1);

    // =========================== step 1 =========================== //
    x_step_ppm_coef_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_Cn, N_grid, Nx, Ny);

    x_step_intC_ppm_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_Cn, ctx.v_d_Cn_bar_x,
        ctx.v_d_xhf, ctx.v_d_x_inter1, N_grid, Nx, Ny, dx);

    dcdx_bar_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_Cn, ctx.v_d_dCndx_bar,
        ctx.v_d_xhf, ctx.v_d_x_inter1, N_points_inter1,
        Nx, Ny, dx);

    // 源项 f≡0，这几步结果全为 0，但保留结构
    get_source_term_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_x, ctx.v_d_y, ctx.v_d_f1_p1, dx, dy, tnphf, K, N_grid);

    get_source_term_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_x, ctx.v_d_y, ctx.v_d_f1_p0, dx, dy, tn, K, N_grid);

    x_step_ppm_coef_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_f1_p0, N_grid, Nx, Ny);

    x_step_intC_ppm_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_f1_p0, ctx.v_d_f1_p0_bar,
        ctx.v_d_xhf, ctx.v_d_x_inter1, N_grid, Nx, Ny, dx);

    x_A_LHS_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_coA_x, ctx.v_d_coB_x, ctx.v_d_coC_x,
        N_grid, Nx, K, dt_mul_dx_square_inverse);

    x_b_RHS_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_b, ctx.v_d_Cn_bar_x, ctx.v_d_dCndx_bar,
        ctx.v_d_f1_p0_bar, ctx.v_d_f1_p1,
        N_grid, Nx, K, dt_step, dt_mul_dx_inverse);

    x_b_rhs_compact_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_b, ctx.v_d_Cptb, N_grid, Ny, Nx);

    x_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_coA_x, ctx.v_d_coB_x, ctx.v_d_coC_x,
        ctx.v_d_Cptb, ctx.v_d_Cnphf_x,
        ctx.p, ctx.q, N_grid, Nx, Ny);

    // =========================== step 2 =========================== //
    y_step_ppm_coef_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_Cnphf_x, N_grid, Nx, Ny);

    y_step_intC_ppm_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_Cnphf_x, ctx.v_d_Cn_bar_y,
        ctx.v_d_yhf, ctx.v_d_y_inter2, N_grid, Nx, Ny, dy);

    dcdy_bar_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_Cnphf_x, ctx.v_d_dCndy_bar,
        ctx.v_d_yhf, ctx.v_d_y_inter2, N_points_inter2,
        Nx, Ny, dy);

    get_source_term_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_x, ctx.v_d_y, ctx.v_d_f2_p1, dx, dy, tnp1, K, N_grid);

    get_source_term_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_x, ctx.v_d_y, ctx.v_d_f2_p0, dx, dy, tn, K, N_grid);

    y_step_ppm_coef_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_f2_p0, N_grid, Nx, Ny);

    y_step_intC_ppm_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_f2_p0, ctx.v_d_f2_p0_bar,
        ctx.v_d_yhf, ctx.v_d_y_inter2, N_grid, Nx, Ny, dy);

    y_A_LHS_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_coA_y, ctx.v_d_coB_y, ctx.v_d_coC_y,
        N_grid, Ny, K, dt_mul_dy_square_inverse);

    y_b_RHS_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_b, ctx.v_d_Cn_bar_y, ctx.v_d_dCndy_bar,
        ctx.v_d_f2_p1, ctx.v_d_f2_p0_bar,
        N_grid, Ny, K, dt_step, dt_mul_dy_inverse);

    y_b_rhs_compact_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_b, ctx.v_d_Cptb, N_grid, Ny);

    y_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_coA_y, ctx.v_d_coB_y, ctx.v_d_coC_y,
        ctx.v_d_Cptb, ctx.v_d_Cnp1_y,
        ctx.p, ctx.q, N_grid, Nx, Ny);

    // =========================== step 3 =========================== //
    x_step_ppm_coef_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_Cnp1_y, N_grid, Nx, Ny);

    x_step_intC_ppm_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_Cnp1_y, ctx.v_d_Cnphf_bar_x,
        ctx.v_d_xhf, ctx.v_d_x_inter1, N_grid, Nx, Ny, dx);

    dcdx_bar_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_Cnphf_bar_x, ctx.v_d_dCnphfdx_bar,
        ctx.v_d_xhf, ctx.v_d_x_inter3, N_points_inter1,
        Nx, Ny, dx);

    // ************** source term at t^(n+1) ************
    get_source_term_on_gpu<<<numBlocks, threadsPerBlock>>>(
    ctx.v_d_x, ctx.v_d_y, ctx.v_d_f1_p1, dx, dy, tnp1, K, N_grid);

   // 近似认为 f^{n+1/2} ≈ f^{n+1}，把数值拷到 f1_p0 中
   cudaMemcpy(ctx.v_d_f1_p0, ctx.v_d_f1_p1,
           N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);


    x_step_ppm_coef_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_f1_p0, N_grid, Nx, Ny);

    x_step_intC_ppm_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_CL, ctx.v_d_CR, ctx.v_d_C6, ctx.v_d_DltC,
        ctx.v_d_f1_p0, ctx.v_d_f1_p0_bar,
        ctx.v_d_xhf, ctx.v_d_x_inter1, N_grid, Nx, Ny, dx);

    x_b_RHS_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_b, ctx.v_d_Cnphf_bar_x, ctx.v_d_dCnphfdx_bar,
        ctx.v_d_f1_p0_bar, ctx.v_d_f1_p1,
        N_grid, Nx, K, dt_step, dt_mul_dx_inverse);

    x_b_rhs_compact_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_b, ctx.v_d_Cptb, N_grid, Ny, Nx);

    x_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>(
        ctx.v_d_coA_x, ctx.v_d_coB_x, ctx.v_d_coC_x,
        ctx.v_d_Cptb, ctx.v_d_Cnp1,
        ctx.p, ctx.q, N_grid, Nx, Ny);

    // 更新 Cn <- Cnp1
    cudaMemcpy(ctx.v_d_Cn, ctx.v_d_Cnp1,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
}

// 粗传播器：一步 DT
void CoarsePropagator(GPUContext &ctx, FLOAT tn, FLOAT DT)
{
    GpuOneStep(ctx, tn, DT);
}

// 细传播器：在 [tn, tn+DT] 内用 J 个小步 dt = DT/J
void FinePropagator(GPUContext &ctx, FLOAT tn, FLOAT DT, int J)
{
    FLOAT dt = DT / (FLOAT)J;
    for (int j = 0; j < J; ++j) {
        FLOAT t_local = tn + j * dt;
        GpuOneStep(ctx, t_local, dt);
    }
}

/*********************************
// Main Code
**********************************/

int main(int argc, char *argv[]) {
  
  FLOAT K;
  FLOAT dx, dy, dt;
  FLOAT T_min, T_max; 
  INT Nx, Ny, Nt, T_span, N_grid, N_points, N_points_inter1, N_points_inter2;

  FLOAT *v_h_Cn, *v_h_Cnp1, *v_h_C_Exact;
  FLOAT *v_d_CL, *v_d_CR, *v_d_C6, *v_d_DltC;
  FLOAT *v_d_Cn, *v_d_Cnphf_x, *v_d_Cn_bar_x, *v_d_dCndx_bar;
  FLOAT *v_d_Cn_bar_y, *v_d_dCndy_bar, *v_d_Cnp1_y;
  FLOAT *v_d_Cnphf_bar_x, *v_d_dCnphfdx_bar, *v_d_Cnp1;
  FLOAT *v_d_x, *v_d_y, *v_d_xhf, *v_d_yhf, *v_d_xhf_bar1, *v_d_yhf_bar2, *v_d_xhf_bar3;
  FLOAT *v_d_x_inter1, *v_d_y_inter2, *v_d_x_inter3;
  FLOAT *v_d_coA_y, *v_d_coB_y, *v_d_coC_y, *v_d_Cptb, *v_d_b, *p, *q;
  FLOAT *v_d_coA_x, *v_d_coB_x, *v_d_coC_x;
  FLOAT *v_d_f1_p0, *v_d_f1_p1, *v_d_f1_p0_bar, *v_d_f2_p0, *v_d_f2_p1, *v_d_f2_p0_bar;
  FLOAT *v_d_C_Exact;
  
  if (argc < 2) {
    cout << "please input config file name" << endl;
    return 1;
  }

  /*************************************************
    read  parameters from config file
  *************************************************/
  ifstream configFile;
  configFile.open(argv[1]);
  string strLine;
  string strKey, strValue;
  size_t pos;
  if (configFile.is_open()) {
    cout << "open config file ok" << endl;
    while (!configFile.eof()) {
      getline(configFile, strLine);
      if (strLine.empty()) continue;
      pos = strLine.find(':');
      if (pos == string::npos) continue;
      strKey   = strLine.substr(0, pos);
      strValue = strLine.substr(pos + 1);
      if (strKey.compare("T_min") == 0) {
        sscanf(strValue.c_str(), "%lf", &T_min);
      }
      if (strKey.compare("T_max") == 0) {
        sscanf(strValue.c_str(), "%lf", &T_max);
      }
      if (strKey.compare("N") == 0) {
        sscanf(strValue.c_str(), "%d", &Nx);
        Ny = Nx;
      }
      if (strKey.compare("Nt") == 0) {
        sscanf(strValue.c_str(), "%d", &Nt);
      }
      if (strKey.compare("T_span") == 0) {
        sscanf(strValue.c_str(), "%d", &T_span);
      }
      if (strKey.compare("K") == 0) {
        sscanf(strValue.c_str(), "%lf", &K);
      }
    }
  } else {
    cout << "Cannot open config file!" << endl;
    return 1;
  }
  configFile.close();

  // discretization
  dx = (X_max - X_min) / Nx;
  dy = (Y_max - Y_min) / Ny;
  dt = (T_max - T_min) / Nt;

  N_grid          = Nx * Ny;
  N_points        = (Nx + 1) * (Ny + 1);
  N_points_inter2 = Nx * (Ny + 1);
  N_points_inter1 = (Nx + 1) * Ny;
  
  // GPU launch parameters
  int numBlocks;
  int threadsPerBlock;
  int maxThreadsPerBlock;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock = 128;
  if (N_points < maxThreadsPerBlock) {
    threadsPerBlock = N_points;
    numBlocks = 1;
  } else {
    threadsPerBlock = maxThreadsPerBlock;
    numBlocks = (N_points + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
  }

  // Host memory
  v_h_Cn     = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_Cnp1   = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_C_Exact= (FLOAT *)malloc(N_grid * sizeof(FLOAT));

  // Device memory
  cudaMalloc((void **)&v_d_x,           N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_y,           N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_xhf,         N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_yhf,         N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_xhf_bar1,    N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_yhf_bar2,    N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_xhf_bar3,    N_points * sizeof(FLOAT));

  cudaMalloc((void **)&v_d_x_inter1,    N_points_inter1 * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_y_inter2,    N_points_inter2 * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_x_inter3,    N_points_inter1 * sizeof(FLOAT));

  cudaMalloc((void **)&v_d_Cn,          N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cnphf_x,     N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cn_bar_x,    N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_dCndx_bar,   N_points_inter1 * sizeof(FLOAT));

  cudaMalloc((void **)&v_d_Cn_bar_y,    N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cnp1_y,      N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_dCndy_bar,   N_points_inter2 * sizeof(FLOAT));

  cudaMalloc((void **)&v_d_C_Exact,     N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cnp1,        N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cnphf_bar_x, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_dCnphfdx_bar,N_points_inter1 * sizeof(FLOAT));

  cudaMalloc((void **)&v_d_CL,          N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_CR,          N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C6,          N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_DltC,        N_grid * sizeof(FLOAT));

  cudaMalloc((void **)&p,               N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&q,               N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_b,           N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cptb,        N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coA_x,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coB_x,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coC_x,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coA_y,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coB_y,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coC_y,       N_grid * sizeof(FLOAT));

  cudaMalloc((void **)&v_d_f1_p0,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_f1_p1,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_f1_p0_bar,   N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_f2_p0,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_f2_p1,       N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_f2_p0_bar,   N_grid * sizeof(FLOAT));

  clock_t start = clock();

  // 预处理：Euler网格节点
  get_Euler_cell_center_points_on_gpu<<<numBlocks, threadsPerBlock>>>(
      v_d_x, v_d_y, Nx, Ny, N_grid, dx, dy);

  get_Euler_half_points_on_gpu<<<numBlocks, threadsPerBlock>>>(
      v_d_xhf, v_d_yhf, Nx, Ny, N_points, dx, dy);

  // 初始条件和最终时刻的解析解
  get_exact_solution_on_gpu<<<numBlocks, threadsPerBlock>>>(
      v_d_x, v_d_y, v_d_Cn, dx, dy, 0.0, K, N_grid);
  cudaMemcpy(v_h_Cn, v_d_Cn,
             N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
  output_result(v_h_Cn, 0, Nx, Ny, dt);
  // === 新增：备份初始条件到设备 d_C0，用于后面参考解和 Parareal 初始化 === 
  FLOAT *d_C0;
  cudaMalloc((void**)&d_C0, N_grid * sizeof(FLOAT));
  cudaMemcpy(d_C0, v_d_Cn, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);


  get_exact_solution_on_gpu<<<numBlocks, threadsPerBlock>>>(
      v_d_x, v_d_y, v_d_C_Exact, dx, dy, T_max, K, N_grid);
  cudaMemcpy(v_h_C_Exact, v_d_C_Exact,
             N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);

  // ====================== 构造 GPUContext ======================
  GPUContext ctx;
  ctx.Nx  = Nx;
  ctx.Ny  = Ny;
  ctx.N_grid = N_grid;
  ctx.N_points = N_points;
  ctx.N_points_inter1 = N_points_inter1;
  ctx.N_points_inter2 = N_points_inter2;
  ctx.dx  = dx;
  ctx.dy  = dy;
  ctx.K   = K;
  ctx.numBlocks       = numBlocks;
  ctx.threadsPerBlock = threadsPerBlock;

  ctx.v_d_x  = v_d_x;
  ctx.v_d_y  = v_d_y;
  ctx.v_d_xhf = v_d_xhf;
  ctx.v_d_yhf = v_d_yhf;
  ctx.v_d_xhf_bar1 = v_d_xhf_bar1;
  ctx.v_d_yhf_bar2 = v_d_yhf_bar2;
  ctx.v_d_xhf_bar3 = v_d_xhf_bar3;
  ctx.v_d_x_inter1 = v_d_x_inter1;
  ctx.v_d_y_inter2 = v_d_y_inter2;
  ctx.v_d_x_inter3 = v_d_x_inter3;

  ctx.v_d_Cn            = v_d_Cn;
  ctx.v_d_Cnp1          = v_d_Cnp1;
  ctx.v_d_Cnphf_x       = v_d_Cnphf_x;
  ctx.v_d_Cn_bar_x      = v_d_Cn_bar_x;
  ctx.v_d_dCndx_bar     = v_d_dCndx_bar;
  ctx.v_d_Cn_bar_y      = v_d_Cn_bar_y;
  ctx.v_d_Cnp1_y        = v_d_Cnp1_y;
  ctx.v_d_dCndy_bar     = v_d_dCndy_bar;
  ctx.v_d_Cnphf_bar_x   = v_d_Cnphf_bar_x;
  ctx.v_d_dCnphfdx_bar  = v_d_dCnphfdx_bar;

  ctx.v_d_CL   = v_d_CL;
  ctx.v_d_CR   = v_d_CR;
  ctx.v_d_C6   = v_d_C6;
  ctx.v_d_DltC = v_d_DltC;

  ctx.v_d_coA_x = v_d_coA_x;
  ctx.v_d_coB_x = v_d_coB_x;
  ctx.v_d_coC_x = v_d_coC_x;
  ctx.v_d_coA_y = v_d_coA_y;
  ctx.v_d_coB_y = v_d_coB_y;
  ctx.v_d_coC_y = v_d_coC_y;
  ctx.v_d_Cptb  = v_d_Cptb;
  ctx.v_d_b     = v_d_b;
  ctx.p         = p;
  ctx.q         = q;

  ctx.v_d_C_Exact   = v_d_C_Exact;
  ctx.v_d_f1_p0     = v_d_f1_p0;
  ctx.v_d_f1_p1     = v_d_f1_p1;
  ctx.v_d_f1_p0_bar = v_d_f1_p0_bar;
  ctx.v_d_f2_p0     = v_d_f2_p0;
  ctx.v_d_f2_p1     = v_d_f2_p1;
  ctx.v_d_f2_p0_bar = v_d_f2_p0_bar;

  // ====================== 参考细解：串行 Nt 个 dt 步 ======================
 int Nt_fine = Nt;
FLOAT dt_fine = dt;

FLOAT *d_Uref_T;
cudaMalloc((void**)&d_Uref_T, N_grid * sizeof(FLOAT));

// === 先把 ctx.v_d_Cn 设回 t=0 初值 ===
cudaMemcpy(ctx.v_d_Cn, d_C0,
           N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

for (int n = 0; n < Nt_fine; ++n) {
    FLOAT tn = T_min + n * dt_fine;
    GpuOneStep(ctx, tn, dt_fine);

    if ((n+1) % T_span == 0) {
      cudaMemcpy(v_h_Cnp1, ctx.v_d_Cn,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
      output_result(v_h_Cnp1, n+1, Nx, Ny, dt_fine);
    }
}

// 参考末时刻解（细解的 T_max）
cudaMemcpy(d_Uref_T, ctx.v_d_Cn,
           N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

  // ====================== Parareal 参数与数据结构 ======================
  int Nt_coarse = 40;                     // 粗时间段数，可根据需要调
  FLOAT DT = (T_max - T_min) / Nt_coarse;
  int J_fine = Nt_fine / Nt_coarse;       // 每粗段内细步数，保证 Nt_fine 能被整除
  int Kmax = 8;                           // Parareal 最大迭代次数

  FLOAT *d_Uk, *d_Uk1, *d_F_np1, *d_Gk, *d_Gk1;
  cudaMalloc((void**)&d_Uk,  (Nt_coarse+1) * N_grid * sizeof(FLOAT));
  cudaMalloc((void**)&d_Uk1, (Nt_coarse+1) * N_grid * sizeof(FLOAT));
  cudaMalloc((void**)&d_F_np1, N_grid * sizeof(FLOAT));
  cudaMalloc((void**)&d_Gk,    N_grid * sizeof(FLOAT));
  cudaMalloc((void**)&d_Gk1,   N_grid * sizeof(FLOAT));

 // ========== Parareal 初始化：k=0 粗解 ==========
// Uk(:,0) = 初值（用 d_C0）
cudaMemcpy(d_Uk, d_C0,
           N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

// 同时把 ctx.v_d_Cn 也重置为初值，方便后面使用
cudaMemcpy(ctx.v_d_Cn, d_C0,
           N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

for (int n = 0; n < Nt_coarse; ++n) {
    FLOAT tn = T_min + n * DT;
    cudaMemcpy(ctx.v_d_Cn, d_Uk + n * N_grid,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

    CoarsePropagator(ctx, tn, DT);

    cudaMemcpy(d_Uk + (n+1)*N_grid, ctx.v_d_Cn,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
}

  for (int n = 0; n < Nt_coarse; ++n) {
    FLOAT tn = T_min + n * DT;
    cudaMemcpy(ctx.v_d_Cn, d_Uk + n * N_grid,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

    CoarsePropagator(ctx, tn, DT);

    cudaMemcpy(d_Uk + (n+1)*N_grid, ctx.v_d_Cn,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
  }

  // ========== Parareal 迭代 ==========
  double *err = new double[Kmax];

  for (int k = 0; k < Kmax; ++k) {

    // Uk1(:,0) = 初值
    cudaMemcpy(d_Uk1, d_Uk,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

    for (int n = 0; n < Nt_coarse; ++n) {
      FLOAT tn = T_min + n * DT;

       // G_{k+1}: 从 Uk1(:,n) 出发做一层粗步
      cudaMemcpy(ctx.v_d_Cn, d_Uk1 + n * N_grid,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
      CoarsePropagator(ctx, tn, DT);
      cudaMemcpy(d_Gk1, ctx.v_d_Cn,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

      // G_k: 从 Uk(:,n) 出发做一层粗步
      cudaMemcpy(ctx.v_d_Cn, d_Uk + n * N_grid,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
      CoarsePropagator(ctx, tn, DT);
      cudaMemcpy(d_Gk, ctx.v_d_Cn,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

      // F_n^k: 从 Uk(:,n) 出发，在 [tn, tn+DT] 内做 J_fine 个细步
      cudaMemcpy(ctx.v_d_Cn, d_Uk + n * N_grid,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
      FinePropagator(ctx, tn, DT, J_fine);
      cudaMemcpy(d_F_np1, ctx.v_d_Cn,
                 N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);

      // 校正：Uk1(:,n+1) = G_{k+1} + (F - G_k)
      parareal_correction_kernel<<<ctx.numBlocks, ctx.threadsPerBlock>>>(
          d_Gk1, d_Gk, d_F_np1,
          d_Uk1 + (n+1) * N_grid, N_grid);
    }

    // 计算第 k 次迭代末时刻与参考细解的误差
    FLOAT *h_Uk1_T  = (FLOAT*)malloc(N_grid * sizeof(FLOAT));
    FLOAT *h_Uref_T = (FLOAT*)malloc(N_grid * sizeof(FLOAT));
    cudaMemcpy(h_Uk1_T,  d_Uk1 + Nt_coarse * N_grid,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Uref_T, d_Uref_T,
               N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);

    double err_inf = 0.0;
    for (INT i = 0; i < N_grid; ++i) {
      double diff = fabs(h_Uk1_T[i] - h_Uref_T[i]);
      if (diff > err_inf) err_inf = diff;
    }
    err[k] = err_inf;
    printf("Parareal iter %2d: err(T) = %.3e\n", k+1, err[k]);

    free(h_Uk1_T);
    free(h_Uref_T);

    // 准备下一次迭代：Uk ← Uk1
    cudaMemcpy(d_Uk, d_Uk1,
               (Nt_coarse+1) * N_grid * sizeof(FLOAT),
               cudaMemcpyDeviceToDevice);
  }

  clock_t end = clock();
  double time = (end - start) / (double)CLOCKS_PER_SEC;
  cout << "K:"  << K  << fixed << setprecision(16) << endl;
  cout << "dx:" << dx << endl;
  cout << "dy:" << dy << endl;
  cout << "dt:" << dt << endl;
  cout << "Nx:" << Nx << endl;
  cout << "Ny:" << Ny << endl;
  cout << "Nt:" << Nt << endl;
  cout << "Total time (including Parareal): " << time << " s" << endl;

  // Free device memory
  cudaFree(v_d_x);          cudaFree(v_d_y);
  cudaFree(v_d_xhf);        cudaFree(v_d_yhf);
  cudaFree(v_d_xhf_bar1);   cudaFree(v_d_yhf_bar2);
  cudaFree(v_d_xhf_bar3);
  cudaFree(v_d_x_inter1);   cudaFree(v_d_y_inter2);
  cudaFree(v_d_x_inter3);

  cudaFree(v_d_Cn);         cudaFree(v_d_Cnphf_x);
  cudaFree(v_d_Cn_bar_x);   cudaFree(v_d_dCndx_bar);
  cudaFree(v_d_Cn_bar_y);   cudaFree(v_d_Cnp1_y);
  cudaFree(v_d_dCndy_bar);  cudaFree(v_d_Cnp1);
  cudaFree(v_d_Cnphf_bar_x);
  cudaFree(v_d_dCnphfdx_bar);

  cudaFree(v_d_CL);         cudaFree(v_d_CR);
  cudaFree(v_d_C6);         cudaFree(v_d_DltC);
  cudaFree(v_d_coA_y);      cudaFree(v_d_coB_y);
  cudaFree(v_d_coC_y);      cudaFree(v_d_coA_x);
  cudaFree(v_d_coB_x);      cudaFree(v_d_coC_x);
  cudaFree(v_d_Cptb);       cudaFree(v_d_b);
  cudaFree(p);              cudaFree(q);

  cudaFree(v_d_f1_p0);      cudaFree(v_d_f1_p1);
  cudaFree(v_d_f1_p0_bar);
  cudaFree(v_d_f2_p0);      cudaFree(v_d_f2_p1);
  cudaFree(v_d_f2_p0_bar);
  cudaFree(v_d_C_Exact);

  cudaFree(d_Uref_T);
  cudaFree(d_Uk);
  cudaFree(d_Uk1);
  cudaFree(d_F_np1);
  cudaFree(d_Gk);
  cudaFree(d_Gk1);

  // Free host memory
  free(v_h_Cn);  free(v_h_Cnp1);  free(v_h_C_Exact);
  delete [] err;

  return 0;
}
