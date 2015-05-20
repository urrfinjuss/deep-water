//#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>

#define pi 2*acos(0)
#define FMODE FFTW_PATIENT

typedef struct vector {
  fftw_complex *r, *v;
} vect;

typedef struct work {
  int n;
  double	y0;
  fftw_complex *Q;
  fftw_complex *V;
} work, *work_ptr;

typedef struct control {
  long int incr_step;	
  double t_stop;
  double T;  // current time
  double H;  // energy
  double M;  // mean level
  double Px;  // momentum x
  double Py;  // momentum y
} ctrl, *ctrl_ptr;

typedef struct params {
  double res1, pos1, phs1, sft1;
  double res2, phs2;
  int nthreads, n_order;

  int n, k_cut, num_iter, e_n, ionum; 
  int ref_lvl, skip, pow;
  long int step_start, max_steps, step_io, step_current;
  int stokes, d_cnt, d_poles;  
  int rflag, pflag;
  double g, cfl, c, tmax, t_skip, dt, nu, sigma;
  char runname[80];
  char resname[80];
  char resnamev[80];
  ctrl *current; 
  work_ptr w_ptr;

} params, *params_ptr;

// 	prep.c
extern void err_msg(char* str);
extern void save_ascii(params_ptr in, work_ptr wrk, char* fname);
extern void save_binary_data(params_ptr in, work_ptr wrk, char* fname);
extern void read_binary_data(params_ptr in, work_ptr wrk, char* fname);
extern void save_array(params_ptr in, fftw_complex *arr, char* fname);
extern void save_fourier(params_ptr in, fftw_complex *arr, char* fname);

//  --  ffts.c
extern void init_input(int *argc, char **argv, params_ptr in, work_ptr wrk);
extern void init_arrays(params_ptr in, work_ptr wrk);
extern void init_fftw(params_ptr in, work_ptr wrk);
extern void proc_one_serial();
extern void* pthread_id0(void *arg);
extern void* pthread_id1(void *arg);
extern void* pthread_id0_rk6(void *arg);
extern void* pthread_id1_rk6(void *arg);
extern void evolve_rk4_serial();
extern void evolve_rk_threads();
extern void bench_fft(params_ptr in, work_ptr wrk);
extern void bench_rk4();
extern void refine();


extern void init(int *argc, char **argv, params_ptr in, work_ptr wrk) ;
extern void hilbert(fftw_complex *in, fftw_complex *out);
extern void project(fftw_complex *in, fftw_complex *out);
extern void pproject(fftw_complex *in, fftw_complex *out);
extern void ffilter(fftw_complex *in);
extern void reconstruct_surface(work_ptr in);
extern void compute_inverse(fftw_complex *in, fftw_complex *out);
extern void compute_zu(work_ptr in);
extern void reconstruct_potential(work_ptr in); 
extern void update_dfunc(work_ptr in);
extern void fft(fftw_complex *in, fftw_complex *out);
extern void ifft(fftw_complex *in, fftw_complex *out);
extern void der(fftw_complex *in, fftw_complex *out);
extern void rhs(work_ptr in, int m);
extern void rk4(params_ptr in);
extern double find_y0(work_ptr in);
extern void generate_derivatives(work_ptr in, int m);

extern void prep_control_params(params_ptr in, work_ptr wrk);
extern void hamiltonian(params_ptr in);
extern void mean_level(params_ptr in);
extern void momentumx(params_ptr in);
extern void momentumy(params_ptr in);
extern void read_restart(params_ptr in, work_ptr wrk);

extern int array_max(fftw_complex *input_array, int N);
extern double* array_enlarge_double (double *input_array, int Nold, int Nnew);
extern fftw_complex* array_enlarge_complex(fftw_complex *input_array, int Nold, int Nnew);
extern fftw_complex* expand_complex (fftw_complex *iarray, int Nnew);
extern void expand_wrk(work_ptr wrk, int Nnew);


extern void init_prep(params_ptr in);
extern void free_prep(params_ptr in);
extern void write_out(fftw_complex* in, char* str, double time);
extern void write_out4pade(fftw_complex* in, char* str, double time);
extern void write_out2(fftw_complex* in, char* str, double time);
extern void write_outwz(fftw_complex* in, char* str, double time);
extern void write_spec(fftw_complex* in, char* str, double time);
extern void write_combined(work_ptr wrk, char* str, double time);
extern void write_combined2(work_ptr wrk, char* str, double time);
extern void read_data(params_ptr in);
extern void read_poles(params_ptr in, work_ptr wrk);
extern void stokes_wave(double tol, params_ptr in);


