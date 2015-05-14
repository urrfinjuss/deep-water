#include "header.h"

static int N;
static FILE *fh_out;
static double *k, *u, g;
static fftw_complex *a[4], *RS;




void save_array(params_ptr in, fftw_complex *arr, char* fname) {
  fh_out = fopen(fname, "w");
  fprintf(fh_out, "#1.u 2. re Z 3. im Z\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", (in->current)->T);
  for (int j = 0; j < in->n; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\n", 2*pi*j/(in->n) - pi, creal(arr[j]), cimag(arr[j]));
  }
  fclose(fh_out);
}

void save_fourier(params_ptr in, fftw_complex *arr, char* fname) {
  fh_out = fopen(fname, "w");
  fprintf(fh_out, "#1.k 2. abs(Rk)\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", (in->current)->T);
  for (int j = 0; j < in->n; j++) {
    if (j < (in->n)/2) fprintf(fh_out, "%d\t%.15e\n", j, cabs(arr[j]));
    else fprintf(fh_out, "%d\t%.15e\n", j - (in->n), cabs(arr[j]));
  }
  fclose(fh_out);
}

void err_msg(char* str)
{
  printf("%s\n",str);
  exit(1);
}


void save_binary_data(params_ptr in, work_ptr wrk, char* fname) {
   fh_out = fopen(fname, "wb");
   fwrite(in, sizeof(struct params), 1, fh_out);
   fwrite(&((in->current)->T), sizeof(double), 1, fh_out);
   fwrite(wrk->Q, sizeof(fftw_complex), in->n, fh_out);
   fwrite(wrk->V, sizeof(fftw_complex), in->n, fh_out);
   fclose(fh_out);
}

void read_binary_data(params_ptr in, work_ptr wrk, char* fname) {
   struct params dummy;
   FILE *fhlog;
   int m;
   fh_out = fopen(fname, "r");
   if (fh_out == NULL) {
	printf("Cannot open file %s\n", fname);
	err_msg("");
   }
   m = fread(&dummy, sizeof(struct params), 1, fh_out);
   m = fread(&((in->current)->T), sizeof(double), 1, fh_out);
   if (dummy.n == in->n) {
     m = fread(wrk->Q, sizeof(fftw_complex), dummy.n, fh_out); 
     m = fread(wrk->V, sizeof(fftw_complex), dummy.n, fh_out);
     fclose(fh_out);
     fhlog = fopen("run.log","a");
     fprintf(fhlog, "\nRestart Log:\n");
     fprintf(fhlog, "Grid from binary restart matches input configuration\n");
     fclose(fhlog);
     in->step_start = dummy.step_start;
     in->step_io = dummy.step_io;
     in->max_steps = dummy.max_steps;
     in->cfl = dummy.cfl;
     in->dt = dummy.dt;
   } else {
     fhlog = fopen("run.log","a");
     fprintf(fhlog, "\nRestart Log:\n");
     fprintf(fhlog, "Grid from binary restart is %d modes and %d in input configuration\n", dummy.n, in->n);
     fprintf(fhlog, "High Fourier modes will be either set to zero or eliminated.\n");
     fclose(fhlog);
     err_msg("Dead End\nFourier interpolation will be here");
   }
}

void save_ascii(params_ptr in, work_ptr wrk, char* fname) {
  fh_out = fopen(fname, "w");
  fprintf(fh_out, "#1.u 2. re Q 3. im Q 4. re V 5. im V\n");
  fprintf(fh_out, "# N = %d Current Time = %.15f\n\n", in->n, (in->current)->T);
  //fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < in->n; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n", 2*pi*j/(in->n) - pi, creal(wrk->Q[j]), cimag(wrk->Q[j]),  creal(wrk->V[j]), cimag(wrk->V[j]) );
  }
  fclose(fh_out);
}


/*void init_prep(params_ptr in) {
  N = in->n;
  u = fftw_malloc(sizeof(double)*N);
  k = fftw_malloc(sizeof(double)*N);
  for (int j = 0; j < in->n; j ++) {
    u[j] = 2*pi*j/N;
    k[j] = j;
    if (j > N/2) k[j] = j - N;
  }
  for (int j = 0; j < 4; j++) a[j] = fftw_malloc(N*sizeof(fftw_complex));
  RS = fftw_malloc(N*sizeof(fftw_complex));
  g = in->g;
}

void free_prep(params_ptr in) {
  free(u); free(k);
  for (int j = 0; j < 4; j++) free(a[j]);
  free(RS);
}

void write_out(fftw_complex* in, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3.re v 4. im v\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\n", u[j], creal(in[j]), cimag(in[j]) );
  }
  fclose(fh_out);
}

void write_out4pade(fftw_complex* in, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3.re v 4. im v\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\n", u[j], creal(in[j]-in[0]), cimag(in[j]-in[0]) );
  }
  fclose(fh_out);
}

void write_spec(fftw_complex* in, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3.re v 4. im v\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\n",  k[j], 
                        creal(in[j]), cimag(in[j]) );
  }
  fclose(fh_out);
}

void write_out2(fftw_complex* in, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3.im z 4. re z\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\t%.15e\n", u[j], k[j], 
                        cimag(in[j]-in[0]), creal(in[j]-in[0]) );
  }
  fclose(fh_out);
}

void write_outwz(fftw_complex* in, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3.re v 4. im v\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\t%.15e\n", u[j], k[j], 
                        creal(in[j]), cimag(in[j]) );
  }
  fclose(fh_out);
}

void write_combined(work_ptr wrk, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2.k 3. x 4. y 5., 6. v  7., 8. q 9., 10. q_t 11., 12. q_u\n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n", 
            u[j], k[j], creal(wrk->Z[j]), cimag(wrk->Z[j]), creal(wrk->V[j]), cimag(wrk->V[j]), creal(wrk->Q[j]), 
            cimag(wrk->Q[j]), creal(wrk->Q_t[j]), cimag(wrk->Q_t[j]), creal(wrk->Q_u[j]), cimag(wrk->Q_u[j]));
  }
  fclose(fh_out);
}

void write_combined2(work_ptr wrk, char* str, double time) {
  fh_out = fopen(str, "w");
  fprintf(fh_out, "#1.u 2., 3. r 4., 5. v \n");
  fprintf(fh_out, "# Current time = %.16e\n\n", time);
  for (int j = 0; j < N; j++) {
    fprintf(fh_out, "%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n", 
            u[j], creal(wrk->R[j]), cimag(wrk->R[j]), creal(wrk->V[j]), cimag(wrk->V[j]));
  }
  fclose(fh_out);
}


void stokes_wave( double tol, params_ptr in) {
  work_ptr wrk = in->w_ptr;
  fftw_complex num = 0, den = 0, M = 0; 
  double c = in->c;
  double b = 2.*g/pow(c,2), overN = 1./N;

  reconstruct_surface(wrk);
  fft(wrk->Z, a[0]);
  for (int j = 0; j < N; j++ ) a[0][j] = 1.I*cimag(a[0][j])*overN;
  ifft(a[0], wrk->Z); der(wrk->Z, wrk->Z_u);
  memcpy(a[3], wrk->Z_u, N*sizeof(fftw_complex));
  while (fabs(creal(M)-1) > tol) {
    for (int j = 0; j < N; j++) a[0][j] = b*cimag(wrk->Z[j])*(1. + a[3][j]);
    project(a[0], RS);
    fft(RS, a[1]);
    fft(a[3], a[2]); M = 0; num = 0; den = 0;
    for (int j = 0; j < N; j++) {
      num += (a[2][j])*(a[2][j]);
      den += (a[2][j])*(a[1][j]);  
    } 
    M = 1.*num/den; 
    for (int j = 0; j < N; j++) {
      a[3][j] = RS[j]*cpow(M, 1.5);
      wrk->Z_u[j] = 1. + a[3][j];
    }
    reconstruct_surface(wrk);
    mean_level(in);
    printf("%.16e\t%16e\n", creal(M), in->current.M);
  }
  printf("Writing Stokes wave to ini_stokes.dat\t");
  FILE *fhstk = fopen("ini_stokes.dat", "w");
  fprintf(fhstk,"%%# 1. u  2. re(R) 3. im(R) 4. re(Z) 5. im(Z)");
  for (int j = 0; j < N; j++) {   
    wrk->R[j] = 1./(wrk->Z_u[j]);
    wrk->V[j] = 1.I*c*(wrk->Z_u[j] - 1.)/(wrk->Z_u[j]);
    fprintf(fhstk, "%e\t%e\t%e\t%e\t%e\n", u[j], creal(wrk->R[j]), cimag(wrk->R[j]), creal(wrk->Z[j]), cimag(wrk->Z[j]));
  }
  fclose(fhstk);
  printf("Done\n");
}*/
