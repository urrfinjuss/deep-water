#include "header.h"

static fftw_complex *aux[4];
static fftw_complex *U,  *B, *Q0, *V0;
static fftw_complex *dU, *dQ, *dV;
static fftw_complex *rhq[7], *rhv[7];
static fftw_plan fwd_dft0, fwd_dft1, fwd_dft2, fwd_dft3;
static fftw_plan bwd_dft0, bwd_dft1, bwd_dft2, bwd_dft3;
static long int *k; 
static int wflag, benchflag;
static int n_order;
static params_ptr pms;
static work_ptr arr;
static double lq, lv;
static struct control current;
static pthread_barrier_t wbar;
static pthread_mutex_t tmut;
static double thresh = 13.5;
static int ref_flag = 0;
//static long int nsteps;



void refine() {
	long int max_steps = pms->max_steps;
	printf("Refinement at time %.9f on step %ld of %ld\n",current.T, pms->step_current, pms->max_steps);
	read_binary_data(pms, arr, "restart.bin");
	pms->max_steps = max_steps;
	printf("Restarting from %.9f on step %ld of %ld with double modes\n", current.T, pms->step_current, pms->max_steps);


	/*save_array(pms, arr->Q, "Qbf.txt");
	save_array(pms, arr->V, "Vbf.txt");*/
	save_fourier(pms, aux[0], "Qbf.spc");
	save_fourier(pms, aux[1], "Vbf.spc");

	double t1 = (pms->ionum)*(pms->dt);
	long int nsteps;

	printf("\nParameters before refinement:\nSpatial step = %.8e\nTemporal step = %.8e\nIO every %.8e\n", 1./(pms->n), 
					pms->dt, (pms->step_io)*(pms->dt));
	pms->n = 2*(pms->n);	
	pms->dt = (pms->dt)/3;
	if ((pms->max_steps) != (pms->step_current)) pms->max_steps = 3*(pms->max_steps - pms->step_current);

	printf("\nParameters after refinement:\nSpatial step = %.8e\nTemporal step = %.8e\nIO every %.8e\n", 1./(pms->n), 
					pms->dt, (pms->step_io)*(pms->dt));

	// reinit arrays
	memcpy(aux[0], arr->Q, sizeof(fftw_complex)*(pms->n)/2);
	memcpy(aux[1], arr->V, sizeof(fftw_complex)*(pms->n)/2);
	free(arr->Q);
	free(arr->V);
	fftw_execute(fwd_dft0);
	fftw_execute(fwd_dft1);
	for (int j = 0; j < (pms->n)/2; j++) {
	  aux[0][j] = 2.*aux[0][j]/(pms->n);
	  aux[1][j] = 2.*aux[1][j]/(pms->n);
	}
	free(aux[2]);
	free(aux[3]);
	aux[2] = malloc(sizeof(fftw_complex)*(pms->n));
	aux[3] = malloc(sizeof(fftw_complex)*(pms->n));
	fftw_plan pr2 = fftw_plan_dft_1d( pms->n, aux[2], aux[2], FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_plan pr3 = fftw_plan_dft_1d( pms->n, aux[3], aux[3], FFTW_BACKWARD, FFTW_ESTIMATE);
	memset(aux[2], 0, sizeof(fftw_complex)*(pms->n));
 	memset(aux[3], 0, sizeof(fftw_complex)*(pms->n));
	for (int j = (pms->n)-1; j > 3*(pms->n)/4; j--) {
	  aux[2][j] = aux[0][j-(pms->n)/2];
	  aux[3][j] = aux[1][j-(pms->n)/2];	
	}
	aux[2][0] = aux[0][0];
	aux[3][0] = aux[1][0];
	save_fourier(pms, aux[2], "Qaft.spc");
	save_fourier(pms, aux[3], "Vaft.spc");
	fftw_execute(pr2);
	fftw_execute(pr3);
	fftw_destroy_plan(pr2);
	fftw_destroy_plan(pr3);

	arr->Q = malloc(sizeof(fftw_complex)*(pms->n));
	arr->V = malloc(sizeof(fftw_complex)*(pms->n));
	memcpy(arr->Q, aux[2], sizeof(fftw_complex)*(pms->n));
	memcpy(arr->V, aux[3], sizeof(fftw_complex)*(pms->n));
	//save_array(pms, arr->Q, "Qaft.txt");
	//save_array(pms, arr->V, "Vaft.txt");
	free(aux[0]);	free(U);	free(B);
	free(aux[1]);	free(Q0);	free(V0);
	free(dU);	free(dQ);	free(dV);
	aux[0] = malloc(sizeof(fftw_complex)*(pms->n));
	aux[1] = malloc(sizeof(fftw_complex)*(pms->n));
	U = malloc(sizeof(fftw_complex)*(pms->n));
	B = malloc(sizeof(fftw_complex)*(pms->n));
	Q0 = malloc(sizeof(fftw_complex)*(pms->n));
	V0 = malloc(sizeof(fftw_complex)*(pms->n));

	dU = malloc(sizeof(fftw_complex)*(pms->n));
	dQ = malloc(sizeof(fftw_complex)*(pms->n));
	dV = malloc(sizeof(fftw_complex)*(pms->n));

    	for (int j = 0; j < pms->n_order+1; j ++) {
	  free(rhq[j]);
	  free(rhv[j]);
    	  rhq[j] = fftw_malloc(sizeof(fftw_complex)*(pms->n));
	  rhv[j] = fftw_malloc(sizeof(fftw_complex)*(pms->n));
	}
	free(k);
        k = fftw_malloc(sizeof(long int)*(pms->n));
    	for (int j = 0; j < pms->n; j++) {
	  k[j] = j;
	  if (j > (pms->n)/2) k[j] = j - (pms->n);
        }
	fftw_destroy_plan(fwd_dft0);
	fftw_destroy_plan(fwd_dft1);
	fftw_destroy_plan(bwd_dft0);
	fftw_destroy_plan(bwd_dft1);
	fftw_destroy_plan(bwd_dft2);
        fwd_dft0 = fftw_plan_dft_1d(pms->n, aux[0], aux[0], FFTW_FORWARD, FMODE);
        fwd_dft1 = fftw_plan_dft_1d(pms->n, aux[1], aux[1], FFTW_FORWARD, FMODE);
        bwd_dft0 = fftw_plan_dft_1d(pms->n, aux[0], aux[0], FFTW_BACKWARD, FMODE);
        bwd_dft1 = fftw_plan_dft_1d(pms->n, aux[1], aux[1], FFTW_BACKWARD, FMODE);
        bwd_dft2 = fftw_plan_dft_1d(pms->n, aux[2], aux[2], FFTW_BACKWARD, FMODE);
	printf("New Plans Created\n");
	//err_msg("Complete");

	

}

void reinit_arrays(){

}


double ref_criterion(fftw_complex *in) {
	int win = 16;	
	double qmax = 0;
	double qmin = 0;
	for (int j = 0; j < win; j++) {
	  qmax += cabs(in[(pms->n)-j-1]);
	  qmin += cabs(in[(pms->n)/2+(pms->n)/32+j]);
	}
	return log10(qmax/qmin);
}


void restore_surface(char *fname1, char *fname2) {
	for (int j = 0; j < pms->n; j++) aux[1][j] = cpow(arr->Q[j],-2);
	fftw_execute(fwd_dft1);
	for (int j = 1; j < pms->n; j++) aux[1][j] = -1.I*aux[1][j]/(pms->n)/k[j];	
	aux[1][0] = 0.;
	for (int j = 1; j < (pms->n)/2; j++) aux[2][j] = 0.5I*conj(aux[1][(pms->n)-j]); // aux[1][j] = 0!!
	for (int j = (pms->n)/2-1; j > 0; j--) aux[1][0] += - 2.I*fabs(k[j])*cabs(aux[2][j])*cabs(aux[2][j]);
	save_fourier(pms, aux[1], fname2);
	fftw_execute(bwd_dft1);
	save_array(pms, aux[1], fname1);
	//memcpy(Z, aux[1], sizeof(fftw_complex)*(pms->n));
}

void bench_rk4() {
/* 
	Benchmark the time difference to complete between serial and threaded 
   	Runge-Kutta 4 iterations. Number of threads for DFT are given by $nthreads
   	input parameter 
*/
	memcpy(aux[3], arr->Q, sizeof(fftw_complex)*(pms->n));
	memcpy(aux[3], arr->V, sizeof(fftw_complex)*(pms->n));
	struct timespec start, finish;
      	double elapsed_serial, elapsed_threads, elapsed_threads_6;

	bench_fft(pms, arr);
	printf("Running serial RK4 ... ");

	clock_gettime(CLOCK_MONOTONIC, &start);
	evolve_rk4_serial();
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_serial = (finish.tv_sec - start.tv_sec);
	elapsed_serial += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Complete\nRunning threaded RK4 ... ");

	memcpy(arr->Q, aux[3], sizeof(fftw_complex)*(pms->n));	
	memcpy(arr->V, aux[3], sizeof(fftw_complex)*(pms->n));

	pms->n_order = 4;
	clock_gettime(CLOCK_MONOTONIC, &start);
	evolve_rk_threads();
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_threads = (finish.tv_sec - start.tv_sec);
	elapsed_threads += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Complete\nRunning threaded RK4 ... ");

	memcpy(arr->Q, aux[3], sizeof(fftw_complex)*(pms->n));	
	memcpy(arr->V, aux[3], sizeof(fftw_complex)*(pms->n));

	pms->n_order = 6;
	clock_gettime(CLOCK_MONOTONIC, &start);
	evolve_rk_threads();
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed_threads_6 = (finish.tv_sec - start.tv_sec);
	elapsed_threads_6 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("Complete\n\n");
	printf("Benchmark results:\n");
	printf("Serial RK4: %.8f s \n", elapsed_serial);
	printf("Threaded RK4: %.8f s \n", elapsed_threads);
	printf("Threaded RK6: %.8f s \n", elapsed_threads_6);

}

void evolve_rk_threads(){
	void *status;
	pthread_t tid[2];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        //while (pms->step_current < pms->max_steps) {
        while (current.T <= pms->tmax) {
          pthread_barrier_init(&wbar, NULL, 2);	
	  if ((pms->n_order) == 4) {
		pthread_create(&tid[0], &attr, &pthread_id0, NULL);
		pthread_create(&tid[1], &attr, &pthread_id1, NULL);
	  } else if ((pms->n_order) == 6) {
		pthread_create(&tid[0], &attr, &pthread_id0_rk6, NULL);
		pthread_create(&tid[1], &attr, &pthread_id1_rk6, NULL);
	  } else err_msg("Method order unknown. Currently only Runge-Kutta 4 and 6 are supported.");
	  pthread_join(tid[0], NULL);
	  pthread_join(tid[1], NULL);
          pthread_barrier_destroy(&wbar);
	  if (ref_flag == 1) refine();
	}
	printf("Simulation Complete. Final time %.9f on step %ld of %ld\n", current.T, pms->step_current, pms->max_steps);

}

void prepare_id0() {
	// thread zero does 4 DFTs, 
	// processes dQ, B with surface tension
        memcpy(aux[0], arr->Q, sizeof(fftw_complex)*(pms->n));
        fftw_execute(fwd_dft0);
	for (int j = 0; j < pms->n; j++) {
	  aux[0][j] = 1.I*k[j]*aux[0][j]/(pms->n);
	}
	fftw_execute(bwd_dft0);
	memcpy(dQ, aux[0], sizeof(fftw_complex)*(pms->n));
	for (int j = 0; j < pms->n; j ++) {
	  aux[0][j] = creal(arr->V[j]*conj(arr->V[j]) + 4.*(pms->sigma)*cimag(dQ[j]*conj(arr->Q[j])) );
	}
        fftw_execute(fwd_dft0);
	memset(&aux[0][1], 0, sizeof(fftw_complex)*(pms->n)/2);
        for (int j = 0; j < pms->n; j++) {
	  aux[0][j] = 1.I*k[j]*aux[0][j]/(pms->n);
	}
        fftw_execute(bwd_dft0);
	memcpy(B, aux[0], sizeof(fftw_complex)*(pms->n));

}
void prepare_id1() {  	
	// thread one does 5 DFTs, 
	// processes dV, U and dU
        memcpy(aux[1], arr->V, sizeof(fftw_complex)*(pms->n));
        fftw_execute(fwd_dft1);
	for (int j = 0; j < pms->n; j++) {
	  aux[1][j] = 1.I*k[j]*aux[1][j]/(pms->n);
	}
        fftw_execute(bwd_dft1);
	memcpy(dV, aux[1], sizeof(fftw_complex)*(pms->n));
	for (int j = 0; j < pms->n; j ++) {
	  aux[1][j] = 2.*creal(conj(arr->V[j])*(arr->Q[j])*(arr->Q[j]));
	}
        fftw_execute(fwd_dft1);
	memset(&aux[1][1], 0, sizeof(fftw_complex)*(pms->n)/2);
	for (int j = 0; j < pms->n; j++) {
	  aux[1][j] = aux[1][j]/(pms->n);
	  aux[2][j] = 1.I*k[j]*aux[1][j];
	}
	fftw_execute(bwd_dft1);
	fftw_execute(bwd_dft2);
	memcpy( U, aux[1], sizeof(fftw_complex)*(pms->n));
	memcpy(dU, aux[2], sizeof(fftw_complex)*(pms->n));

}

//---------------------  Classical Runge Kutta 4

void * pthread_id0(void *arg) {
	long int mm = pms->step_start;
	// process arrays for Q in RK4
	pthread_barrier_wait(&wbar);
	//while (mm < pms->max_steps) {
	while (current.T <= pms->tmax) {
	  memcpy(Q0, arr->Q, sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);
	  prepare_id0();
	  pthread_barrier_wait(&wbar);
  	  for (int j = 0; j < pms->n; j++) rhq[0][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);
  	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + 0.5*(pms->dt)*rhq[0][j];
	  pthread_barrier_wait(&wbar);
	  prepare_id0();
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) rhq[1][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + 0.5*(pms->dt)*rhq[1][j];
	  pthread_barrier_wait(&wbar);
	  prepare_id0();
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) rhq[2][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + (pms->dt)*rhq[2][j];
	  pthread_barrier_wait(&wbar);
	  prepare_id0();
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) {
	    rhq[3][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	    aux[0][j] = (Q0[j] + ((pms->dt)/6.)*(rhq[0][j]+2.*rhq[1][j]+2.*rhq[2][j]+rhq[3][j]))/(pms->n);
	  }
          fftw_execute(fwd_dft0);
	  lq = ref_criterion(aux[0]);
	  pthread_barrier_wait(&wbar);
	  if ((lv < thresh)||(lq < thresh)) {
		break;
	  }
	  pthread_barrier_wait(&wbar);
	  memset(&aux[0][1], 0, sizeof(fftw_complex)*(pms->n)/2);
          fftw_execute(bwd_dft0);
 	  memcpy(arr->Q, aux[0], sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);
	  mm += 1;
	  pthread_barrier_wait(&wbar);
	}

}

void * pthread_id1(void *arg) {
	char str[80], str_fourier[80];
	pms->step_current = pms->step_start;
	current.incr_step = 0;
	// process arrays for V in RK4

	//printf("%ld\t%.9f\t%ld\n", nn, current.T, nn/(pms->step_io));
	printf("T = %.9f\t%ld\n", current.T, (pms->step_current)/(pms->step_io));
	sprintf(str, "data/Z%03ld.txt", (pms->step_current)/(pms->step_io));
	sprintf(str_fourier, "data/Z%03ld.spc", (pms->step_current)/(pms->step_io));
	restore_surface(str, str_fourier);

	pthread_barrier_wait(&wbar);
	//while (pms->step_current < pms->max_steps) {
	while (current.T <= pms->tmax) {
	  memcpy(V0, arr->V, sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);
	  prepare_id1();
	  pthread_barrier_wait(&wbar);
  	  for (int j = 0; j < pms->n; j++) rhv[0][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.); 
	  pthread_barrier_wait(&wbar);
  	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + 0.5*(pms->dt)*rhv[0][j]; 
	  pthread_barrier_wait(&wbar);
	  prepare_id1();
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) rhv[1][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + 0.5*(pms->dt)*rhv[1][j]; 
	  pthread_barrier_wait(&wbar);
	  prepare_id1();
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) rhv[2][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + (pms->dt)*rhv[2][j]; 
	  pthread_barrier_wait(&wbar);
	  prepare_id1();
	  pthread_barrier_wait(&wbar);
	  for (int j = 0; j < pms->n; j++) {
	    rhv[3][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	    aux[1][j] = (V0[j] + ((pms->dt)/6.)*(rhv[0][j]+2.*rhv[1][j]+2.*rhv[2][j]+rhv[3][j]))/(pms->n);
	  }
          fftw_execute(fwd_dft1);
	  lv = ref_criterion(aux[1]);
	  pthread_barrier_wait(&wbar);
	  if ((lv < thresh)||(lq < thresh)) {
		printf("Spectrum too wide.\n");
		ref_flag = 1;
		break;
	  }
	  pthread_barrier_wait(&wbar);
	  memset(&aux[1][1], 0, sizeof(fftw_complex)*(pms->n)/2);
          fftw_execute(bwd_dft1);
	  memcpy(arr->V, aux[1], sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);
	  pms->step_current += 1;
	  current.incr_step += 1;
	  current.T = current.t_stop+(current.incr_step)*(pms->dt);
	  if ( (pms->step_current % (pms->step_io) ) == 0) {
		pms->step_start = pms->step_current;
		current.incr_step = 0;
		current.t_stop = current.T;
	        sprintf(str, "data/Z%03ld.txt", (pms->step_current)/(pms->step_io));
	        sprintf(str_fourier, "data/Z%03ld.spc", (pms->step_current)/(pms->step_io));
		restore_surface(str, str_fourier);
	 	printf("T = %.9f\t%ld\t Max to min ratios Qk = %.5f\tVk = %.5f\tWriting binary restart ... ", current.T, (pms->step_current)/(pms->step_io), lq, lv);
		fflush(stdout);
		save_binary_data(pms, arr, "restart.bin");
		printf("Complete\n");
	  }

	  pthread_barrier_wait(&wbar);
	}
	if (ref_flag == 1) printf("Refining\n");
	//printf("Final time threaded %.15f\n", current.T);
	//sprintf(str, "data/out%03d.txt", nn/(pms->ionum));
	//save_ascii(pms, arr, str); // threaded iterations
}


void evolve_rk4_serial(){
	char str[80];
	long int nn = pms->step_start;
	long int mm = 0;
	long int kk = 0;

	//sprintf(str, "out%03d.txt", 0);
	//save_ascii(pms, arr, str);
	current.T = 0; 
	while (nn < pms->max_steps) {
          memcpy(Q0, arr->Q, sizeof(fftw_complex)*(pms->n));
          memcpy(V0, arr->V, sizeof(fftw_complex)*(pms->n));
	  proc_one_serial(pms, arr);
  	  for (int j = 0; j < pms->n; j++) {
	    rhq[0][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	    rhv[0][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.); 
	    arr->Q[j] = Q0[j] + 0.5*(pms->dt)*rhq[0][j];
	    arr->V[j] = V0[j] + 0.5*(pms->dt)*rhv[0][j]; 
	  }
	  proc_one_serial(pms, arr);	
	  for (int j = 0; j < pms->n; j++) {
	    rhq[1][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	    rhv[1][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	    arr->Q[j] = Q0[j] + 0.5*(pms->dt)*rhq[1][j];
	    arr->V[j] = V0[j] + 0.5*(pms->dt)*rhv[1][j]; 
	  }
	  proc_one_serial(pms, arr);	
	  for (int j = 0; j < pms->n; j++) {
	    rhq[2][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	    rhv[2][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	    arr->Q[j] = Q0[j] + (pms->dt)*rhq[2][j];
	    arr->V[j] = V0[j] + (pms->dt)*rhv[2][j]; 
	  }
	  proc_one_serial(pms, arr);	
	  for (int j = 0; j < pms->n; j++) {
	    rhq[3][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	    rhv[3][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	    arr->Q[j] = Q0[j] + ((pms->dt)/6.)*(rhq[0][j]+2.*rhq[1][j]+2.*rhq[2][j]+rhq[3][j]);
	    arr->V[j] = V0[j] + ((pms->dt)/6.)*(rhv[0][j]+2.*rhv[1][j]+2.*rhv[2][j]+rhv[3][j]);
	  }
	  memcpy(aux[0],arr->Q, sizeof(fftw_complex)*(pms->n));
	  memcpy(aux[1],arr->V, sizeof(fftw_complex)*(pms->n));
          fftw_execute(fwd_dft0);
          fftw_execute(fwd_dft1);
	  memset(&aux[0][1], 0, sizeof(fftw_complex)*(pms->n)/2);
	  memset(&aux[1][1], 0, sizeof(fftw_complex)*(pms->n)/2);
	  for (int j = 0; j < pms->n; j++) {
	    aux[0][j] = aux[0][j]/(pms->n);
	    aux[1][j] = aux[1][j]/(pms->n);
	  }
          fftw_execute(bwd_dft0);
          fftw_execute(bwd_dft1);
	  memcpy(arr->Q, aux[0], sizeof(fftw_complex)*(pms->n));
	  memcpy(arr->V, aux[1], sizeof(fftw_complex)*(pms->n));
          nn += 1;	
	  current.T = nn*(pms->dt);
         /* if (nn > mm) {
	    kk += 1;
	    sprintf(str, "out%03ld.txt", kk);
	    printf("T = %.8f\n", current.T);
	    save_ascii(pms, arr, str);
	    mm += (pms->skip);	
	  }*/
	}
	//printf("Final time threaded %.15f\n", current.T);
	save_ascii(pms, arr, "serial_last.txt"); // serial iterations
}


void proc_one_serial() {
	for (int j = 0; j < pms->n; j ++) {
	  U[j] = 2.*creal(conj(arr->V[j])*(arr->Q[j])*(arr->Q[j]));
	}	

	memcpy(aux[0], arr->Q, sizeof(fftw_complex)*(pms->n));	
	memcpy(aux[1], U, sizeof(fftw_complex)*(pms->n));
        fftw_execute(fwd_dft0);
        fftw_execute(fwd_dft1);
        memset(&aux[1][1], 0, sizeof(fftw_complex)*(pms->n)/2);

	for (int j = 0; j < pms->n; j++) {
	  aux[0][j] = 1.I*k[j]*aux[0][j]/(pms->n);
	  aux[1][j] = aux[1][j]/(pms->n);
	  aux[2][j] = 1.I*k[j]*aux[1][j];
	}
	fftw_execute(bwd_dft0);
	fftw_execute(bwd_dft1);
	fftw_execute(bwd_dft2);

	memcpy(dQ, aux[0], sizeof(fftw_complex)*(pms->n));
	memcpy( U, aux[1], sizeof(fftw_complex)*(pms->n));
	memcpy(dU, aux[2], sizeof(fftw_complex)*(pms->n));

	for (int j = 0; j < pms->n; j ++) {
	  aux[0][j] = creal(arr->V[j]*conj(arr->V[j]) + 4.*(pms->sigma)*cimag(dQ[j]*conj(arr->Q[j])));
	}
        memcpy(aux[1], arr->V, sizeof(fftw_complex)*(pms->n));
        
        fftw_execute(fwd_dft0);
        fftw_execute(fwd_dft1);
        memset(&aux[0][1], 0, sizeof(fftw_complex)*(pms->n)/2);
        for (int j = 0; j < pms->n; j++) {
	  aux[0][j] = 1.I*k[j]*aux[0][j]/(pms->n);
	  aux[1][j] = 1.I*k[j]*aux[1][j]/(pms->n);
	}
        fftw_execute(bwd_dft0);
        fftw_execute(bwd_dft1);

	memcpy( B, aux[0], sizeof(fftw_complex)*(pms->n));
	memcpy(dV, aux[1], sizeof(fftw_complex)*(pms->n));

	/*save_array(pms, arr->Q,  "Q.txt");
	save_array(pms, dQ,"dQ.txt");
	save_array(pms, U,  "U.txt");
	save_array(pms, dU,"dU.txt");
	save_array(pms, B,  "B.txt");
	save_array(pms, arr->V,"V.txt");
	save_array(pms, dV, "dV.txt");*/
}

void read_pole_data(params_ptr in, work_ptr wrk) {
	char val[256], line[256], *dm, tmp[256], p1[96], p2[96];
	double u, y, chi[in->d_poles], gam[in->d_poles];
	FILE *fh = fopen(in->resname,"r");
	if (fh == NULL) err_msg("Cannot read restart from poles.");
	for (int j = 0; j < 2; j++) dm = fgets(line, 256, fh); 	
	sscanf(line, "# N = %s\tL2_residual = %s\ty0 = %s\n", tmp, tmp, val);
	y = strtod(val, NULL); dm = fgets(line, 256, fh); 	
	for (int j = 0; j < in->d_poles; j++) {	
	  dm = fgets(line, 256, fh);
	  sscanf(line, "%s\t%s\t%s\t%s\n", p1, p2, tmp, tmp);
	  chi[j] = strtod(p1,NULL);
	  gam[j] = strtod(p2,NULL);
	  //printf("%s", line);
	}
	fclose(fh);
  	//err_msg("Complete");
	for (int j = 0; j < in->n; j++) {
	  U[j] = 1.I*y;
	  B[j] = 1.;
	  u = pi*(2.*j/(in->n) - 1);
	  for (int p = 0; p < in->d_poles; p++) {
	    U[j] += gam[p]/(tan(0.5*u) - 1.I*chi[p]);
	    B[j] += -0.5*gam[p]/cpow(sin(0.5*u) - 1.I*chi[p]*cos(0.5*u), 2);
	  }
	  wrk->V[j] = 1.I*(in->c)*(1. - 1./B[j]);
          B[j] += 0.5*(in->res1)*cexp(1.I*pi*(in->phs1))/cpow(sin(0.5*(u+pi-(in->sft1)*pi))-1.I*(in->pos1)*cos(0.5*(u+pi-(in->sft1)*pi)),2); 
	  wrk->Q[j] = 1./csqrt(B[j]);
	}
	printf("Succesfully read Pade data for Stokes wave\n");
	save_binary_data(in, wrk, "restart.bin");
}


void bench_fft(params_ptr in, work_ptr wrk) {
	struct timespec start, finish;
      	double elapsed;

	clock_gettime(CLOCK_MONOTONIC, &start);
	int mm = 10000;
        for (int j = 0; j < mm; j++) {
	   fftw_execute(fwd_dft0);
	   if ((j % 20) == 0) memset(aux[0], 0, sizeof(fftw_complex));
        }
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	printf("Threaded 1D Complex DFT %d modes on %d thread(-s)\n", in->n, in->nthreads);
	printf("Execution time for 1 DFT is %.8f ms \n", elapsed*1000/mm);
}

void init_fftw(params_ptr in, work_ptr wrk){
	char str[80];
	fftw_init_threads();
	fftw_plan_with_nthreads(in->nthreads);
	sprintf(str, "dft.fftw.%d", in->nthreads);
	fftw_import_wisdom_from_filename(str);
	printf("Creating Wisdom .... ");
	fflush(stdout);
        fwd_dft0 = fftw_plan_dft_1d( in->n, aux[0], aux[0], FFTW_FORWARD, FMODE);
        fwd_dft1 = fftw_plan_dft_1d( in->n, aux[1], aux[1], FFTW_FORWARD, FMODE);
        bwd_dft0 = fftw_plan_dft_1d( in->n, aux[0], aux[0], FFTW_BACKWARD, FMODE);
        bwd_dft1 = fftw_plan_dft_1d( in->n, aux[1], aux[1], FFTW_BACKWARD, FMODE);
        bwd_dft2 = fftw_plan_dft_1d( in->n, aux[2], aux[2], FFTW_BACKWARD, FMODE);
	if(fftw_export_wisdom_to_filename(str)!=0) printf("Complete.\nExported Wisdom to %s.\n", str);
	if (benchflag == 1) {
	  bench_rk4(in, wrk);
          err_msg("\n");
	}
	//fftw_cleanup_threads();
}

void init_arrays(params_ptr in, work_ptr wrk){
    // initial structure arrays
    wrk->Q = fftw_malloc(sizeof(fftw_complex)*(in->n));
    wrk->V = fftw_malloc(sizeof(fftw_complex)*(in->n));
    // initialize static arrays
    Q0 = fftw_malloc(sizeof(fftw_complex)*(in->n));
    V0 = fftw_malloc(sizeof(fftw_complex)*(in->n));
    k = fftw_malloc(sizeof(long int)*(in->n));
    B = fftw_malloc(sizeof(fftw_complex)*(in->n));
    U = fftw_malloc(sizeof(fftw_complex)*(in->n));
    //Z = fftw_malloc(sizeof(fftw_complex)*(in->n));
    dU = fftw_malloc(sizeof(fftw_complex)*(in->n));
    dQ = fftw_malloc(sizeof(fftw_complex)*(in->n));
    dV = fftw_malloc(sizeof(fftw_complex)*(in->n));
    for (int j = 0; j < 4; j ++) {
    	aux[j] = fftw_malloc(sizeof(fftw_complex)*(in->n));
    }
    for (int j = 0; j < in->n_order+1; j ++) {
    	rhq[j] = fftw_malloc(sizeof(fftw_complex)*(in->n));
	rhv[j] = fftw_malloc(sizeof(fftw_complex)*(in->n));
    }
    for (int j = 0; j < in->n; j++) {
	k[j] = j;
	if (j > (in->n)/2) k[j] = j - (in->n);
    }
}


void init_input(int *argc, char **argv, params_ptr in, work_ptr wrk) {
	char str[80], line[80], value[80], param[80], dm[80], lval[80];
        int threads_old, n_old, num;
  	sprintf(str, "Usage:\n\t%s input", argv[0]);
  	if (*argc != 2) err_msg(str);
  	sprintf(str, "%s.cfg", argv[1]);
  	FILE *fh = fopen(str,"r"); 
  	if (fh == NULL) err_msg("Cannot open file");
  	while (fgets(line, 80, fh)!=NULL) {
    		sscanf(line, "%s\t%s", param, value);
	    	if (strcmp(param,"#runname=") == 0) sprintf(in->runname,"%s", value);
    		if (strcmp(param,"#resname=") == 0) sprintf(in->resname,"%s", value);
    		if (strcmp(param,"#npoints=") == 0) in->n = atoi(value);
    		if (strcmp(param,"#kcutoff=") == 0) in->k_cut = atoi(value);
    		if (strcmp(param,"#cfl_con=") == 0) in->cfl = atof(value);
    		if (strcmp(param,"#gravity=") == 0) in->g = atof(value);
    		if (strcmp(param,"#velocit=") == 0) in->c = atof(value);
    		if (strcmp(param,"#fintime=") == 0) in->tmax = atof(value);
    		if (strcmp(param,"#skip_it=") == 0) in->skip = atoi(value);
    		if (strcmp(param,"#data_pr=") == 0) in->ionum = atoi(value);
    		if (strcmp(param,"#readdat=") == 0) in->rflag = atoi(value);
    		if (strcmp(param,"#readpls=") == 0) in->pflag = atoi(value);
    		if (strcmp(param,"#n_poles=") == 0) in->d_poles = atoi(value);
    		if (strcmp(param,"#pertres=") == 0) in->res1 = atof(value);
    		if (strcmp(param,"#pertphs=") == 0) in->phs1 = atof(value);
    		if (strcmp(param,"#pertpos=") == 0) in->pos1 = atof(value);
    		if (strcmp(param,"#pertsft=") == 0) in->sft1 = atof(value);
    		if (strcmp(param,"#vprtres=") == 0) in->res2 = atof(value);
    		if (strcmp(param,"#vprtphs=") == 0) in->phs2 = atof(value);
    		if (strcmp(param,"#threads=") == 0) in->nthreads = atoi(value);
    		if (strcmp(param,"#nmorder=") == 0) in->n_order = atoi(value);
    		if (strcmp(param,"#surftsn=") == 0) in->sigma = atof(value);
    		if (strcmp(param,"#benchft=") == 0) benchflag = atoi(value);
  	}
	fclose(fh);
  	FILE *fhlog = fopen("run.log","w"); 
	fprintf(fhlog, "Run Name: %s\n", in->runname );
	fprintf(fhlog, "Benchmark FFTs run (yes/no) %d\n", benchflag);
  	fprintf(fhlog, "Number Fourier modes at start\t%d\n", in->n );
  	fprintf(fhlog, "Sampling Fourier mode is # %d\n", in->k_cut );
  	fprintf(fhlog, "CFL condition = %f\n", in->cfl );
  	fprintf(fhlog, "Free Fall Acceleration is %.5f\n", in->g );
  	fprintf(fhlog, "Surface tension sigma %.5e\n", in->sigma );
  	fprintf(fhlog, "Number of threads to use\t%d\n", in->nthreads);
  	fprintf(fhlog, "Order of the time integrator\t%d\n", in->n_order);
  	fprintf(fhlog, "Restart Name: %s\n", in->resname );
  	fprintf(fhlog, "Velocity of Stokes wave = %.14e\n", in->c );
  	fprintf(fhlog, "T_max = %f\n", in->tmax );
  	fprintf(fhlog, "Read Restart data from restart file (yes/no) %d\n", in->rflag );
  	fprintf(fhlog, "Read Restart data from poles (yes/no) %d\n", in->pflag );
        fprintf(fhlog, "Perturbation in Z':\n");
  	fprintf(fhlog, "Magnitude of perturbation residue is %.5e\n", in->res1 );
  	fprintf(fhlog, "Phase of perturbation residue is %.5e\n", in->phs1);
  	fprintf(fhlog, "Position of perturbation above the real axis %.5e\n", in->phs1);
  	fprintf(fhlog, "Shift of perturbation on the horizontal %.5e\n", in->sft1);
        fprintf(fhlog, "Perturbation in complex velocity V:\n");
	fprintf(fhlog, "Magnitude of perturbation residue is %.5e\n", in->res2 );
  	fprintf(fhlog, "Phase of perturbation residue is %.5e\n", in->phs2);
	fclose(fhlog);
        init_arrays(in, wrk);
        if ((in->rflag)&&(in->pflag)) err_msg("Make up your mind! Pick type of file to read.");
	(in->current) = &current;
	arr = wrk;  pms = in;
  	n_order = pms->n_order;
	long double t1;
	long int nsteps;
	if (in->pflag == 1) {
		printf("Reading data from %s\n", in->resname);
		read_pole_data(in, wrk);
		current.T = 0;
		current.t_stop = 0.;
		t1 = ((pms->tmax)-current.T)/(pms->ionum);
		pms->dt = (pms->cfl)/(pms->n);
		nsteps = floor(t1/(pms->dt))+1;
		pms->dt = t1/nsteps;
		pms->cfl = (pms->n)*(pms->dt);
		pms->step_start = 0;
		pms->step_current = 0;
		pms->step_io = nsteps;
		pms->max_steps = (pms->ionum)*nsteps;
	}
	if (in->rflag == 1) {
		printf("Reading data from binary %s\n", in->resname);
		read_binary_data(in, wrk, in->resname);
		printf("Starting at t = %.9f\n", current.T);
		printf("Starting step is %ld\nI/O every %ld steps\n", pms->step_start, pms->step_io);

	}
	//dt = pms->dt;
	//nsteps = (pms->ionum)*nsteps;
	printf("Total number of steps: %ld\nOutput every %ld steps\n", pms->max_steps, pms->step_io);

  	fhlog = fopen("run.log","a"); 
	fprintf(fhlog, "Actual CFL value = %.9f\n", pms->cfl );
	fprintf(fhlog, "Actual time-step = %.9f\n", pms->dt );
	fclose(fhlog);


}


//---------------------  Runge Kutta 6

void * pthread_id0_rk6(void *arg) {
	long int mm = 0;
	// process arrays for Q in RK6
	pthread_barrier_wait(&wbar); // 1
	while (mm < pms->max_steps) {
	  memcpy(Q0, arr->Q, sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);  //2
	  prepare_id0();
	  pthread_barrier_wait(&wbar);  //3
  	  for (int j = 0; j < pms->n; j++) rhq[0][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);  //4
  	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + ((pms->dt)/3.)*rhq[0][j];
	  pthread_barrier_wait(&wbar);  //5 
	  prepare_id0();
	  pthread_barrier_wait(&wbar);  //6
	  for (int j = 0; j < pms->n; j++) rhq[1][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);  //7
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + (2.*(pms->dt)/3.)*rhq[1][j];
	  pthread_barrier_wait(&wbar);  //8
	  prepare_id0();
	  pthread_barrier_wait(&wbar);  //9
	  for (int j = 0; j < pms->n; j++) rhq[2][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);  //10
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + ((pms->dt)/12.)*(rhq[0][j]+4.*rhq[1][j]-rhq[2][j]);
	  pthread_barrier_wait(&wbar);  //11
	  prepare_id0();
	  pthread_barrier_wait(&wbar);  //12
	  for (int j = 0; j < pms->n; j++) rhq[3][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);  //13
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + ((pms->dt)/16.)*(-rhq[0][j]+18.*rhq[1][j]-3.*rhq[2][j]-6.*rhq[3][j]);
	  pthread_barrier_wait(&wbar);	//14
	  prepare_id0();
	  pthread_barrier_wait(&wbar);	//15
	  for (int j = 0; j < pms->n; j++) rhq[4][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);	//16
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + ((pms->dt)/8.)*(9.*rhq[1][j]-3.*rhq[2][j]-6.*rhq[3][j]+4.*rhq[4][j]);
	  pthread_barrier_wait(&wbar);	//17
	  prepare_id0();
	  pthread_barrier_wait(&wbar);	//18
	  for (int j = 0; j < pms->n; j++) rhq[5][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	  pthread_barrier_wait(&wbar);	//19
	  for (int j = 0; j < pms->n; j++) arr->Q[j] = Q0[j] + ((pms->dt)/44.)*(9.*rhq[0][j]-36.*rhq[1][j]+63.*rhq[2][j]+72.*rhq[3][j]-64.*rhq[5][j]);
	  pthread_barrier_wait(&wbar);	//20
	  prepare_id0();
	  pthread_barrier_wait(&wbar);	//21
	  for (int j = 0; j < pms->n; j++) {
	    rhq[6][j] = 0.5I*(2*dQ[j]*U[j] - arr->Q[j]*dU[j]);
	    aux[0][j] = (Q0[j] + ((pms->dt)/120.)*(11.*(rhq[0][j] + rhq[6][j]) + 81.*(rhq[2][j]+rhq[3][j]) -32.*(rhq[4][j]+rhq[5][j]))   )/(pms->n);
	  }
          fftw_execute(fwd_dft0);
	  memset(&aux[0][1], 0, sizeof(fftw_complex)*(pms->n)/2);
          fftw_execute(bwd_dft0);
 	  memcpy(arr->Q, aux[0], sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);	//22
	  mm += 1;
	  pthread_barrier_wait(&wbar);	//23
	}

}

void * pthread_id1_rk6(void *arg) {
	char str[80];
	long int nn = 0;
	// process arrays for V in RK6

	save_ascii(pms, arr, "data/out000.txt");
	pthread_barrier_wait(&wbar);	//1
	while (nn < pms->max_steps) {
	  memcpy(V0, arr->V, sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);	//2
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//3
  	  for (int j = 0; j < pms->n; j++) rhv[0][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.); 
	  pthread_barrier_wait(&wbar);	//4
  	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + ((pms->dt)/3.)*rhv[0][j]; 
	  pthread_barrier_wait(&wbar);	//5
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//6
	  for (int j = 0; j < pms->n; j++) rhv[1][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);	//7
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + (2.*(pms->dt)/3.)*rhv[1][j]; 
	  pthread_barrier_wait(&wbar);	//8
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//9
	  for (int j = 0; j < pms->n; j++) rhv[2][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);	//10
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + ((pms->dt)/12.)*(rhv[0][j]+4.*rhv[1][j]-rhv[2][j]); 
	  pthread_barrier_wait(&wbar);	//11
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//12
	  for (int j = 0; j < pms->n; j++) rhv[3][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);	//13
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + ((pms->dt)/16.)*(-rhv[0][j]+18.*rhv[1][j]-3.*rhv[2][j]-6.*rhv[3][j]);
	  pthread_barrier_wait(&wbar);	//14
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//15
	  for (int j = 0; j < pms->n; j++) rhv[4][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);	//16
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + ((pms->dt)/8.)*(9.*rhv[1][j]-3.*rhv[2][j]-6.*rhv[3][j]+4.*rhv[4][j]);
	  pthread_barrier_wait(&wbar);	//17
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//18
	  for (int j = 0; j < pms->n; j++) rhv[5][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	  pthread_barrier_wait(&wbar);	//19
	  for (int j = 0; j < pms->n; j++) arr->V[j] = V0[j] + ((pms->dt)/44.)*(9.*rhv[0][j]-36.*rhv[1][j]+63.*rhv[2][j]+72.*rhv[3][j]-64.*rhv[5][j]);
	  pthread_barrier_wait(&wbar);	//20
	  prepare_id1();
	  pthread_barrier_wait(&wbar);	//21

	  for (int j = 0; j < pms->n; j++) {
	    rhv[6][j] = 1.I*(U[j]*dV[j] - B[j]*(arr->Q[j])*(arr->Q[j])) + (pms->g)*((arr->Q[j])*(arr->Q[j]) - 1.);
	    aux[1][j] = (V0[j] + ((pms->dt)/120.)*(11.*(rhv[0][j] + rhv[6][j]) + 81.*(rhv[2][j]+rhv[3][j]) -32.*(rhv[4][j]+rhv[5][j]))   )/(pms->n);
	  }
          fftw_execute(fwd_dft1);
	  memset(&aux[1][1], 0, sizeof(fftw_complex)*(pms->n)/2);	
          fftw_execute(bwd_dft1);
	  memcpy(arr->V, aux[1], sizeof(fftw_complex)*(pms->n));
	  pthread_barrier_wait(&wbar);	//22
	  nn += 1;
	  current.T = nn*(pms->dt);
	  if ( (nn % (pms->step_io) ) == 0) {
		pms->step_start = nn;
	        sprintf(str, "data/out%03ld.txt", nn/(pms->step_io));
		save_ascii(pms, arr, str);
	 	printf("%ld\t%.9f\t%ld\tWriting binary restart ... ", nn, current.T, nn/(pms->step_io));
		fflush(stdout);
		save_binary_data(pms, arr, "restart.bin");
		printf("Complete\n");
		
	  }

	  pthread_barrier_wait(&wbar);	//23
	}
	printf("Final time threaded %.15f\n", current.T);
	//sprintf(str, "data/out%03d.txt", nn/(pms->ionum));
	//save_ascii(pms, arr, "rk6.txt"); // threaded iterations
}
