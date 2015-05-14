#include "header.h"

int main(int argc, char** argv)
{
  work	  wrk;
  params  ctrl;
  char	  s1[80], s2[80];  
  int 	  it = 0, j_out = 0, reflag, rff = 0; 
  double  P0x, P0y, H0;
  //FILE 	  *fh = fopen("time_dep.txt", "w");

  init_input(&argc, argv, &ctrl, &wrk);
  init_fftw(&ctrl, &wrk);
  save_ascii(&ctrl, &wrk, "data.txt");

  
  evolve_rk_threads();

  //proc_one_serial(&ctrl, &wrk);
  


  /*pthread_zero(&ctrl, &wrk);
  pthread_one(&ctrl, &wrk);
  pthread_two(&ctrl, &wrk);*/
  
  /*
  init(&argc, argv, &ctrl, &wrk);
  init_prep(&ctrl); ctrl.current.T = 0.;
  reconstruct_surface(&wrk);
  write_out(wrk.V, "vstart.txt", 0);

  if (ctrl.stokes != 0.) { 
    stokes_wave(1e-15, &ctrl);
    generate_derivatives(&wrk, ctrl.d_cnt);
  }

  fftw_complex* rk  = fftw_malloc(sizeof(fftw_complex)*(ctrl.n));
  fftw_complex* vk  = fftw_malloc(sizeof(fftw_complex)*(ctrl.n));

  prep_control_params(&ctrl, &wrk);
  P0x = ctrl.current.Px; 
  P0y = ctrl.current.Px; 
  H0 = ctrl.current.H;
  printf("y0 = %.15e from FFT\n", find_y0(&wrk));
  printf("t = %.15e\tH = %.15e\tM = %.15e\n", 0., ctrl.current.H, ctrl.current.M);
  fprintf(fh, "#1.time 2. error in hamiltonian\n\n");
  for (int j = 0; j < ctrl.num_iter + 1; j++) {
    if (j >= j_out) { 
      prep_control_params(&ctrl, &wrk);
      if (it >=  1) {
        fprintf(fh, "%.15e\t%.15e\n", ctrl.current.T,  ((ctrl.current.H)- H0)/H0);
        printf("t = %.15e\tH = %.15e\tM = %.15e\n", ctrl.current.T, ctrl.current.H, ctrl.current.M);
        fhlog = fopen("run.log","a");
        fprintf(fhlog, "t = %.15e\tH = %.15e\tM = %.15e\n", ctrl.current.T, ctrl.current.H, ctrl.current.M);
        fclose(fhlog);
      }
      update_dfunc(&wrk);

      sprintf(s1,"test_out/fz%03d.txt", it);
      write_outwz(wrk.Z, s1, (ctrl.current.T));
      sprintf(s1,"test_out/v%03d.txt", it);
      write_out4pade(wrk.V, s1, (ctrl.current.T));


      j_out += ctrl.skip; it += 1;
    }
    if (j != ctrl.num_iter) {
      if (ctrl.n < ctrl.k_cut) {
        //printf("Reached Maximum allowed Fourier modes Nmax = %d\n", ctrl.k_cut);
        reflag = refine(&wrk, &ctrl, j);
      }
      if (reflag == 1) {
        rff = rff+1;
        update_dfunc(&wrk);
        free(rk); free(vk);
	rk  = fftw_malloc(sizeof(fftw_complex)*(ctrl.n));
	vk  = fftw_malloc(sizeof(fftw_complex)*(ctrl.n));
        fft(wrk.R, rk);
        fft(wrk.V, vk); 
        sprintf(s1,"rk%03d.txt", rff);
        write_outwz(rk, s1, (ctrl.current.T));
        sprintf(s1,"vk%03d.txt", rff);
        write_out(vk, s1, (ctrl.current.T));
        init_prep(&ctrl);
        reflag = 0;
      }
      rk4(&ctrl);
      ctrl.current.T += ctrl.dt;
    }
    if (ctrl.n > ctrl.k_cut) {
      printf("Reached Maximum allowed Fourier modes Nmax = %d\n", 2*ctrl.k_cut);
      fhlog = fopen("run.log","a");
      fprintf(fhlog, "Reached Maximum allowed Fourier modes Nmax = %d\n", 2*ctrl.k_cut);
      fclose(fhlog);
      err_msg("Complete");
      break;
    }
  }
  prep_control_params(&ctrl, &wrk);
  fprintf(fh, "%.15e\t%.15e\n", ctrl.current.T, ((ctrl.current.H)- H0)/H0);
  printf("t = %.15e\tH = %.15e\tM = %.15e\n", ctrl.current.T, ctrl.current.H, ctrl.current.M);
  fhlog = fopen("run.log","a");  
  fprintf(fhlog, "t = %.15e\tH = %.15e\tM = %.15e\n", ctrl.current.T, ctrl.current.H, ctrl.current.M);
  fclose(fhlog);

  fclose(fh);*/
}
