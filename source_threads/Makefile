FSW     = fsw.x

#List of sources
C_FSW   = main_alt.c ffts_alt.c prep.c

# Compilers
CC      = gcc
LINK    = gcc
OPT     = -march=native -Ofast -ffast-math -std=gnu99 -Wpointer-arith -Wcast-align -fno-schedule-insns -fschedule-insns2 -fstrict-aliasing -funroll-loops -fprefetch-loop-arrays
#-----------------------------
#generic

LIB_MPI         =
LIB_FFT         = -L/home/orange/fftw3libd/lib  -lfftw3_threads -lfftw3 -lm -lpthread
INC_MPI         =
INC_FFT         = -I/home/orange/fftw3libd/include
LIB_ADD         =

ifeq ($(HOSTNAME), pequena)
        LIB_MPI         =
        LIB_FFT         = -lfftw3_mpi -lfftw3 -lm
        INC_MPI         =
        INC_FFT         =
endif
ifeq ($(HOSTNAME), metropolis.RR)
        LIB_MPI         =
        LIB_FFT         = -L/opt/local/fftw/3.3.1/mvapich2/1.7/intel/12.1/lib -lfftw3_mpi -lfftw3 -lm
        INC_MPI         =
        INC_FFT         =
endif

#-----------------------------

OBJ_FSW         = $(C_FSW:.c=.o) $(F_FSW:.f=.o)


LIB_FSW         = $(LIB_MPI) $(LIB_FFT) $(LIB_ADD)
INC_FSW         = $(INC_MPI) $(INC_FFT)

#-----------------------------

default: fsw

.f.o:
	$(FC) $(FFLAGS) -c $<


fsw:
	$(CC) $(OPT) $(DEF_FSW) $(INC_FSW) -c $(C_FSW)
	$(LINK) $(OPT) $(OBJ_FSW) $(LIB_FSW) -o $(FSW)
	cp fsw.x ../debug/fsw.x

hostname:
	@echo $(HOSTNAME) $(INC_FFT)

clean:
	@echo "cleaning ..."
	rm -f *~ *.o
	rm -f ../debug/fsw.x

#-------------------------------


