FSW     = fsw.x

#List of sources
C_FSW   = main_alt.c ffts_alt.c prep.c

# Compilers
CC      = icc
LINK    = icc
OPT     = -xhost -ip -ipo -O3 -std=gnu99 
#-----------------------------
#generic
#icc -I$TACC_FFTW3_INC -std=gnu99 main_alt.o prep.o ffts_alt.o -Wl,-rpath,$TACC_FFTW3_LIB  -L$TACC_FFTW3_LIB -lfftw3_threads -lfftw3

LIB_MPI         =
LIB_FFT         = -Wl,-rpath,$(TACC_FFTW3_LIB)  -L$(TACC_FFTW3_LIB) -lfftw3_threads -lfftw3 -lm -lpthread
INC_MPI         =
INC_FFT         = -I$(TACC_FFTW3_INC)
LIB_ADD         =


ifeq "$(HOSTNAME)" "orange-machina"
	CC=gcc-4.8
	LINK=gcc-4.8
	OPT=-Ofast -flto -std=gnu99
	LIB_FFT=-L/home/orange/usr/lib -lfftw3_threads -lfftw3 -lm -lpthread
	INC_FFT=-I/home/orange/usr/include
endif

#-----------------------------

OBJ_FSW         = $(C_FSW:.c=.o) $(F_FSW:.f=.o)


LIB_FSW         = $(LIB_MPI) $(LIB_FFT) $(LIB_ADD)
INC_FSW         = $(INC_MPI) $(INC_FFT)

#-----------------------------

all:
	@echo $(HOSTNAME)

default: fsw

.f.o:
	$(FC) $(FFLAGS) -c $<


fsw:
	@echo $(HOSTNAME)
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


