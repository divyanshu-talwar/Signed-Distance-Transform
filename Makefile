#OS X compilation flags
#CPPFLAGS=-Wno-deprecated-declarations -I. -I/usr/local/include #-fopenmp
#LDFLAGS= -lstdc++ -O2 -L/usr/local/lib -lIL -lILU

#Linux compilation flags
CPPFLAGS=-I../devil/include -Xcompiler -fopenmp 
LDFLAGS= -O2 -L../devil/lib -lm -lstdc++ -lIL -lILU 

gpu:
	nvcc -o sdt_gpu sdt_gpu.cu ${CPPFLAGS} ${LDFLAGS}

clean:
	-rm -f sdt_gpu
