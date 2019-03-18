CXX           = clang++
CXXFLAGS        = -std=c++14 -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion
INCPATH       = -Iinclude -I/usr/local/cuda-7.0/include -I/opt/cuda/include -I/opt/local/include -I/usr/local/cuda/include -I/home/fengwang/usr/include -I/home/fengwang/opt/cuda/include -I/usr/local/include  -I/usr/local/cuda-7.0/include
LINK          = $(CXX)
LFLAGS        = -lc++ -lc++abi -O3 -pthread
DEL_FILE      = rm -f
DEL_DIR       = rmdir
MOVE          = mv -f
MAKE_DIR      = mkdir
CUDACXX      = nvcc
CUDACXX64      = nvcc
CUDACXXFLAGS = -m64 -gencode arch=compute_61,code=sm_61 -O2 -Iinclude
CUDALIB   = -L/opt/cuda/lib64 -L/usr/local/cuda-7.0/lib64 -L/usr/local/cuda/lib -L/home/fengwang/opt/cuda/lib64 -L/Developer/NVIDIA/CUDA-7.0/lib
CUDALFLAGS   = -L/opt/cuda/lib64 -L/usr/local/cuda-7.0/lib64 -L/usr/local/cuda/lib -L/home/fengwang/opt/cuda/lib64 -L/Developer/NVIDIA/CUDA-7.5/lib -lcudart -lcurand -lcublas
CUDALIB64   = -L/opt/cuda/lib64 -L/usr/local/cuda-7.0/lib64 -L/usr/local/cuda/lib -L/home/fengwang/opt/cuda/lib64 -L/Developer/NVIDIA/CUDA-7.5/lib
CUDALINK     = g++

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin
LIB_DIR       = ./lib
LOG_DIR       = ./log

default: direct_cuda_pattern

direct_cuda_pattern: src/direct_cuda_pattern.cc
	$(CXX) $(CXXFLAGS) -c src/direct_cuda_pattern.cc -o $(OBJECTS_DIR)/direct_cuda_pattern.o $(INCPATH)
	$(CUDACXX64) src/cuda_pattern.cu -o $(OBJECTS_DIR)/_cuda_pattern.o $(INCPATH) -m64 -dc -gencode arch=compute_61,code=sm_61 --relocatable-device-code true  -Xptxas -v
	$(CUDACXX64) -dlink $(OBJECTS_DIR)/_cuda_pattern.o -arch=sm_61 -o $(OBJECTS_DIR)/cuda_pattern.o -m64 -lcudadevrt $(CUDALIB64) -rdc=true -Xptxas -v
	$(CXX) -o $(BIN_DIR)/direct_cuda_pattern $(OBJECTS_DIR)/_cuda_pattern.o $(OBJECTS_DIR)/direct_cuda_pattern.o $(OBJECTS_DIR)/cuda_pattern.o  -lcudadevrt -lcublas -lcudart -m64 $(CUDALIB64) $(LFLAGS)

