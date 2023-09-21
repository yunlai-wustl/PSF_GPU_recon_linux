INCLUDE1= ../Solution_Items
INCLUDE2= ../image_update_CUDA
INCLUDE3= ../proj_functions_CUDA

OBJ=../Objects
TARGET = EM_recon_psf_generic_no_effective_DOI_half_TOR_width_test
CXX = g++
NVCC = nvcc 

src=$(wildcard ${INCLUDE1}/*.cpp ./*.cpp)
src2=$(wildcard ${INCLUDE2}/*.cu ${INCLUDE3}/*.cu)
objects = $(patsubst %.cpp, ${OBJ}/%.o, $(notdir ${src}))
objects2 = $(patsubst %.cu, ${OBJ}/%.o, $(notdir ${src2}))
objects += $(objects2)

NVCC_OPTS=-g -pg -Xcompiler -Wextra -m64 -l cuda -std=c++11  -I$(INCLUDE1) -D_GLIBCXX_USE_CXX11_ABI=0 --expt-relaxed-constexpr
GCC_OPTS=-m64 -g -Wall -Wextra -m64 -I$(INCLUDE1) -std=c++11


${TARGET}:${objects}
	$(NVCC) $(NVCC_OPTS)  $(objects) -O2 -o $@

${OBJ}/%.o: ./%.cpp
	$(NVCC) $(NVCC_OPTS) -c $< -o $@

${OBJ}/%.o: ${INCLUDE3}/%.cu
	$(NVCC) $(NVCC_OPTS) -c $< -o $@

${OBJ}/%.o: ${INCLUDE2}/%.cu
	$(NVCC) $(NVCC_OPTS) -c $< -o $@

${OBJ}/%.o: ${INCLUDE1}/%.cpp
	$(NVCC) $(NVCC_OPTS) -c $< -o $@

clean:
	rm -rf ${objects} ${objects2}
	rm -rf ${TARGET}
