ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

#
# Compile flags
#

CFLAGS   = -I. -O3 -std=c11   -fPIC
CXXFLAGS = -I. -O3 -std=c++11 -fPIC
LDFLAGS  =

# CUDA
ifdef CUDA
	CUDA_OBJECTS = ggml-cuda.o
	CFLAGS   += -DGGML_CUDA
	CXXFLAGS += -DGGML_CUDA
	LDFLAGS += -L/usr/local/cuda/lib64 -lcudart -lcublas
endif

# Performance
ifdef PERF
	CFLAGS 	 += -DGGML_PERF
	CXXFLAGS += -DGGML_PERF
endif

# Debug
ifdef DEBUG
	CXXFLAGS += -DGGML_DEBUG
endif

# OS specific
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# Architecture specific
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	ifeq ($(UNAME_S),Darwin)
		CFLAGS += -mf16c
		AVX1_M := $(shell sysctl machdep.cpu.features)
		ifneq (,$(findstring FMA,$(AVX1_M)))
			CFLAGS += -mfma
		endif
		ifneq (,$(findstring AVX1.0,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell sysctl machdep.cpu.leaf7_features)
		ifneq (,$(findstring AVX2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
	else ifeq ($(UNAME_S),Linux)
		AVX1_M := $(shell grep "avx " /proc/cpuinfo)
		ifneq (,$(findstring avx,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell grep "avx2 " /proc/cpuinfo)
		ifneq (,$(findstring avx2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell grep "fma " /proc/cpuinfo)
		ifneq (,$(findstring fma,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell grep "f16c " /proc/cpuinfo)
		ifneq (,$(findstring f16c,$(F16C_M)))
			CFLAGS += -mf16c
		endif
		SSE3_M := $(shell grep "sse3 " /proc/cpuinfo)
		ifneq (,$(findstring sse3,$(SSE3_M)))
			CFLAGS += -msse3
		endif
	else ifeq ($(UNAME_S),Haiku)
		AVX1_M := $(shell sysinfo -cpu | grep "AVX ")
		ifneq (,$(findstring avx,$(AVX1_M)))
			CFLAGS += -mavx
		endif
		AVX2_M := $(shell sysinfo -cpu | grep "AVX2 ")
		ifneq (,$(findstring avx2,$(AVX2_M)))
			CFLAGS += -mavx2
		endif
		FMA_M := $(shell sysinfo -cpu | grep "FMA ")
		ifneq (,$(findstring fma,$(FMA_M)))
			CFLAGS += -mfma
		endif
		F16C_M := $(shell sysinfo -cpu | grep "F16C ")
		ifneq (,$(findstring f16c,$(F16C_M)))
			CFLAGS += -mf16c
		endif
	else
		CFLAGS += -mfma -mf16c -mavx -mavx2
	endif
endif
ifeq ($(UNAME_M),amd64)
	CFLAGS += -mavx -mavx2 -mfma -mf16c
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mpower9-vector
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif
ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif
ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, 2, 3
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

#
# Print build information
#

$(info I llm.cpp build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )

default: run 

#
# Build library
#

ggml-cuda.o: ggml-cuda.cu ggml-cuda.h
ifdef CUDA
	nvcc -c ggml-cuda.cu -o ggml-cuda.o
endif

ggml.o: ggml.c ggml.h
	$(CC)  $(CFLAGS)   -c ggml.c -o ggml.o

utils.o: utils.cpp utils.h
	$(CXX) $(CXXFLAGS) -c utils.cpp -o utils.o

clean:
	rm -f *.o run 

run: run.cpp $(CUDA_OBJECTS) ggml.o utils.o
ifdef CUDA
	nvcc -DGGML_CUDA run.cpp $(CUDA_OBJECTS) ggml.o utils.o -o run $(LDFLAGS)
else
	$(CXX) $(CXXFLAGS) run.cpp $(CUDA_OBJECTS) ggml.o utils.o -o run $(LDFLAGS)
endif