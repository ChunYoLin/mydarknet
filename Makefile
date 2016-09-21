ARCH= --gpu-architecture=compute_20 --gpu-code=compute_20

VPATH=./src/
EXEC=darknet
EXEC_GPU=darknet_GPU
EXEC_OPENCV=darknet_OPENCV
OBJDIR=./obj/

CC=gcc -pg -Wall
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread -lstdc++ 
LDFLAGSGPU= -lm -pthread -lstdc++  
COMMON= 
COMMONGPU= 
CFLAGS=-Wall -Wfatal-errors 
CFLAGSGPU=-Wall -Wfatal-errors -DOPENCV 
GPU=1
OPENCV=1
DEBUG=0

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif


COMMONGPU+= -DOPENCV
CFLAGSGPU+= -DOPENCV
LDFLAGSGPU+= `pkg-config --libs opencv` 
COMMONGPU+= `pkg-config --cflags opencv` 
COMMONGPU+= -DGPU -I/usr/local/cuda/include/
CFLAGSGPU+= -DGPU
LDFLAGSGPU+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand



OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o  activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o imagenet.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o layer.o compare.o classifier.o local_layer.o thpool.o image.o
ifeq ($(GPU), 1) 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o softmax_layer_kernels.o network_kernels.o avgpool_layer_kernels.o yolo_kernels.o coco_kernels.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj results $(EXEC)

G: obj results $(EXEC_GPU)
	@echo $(EXEC_GPU)

$(EXEC): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(EXEC_GPU): $(OBJS) 
	$(CC) $(COMMONGPU) $(CFLAGSGPU) $^ -o $@ $(LDFLAGSGPU)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC) $(EXEC_GPU) 

