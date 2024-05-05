# llm.cpp

Run a ChatGPT-like model using CPU and GPU. The goal of this project is to optimize the performance of LLM generation on consumer hardware. We plan to implement the following features:
- [X] CPU inference
- [X] Multi-threading
- [ ] GPU acceleration
- [ ] CPU/GPU hybrid computation

```
== Running in chat mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to LLaMA.
 - If you want to submit another line, end your input in '\'.

> What is the capital of the United States ?
The capital of the United States is Washington D.C..
> ^C
== Quit ==
```

## Getting Started

### Requirements
- gcc compiler


### Clone the repository

```bash
git clone https://github.com/PopoDev/llm.cpp
cd llm.cpp
```

### Download the model

The model used is the Alpaca 7b model with 4-bit quantization available [here](https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/blob/main/ggml-alpaca-7b-q4.bin).

```bash
wget https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/resolve/main/ggml-alpaca-7b-q4.bin
```

### Run the model
```bash
make run
./run
```


## Performance Analysis

Use the flag `PERF` to enable the performance analysis.

```bash
PERF=1 make run
```

The output will be like this:

```
perf_total_per_op_us[            NONE] =    0.000 ms | 0.00% 
perf_total_per_op_us[             DUP] =    0.000 ms | 0.00% 
perf_total_per_op_us[             ADD] =    0.000 ms | 0.00% 
perf_total_per_op_us[             SUB] =    0.000 ms | 0.00% 
perf_total_per_op_us[             MUL] =    2.000 ms | 0.22% 
perf_total_per_op_us[             DIV] =    0.000 ms | 0.00% 
perf_total_per_op_us[             SQR] =    0.000 ms | 0.00% 
perf_total_per_op_us[            SQRT] =    0.000 ms | 0.00% 
perf_total_per_op_us[             SUM] =    0.000 ms | 0.00% 
perf_total_per_op_us[            MEAN] =    0.000 ms | 0.00% 
perf_total_per_op_us[          REPEAT] =    0.000 ms | 0.00% 
perf_total_per_op_us[             ABS] =    0.000 ms | 0.00% 
perf_total_per_op_us[             SGN] =    0.000 ms | 0.00% 
perf_total_per_op_us[             NEG] =    0.000 ms | 0.00% 
perf_total_per_op_us[            STEP] =    0.000 ms | 0.00% 
perf_total_per_op_us[            RELU] =    0.000 ms | 0.00% 
perf_total_per_op_us[            GELU] =    0.000 ms | 0.00% 
perf_total_per_op_us[            SILU] =    0.000 ms | 0.00% 
perf_total_per_op_us[            NORM] =    0.000 ms | 0.00% 
perf_total_per_op_us[        RMS_NORM] =    7.000 ms | 0.78% 
perf_total_per_op_us[         MUL_MAT] =  870.000 ms | 97.10% 
perf_total_per_op_us[           SCALE] =    0.000 ms | 0.00% 
perf_total_per_op_us[             CPY] =   14.000 ms | 1.56% 
perf_total_per_op_us[         RESHAPE] =    0.000 ms | 0.00% 
perf_total_per_op_us[            VIEW] =    0.000 ms | 0.00% 
perf_total_per_op_us[         PERMUTE] =    0.000 ms | 0.00% 
perf_total_per_op_us[       TRANSPOSE] =    0.000 ms | 0.00% 
perf_total_per_op_us[        GET_ROWS] =    3.000 ms | 0.33% 
perf_total_per_op_us[   DIAG_MASK_INF] =    0.000 ms | 0.00% 
perf_total_per_op_us[        SOFT_MAX] =    0.000 ms | 0.00% 
perf_total_per_op_us[            ROPE] =    0.000 ms | 0.00% 
perf_total_per_op_us[      CONV_1D_1S] =    0.000 ms | 0.00% 
perf_total_per_op_us[      CONV_1D_2S] =    0.000 ms | 0.00% 
perf_total_per_op_us[      FLASH_ATTN] =    0.000 ms | 0.00% 
perf_total_per_op_us[        FLASH_FF] =    0.000 ms | 0.00% 
```