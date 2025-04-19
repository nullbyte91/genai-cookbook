# genai-cookbook
## 📌 Table of Contents
1. [Introduction](#-introduction)
2. [🧭 Strategy & Scoping](#-strategy--scoping)
3. [🧱 Data Foundation](#-data-foundation)
4. [🔁 Iterative Development & Evaluation](#-iterative-development--evaluation)
   - [3.1 Foundation Model Selection & Baseline Evaluation](#31-foundation-model-selection--baseline-evaluation)
   - [3.2 Application Prototyping](#32-application-prototyping)
   - [3.3 Prompt Engineering & Optimization](#33-prompt-engineering--optimization)
   - [3.4 RAG System Development & Tuning](#34-rag-system-development--tuning)
   - [3.5 Fine-tuning](#35-fine-tuning)
     - [3.5.1 Dataset Preparation](#351-dataset-preparation)
     - [3.5.2 Training Setup](#352-training-setup)
     - [3.5.3 Model Initialisation](#353-model-initialisation)
     - [3.5.4 Selection of Fine-Tuning Technique](#354-selection-of-fine-tuning-technique)
   - [3.6 Comprehensive Evaluation](#36-comprehensive-evaluation)
5. [🚀 Productionization & Operations (LLMOps)](#-productionization--operations-llmops)
6. [🔍♻️ Continuous Monitoring & Improvement](#-continuous-monitoring--improvement)
7. [📚 Appendix](#-appendix-foundational-concepts)
7. [🤝 Contributing](#-contributing)
8. [📜 License](#-license)

## 📘 Introduction
This repository is a research-driven framework for adopting and fine-tuning Large Language Models (LLMs) effectively and responsibly.

Instead of jumping directly into fine-tuning, this project explores a <b>layered approach to LLM system design</b>, combining industry best practices, cost-effective techniques, and deep evaluations. It aims to help you decide <b>when to use prompt engineering, retrieval-augmented generation (RAG), or full/parameter-efficient fine-tuning</b>, based on your task, domain, and constraints.

The goal is to build a modular, extensible pipeline that supports experimentation, benchmarking, and eventual deployment of high-performing, domain-specific LLMs — while maintaining reproducibility, traceability, and practical relevance.


## 🧭 Strategy & Scoping
*(Coming Soon)*

## 🧱 Data Foundation
*(Coming Soon)*

## 🔁 Iterative Development & Evaluation

### 3.1 Foundation Model Selection & Baseline Evaluation
Following the establishment of a robust data foundation, the core development phase focuses on selecting an appropriate foundation model and adapting it to the specific use case through various techniques, all managed within an LLMOps framework.

The choice of the base LLM significantly impacts the application's capabilities, cost, and development trajectory. Key considerations include:
* <b>Proprietary vs. Open-Source:</b> Developers must choose between proprietary models offered via APIs (e.g., OpenAI's GPT series, Anthropic's Claude) and open-source models (e.g., Meta's Llama series, Mistral models 55) that allow for greater customization and local deployment.

* <b>Evaluation Criteria:</b> While general benchmarks provide a starting point, the "best" model is highly context-dependent. Performance can vary significantly across different tasks and domains. Therefore, it is crucial to create targeted evaluation sets that reflect the specific, real-world tasks the LLM will perform. Evaluating shortlisted models against these custom datasets before making a final selection is essential to ensure suitability and avoid investing resources in a suboptimal foundation.

* <b>Other Factors:</b> Cost structures (API call fees vs. hosting/compute costs), flexibility for fine-tuning, data privacy implications (especially with API-based models), built-in safety and alignment features, and licensing restrictions are also critical decision factors. Model Gardens and Hubs, such as Hugging Face and Google's Vertex AI Model Garden, serve as valuable resources for discovering and initially assessing available models.

---

### 3.2 Application Prototyping
---

### 3.3 Prompt Engineering
---

### 3.4 RAG System Development & Tuning
---
### 3.5 Fine-Tuning
---
#### 3.5.1 Dataset Preparation
---
#### 3.5.2 Training Setup
---
#### 3.5.3 Model Initialisation
---
#### 3.5.4 Memory Optimization Strategies
Imagine the standard deep learning training loop - forward pass, backward pass, optimizer step - as a journey for your data and model weights within the GPU. Let's track the memory usage at each stage:

##### Phase 1: Setting the Stage - Initialization & Model Loading
Before any training happens, we need to load the essentials onto the GPU. This establishes the static memory baseline - the minimum VRAM required just to exist.

1. <b>CUDA Context & Framework Overhead</b>: 
The deep learning framework (like PyTorch or TensorFlow) reserves a chunk of memory for its CUDA context. This isn't just an empty space; it holds compiled CUDA kernels, GPU libraries (cuDNN, cuBLAS), and the overhead from the framework's own memory management system.
2. <b>Model Parameters</b>: This is often the most significant static component. It's the memory needed to store all the weights and biases of your pre-trained transformer. This includes,
*Input Embeddings (embed_p)
* Transformer Layer Weights (other_p - attention, linear layers, normalization)
* Task-Specific Head (lm_p - language model head, classification head) * The size depends heavily on the model's architecture (depth, width, vocab size) and, crucially, the precision used:

$$
\mathrm{param\_memory} =
\begin{cases}
4~\text{bytes,} & \text{FP32} \\
2~\text{bytes,} & \text{FP16 or BF16} \\
1~\text{byte,} & \text{INT8} \\
0.5~\text{bytes,} & \text{NF4 (QLoRA)}
\end{cases}
$$

3. <b>Memory Padding:</b> Allocators don't usually grab exactly the requested memory. For efficiency (coalesced access, easier management, kernel requirements, MMU alignment), they round up allocations to align with hardware boundaries (like CUDA page sizes). This means some allocated memory might be unoccupied padding, slightly inflating the actual usage beyond the sum of parameter sizes.

$$
m_{\mathrm{base}} = m_{\mathrm{cuda_ctx}} + m_{\mathrm{framework}} + m_{\mathrm{libs}}
$$

> **_Note_** This initial phase sets a potentially high floor for memory usage, dominated by model parameters (mp​) and framework overhead (mb​ase). Optimizations here involve shrinking the model itself (Quantization, Parameter-Efficient Fine-Tuning - PEFT).

##### Phase 2: Data Arrives - Loading & Embedding
Now, the training loop begins. Batches of input data (tokenized sequences, often shaped [batch_size, sequence_length] or [bs, sl]) are moved to the GPU.

1. <b>Input Embedding Lookup:</b> The model's embedding layer (embed_p) converts token IDs into dense vectors.
2. <b>First Activations:</b> This lookup generates the first set of intermediate results, or activations. Their size is typically bs×sl×hidden_dim, using the compute precision (e.g., FP16/BF16 in mixed-precision).

$$
m_{\mathrm{embed}} = B \times S \times H \times \mathrm{dtype_size}
$$

> **_Note_**  This marks the start of dynamic memory allocation. Activation memory scales directly with batch size (bs) and sequence length (sl), making these key parameters to tune for memory management.

##### Phase 3: The Forward Pass - Climbing the Activation Mountain
The input embeddings flow through the transformer layers (other_p). Each layer computes outputs (more activations) that become the input for the next. Finally, the LM head (lm_p) generates logits (often bs×sl×vocab_size), and the loss is calculated.

1. Intermediate Activations (mout​): This is the memory holding the outputs of all intermediate layers (self-attention outputs, MLP outputs, layer norm outputs, etc.).
2. LM Head Outputs & Loss (mlm​): Includes the final logits, potentially targets moved to the GPU, and temporary variables used during loss calculation.

Unless gradient checkpointing a optimization techniques is used, all these intermediate activations (mout​) must be kept in memory simultaneously. Why we need them? They are needed during the backward pass to calculate gradients.

$$
m_{\mathrm{out}} = \sum_{l=1}^{L} A_l
$$

$$
\mathrm{logits_shape} = B \times S \times V
$$

> **_Note_** For deep models, long sequences, or large batches, activation memory (mout​) often becomes the dominant dynamic memory consumer, potentially exceeding parameter or optimizer state memory. Its sheer size makes it the prime target for techniques like Gradient Checkpointing.

##### Phase 4: The Backward Pass - Gradients and the Optimizer State Surprise
Starting from the loss, gradients are computed backward through the network.
Gradient Computation: The derivative of the loss is calculated with respect to each layer's output and, critically, with respect to each trainable model parameter.
1. <b>Gradient Storage (mg​)</b>: Memory is allocated to store these gradients (dLoss/dWeight) for all trainable parameters. The size is proportional to the number of parameters being updated.
2. Optimizer State Allocation (mos​)</b>: The One-Time Hit: During the first backward pass, the optimizer allocates memory for its internal states. For Adam/AdamW, this typically means:
* Momentum (m): First moment estimate (average of past gradients).
* Variance (v): Second moment estimate (average of past squared gradients). These are usually stored in FP32for numerical stability, even in 

$$
m_g = P_{\mathrm{train}} \times \mathrm{dtype_size}
$$

$$
m_{\mathrm{os}} = P_{\mathrm{train}} \times 8
$$


> **_Note_** Gradient memory (mg​) scales with trainable parameters, making PEFT highly effective here. Optimizer states (mos​) represent a significant, persistent memory cost established early on, making optimizer choice critical.

##### Phase 5: The Optimizer Step - Updating Weights
The optimizer uses the computed gradients (mg​) and its stored states (mos​) to update the model parameters (mp​).
1. Parameter Update: Calculations are performed (e.g., applying momentum, variance correction, weight decay) and the results are written back to the model's parameters.
2. Memory Fluctuation: Memory usage might slightly decrease after this step if the gradient buffers (mg​) are released (this depends on the framework's memory management). However, the large optimizer state memory (mos​) remains.

$$
m_{\mathrm{after_update}} \approx m_{\mathrm{base}} + m_p + m_{\mathrm{os}} + c_{\mathrm{up}}
$$


> **_Note_** The optimizer step completes the cycle. The persistent nature of optimizer states (mos​) highlights the value of memory-efficient optimizers.

The Final GPU Memory,

$$
\mathrm{Peak\_GPU\_Memory} \approx m_{\mathrm{base}} + m_p + m_{\mathrm{os}} + \max(m_{\mathrm{out}}, m_g) + c_{\mathrm{up}}
$$

The followed section aims to provide a comprehensive and in-depth overview of the techniques, tools, benchmarks, and best practices for optimizing the fine-tuning process of LLMs, specifically targeting hardware environments with limited compute capacity.

The primary consumers of GPU VRAM during the training process are the model's parameters, the optimizer states required for parameter updates, the gradients computed during backpropagation, and the intermediate activations generated during the forward pass.12 Various strategies have been developed to target one or more of these components, thereby reducing the overall memory footprint.

#### 3.5.5.1 Gradient Checkpointing (Activation Recomputation)
Gradient checkpointing, also known as activation recomputation, is a technique specifically designed to reduce the memory consumed by storing intermediate activations. In standard backpropagation, all activations calculated during the forward pass are typically stored in VRAM to be readily available for gradient computation in the backward pass.

Gradient checkpointing mitigates this by strategically saving (checkpointing) activations only at specific layers within the network. During the backward pass, if the activations for a non-checkpointed layer are required, they are recomputed on-the-fly, starting from the nearest preceding checkpoint. This recomputation avoids storing all activations simultaneously.

The fundamental trade-off inherent in gradient checkpointing is between memory usage and computational time. By discarding and recomputing activations, VRAM usage is significantly reduced, with the extent of savings dependent on the checkpointing frequency. However, the recomputation introduces additional forward passes during the backward step, increasing the overall computation time per training iteration. This slowdown is often estimated to be between 20% and 50%.

For smaller models or scenarios where memory is not the absolute bottleneck, the computational overhead might negate the benefits of a larger batch size, potentially leading to slower overall training.36 However, for large models (e.g., 8B parameters) fine-tuned on multi-GPU setups, gradient checkpointing has been shown to dramatically increase the feasible batch size (e.g., from 2 to 12 or 14 per GPU), even if the time per epoch might still be longer compared to a smaller-batch run without checkpointing.

To enable gradient checkpointing in the 🤗 Trainer, pass the corresponding a flag to 🤗 TrainingArguments:
```python
training_args = TrainingArguments(
    .., gradient_checkpointing=True, **default_args
)
```
---
#### 3.5.5.2 Gradient Accumulation
Gradient accumulation is a technique used to compute gradients in smaller chunks rather than processing the entire batch at once. In this method, the model performs forward and backward passes on smaller mini-batches, accumulating the gradients over several iterations. Once the desired number of mini-batches has been processed, the optimizer updates the model parameters. This approach allows the use of larger effective batch sizes than what the GPU memory can typically handle. However, it's worth noting that the extra forward and backward passes required for accumulation can lead to slower training times.

```python
training_args = TrainingArguments(
      .., gradient_accumulation_steps=4, **default_args)
```

Let's compare the memory and compute trade-offs of Gradient Checkpointing and Gradient Accumulation

```bash
# Baseline
==================================================
Starting training with config:
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 5e-05
  weight_decay: 0.0
  fp16: False
  bf16: False
  gradient_checkpointing: False
  max_grad_norm: 1.0
  model_type: gpt2
  hidden_size: 1024
  num_hidden_layers: 24
  num_attention_heads: 16
==================================================

==================================================
Training completed!
Total training time: 468.57 seconds
Epochs completed: 3
Steps completed: 1101
Peak GPU memory: 8114.69 MB
Average GPU memory: 8032.53 MB
==================================================

# With Gradient checkpointing & Accumulation

==================================================
Starting training with config:
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 5e-05
  weight_decay: 0.0
  fp16: False
  bf16: False
  gradient_checkpointing: True
  max_grad_norm: 1.0
  model_type: gpt2
  hidden_size: 1024
  num_hidden_layers: 24
  num_attention_heads: 16
==================================================

==================================================
Training completed!
Total training time: 528.48 seconds
Epochs completed: 3
Steps completed: 1101
Peak GPU memory: 7960.75 MB
Average GPU memory: 7894.72 MB
==================================================
```

This comparison demonstrates that for GPT-2 Medium, the memory savings from gradient checkpointing are relatively modest compared to the performance impact. These optimizations would likely show more significant memory benefits on larger models or with larger batch sizes.

---

#### 3.5.5.3 Mixed Precision Training (FP16/BF16)
Mixed precision training involves performing computations and storing weights and activations using lower-precision floating-point formats, namely 16-bit formats like FP16 (half-precision) or BF16 (Brain Floating Point), instead of the standard 32-bit single-precision (FP32).

This approach offers two primary benefits:
* Reduced Memory Footprint: Storing parameters, activations, and gradients in 16-bit formats effectively halves the memory required compared to FP32, freeing up significant VRAM.15
* Faster Computation: Modern GPUs, particularly those with NVIDIA Tensor Cores, are optimized for lower-precision computations, leading to substantial speedups in matrix multiplications and convolutions.34

While both FP16 and BF16 use 16 bits, they differ in their representation, leading to distinct characteristics
* Numerical Range: BF16 allocates more bits to the exponent, providing a numerical range similar to FP32. This makes it less susceptible to underflow and overflow issues during training, often eliminating the need for loss scaling.
* Precision: FP16 allocates more bits to the mantissa, offering higher precision (ability to represent numbers with finer detail) than BF16.38
* Loss Scaling: Due to its narrower range, FP16 training often requires loss scaling – multiplying the loss by a scaling factor before backpropagation and unscaling the gradients before the optimizer step – to prevent gradients from becoming zero (underflow). BF16 typically avoids this complexity.
* Conversion Speed: BF16 conversion to/from FP32 is generally faster as it shares the same exponent range, essentially involving truncation or padding of the mantissa. FP16 conversion is more complex.
* Hardware Support: FP16 support is widespread. BF16 support is typically found on newer architectures like NVIDIA Ampere (A100, RTX 30 series) and subsequent generations.

For example, 
```python
# FP16
training_args = TrainingArguments(.., fp16=True, **default_args)

# BF16
training_args = TrainingArguments(.., bf16=True, **default_args)
```
These implementation is usually handled by Automatic Mixed Precision (AMP) libraries, such as PyTorch's built-in torch.amp or NVIDIA's Apex library.

There is one more type called tf32 t has the same numerical range as fp32 (8-bits), but instead of 23 bits precision it has only 10 bits (same as fp16) and uses only 19 bits in total. It’s “magical” in the sense that you can use the normal fp32 training and/or inference code and by enabling tf32 support you can get up to 3x throughput improvement. All you need to do is to add the following to your code:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

You can enable this mode in the 🤗 Trainer:
```python
training_args = TrainingArguments(.., tf32=True, **default_args)
```

Let's compare the memory and compute trade-offs of Gradient Checkpointing, Gradient Accumulation and the Mixed Precision, 
```bash
# Baseline
==================================================
Starting training with config:
  batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 5e-05
  weight_decay: 0.0
  fp16: False
  bf16: False
  gradient_checkpointing: False
  max_grad_norm: 1.0
  model_type: gpt2
  hidden_size: 1024
  num_hidden_layers: 24
  num_attention_heads: 16
==================================================
==================================================
Training completed!
Total training time: 470.13 seconds
Epochs completed: 3
Steps completed: 1101
Peak GPU memory: 8148.88 MB
Average GPU memory: 8071.29 MB
==================================================

#  With Gradient checkpointing, Accumulation and Mixed Precision
==================================================
Training completed!
Total training time: 387.06 seconds
Epochs completed: 3
Steps completed: 1101
Peak GPU memory: 7902.31 MB
Average GPU memory: 7801.63 MB
==================================================
```
FP16 fine-tuning is the primary contributor to compute time in this process.
---
#### 3.5.5.4 Efficient Optimizers and State Compression
Standard optimizers like AdamW contribute significantly to VRAM usage because they maintain auxiliary information, or "states," for each model parameter being trained. For AdamW using FP32, this typically involves storing momentum and variance estimates, requiring an additional 8 bytes per parameter on top of the parameter itself, totaling 12 bytes/param. Even with mixed precision where parameters might be FP16, the optimizer states are often kept in FP32 for stability, still consuming considerable memory (e.g., 8 bytes/param for states).

Several optimizers offer improved memory efficiency:
* AdamW: The standard baseline, known for good performance but high memory consumption (e.g., for a 3B parameter model, ~24GB VRAM just for the optimizer states if using FP32 states).
* Adafactor: Developed to be more memory-efficient than AdamW.35 It avoids storing full momentum and variance vectors, instead using factorized or aggregated statistics, reducing memory usage to slightly more than 4 bytes per parameter (e.g., >12GB for a 3B model).
```python
training_args = TrainingArguments(.., optim="adafactor", **default_args)
```
* 8-bit Optimizers: Implemented in libraries like bitsandbytes, these optimizers (e.g., AdamW8bit) store the optimizer states (momentum, variance) using 8-bit precision.16 The states are dequantized only during the parameter update step. This drastically reduces the memory required for optimizer states to approximately 2 bytes per parameter (e.g., ~6GB for a 3B model), offering significant savings compared to AdamW or even Adafactor.
```python
training_args = TrainingArguments(.., optim="adamw_bnb_8bit", **default_args)
```
* Paged optimizers: Address the GPU memory bottleneck by offloading optimizer states such as moment estimates in Adam to CPU RAM when theyre not actively needed on the GPU. This approach is inspired by virtual memory management in operating systems, where data is paged between RAM and disk as needed.

```python
training_args = TrainingArguments(.., optim="paged_adamw_8bit", **default_args)
```

```python
# Benchmark config
optimizer_configs = [
    {"name": "baseline", "fp16": False, "gradient_checkpointing": False},
    {"name": "adafactor",
     "fp16": True, 
     "gradient_checkpointing": True, 
     "gradient_accumulation_steps": 1,
     "optim": "adafactor"},
    {"name": "adamw_bnb_8bit",
     "fp16": True, 
     "gradient_checkpointing": True, 
     "gradient_accumulation_steps": 1,
     "optim": "adamw_bnb_8bit"},
    {"name": "paged_adamw_8bit",
     "fp16": True, 
     "gradient_checkpointing": True, 
     "gradient_accumulation_steps": 1,
     "optim": "paged_adamw_8bit"},
]
```
| Optimizer           | Train Time (s) | Avg GPU (MB) | Peak GPU (MB) | Train Loss | Eval Loss (Final) | Throughput (steps/s) |
|---------------------|----------------|---------------|----------------|-------------|--------------------|------------------------|
| AdamW (baseline)    | 464.05         | 12141.47      | 12197.06       | 0.6911      | 0.6898             | 2.37                   |
| Adafactor           | 387.97         | 8409.51       | 8513.69        | 0.7478      | 0.6900             | 2.83                   |
| AdamW BnB 8-bit     | 309.66         | 8794.61       | 8865.25        | 0.7591      | 0.6915             | 3.55                   |
| Paged AdamW 8-bit   | 321.74         | 9143.26       | 9371.75        | 0.7591      | 0.6915             | 3.42                   |

Obseervations are,
##### Speed vs Accuracy Tradeoff:
* AdamW (fp32) gives the best final training loss (0.6911) and slightly better eval loss but at highest memory and time cost.
* 8-bit optimizers (AdamW BnB & Paged AdamW) achieved ~33% faster training, using ~30% less GPU memory.

##### Adafactor:
* Performs very well on memory, lowest usage.
* Slightly higher train/eval loss compared to AdamW, but faster and more memory efficient.
* Good choice for low-resource environments.

##### 8-bit Optimizers (bnb_8bit & paged_adamw_8bit):
* Ideal if you're optimizing for training speed and memory efficiency without significantly compromising on final performance.
* PagedAdamW uses slightly more memory than adamw_bnb_8bit, likely due to the paging mechanism overhead.

[Want to try out - benchmark code](./finetune/experiments/optimization/benchmark_v1.0.py)

---

#### 3.5.5.5 Sharding & Offloading Techniques
Memory offloading strategies involve moving parts of the model's state – parameters, gradients, and optimizer states – from the limited GPU VRAM to the host system's larger memory pools, typically CPU RAM or, more recently, fast NVMe (Non-Volatile Memory Express) SSD storage.

This allows training models that are far too large to fit entirely within the collective memory of the available GPUs.

DeepSpeed, particularly through its ZeRO optimizer stages, provides sophisticated offloading capabilities:
* ZeRO-Offload: Introduced as part of ZeRO-2, this system focuses on offloading the optimizer states and gradient computations to the CPU RAM. This frees up GPU memory primarily consumed by these components.

* ZeRO-Infinity: An extension built upon ZeRO-3, ZeRO-Infinity enables offloading of all model states – parameters, gradients, and optimizer states – to either CPU RAM or NVMe storage. It employs intelligent partitioning and scheduling to manage data movement between the GPU and the host memory/storage, aiming for better bandwidth utilization and overlap between computation and communication compared to ZeRO-Offload.ZeRO-Infinity requires the use of ZeRO stage 3.

> **_Note_** FSDP, or "Fully Sharded Data Parallel," was initially developed by Facebook AI Research and introduced through the Fairscale library. Native support was later integrated into PyTorch starting with version 1.11.
Its core functionality mirrors that of DeepSpeed ZeRO — efficiently sharding optimizer states, gradients, and model parameters to reduce memory usage. It also includes support for CPU offloading. A notable advantage of FSDP is that it can be used as a drop-in replacement for DistributedDataParallel, simplifying integration into existing distributed training workflows.

It is a good idea to check whether you have enough GPU and CPU memory to fit your model. DeepSpeed provides a tool for estimating the required CPU/GPU memory.

```bash
python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'

# Output
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 354M total params, 51M largest layer params.
  per CPU  |  per GPU |   Options
    8.92GB |   0.19GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
    8.92GB |   0.19GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
    7.93GB |   0.85GB | offload_param=none, offload_optimizer=cpu , zero_init=1
    7.93GB |   0.85GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.29GB |   6.14GB | offload_param=none, offload_optimizer=none, zero_init=1
    1.98GB |   6.14GB | offload_param=none, offload_optimizer=none, zero_init=0
```
If my GPU has ≥6 GB VRAM, go for:
offload_param=none, offload_optimizer=none, zero_init=1
→ Fastest and most efficient for training speed.

If am limited on GPU memory (e.g., <4 GB VRAM), use:
offload_param=cpu, offload_optimizer=cpu, zero_init=1
→ Will be slower, but lets you train without OOM errors.

Let's compare the memory and compute trade-offs of various offloding techniques, 

### GPT2-Medium Optimization Benchmark Results

| Strategy                        | Train Time (s) | Avg GPU (MB) | Peak GPU (MB) | Train Loss | Eval Loss (Final) | Throughput (steps/s) |
|--------------------------------|------------------:|----------------:|-----------------:|---------------:|----------------------:|------------------------:|
| **AdamW (baseline)**           | 151.18            | 8103.41         | 8209.62          | 1.0705         | 0.7026                | 2.428                   |
| **FP16 + Gradient Checkpoint** | 97.29             | 8049.62         | 8235.88          | 1.1514         | —                     | 0.936                   |
| **ZeRO-1**                     | 85.86             | 10455.24        | 10710.50         | 1.5377         | —                     | 1.06                    |
| **ZeRO-2 + Offload Optimizer** | 162.99            | 10678.60        | 10798.94         | 1.5141         | —                     | 0.558                   |
| **ZeRO-3 + Full Offload**      | 258.11            | 3720.13         | 3809.00          | 1.6483         | —                     | 0.353                   |

[Want to try out - benchmark code](./finetune/experiments/optimization/deepspeed_exp/benchmark_v1.0.py)

#### 3.5.5.6 Zeroth-Order Optimization (MeZO, DiZO, SubZero)
Zeroth-Order (ZO) optimization presents a fundamentally different approach to training neural networks, potentially offering extreme memory efficiency by avoiding the standard backpropagation algorithm.

1. The core idea behind ZO methods is to estimate the gradient of the loss function with respect to the model parameters using only forward passes, treating the model essentially as a black box.6 Instead of calculating the exact gradient via backpropagation, ZO techniques typically work by:
Perturbing the model parameters θ slightly in one or more random directions (e.g., using a random vector z).
2. Evaluating the model's loss function L using these perturbed parameters (e.g., L(θ+ϵz) and potentially L(θ−ϵz)).
3. Using the observed changes in the loss value to estimate the gradient ∇L(θ). A common technique is Simultaneous Perturbation Stochastic Approximation (SPSA), which uses two forward passes with opposing perturbations to estimate the gradient.81

The primary memory saving of ZO methods comes from completely bypassing the need for backpropagation. Backpropagation requires storing intermediate activations from the forward pass to compute gradients, which is the dominant memory consumer in standard LLM fine-tuning. By relying solely on forward passes, ZO methods eliminate this activation memory overhead, potentially reducing memory costs significantly (e.g., up to 12x reduction claimed relative to FO methods, primarily due to activation savings). This makes ZO attractive for fine-tuning extremely large models on memory-constrained hardware.

##### MeZO (Memory-Efficient ZO): 
A foundational approach applying ZO to LLMs. It uses SPSA with full-parameter perturbation to estimate the gradient and updates parameters using ZO-SGD.81 While demonstrating the feasibility of ZO fine-tuning, MeZO often suffers from slow convergence. The variance of its gradient estimate typically scales with the model dimension, which is very high for LLMs, requiring many iterations for stable updates.6

##### DiZO (Divergence-aware ZO): 
Aims to accelerate MeZO's convergence. It analyzes the layer-wise differences ("divergence") between typical ZO updates and standard First-Order (FO) updates obtained via backpropagation. Based on this analysis, DiZO introduces a layer-wise adaptation mechanism (e.g., scaling factors) to make the ZO gradient estimates more closely align with FO gradients, particularly in terms of update magnitudes across layers.6 This is claimed to significantly reduce the number of training iterations needed for convergence, cutting training time (GPU hours) by up to 48% compared to 
MeZO and sometimes achieving accuracy comparable to or even exceeding FO fine-tuning.6

##### SubZero (Subspace ZO):
Addresses the high-dimensionality challenge differently. Instead of perturbing all parameters simultaneously, SubZero performs ZO optimization within randomly selected, low-dimensional subspaces of the parameter space, often using layer-wise low-rank perturbations.80 This approach aims to reduce the variance of the gradient estimate compared to full-parameter methods like MeZO, leading to faster convergence. It also further reduces memory consumption compared to MeZO by only needing to store and manipulate the low-rank perturbation matrices. SubZero is designed to be compatible with various fine-tuning schemes, including full-parameter tuning and PEFT methods like LoRA.

---
#### 3.5.5.7
FlashAttention-2 is a highly optimized version of the standard attention mechanism, designed to significantly accelerate inference by:
* Further parallelizing attention computation across the sequence length
* Efficiently distributing workloads among GPU threads to minimize communication and reduce shared memory read/write overhead.

> **_Note_** FlashAttention can only be used for models with the fp16 or bf16 torch type, so make sure to cast your model to the appropriate type first. The memory-efficient attention backend is able to handle fp32 models.

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```
#### 3.5.5 Selection of Fine-Tuning Technique

#### 3.5.6 Practical Examples
1. [Fine-Tuning GPT-2 for AG News Classification](./finetune/experiments/gpt2_ag_news_classifier/README.md)
2. [Fine-Tuning GPT-2 for code generation](./finetune/experiments/gpt2_py_code_generation/README.md)

### 3.6 Comprehensive Evaluation

## 🚀 Productionization & Operations (LLMOps)
*(Coming Soon)*

## 🔍♻️ Continuous Monitoring & Improvement
*(Coming Soon)*

## 📚 Appendix
### Archiecture
1. [The Transformer Architecture](./appendix/transformer_archiecture_overview.md)

## Reference
### Finetune
1. [Data Preparation Guide by Unsloth](https://docs.unsloth.ai/basics/datasets-guide)
2. [LLMDataHub](https://github.com/Zjh-819/LLMDataHub)

### General
1. [awesome-llms-fine-tuning repo 1](https://github.com/Curated-Awesome-Lists/awesome-llms-fine-tuning)
2. [awesome-llms-fine-tuning repo 2](https://github.com/pdaicode/awesome-LLMs-finetuning)

## 🤝 Contributing
We welcome contributors! See CONTRIBUTING.md for guidelines.

## 📜 License
This project uses:
Code: [Apache-2.0](LICENSE.txt)
Content: [LICENSE.CC-BY-NC](LICENSE.CC-BY-NC.txt)

