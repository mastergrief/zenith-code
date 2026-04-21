# Litespark — Introduction & Experimental Setup

← back to [LITESPARK.md](LITESPARK.md)

1Introduction
The exponential growth of large language models (LLMs) since GPT-3’s release in 2020 [1] has fundamentally transformed the way we use artificial intelligence (AI) in daily lives. Since then, a plethora of LLMs have been developed [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], demonstrating significant progress towards the pursuit of artificial general intelligence (AGI). However, this progress has come at an unprecedented computational and environmental cost. Training modern LLMs now requires months of compute time, tens of millions of dollars, and energy consumption equivalent to powering thousands of homes for years [20].

The scale of this challenge has intensified over the years. For example, GPT-3 175B model was trained for 3,640 PF-days [1], which would take around 
50
−
70
 days on 
1
,
000
 NVIDIA V100 GPUs. Llama 3.1-405B reportedly consumed 
30.84
 million GPU-hours [21], equivalent to 
80
 days of training on 
16
,
000
 H100 GPUs. Simultaneously, energy consumption has exploded: from GPT-3’s 
1
,
287
 MWh [22] to Llama 3.1-405B’s approximately 21.6 GWh (based on 700 W consumption per H100 GPU) [21]. The environmental impact of model training has increased dramatically. Llama-3.1-405B’s training would leave a carbon footprint reaching 8,930 tonnes CO2eq [21], representing a 16-fold increase over GPT-3 175B’s 552.1 tonnes CO2eq [22].

Both of these challenges – extended training times and massive energy consumption – stem from the same fundamental issue: inefficient utilization of computational resources during transformer training. Despite consuming full power, GPUs during standard LLM pre-training often operate at suboptimal utilization rates of 
30
%
−
50
%
. This inefficiency creates a compound problem: training takes longer than necessary while simultaneously wasting energy: organizations face both extended time-to-market delays and inflated energy costs. Running ablations in the model development phase becomes slower and prohibitively costly, effectively limiting the breadth of scientific exploration.

The suboptimal performance in LLM pre-training stems largely from bottlenecks in the core transformer architecture, particularly in the attention and MLP (Multi-Layer Perceptron) layers that constitute the majority of computational operations. The attention mechanism [23] suffers from inherent limitations in memory bandwidth that prevent GPUs from achieving maximum computational throughput. Traditional attention implementations are memory-bound rather than compute-bound, causing expensive GPU compute units to remain idle while waiting for data transfers [24]. Similarly, standard MLP layers often fail to fully utilize modern GPU capabilities, particularly the specialized Tensor Core units designed for high-throughput operations [25]. These architectural inefficiencies translate directly into wasted energy: every second a GPU operates below capacity represents energy consumed without proportional computational gains.

Recent research has demonstrated that algorithmic improvements can address both challenges simultaneously. Techniques like FlashAttention achieve 
2
x–
3
x training speedup while maximizing GPU utilization [24, 26, 27]. Mixture-of-Experts approaches reduce training time and computational requirements by 4x–7x through sparse activation patterns [28].

In this technical report, we introduce Litespark, a novel pre-training framework that simultaneously addresses both training time and energy efficiency challenges through targeted optimizations to the transformer architecture’s attention and MLP layers. Our approach focuses on maximizing Model FLOPs Utilization (MFU) while maintaining compatibility with standard transformer implementations. The optimizations occur in two steps.

• Architectural optimization: optimizes the attention and MLP blocks in the transformer architecture.
• Algorithmic optimization: optimizes the forward and backward pass operations to increase FLOPs per GPU.
Litespark offers 2x–6x enhancement in training throughput, and 
55
%
−
83
% reduction in the energy consumption during the pre-training process. Notably, these optimizations add on top of the performance improvements from known existing techniques like flash-attention, quantization, model pruning etc. Furthermore, the optimizations are model- and hardware-agnostic, and can be incorporated into any model architectures and hardware families including GPUs and ASICs.

The report is organized as follows. In section 2, we describe the setup for running benchmarking experiments comparing the performance of pre-training models with Litespark vs. Llama baselines. Section 3 showcases the main results in terms of enhanced throughput and energy efficiency. Section 4 points out some future directions of research, and we conclude in section 5.

2Experimental Setup
To evaluate the effectiveness of the Litespark framework, we conducted comprehensive benchmarking experiments comparing our optimized implementation against baseline Llama models across multiple scales and configurations. Our experimental design focuses on measuring both training acceleration and energy efficiency improvements while ensuring fair comparison through identical model architectures, datasets, and training hyperparameters. The evaluation covers scenarios from single-node training to large-scale distributed setups, enabling assessment of how our optimizations perform across the full spectrum of practical deployment scenarios.

2.1Hardware infrastructure
All experiments were conducted on an Amazon SageMaker Hyperpod cluster equipped with NVIDIA H200 GPUs. The H200 represents a recent generation of data center GPUs, featuring 141GB of HBM3e memory and peak theoretical performance of 989 TFLOPS for BF16 operations [29]. Our multi-node distributed training setup utilized high-bandwidth InfiniBand interconnects for intra-node communication between GPUs and AWS Elastic Fabric Adapter (EFA) with NCCL for inter-node communication to minimize communication overhead during parameter synchronization.

We evaluated scalability across multiple node configurations: 1, 2, 16, 32, and 64 nodes for different model sizes. Each node contained 8 H200 GPUs, enabling evaluation of training performance from single-node (8 GPUs) to large-scale distributed scenarios (512 GPUs for the largest configuration). This range allows us to assess both the baseline efficiency improvements and how our optimizations scale with increasing distributed training complexity.

2.2Dataset
Training was performed on the SlimPajama-627B dataset [30], a refined version of the RedPajama dataset [31], containing 627 billion tokens of high-quality text data. SlimPajama consists of web pages, books, academic papers, code repositories, and reference materials, representing a diverse and representative sample for general-purpose language model pre-training. The dataset has been preprocessed to remove low-quality content and deduplicated to improve training efficiency. This dataset choice allows for direct comparison with other published LLM training results while ensuring sufficient scale to evaluate performance across extended training runs.

2.3Tokenizer
The training dataset was pre-processed utilizing a SentencePiece-based tokenizer [32] with a vocabulary size of 32,000 tokens. This tokenizer choice ensures consistency with the Llama model family and enables direct performance comparisons without introducing tokenization-related variations. The tokenizer employs byte-pair encoding (BPE) [33] to handle out-of-vocabulary words and maintains compatibility with the original Llama tokenization scheme, ensuring that our optimizations can be fairly evaluated against baseline implementations using identical text preprocessing.

2.4Model architecture
We have chosen two model configurations based on the Llama architecture to enable direct performance comparison, as shown in Table 1.

Model size	n_layers	hidden_size	n_heads	kv_heads	intermediate_size
3B	28	2,048	16	2	11,008
30B	60	6,656	64	64	17,920
Table 1:Model configuration
Both models utilize standard Llama architectural components including RMSNorm for layer normalization [34], SwiGLU activation functions [35] in the MLP layers, and Rotary Positional Embeddings (RoPE) for position encoding [36]. The 3B model employs grouped query attention (GQA) [37] to reduce memory overhead, while the 30B model uses standard multi-head attention. These configurations were chosen to represent both smaller models suitable for research experimentation and larger models representative of production deployments.

2.5Pre-training configuration
2.5.1Distributed training setup
We employed a combination of data parallelism and tensor parallelism to distribute training across multiple nodes and GPUs [38, 39]. Data parallelism replicates the model across different GPU groups, while tensor parallelism splits individual layers across GPUs to handle models that exceed single-GPU memory capacity.

2.5.2Optimizer settings
All models were trained in BF16 mixed precision using the AdamW optimizer [40] with 
β
1
=
0.90
, 
β
2
=
0.95
, weight decay of 0.01, and gradient clipping threshold of 1.0. We used ZeRO Stage 1 optimization [41] to distribute optimizer states across GPUs while maintaining model replicas.

2.5.3Learning rate schedule
Training employed a cosine learning rate scheduler with maximum learning rate of 
1.2
×
10
−
3
, minimum learning rate of 
1.0
×
10
−
5
, and 
2
,
000
 warmup steps. The global batch size was set to 
256
 across all configurations, with micro-batch sizes adjusted based on memory constraints and node configuration.

Hyperparameter	Value
Optimizer	AdamW (
β
1
=
0.90
, 
β
2
=
0.95
)
ZeRO stage	1
Learning rate scheduler	Cosine
Max learning rate	
1.2
×
10
−
3
Min learning rate	
1.0
×
10
−
5
Warmup steps	
2
,
000
Batch size	
256
Weight decay	
0.01
Gradient clipping threshold	
1.0
Table 2:Pre-training hyperparameters
2.5.4Evaluation metrics
We measured training throughput (tokens per second), computational efficiency (TFLOPs per GPU), training time per iteration, Model FLOPs Utilization (MFU), and total energy consumption in MWh per 500 billion tokens processed. Energy measurements were calculated by integrating GPU power consumption over training time as reported by Wandb telemetry [42]. Values represent direct GPU energy consumption during training.

