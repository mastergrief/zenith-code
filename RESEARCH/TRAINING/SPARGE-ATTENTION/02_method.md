# SpargeAttention — Method

← back to [SPARGE-ATTENTION.md](SPARGE-ATTENTION.md)

3SpargeAttn
SpargeAttn contains a two-stage online filter to implement sparse FlashAttention. First, as shown in Step1 and Step2 in Fig. 3, we design a fast and accurate method to predict the sparse block in the attention map, thereby skipping the corresponding products of 
Q
i
​
K
j
⊤
 and 
P
~
i
​
j
​
V
j
. Second, as shown in Step3 in Fig. 3, we design a sparse online softmax method to further skip the products of 
P
~
i
​
j
​
V
j
.

Refer to caption
Figure 4:Exemplary patterns of the query and key in the attention of various models.
3.1Sparse FlashAttention
SpargeAttn adopts the tiling strategy of FlashAttention (dao2023flashattention), and skip computing the blocks that are filtered out. Consider an attention operation 
S
=
Q
​
K
⊤
/
d
,
P
=
σ
​
(
S
)
,
O
=
P
​
V
, where 
σ
​
(
S
)
i
​
j
=
exp
⁡
(
S
i
​
j
)
/
∑
k
exp
⁡
(
S
i
​
k
)
 is the softmax operation. Let 
N
 be the sequence length and 
d
 be the dimensionality of each head; the matrices 
Q
, 
K
, and 
V
 each have dimensions 
N
×
d
, while the matrix 
S
 and 
P
 is 
N
×
N
. FlashAttention proposes to tile 
Q
, 
K
, and 
V
 from the token dimension into blocks 
{
Q
i
}
,
{
K
i
}
,
{
V
i
}
 with block sizes 
b
q
, 
b
k
, 
b
k
, respectively. Then, it uses online softmax (milakov2018online) to progressively compute each block of 
O
, i.e., 
O
i
:

S
i
​
j
=
Q
i
​
K
j
⊤
/
d
,
(
m
i
​
j
,
P
~
i
​
j
)
=
σ
~
​
(
m
i
,
j
−
1
,
S
i
​
j
)
,
l
i
​
j
=
exp
⁡
(
m
i
,
j
−
1
−
m
i
​
j
)
​
l
i
,
j
−
1
+
rowsum
​
(
P
~
i
​
j
)
,
O
i
​
j
=
diag
​
(
exp
⁡
(
m
i
,
j
−
1
−
m
i
​
j
)
)
​
O
i
,
j
−
1
+
P
~
i
​
j
​
V
j
(1)
where 
m
i
​
j
 and 
l
i
​
j
 are 
b
q
×
1
 vectors, which are initialized to 
−
∞
 and 
0
 respectively. The 
σ
~
​
(
)
 is an operator similar to softmax.: 
m
i
​
j
=
max
⁡
{
m
i
,
j
−
1
,
rowmax
​
(
S
i
​
j
)
}
,
P
~
i
​
j
=
exp
⁡
(
S
i
​
j
−
m
i
​
j
)
. Finally, the output 
O
i
 can be computed by 
O
i
=
diag
​
(
l
i
​
j
)
−
1
​
O
i
​
j
.

Implementing sparse FlashAttention is intuitive. By skipping certain block matrix multiplications of 
Q
i
​
K
j
⊤
 and 
P
~
i
​
j
​
V
j
, we can accelerate the attention computation. We formulate sparse attention based on FlashAttention in the following definitions.

Definition 1 (Block Masks).
Let 
M
g
 and 
M
p
​
v
 be binary masks of dimensions 
⌈
N
/
b
q
⌉
×
⌈
N
/
b
k
⌉
, where each value is either 0 or 1. These masks determine which computations are skipped in the sparse attention mechanism.

Definition 2 (Sparse FlashAttention).
The computation rules for sparse FlashAttention based on the masks are defined as follows:

Q
i
​
K
j
⊤
,
P
~
i
​
j
​
V
j
​
are
​
skipped
​
if
​
M
g
​
[
i
,
j
]
=
0
.
(2)
P
~
i
​
j
​
V
j
​
is
​
skipped
​
if
​
M
p
​
v
​
[
i
,
j
]
=
0
.
(3)
Algorithm 1 Implementation of SpargeAttn.
1:  Input: Matrices 
Q
​
(
FP16
)
,
K
​
(
FP16
)
,
V
​
(
FP16
)
∈
ℝ
N
×
d
, block size 
b
q
,
b
k
​
v
, count of GPU Warps 
c
w
, hyper-parameters 
τ
,
θ
,
 and 
λ
.
2:  Divide 
Q
 to 
T
m
=
N
/
b
q
 blocks 
{
Q
i
}
; divide 
K
, 
V
 to 
T
n
=
N
/
b
k
​
v
 blocks 
{
K
i
}
 and 
{
V
i
}
.
3:  
Q
^
i
,
K
^
j
,
δ
Q
,
δ
K
=
Quant
​
(
Q
i
,
K
j
)
 ; // per-block quantization in SageAttention.
4:  
q
=
{
q
i
}
=
{
mean
​
(
Q
i
,
axis
=
0
)
}
 ;   
k
=
{
k
j
}
=
{
mean
​
(
K
j
,
axis
=
0
)
}
 ;
5:  
S
^
=
q
​
k
⊤
;
s
q
​
i
=
CosSim
​
(
Q
i
)
;
s
k
​
j
=
CosSim
​
(
K
j
)
;
S
^
​
[
:
,
j
]
=
−
∞
,
If
​
s
k
​
j
<
θ
;
6:  
P
^
​
[
i
]
=
Softmax
​
(
S
^
​
[
i
]
)
 ;   
M
​
[
i
,
:
]
=
TopCdf
​
(
P
^
​
[
i
]
,
τ
)
 ;   
M
​
[
i
,
:
]
=
1
,
If
​
s
q
​
i
<
θ
 ;   
M
​
[
:
,
j
]
=
1
,
If
​
s
k
​
j
<
θ
 ;
7:  for 
i
=
1
 to 
T
m
 do
8:  Load 
Q
^
i
 and 
δ
Q
​
[
i
]
 into a SM ;
9:  for j in [1, 
T
n
] do
10:   if 
M
​
[
i
,
j
]
!
=
0
 then
11:    Load 
K
^
j
, 
V
^
j
, and 
δ
K
​
[
j
]
 into the SM ;
12:    
S
i
​
j
=
Matmul
​
(
Q
^
i
,
K
^
j
T
)
×
δ
Q
×
δ
K
 ; // dequantization of SageAttention.
13:    
m
local
=
rowmax
​
(
S
i
​
j
)
;
m
i
​
j
=
max
​
(
m
i
,
j
−
1
,
m
local
)
;  
P
~
i
​
j
=
exp
​
(
S
i
​
j
−
m
i
​
j
)
;  
l
i
​
j
=
e
m
i
,
j
−
1
−
m
i
​
j
​
l
i
,
j
−
1
+
rowsum
​
(
P
~
i
​
j
)
;
14:    
i
w
=
range
​
(
c
w
)
 ;   
I
w
=
[
i
w
∗
b
q
c
w
:
(
i
w
+
1
)
∗
b
q
c
w
]
 ;
15:    if 
max
⁡
(
m
local
​
[
I
w
]
−
m
i
​
j
​
[
I
w
]
)
>
λ
 then
16:     
O
i
​
j
​
[
I
w
]
=
diag
​
(
e
m
i
,
j
−
1
​
[
I
w
]
−
m
i
​
j
​
[
I
w
]
)
​
O
i
,
j
−
1
​
[
I
w
]
+
 
Matmul
​
(
P
~
i
​
j
​
[
I
w
]
,
V
j
)
 ; // Paralleled by 
c
w
 warps.
17:    end if
18:   end if
19:  end for
20:  
O
i
=
diag
​
(
l
i
,
T
n
)
−
1
​
O
i
,
T
n
 ;
21:  Write 
O
i
 ;
22:  end for
23:  return 
O
=
{
O
i
}
 ;
3.2Selective Token Compression for Sparse Prediction
Key idea. Although attention maps vary across models, we observe that various models exhibit a common trait: Most neighboring tokens in the query and key matrices of the attention show high similarity (See Fig. 4). Consequently, for blocks composed of highly similar tokens, we can consolidate these tokens into a single representative token for the block. Based on this observation, we propose a pattern-free online prediction method for identifying sparse blocks in 
P
 to skip some computation of 
Q
i
​
K
j
⊤
 and 
P
~
i
​
j
​
V
j
 during the FlashAttention process. Specifically, we first compress blocks exhibiting high self-similarity within 
Q
 and 
K
 into tokens. Then, we swiftly compute a compressed attention map 
P
^
 using the compressed 
Q
 and 
K
. Finally, we selectively compute 
{
Q
i
​
K
j
⊤
,
P
~
i
​
j
​
V
j
}
 for those pairs 
(
i
,
j
)
 where 
{
P
^
​
[
i
,
j
]
}
 accumulates a high score in the compressed attention map. Notably, block selection was only performed in the high self-similarity blocks, which we also refer to as ”selective blocks.” For those non-self-similar blocks, as a good presentation token for the whole block is hard to find, we choose to always compute the non-self-similar block in the attention operation, which we also refer to as ”fix blocks.” Importantly, compressing only the token blocks with high self-similarity is crucial, as omitting computations for fix blocks can result in the loss of critical information. This will be confirmed in Sec. 4 and A.2.

Prediction. As shown in Step1 in Fig. 3, we first compute a mean cosine similarity across tokens for each block of 
Q
 and 
K
. Next, we compress each block into a single token by calculating a mean across tokens. Then, we compute a compressed 
Q
​
K
⊤
 using the compressed 
Q
 and 
K
. Finally, to prevent interference from non-self-similar blocks, i.e., the block similarity less than a hyper-parameter 
θ
, we set the corresponding values in 
S
 to 
−
∞
, and then obtain a compressed attention map through softmax. This algorithm can be expressed as:

q
=
{
q
i
}
=
{
mean
(
Q
i
,
axis
=
0
)
}
k
=
{
k
j
}
=
{
mean
(
K
j
,
axis
=
0
)
}
s
q
​
i
=
CosSim
​
(
Q
i
)
,
s
k
​
j
=
CosSim
​
(
K
j
)
S
^
​
[
i
]
=
q
i
​
k
⊤
;
S
^
​
[
:
,
j
]
=
−
∞
,
If
​
s
k
​
j
<
θ
P
^
​
[
i
]
=
Softmax
(
S
^
​
[
i
]
)
where 
Q
i
∈
ℝ
b
q
×
d
,
q
i
∈
ℝ
1
×
d
,
K
j
∈
ℝ
b
k
×
d
,
k
j
∈
ℝ
1
×
d
 and 
CosSim
​
(
X
)
=
m
​
e
​
a
​
n
​
(
X
​
X
⊤
|
max
⁡
(
X
​
X
⊤
)
|
)
 measures the cosine-similarity within a block.

For each row of 
P
^
, i.e., 
P
^
​
[
i
]
, we select the positions of the top values whose cumulative sum reaches 
τ
⋅
∑
P
^
​
[
i
]
, where 
τ
 is a hyper-parameter. These positions are set to 1 in 
M
g
​
[
i
,
:
]
, while all other positions are set to 0.

M
g
​
[
i
,
:
]
=
TopCdf
​
(
P
^
​
[
i
]
,
τ
)
(4)
where the 
TopCdf
​
(
P
^
​
[
i
]
,
τ
)
 can be formulated as follows.

def Top_Cdf(P[i], tau):
sorted_P, idx = torch.sort(P[i], descending=True)
cusum_P = torch.cumsum(sorted_P, dim=0)
mask = cusum_P <= tau * P[i].sum()
M_i = torch.zeros_like(mask)
M_i[idx] = mask
return M_i
Finally, we need to ensure that calculations involving non-self-similar blocks(fix block) of 
Q
 or 
K
 are not omitted. Therefore, we set all values in the rows of 
M
g
 corresponding to not self-similar blocks of 
Q
 to 1, and all values in the columns of 
M
g
 corresponding to non-self-similar blocks of 
K
 to 1.

M
g
​
[
i
,
:
]
=
1
,
If
​
s
q
​
i
<
θ
;
M
g
​
[
:
,
j
]
=
1
,
If
​
s
k
​
j
<
θ
(5)
3.3Masking of the First Stage
Masking. The 
M
g
 can be applied in FlashAttention directly to save some computation. In the inner loop of FlashAttention, i.e., during computing attention between a 
Q
i
 and 
{
K
j
}
,
{
V
j
}
, we can skip {
Q
i
​
K
j
⊤
, 
P
~
i
​
j
​
V
j
} when 
M
g
​
[
i
,
j
]
=
0
.

Skip
​
Q
i
​
K
j
⊤
​
and
​
P
~
i
​
j
​
V
j
,
If
​
M
g
​
[
i
,
j
]
=
0
(6)
3.4Sparse Warp Online Softmax
Key idea. We can further identify the small enough values in the attention map during the online softmax process. If all values in 
P
~
i
​
j
 are close enough to zero, the 
P
~
i
​
j
​
V
j
 will be negligible and can be omitted.

To identify which 
P
~
i
​
j
=
exp
⁡
(
S
i
​
j
−
m
i
,
j
)
 (See Sec. 3.1) contains values small enough to be omitted, we note that in every inner loop of FlashAttention, the 
O
i
​
j
 will be scaled by 
exp
⁡
(
m
i
,
j
−
1
−
m
i
​
j
)
 and then plus the 
P
~
i
​
j
​
V
j
:

m
local
=
rowmax
​
(
S
i
​
j
)
,
m
i
​
j
=
max
⁡
{
m
i
,
j
−
1
,
m
local
}
O
i
​
j
=
diag
​
(
exp
⁡
(
m
i
,
j
−
1
−
m
i
​
j
)
)
​
O
i
,
j
−
1
+
P
~
i
​
j
​
V
j
If 
rowmax
​
(
S
i
​
j
)
<
m
i
​
j
, then 
m
i
​
j
=
m
i
,
j
−
1
. Consequently, 
O
i
​
j
=
O
i
,
j
−
1
+
P
~
i
​
j
​
V
j
. Furthermore, if 
rowmax
​
(
S
i
​
j
)
≪
m
i
​
j
 holds true, then all values in 
P
~
i
​
j
=
exp
⁡
(
S
i
​
j
−
m
i
​
j
)
 are close to 0. This results in all values in 
P
~
i
​
j
​
V
j
 being close to 0. This condition implies that 
P
~
i
​
j
​
V
j
 is negligible when 
rowmax
​
(
S
i
​
j
)
 is significantly smaller than 
m
i
​
j
:

O
i
​
j
≈
O
i
,
j
−
1
,
if 
​
max
⁡
(
exp
⁡
(
S
i
​
j
−
m
i
​
j
)
)
→
0
max
⁡
(
exp
⁡
(
S
i
​
j
−
m
i
​
j
)
)
→
0
⇔
max
⁡
(
m
local
−
m
i
​
j
)
<
λ
The above equivalence is satisfied when 
λ
 is small enough.

Therefore, based on the analysis above, we propose a simple yet effective sparse method to further skip the 
P
~
i
​
j
​
V
j
 computation. Specifically, in the inner loop of FlashAttention, the 
S
i
​
j
 will be split by 
c
w
 GPU warps to {
S
i
​
j
[
i
w
∗
b
q
c
w
:
(
i
w
+
1
)
∗
b
q
c
w
,
:
]
}, where 
i
w
 is the index of the GPU warp. Let 
I
w
=
[
i
w
∗
b
q
c
w
:
(
i
w
+
1
)
∗
b
q
c
w
]
. If 
max
⁡
(
m
local
​
[
I
w
]
−
m
i
​
j
​
[
I
w
]
)
<
λ
, where 
λ
 is small enough, then 
O
i
​
j
​
[
I
w
]
≈
O
i
,
j
−
1
​
[
I
w
]
, and we will skip the computation of 
P
~
i
​
j
​
[
I
w
]
​
V
j
 which is used to update 
O
i
​
j
​
[
I
w
]
.

3.5Combined with SageAttention
To further accelerate our implementation of sparse attention, we integrate our method into SageAttention (zhang2024sageattention2; 2024sageattention; zhang2025sageattention2++; zhang2025sageattention2_wksp; zhang2025sageattention3), which proposes a quantized method for accelerating attention. Since quantization (hu2025quant; zhang2025int8train) operations and sparse operations are orthogonal, sparse computation can be directly applied to SageAttention. The complete algorithm is shown in Algorithm 1. Specifically, first, we need to add one judgment at the beginning of the inner loop of SageAttention (Line 10, Algorithm 1) to decide whether to skip the whole inner loop once. Second, we add another judgment before the updating of 
O
i
​
j
 in the inner loop of SageAttention (Line, in Algorithm 1) to decide whether to skip the computation of 
P
~
i
​
j
​
V
j
. Moreover, to minimize the attention map prediction overhead, we implement the prediction using CUDA and adopt some kernel fusion techniques.

3.6Hyper-parameters Determination for Model Layer
Based on the method description in Sec. 3.2 and 3.4, our method incorporates three hyper-parameters: 
τ
∈
(
0
,
1
)
, 
θ
∈
(
−
1
,
1
)
, and 
λ
<
0
. The parameter determination process for each attention layer in any model is straightforward. We aim to identify a set of hyperparameters that not only maximize attention sparsity but also constrain the attention error across five different model inputs. To evaluate attention accuracy, we employ a strict error metric, the Relative L1 distance, defined as 
L
​
1
=
∑
|
O
−
O
′
|
/
∑
|
O
|
. The process begins by setting two L1 error thresholds 
l
1
 and 
l
2
, e.g., 
l
1
=
0.05
,
l
2
=
0.06
. We first conduct a grid search for 
τ
 and 
θ
 to identify the optimal pair that maximizes sparsity while ensuring 
L
​
1
<
l
1
. Subsequently, we perform another grid search for 
λ
 to find the optimal value that further maximizes sparsity while maintaining 
L
​
1
<
l
2
.

Refer to caption
Figure 5:Illustration of different token permutation methods in 
1
×
6
×
6
 space, with block size of 4.
3.7HilbertCurve Permutation
Key idea. Improving sparsity while maintaining accuracy is a key challenge in enhancing the performance of sparse attention. In our algorithm, increasing the self-similarity of key and query blocks can reduce the number of fix blocks. This allows more selective blocks to participate in 
TopCdf
 selection, thereby improving sparsity. Since attention is computationally invariant to token permutations, the problem reduces to finding a permutation that enhances the similarity of adjacent tokens.

Image and video models benefit from strong priors: adjacent pixels are likely to be similar. To better leverage this prior, we propose the HilbertCurve permutation, given 3D visual tokens 
Q
,
K
,
V
∈
ℝ
T
×
H
×
W
×
d
, We use the Hilbert Curve to fill the 3D space and then flatten tokens along the curve into shape 
ℝ
L
×
d
,
L
=
T
×
H
×
W
. Fig. 5 illustrates an example of 
1
×
6
×
6
 visual tokens flattened by row-major order and HilbertCurve. The Hilbert Curve preserves locality effectively, traversing the entire 3D space without crossing rows or columns, thereby increasing the similarity of adjacent tokens and the sparsity of attention.

Table 1:End-to-end metrics across text, image, and video generation models. ✗ indicates an inability to generate results for evaluation. The speed and sparsity are the average for each layer in the model in real generation tasks described in Sec. 4.1. The speed and sparsity of Llama3.1 are measured in the Needle-in-a-Haystack task with a 128K sequence length.
Model (seq_len)
 	
Attention (Sparsity)
Speed (
1
/
t
)
↑
WikiText (Ppl.) 
↓
Longbench 
↑
InfiniteBench 
↑
NIAH 
↑
Llama3.1
(128K)
 	
Full-Attention
156.9	6.013	38.682	0.6594	0.907
Minference (0.5)
 	140.1	10.631	28.860	0.5152	0.832
FlexPrefill (0.5)
 	240.6	6.476	38.334	0.6460	0.858
Minference (0.3)
 	115.7	6.705	34.074	0.6532	0.870
FlexPrefill (0.42)
 	206.9	6.067	38.334	0.6581	0.878
SpargeAttn (0.54)
 	708.1	6.020	39.058	0.6638	0.909
Model (seq_len)
 	
Attention (Sparsity)
Speed (
1
/
t
)
↑
CLIPSIM 
↑
CLIP-T 
↑
VQA-a 
↑
VQA-t 
↑
FScore 
↑
   
CogvideoX
(17K)
 	
Full-Attention
166.0	0.1819	0.9976	80.384	75.946	5.342
Minference (0.5)
 	264.6	0.1728	0.9959	70.486	62.410	2.808
FlexPrefill (0.6)
 	175.3	0.1523	0.9926	1.5171	4.5034	1.652
Minference (0.3)
 	196.9	0.1754	0.9964	77.326	63.525	3.742
FlexPrefill (0.45)
 	142.0	0.1564	0.9917	7.7259	8.8426	2.089
SpargeAttn (0.46)
 	507.9	0.1798	0.9974	78.276	74.846	5.030
    
Mochi
(22K)
 	
Full-Attention
164.2	0.1725	0.9990	56.472	67.663	1.681
Minference (0.5)
 	202.4	0.1629	0.9891	6.668	50.839	0.653
FlexPrefill (0.48)
 	191.3	0.1667	0.9898	0.582	0.0043	✗
Minference (0.3)
 	147.7	0.1682	0.9889	14.541	42.956	0.833
FlexPrefill (0.4)
 	171.7	0.1677	0.9909	2.941	0.7413	✗
SpargeAttn (0.47)
 	582.4	0.1720	0.9990	54.179	67.219	1.807
Model (seq_len)
 	
Attention (Sparsity)
CLIPSIM 
↑
CLIP-T 
↑
VQA-a 
↑
VQA-t 
↑
FScore 
↑
Latency 
↓
Open-Sora-Plan
(38K)
 	
Full-Attention
0.1650	0.9994	81.40	80.60	0.847	629s
SpargeAttn (0.34)
 	0.1686	0.9985	77.59	76.91	0.839	393s
   
Model (seq_len)
 	   
Attention (Sparsity)
  Speed (
1
/
t
)
↑
  FID 
↓
  CLIP 
↑
  IR 
↑
   
  Flux
  (4.5K)
 	   
Full-Attention
  158.2	  166.103	  31.217	  0.8701
   
Minference (0.5)
 	  151.8	  180.650	  30.235	  0.4084
   
FlexPrefill (0.48)
 	  47.7	  443.928	  18.3377	  -2.2657
   
Minference (0.3)
 	  118.9	  170.221	  31.001	  0.7701
   
FlexPrefill (0.41)
 	  40.9	  405.043	  19.5591	  -2.2362
   
SpargeAttn (0.38)
 	  280.3	  163.982	  31.448	  0.9207
   
  Stable-
  Diffusion3.5
  (4.5K)
 	   
Full-Attention
  164.2	  166.101	  32.007	  0.9699
   
Minference (0.5)
 	  186.4	  348.930	  18.3024	  -2.2678
   
FlexPrefill (0.37)
 	  23.1	  350.497	  18.447	  -2.2774
   
Minference (0.3)
 	  150.3	  337.530	  18.099	  -2.2647
   
FlexPrefill (0.35)
 	  22.7	  348.612	  18.147	  -2.2756
   
SpargeAttn (0.31)
 	  293.0	  166.193	  32.114	  0.9727
