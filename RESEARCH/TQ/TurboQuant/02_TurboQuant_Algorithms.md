# TurboQuant — Part 2: Algorithms
_Part 2 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

3TurboQuant: High Performance Quantization
We developed two VQ algorithms, each tailored to a specific objective. The first algorithm is designed to minimize the MSE between the original and reconstructed vectors after quantization. The second algorithm is optimized for unbiased inner product estimation, addressing the bias inherent in MSE-optimal quantizers. These algorithms are detailed in the following subsections.

Furthermore, in ??, we establish information-theoretic lower bounds on the best achievable distortion rates for any vector quantizer. This analysis demonstrates that TurboQuant achieve near-optimality, differing from the lower bound by only a small constant factor across all bit-widths.

3.1MSE Optimal TurboQuant
Let 
𝒙
∈
SS
d
−
1
 be a (worst-case) vector on the unit sphere in dimension 
d
. We aim to quantize 
𝒙
 to 
b
 bits per coordinate while minimizing the reconstruction MSE defined in ??. We start by randomizing this vector by multiplying it with a random rotation matrix 
𝚷
∈
ℝ
d
×
d
. We can generate 
𝚷
 by applying QR decomposition on a random matrix with i.i.d Normal entries.

The resulting rotated vector, 
𝚷
⋅
𝒙
, is uniformly distributed on the unit sphere 
SS
d
−
1
. As shown in ??, each coordinate of 
𝚷
⋅
𝒙
 follows a Beta distribution, which converges to a normal distribution in high dimensions. Furthermore, in high dimensions, distinct coordinates of 
𝚷
⋅
𝒙
 become nearly independent [55], allowing us to apply optimal scalar quantizers to each coordinate independently. Therefore, by ??, our task reduces to designing a scalar quantizer for random variables with the distribution 
f
X
​
(
x
)
=
Γ
​
(
d
/
2
)
π
⋅
Γ
​
(
(
d
−
1
)
/
2
)
​
(
1
−
x
2
)
(
d
−
3
)
/
2
 for 
x
∈
[
−
1
,
1
]
.

The optimal scalar quantization problem, given a known probability distribution, can be framed as a continuous k-means problem in dimension one. Specifically, we aim to partition the interval 
[
−
1
,
1
]
 into 
2
b
 clusters/buckets. The optimal solution adheres to a Voronoi tessellation [42], meaning interval boundaries are the midpoints between consecutive centroids, when arranged in sorted order. Therefore, with 
c
i
’s denoting the centroids in ascending order, we can formulate the scalar quantization as the following k-means optimization problem:

𝒞
​
(
f
X
,
b
)
:=
min
−
1
≤
c
1
≤
c
2
≤
…
≤
c
2
b
≤
1
​
∑
i
=
1
2
b
∫
c
i
−
1
+
c
i
2
c
i
+
c
i
+
1
2
|
x
−
c
i
|
2
⋅
f
X
​
(
x
)
​
𝑑
x
.
(4)
Note that 
𝒞
​
(
f
X
,
b
)
 in ?? denotes the optimal MSE cost function for bit-width 
b
, a quantity we will bound to prove the upper bound on the end-to-end MSE of TurboQuant. The problem in ?? can be solved using iterative numerical methods to achieve any desired precision. We solve ?? for a range of practically relevant bit-widths 
b
 once, and store the results for future uses by the quantizer.

For example, in moderately high dimensions 
d
, where the distribution 
f
X
​
(
x
)
 closely approximates a normal distribution, the optimal quantization centroids for bit-widths 
b
=
1
,
2
 are 
{
±
2
/
π
d
}
 and 
{
±
0.453
d
,
±
1.51
d
}
, respectively.

Therefore the quantizer 
Q
𝚖𝚜𝚎
:
ℝ
d
→
{
0
,
1
}
b
⋅
d
 first computes 
𝚷
⋅
𝒙
 and then computes and stores the indices of the nearest centroids to each coordinate of this vector. The dequantization map 
Q
𝚖𝚜𝚎
−
1
:
{
0
,
1
}
b
⋅
d
→
ℝ
d
 reconstructs the vector by retrieving the centroids corresponding to the stored indices and then rotating the result back to the original basis through multiplication with 
𝚷
⊤
. A pseudocode for these procedures is given in ??.

Algorithm 1 
TurboQuant
𝚖𝚜𝚎
: optimized for MSE
1: input: dimension 
d
 and bit-width 
b
 // Global Parameters for Setting up 
TurboQuant
𝚖𝚜𝚎
2: Generate a random rotation matrix 
𝚷
∈
ℝ
d
×
d
3: Construct codebook by finding centroids 
c
1
,
c
2
,
…
​
c
2
b
∈
[
−
1
,
1
]
 that minimize MSE cost in ??  
4: Procedure 
Quant
𝚖𝚜𝚎
​
(
𝒙
)
5: 
𝒚
←
𝚷
⋅
𝒙
6: 
𝚒𝚍𝚡
j
←
arg
⁡
min
k
∈
[
2
b
]
⁡
|
𝒚
j
−
c
k
|
 for every 
j
∈
[
d
]
{ 
𝚒𝚍𝚡
j
’s are 
b
-bit integers}
7: output: 
𝚒𝚍𝚡
  
8: Procedure 
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
9: 
𝒚
~
j
←
c
𝚒𝚍𝚡
j
 for every 
j
∈
[
d
]
10: 
𝒙
~
←
𝚷
⊤
⋅
𝒚
~
11: output: 
x
~
We are now ready to prove our main theorem for 
TurboQuant
𝚖𝚜𝚎
.

Theorem 1 (performance guarantee: 
TurboQuant
𝚖𝚜𝚎
). For any bit-width 
b
≥
1
 and any vector 
𝐱
∈
SS
d
−
1
, the procedure 
Quant
𝚖𝚜𝚎
​
(
𝐱
)
 in ?? outputs an index vector 
𝚒𝚍𝚡
∈
[
2
b
]
d
. When this index vector is passed to the primitive 
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
, it produces a reconstructed vector 
𝐱
~
∈
ℝ
d
 that satisfies the following distortion bounds:
• MSE defined as 
D
𝚖𝚜𝚎
:=
𝔼
𝒙
~
[
‖
𝒙
−
𝒙
~
‖
2
2
]
 is bounded by 
D
𝚖𝚜𝚎
≤
3
​
π
2
⋅
1
4
b
 for any 
b
≥
0
.
• For small bit-widths, specifically 
b
=
1
,
2
,
3
,
4
 the MSE exhibits finer-grained distortion values: 
D
𝚖𝚜𝚎
≈
0.36
,
0.117
,
0.03
,
0.009
, respectively.
Proof.We start the proof by showing that 
D
𝚖𝚜𝚎
=
d
⋅
𝒞
​
(
f
X
,
b
)
, where 
𝒞
​
(
f
X
,
b
)
 is the optimal MSE cost for scalar quantizer defined in ??. Let 
𝒚
~
 be defined as per line 9 of ??. Since 
𝚷
 is a rotation matrix we can write: 
‖
𝒙
−
𝒙
~
‖
2
=
‖
𝚷
⋅
𝒙
−
𝒚
~
‖
2
. Using the notation 
𝒚
=
𝚷
⋅
𝒙
 as per line 5 of ?? and plugging this into the definition of 
D
𝚖𝚜𝚎
 we can write:
D
𝚖𝚜𝚎
=
𝔼
[
‖
𝒚
−
𝒚
~
‖
2
2
]
=
∑
j
∈
[
d
]
𝔼
[
|
𝒚
j
−
𝒚
~
j
|
2
]
=
∑
j
∈
[
d
]
𝔼
[
|
𝒚
j
−
c
𝚒𝚍𝚡
j
|
2
]
=
d
⋅
𝔼
[
|
𝒚
1
−
c
𝚒𝚍𝚡
1
|
2
]
=
d
⋅
min
−
1
≤
c
1
≤
c
2
≤
…
≤
c
2
b
≤
1
​
∑
i
=
1
2
b
∫
c
i
−
1
+
c
i
2
c
i
+
c
i
+
1
2
|
x
−
c
i
|
2
⋅
f
X
​
(
x
)
​
𝑑
x
=
d
⋅
𝒞
​
(
f
X
,
b
)
.
The third equality above follows from the definition of 
𝒚
~
 in line 9 of ?? and the fourth line above follows because all 
𝒚
j
’s have identical distribution of 
𝒚
j
∼
f
X
​
(
⋅
)
 as shown in ??. The last two lines above follows because 
c
𝚒𝚍𝚡
j
 is chosen to be the nearest centroid to each coordinate 
𝒚
j
 in line 6.

Now we must bound the optimal k-means cost 
𝒞
​
(
f
X
,
b
)
. For moderate values of 
d
, 
f
X
→
𝒩
​
(
0
,
1
/
d
)
. By numerically solving the optimization problem in ?? for values 
b
=
1
,
2
,
3
,
4
 we get that 
𝒞
​
(
f
X
,
b
)
≈
0.36
d
,
0.117
d
,
0.03
d
,
0.009
d
, respectively. For larger bit-widths 
b
>
4
, we can apply the Panter-Dite [44] high-resolution formula for the distortion of a fixed-rate scalar quantizer, yielding the following bound:

𝒞
​
(
f
X
,
b
)
≤
1
12
⋅
(
∫
f
X
​
(
x
)
1
/
3
​
𝑑
x
)
3
⋅
1
4
b
=
3
​
π
2
​
d
⋅
1
4
b
.
This completes the proof. ∎

Entropy Encoding Codebook Pointers.
TurboQuant’s efficiency can be further increased by applying entropy encoding to the indices that point to the closest codebook elements. Specifically, the probability of each codeword index appearing in the quantized vectors can be computed as 
p
ℓ
:=
∫
c
ℓ
−
1
+
c
ℓ
2
c
ℓ
+
c
ℓ
+
1
2
f
X
​
(
x
)
​
𝑑
x
. Optimally coding the indices, reduces the average bit-width to nearly the entropy of the distribution 
{
p
i
}
i
∈
[
2
b
]
. This lossless compression does not affect the distortion and provides a bit-width reduction at no cost. The most significant reduction occurs for 
b
=
4
, where the entropy of 
{
p
i
}
i
∈
[
2
b
]
 is approximately 
3.8
. Detailed calculations for optimal prefix codes reveal that the average bit-width can be reduced by 
5
%
. However, given the limited gain, we have chosen not to incorporate this technique into TurboQuant to maintain simplicity and speed.

3.2Inner-product Optimal TurboQuant
For important applications like nearest neighbor search, having an unbiased inner product estimator is essential. However, 
TurboQuant
𝚖𝚜𝚎
 presented in ?? does not provide unbiased inner product estimates with query vectors. To illustrate this, consider the case with a bit-width of 
b
=
1
. In this scenario, the optimal codebooks that solve the optimization problem in ??, for sufficiently large 
d
, are 
{
±
2
π
​
d
}
. This implies that the quantization map for 
TurboQuant
𝚖𝚜𝚎
 is 
Q
𝚖𝚜𝚎
​
(
𝒙
)
=
𝚜𝚒𝚐𝚗
​
(
𝚷
⋅
𝒙
)
 for any 
𝒙
∈
ℝ
d
, and the dequantization map is 
Q
𝚖𝚜𝚎
−
1
​
(
𝒛
)
=
2
π
​
d
⋅
𝚷
⊤
⋅
𝒛
 for any 
𝒛
∈
{
−
1
,
+
1
}
d
. Therefore, for large enough 
d
, according to ??, we have 
𝔼
[
⟨
𝒚
,
Q
𝚖𝚜𝚎
−
1
​
(
Q
𝚖𝚜𝚎
​
(
𝒙
)
)
⟩
]
=
2
π
⋅
⟨
𝒚
,
𝒙
⟩
, which has a multiplicative bias of 
2
/
π
. This bias diminishes with increasing bit-widths 
b
, as we empirically demonstrate in ??.

To address this bias, we propose a solution that combines 
TurboQuant
𝚖𝚜𝚎
 with an instance of QJL [62]. Specifically, let 
Q
𝚖𝚜𝚎
 be the quantization map corresponding to 
TurboQuant
𝚖𝚜𝚎
 with a bit-width of 
b
−
1
. For any 
𝒙
∈
SS
d
−
1
 the residual vector, defined as 
𝒓
:=
𝒙
−
Q
𝚖𝚜𝚎
−
1
​
(
Q
𝚖𝚜𝚎
​
(
𝒙
)
)
, has a small L2 norm, i.e., on expectation 
𝔼
[
‖
𝒓
‖
]
=
𝒞
​
(
f
X
,
b
−
1
)
 (per ??). We can then apply the QJL quantization map 
Q
𝚚𝚓𝚕
 on this residual vector, resulting in an overall bit-width of 
b
 and providing the following unbiased inner product estimator:

⟨
𝒚
,
Q
𝚖𝚜𝚎
−
1
​
(
Q
𝚖𝚜𝚎
​
(
𝒙
)
)
⟩
+
‖
𝒓
‖
2
⋅
⟨
𝒚
,
Q
𝚚𝚓𝚕
−
1
​
(
Q
𝚚𝚓𝚕
​
(
𝒓
)
)
⟩
.
More formally, the quantization map 
Q
𝚙𝚛𝚘𝚍
:
SS
d
−
1
→
[
2
b
−
1
]
d
×
{
−
1
,
1
}
d
×
ℝ
 is defined as:

Q
𝚙𝚛𝚘𝚍
​
(
𝒙
)
=
[
Q
𝚖𝚜𝚎
​
(
𝒙
)
,
Q
𝚚𝚓𝚕
​
(
𝒙
−
Q
𝚖𝚜𝚎
−
1
​
(
Q
𝚖𝚜𝚎
​
(
𝒙
)
)
)
,
‖
𝒙
−
Q
𝚖𝚜𝚎
−
1
​
(
Q
𝚖𝚜𝚎
​
(
𝒙
)
)
‖
2
]
.
A pseudocode for this procedure is given in ??.

Algorithm 2 
TurboQuant
𝚙𝚛𝚘𝚍
: optimized for inner product
1: input: dimension 
d
 and bit-width 
b
 // Global Parameters for Setting up 
TurboQuant
𝚙𝚛𝚘𝚍
2: Instantiate a 
TurboQuant
𝚖𝚜𝚎
 with bit-width 
b
−
1
 as per ??
3: Generate a random projection matrix 
𝑺
∈
ℝ
d
×
d
 with i.i.d. entries 
𝑺
i
,
j
∼
𝒩
​
(
0
,
1
)
  
4: Procedure 
Quant
𝚙𝚛𝚘𝚍
​
(
𝒙
)
5: 
𝚒𝚍𝚡
←
Quant
𝚖𝚜𝚎
​
(
𝒙
)
6: 
𝒓
←
𝒙
−
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
{ residual vector}
7: 
𝚚𝚓𝚕
←
𝚜𝚒𝚐𝚗
​
(
𝑺
⋅
𝒓
)
{ QJL on residual vector}
8: output: 
(
𝚒𝚍𝚡
,
𝚚𝚓𝚕
,
‖
r
‖
2
)
  
9: Procedure 
DeQuant
𝚙𝚛𝚘𝚍
​
(
𝚒𝚍𝚡
,
𝚚𝚓𝚕
,
γ
)
10: 
𝒙
~
𝚖𝚜𝚎
←
DeQuant
𝚖𝚜𝚎
​
(
𝚒𝚍𝚡
)
11: 
𝒙
~
𝚚𝚓𝚕
←
π
/
2
d
⋅
γ
⋅
𝑺
⊤
⋅
𝚚𝚓𝚕
12: output: 
x
~
𝚖𝚜𝚎
+
x
~
𝚚𝚓𝚕
We prove the main result for 
TurboQuant
𝚙𝚛𝚘𝚍
 in the following theorem.

Theorem 2 (performance guarantee: 
TurboQuant
𝚙𝚛𝚘𝚍
). For any bit-width 
b
≥
1
 and any vector 
𝐱
∈
SS
d
−
1
, the procedure 
Quant
𝚙𝚛𝚘𝚍
​
(
𝐱
)
 in ?? outputs an index vector 
𝚒𝚍𝚡
∈
[
2
b
−
1
]
d
 along with a sign vector 
𝚚𝚓𝚕
∈
{
−
1
,
1
}
d
 and a positive number 
γ
≥
0
. When these vectors and the scalar value are passed to the primitive 
DeQuant
𝚙𝚛𝚘𝚍
​
(
𝚒𝚍𝚡
,
𝚚𝚓𝚕
,
γ
)
, it produces a reconstructed vector 
𝐱
~
∈
ℝ
d
 that for any vector 
𝐲
∈
ℝ
d
 satisfies the following properties:
• Expected inner-product 
𝔼
𝒙
~
[
⟨
𝒚
,
𝒙
~
⟩
]
=
⟨
𝒚
,
𝒙
⟩
• Inner-product distortion defined as 
D
𝚙𝚛𝚘𝚍
:=
𝔼
𝒙
~
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
𝒙
~
⟩
|
2
]
 is bounded by 
D
𝚙𝚛𝚘𝚍
≤
3
​
π
2
⋅
‖
𝒚
‖
2
2
d
⋅
1
4
b
 for any 
b
≥
0
.
• For small bit-widths, specifically 
b
=
1
,
2
,
3
,
4
, 
D
𝚙𝚛𝚘𝚍
 exhibits finer-grained distortion values: 
D
𝚙𝚛𝚘𝚍
≈
1.57
d
,
0.56
d
,
0.18
d
,
0.047
d
, respectively.
Proof.First we compute the conditional expectation of the inner product estimate 
⟨
𝒚
,
𝒙
~
⟩
 conditioned on 
𝒙
~
𝚖𝚜𝚎
 as follows:
𝔼
[
⟨
𝒚
,
𝒙
~
⟩
|
𝒙
~
𝚖𝚜𝚎
]
=
𝔼
𝒙
~
𝚚𝚓𝚕
[
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
+
𝒙
~
𝚚𝚓𝚕
⟩
|
𝒙
~
𝚖𝚜𝚎
]
=
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
⟩
+
𝔼
𝒙
~
𝚚𝚓𝚕
[
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
|
𝒙
~
𝚖𝚜𝚎
]
=
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
⟩
+
⟨
𝒚
,
𝒓
⟩
=
⟨
𝒚
,
𝒙
⟩
,
where the first equality follows from the definition of 
𝒙
~
 in line 12 of the algorithm. The third equality above follows from ?? and last line follows from definition of the residual vector 
𝒓
=
𝒙
−
𝒙
~
𝚖𝚜𝚎
 in line 6. Now we can computed the unconditional expectation using the law of total expectation: 
𝔼
𝒙
~
[
⟨
𝒚
,
𝒙
~
⟩
]
=
𝔼
𝒙
~
𝚖𝚜𝚎
[
𝔼
[
⟨
𝒚
,
𝒙
~
⟩
|
𝒙
~
𝚖𝚜𝚎
]
]
=
𝔼
[
⟨
𝒚
,
𝒙
⟩
]
=
⟨
𝒚
,
𝒙
⟩
, which proves the first claim of the theorem.

We apply the same conditioning on 
𝒙
~
𝚖𝚜𝚎
, when computing the distortion, and then compute the resulting conditional distortion:

𝔼
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
𝒙
~
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
=
𝔼
𝒙
~
𝚚𝚓𝚕
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
𝒙
~
𝚖𝚜𝚎
+
𝒙
~
𝚚𝚓𝚕
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
=
𝔼
𝒙
~
𝚚𝚓𝚕
[
|
⟨
𝒚
,
𝒓
⟩
−
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
=
𝚅𝚊𝚛
​
(
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
|
𝒙
~
𝚖𝚜𝚎
)
≤
π
2
​
d
⋅
‖
𝒓
‖
2
2
​
‖
𝒚
‖
2
2
,
where the second equality above follows from the definitions of 
𝒓
 and 
𝒙
~
𝚖𝚜𝚎
 in lines 6 and 10 of ??. The third line above follows because 
𝔼
[
⟨
𝒚
,
𝒙
~
𝚚𝚓𝚕
⟩
]
=
⟨
𝒚
,
𝒓
⟩
, by ??. The last line follows from the variance bound of QJL estimator shown in ?? and using the fact that 
𝒙
~
𝚚𝚓𝚕
 in line 11 is re-scaled by 
γ
=
‖
𝒓
‖
.

Now by law of total expectation along with the fact that 
𝒓
=
𝒙
−
𝒙
~
𝚖𝚜𝚎
 we can bound the inner product distortion as follows:

D
𝚙𝚛𝚘𝚍
=
𝔼
𝒙
~
𝚖𝚜𝚎
[
𝔼
[
|
⟨
𝒚
,
𝒙
⟩
−
⟨
𝒚
,
𝒙
~
⟩
|
2
|
𝒙
~
𝚖𝚜𝚎
]
]
≤
π
2
​
d
⋅
‖
𝒚
‖
2
2
⋅
𝔼
[
‖
𝒙
−
𝒙
~
𝚖𝚜𝚎
‖
2
2
]
=
π
2
​
d
⋅
‖
𝒚
‖
2
2
⋅
D
𝚖𝚜𝚎
.
The theorem follows by invoking the MSE bounds from ?? with bit-width 
b
−
1
. ∎

