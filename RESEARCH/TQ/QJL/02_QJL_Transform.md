# QJL — Part 2: Quantized JL Transform
_Part 2 of 3. See [`00_INDEX.md`](00_INDEX.md) for full paper TOC._

3 Quantized Johnson-Lindenstrauss (QJL) Transform
Our goal is to save memory space for storing the KV cache while the inner product between query
and key remains undistorted. To achieve this, we first transform the embedding vectors using a
random projection that preserves the inner products, acting as a preconditioning step, and then
quantize the result. Specifically, we project the input vectors onto a random subspace by applying
the Johnson-Lindenstrauss (JL) transform [15], which amounts to multiplying by a random Gaussian
matrix. The inner product of the resulting vectors after applying this projection provides an unbiased
and low-distortion estimator for the inner product of the original vectors [8]. We introduce a 1-bit
Johnson-Lindenstrauss transform, comprising a JL transformation followed by quantization to a single
sign bit, and demonstrate its ability to offer an unbiased and low-distortion inner product estimator.
We complement our binary quantizer by developing an unbiased estimator for the inner product of
the quantized vector with any arbitrary vector. This inner product estimator is asymmetric, as one
of the vectors is quantized to a single bit while the other remains unquantized, making it well-suited
for the KV cache mechanism. The Quantized Johnson-Lindenstrauss (QJL) transformation, acting
as a 1-bit quantizer, alongside our proposed estimator, is formally defined in the following definition:
Definition 3.1 (QJL and inner product estimator). For any positive integers d, m, let S ∈ R
m×d
be a JL transform matrix, i.e., entries of S are i.i.d. samples from the zero mean and unit variance
4
Normal distribution. The QJL is a mapping function HS : R
d → {−1, +1}
m defined as:
HS(k) := sign(Sk) for any k ∈ R
d
. (3)
Furthermore, for any pair of vectors k, q ∈ R
d
the estimator for their inner product ⟨q, k⟩ based on
the aforementioned quantizer is defined as:
ProdQJL(q, k) :=
p
π/2
m
· ∥k∥2 · ⟨Sq, HS(k)⟩. (4)
Now, we show that the inner product estimator ProdQJL(q, k), exactly like the inner product
of JL-transformed vectors without quantization to sign bit, is an unbiased estimator. The crucial
point to note is that if we applied QJL to both vectors q and k in Equation (4), we would obtain an
unbiased estimator for the angle between these vectors, as shown in [6]. However, to estimate the
inner product one needs to apply the cosine function on top of the angle estimator, which results
in a biased estimation. Thus, to achieve an unbiased inner product estimator, it is necessary to
asymmetrically apply quantization to the JL transform of only one of the vectors q and k.
Lemma 3.2 (Inner product estimator ProdQJL is unbiased). For any vectors q, k ∈ R
d
the expected
value of the estimator ProdQJL(q, k) defined in Equation (4) is:
E
S
[ProdQJL(q, k)] = ⟨q, k⟩,
where the expectation is over the randomness of the JL matrix S in Definition 3.1.
Proof. Let s1, s2, . . . sm denote the rows of the JL matrix S. Additionally, let us decompose q to its
projection onto the vector k and its orthogonal component, i.e., q
⊥k
:= q −
⟨q,k⟩
∥k∥
2
2
· k. We can write,
ProdQJL(q, k) =
p
π/2
m
X
i∈[m]
∥k∥2 · s
⊤
i q · sign(s
⊤
i k)
=
p
π/2
m
X
i∈[m]
⟨q, k⟩
∥k∥2
· s
⊤
i k · sign(s
⊤
i k) + ∥k∥2 · s
⊤
i q
⊥k
· sign(s
⊤
i k)
=
p
π/2
m
X
i∈[m]
⟨q, k⟩
∥k∥2
· |s
⊤
i k| + ∥k∥2 · s
⊤
i q
⊥k
· sign(s
⊤
i k).
Since si
’s have identical distributions, we have:
E
S
[ProdQJL(q, k)] = p
π/2

⟨q, k⟩
∥k∥2
· E
h
|s
⊤
1 k|
i
+ ∥k∥2 · E
h
s
⊤
1 q
⊥k
· sign(s
⊤
1 k)
i
.
To calculate the above expectation let us define variables x := s
⊤
1 k and y := s
⊤
1 q
⊥k
. Note that x
and y are both zero-mean Gaussian random variables and because ⟨q
⊥k
, k⟩ = 0. By the following
Fact 3.3, x and y are independent.
Fact 3.3. If x ∈ R
d
is a vector of i.i.d. zero-mean normal entries with variance σ
2 and A ∈ R
m×d
is
a matrix, then A · x is a normal random variable with mean zero and covariance matrix σ
2
· AA⊤.
This implies that the second expectation term above is zero because E

s
⊤
1 q
⊥k
· sign(s
⊤
1 k)

=
E[y · sign(x)] = E[y] · E[sign(x)] = 0. Furthermore, x is a Gaussian random variable with mean
zero and variance ∥k∥
2
2
. Therefore, we have
E
S
[ProdQJL(q, k)] = p
π/2 ·
⟨q, k⟩
∥k∥2
· E
x
[|x|] = ⟨q, k⟩.
where the equality comes from the following Fact 3.4:
5
Fact 3.4 (Moments of Normal Random Variable). If x is a normal random variable with zero mean
and variance σ
2
, then for any integer ℓ, the ℓ-th moment of x is E

|x|
ℓ

= σ
ℓ
· 2
ℓ/2Γ((ℓ + 1)/2)/
√
π.
This completes the proof of Lemma 3.2.
Now we show that the inner product estimator ProdQJL in Definition 3.1, just like the estimators
based on the standard JL transform, has a bounded distortion with high probability.
Lemma 3.5 (Distortion of inner product estimator ProdQJL). For any vectors q, k ∈ R
d
if the
estimator ProdQJL(q, k) is defined as in Equation (4) for QJL with dimension m ≥
4
3
·
1+ε
ε
2 log 2
δ
, then:
Pr
S
[|ProdQJL(q, k) − ⟨q, k⟩| > ε∥q∥2∥k∥2] ≤ δ,
where the probability is over the randomness of the JL matrix S in Definition 3.1.
Proof. First note that, letting s1, s2, . . . sm denote the rows of the JL transform matrix S, we have:
ProdQJL(q, k) = 1
m
X
i∈[m]
p
π/2 · ∥k∥2 · s
⊤
i q · sign(s
⊤
i k).
Since si
’s are i.i.d. the above is indeed the average of m i.i.d. estimators defined as zi
p
:=
π/2 · ∥k∥2 · s
⊤
i
q · sign(s
⊤
i k) for i ∈ [m]. Let us now calculate the ℓ-th moment of zi using Fact 3.4:
E
h
|zi
|
ℓ
i
=
p
π/2 · ∥k∥2
ℓ
· E
h
|s
⊤
i q|
ℓ
i
=
√
π · ∥k∥2∥q∥2
ℓ
·
Γ((ℓ + 1)/2)
√
π
, (5)
where the second equality above follows because s
⊤
i
q is a Gaussian random variable with mean zero
and variance ∥q∥
2
2
along with Fact 3.4. Now we can prove the result by invoking the unbiasedness of
the estimator, Lemma 3.2, along with an appropriate version of Bernstein inequality and using the
moment bounds in Equation (5). More specifically, our moment calculation in Equation (5) implies:
E
h
|zi
|
ℓ
i
= E

|zi
|
2

·
√
π∥k∥2∥q∥2
ℓ−2
·
Γ((ℓ + 1)/2)
Γ(3/2) ≤ E

|zi
|
2

·

2
3
· ∥k∥2∥q∥2
ℓ−2
·
ℓ!
2
Therefore, by invoking a proper version of the Bernstein inequality, for instance Corollary 2.11 from
[5], we have the following:
Pr
S
[|ProdQJL(q, k) − ⟨q, k⟩| > t] ≤ 2 exp 
3
4
·
mt2
∥k∥
2
2
∥q∥
2
2 + ∥k∥2∥q∥2 · t

.
If we set t = ε∥q∥2∥k∥2 the above simplifies to:
Pr
S
[|ProdQJL(q, k) − ⟨q, k⟩| > ε∥q∥2∥k∥2] ≤ 2 exp 
3
4
·
mε2
1 + ε

.
Therefore if m ≥
4
3
·
1+ε
ε
2 log 2
δ
the error bound follows. This completes the proof of Lemma 3.5.
Note that the distortion bound in Lemma 3.5 has remarkably small constants, even smaller than
those of the original unquintized JL transform. This indicates that quantizing one of the vectors to
just a single sign bit does not result in any loss of accuracy. We use these properties of QJL and our
inner product estimator to prove the final approximation bound on our KV cache quantizer.
6
Algorithm 1 QJL Key Cache Quantizer
Input: Stream of key tokens k1, k2, . . . ∈ R
d
, integer m
1: Draw a random sketch S ∈ R
m×d with i.i.d. entries Si,j ∼ N (0, 1) as per Definition 3.1
2: repeat
3: Compute k˜
i ← sign (Ski) and νi ← ∥ki∥2
4: store the quantized vector ˜ki and the key norm νi
in the cache
5: until token stream ends
Procedure EstimateScores(qn)
6: Compute inner product estimators qKg(j) ←
√
π/2
m
· νi
· ⟨Sqn, k˜
j ⟩ for every j ∈ [n]
7: Score ^ ← softmax 
qKg

return Score ^
3.1 Key Cache Quantization via QJL
The key cache is used in the computation of attention scores as shown in Equation (2). To calculate
these scores, we need to compute the inner products of the current query embedding with all key
embeddings in the cache. We design a quantization scheme that allows for a low-distortion estimate
of the inner products between an arbitrary query and all keys in the cache. In this section, we develop
a practical algorithm with provable guarantees based on QJL and the inner product estimator defined
in Definition 3.1.
The quantization scheme presented in Algorithm 1 applies QJL, defined in Definition 3.1, to each
key embedding, mapping them to binary vectors and storing the results in the key cache. We show
in the following theorem that the attention scores calculated by Algorithm 1 have very small (1 ± ε)
relative distortion with high probability:
Theorem 3.6 (Distortion bound on QJL key cache quantizer). For any sequence of key tokens
k1, . . . kn ∈ Rd and any integer m, Algorithm 1 stores binary vectors k˜
1, . . . k˜
n ∈ {−1, +1}
m along
with scalar values ν1, . . . νn in the cache. If the key embeddings have bounded norm maxi∈[n] ∥ki∥2 ≤ r
and m ≥ 2r
2
ε
−2
log n, then for any query embedding qn ∈ Rd with bounded norm ∥qn∥2 ≤ r the
output of the procedure EstimateScores(qn) satisfies the following with probability 1 −
1
poly(n)
sinultaneously for all i ∈ [n]:



Score ^(i) − Score(i)


 ≤ 3ε · Score(i),
where Score is the vector of attention scores defined in Equation (2).
Proof. The proof is by invoking Lemma 3.5 and a union bound. For every j ∈ [n] the estimator
qKg(j) computed in line 6 of Algorithm 1 is in fact equal to the inner product estimator qKg(j) =
ProdQJL(qn, kj ) as defined in Equation (4). Thus by Lemma 3.5 we have the following with probability
at least 1 −
1
n3/(2+2ε)
:


qKg(j) − ⟨qn, kj ⟩


 ≤
ε
r
2
· ∥qn∥2∥kj∥2 ≤ ε,
where the second inequality follows from the preconditions of the theorem regarding the boundedness
of the norms of the query and key embeddings. By union bound, the above inequality holds
simultaneously for all j ∈ [n] with high probability in n. Thus after applying the softmax function
in line 7 of Algorithm 1 we get that with high probability in n:
Score ^(i) ∈ e
±2ε
· Score(i) ∈ (1 ± 3ε) · Score(i).
7
This completes the proof of Theorem 3.6.
This theorem shows that if the query and key embeddings have constant norms, as is common
in practical scenarios, we can quantize each key embedding such that only m ≈ ε
−2
log n bits are
needed to store each key token. This is independent of the embedding dimension of the tokens and
scales only logarithmically with the sequence length.
3.2 Value Cache Quantization
We quantize the value cache using a standard quantization method, i.e., normalizing each token’s
entries and then rounding each entry to a few-bit integer representation. This approach aligns with
prior work, which has shown that standard token-wise quantization is highly effective for the value
cache and results in a minimal accuracy drop [22, 13].
