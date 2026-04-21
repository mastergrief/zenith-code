# Litespark — Results

← back to [LITESPARK.md](LITESPARK.md)

3Results
3.1Training throughput acceleration
Litespark delivers substantial reductions in training time across all configurations, directly addressing the time-to-market challenges facing LLM development. For the 3B parameter model, as shown in Table 3, our framework processes 2x–4x more tokens per second than baseline Llama implementations, translating to proportional reductions in total training time. For example, with 8 H200 GPUs, completing a fixed amount of training that would take Llama 100 hours would require only 50 hours with Litespark.

The time savings become more pronounced at scale, where distributed training traditionally suffers from communication bottlenecks. With 128 GPUs, Litespark’s 
3.81
x speedup for the 3B model means training jobs that previously required weeks can be completed in days. For the 30B model, as shown in Table 4, the 4.73x–6.36x acceleration transforms month-long training cycles into week-long iterations, fundamentally changing the pace of model development and experimentation.

These training time reductions have immediate strategic value beyond energy considerations. Faster training enables rapid iteration during model development, allowing researchers to test more architectural variations and hyperparameter configurations within fixed time budgets. Organizations can respond more quickly to market demands, reduce time-to-deployment for new models, and maintain competitive advantages through faster innovation cycles. The ability to complete training in days rather than weeks also reduces the risk of infrastructure failures derailing long-running experiments.

Num	Model	tokens/sec	TFLOPs/	time/	MFU	Speedup
GPUs			GPU	iteration (sec)	(%)	
8	Litespark	439,644.81	888.06	2.38	89.35	2.00
Llama	218,967.60	442.30	6.44	44.70
16	Litespark	862,899.88	871.51	1.40	88.65	2.17
Llama	396,768.52	400.73	5.49	40.63
128	Litespark	1,387,342.21	175.15	2.23	17.66	3.81
Llama	364,328.66	45.99	7.19	4.67
256	Litespark	964,981.24	61.58	2.43	6.29	2.25
Llama	428,056.25	27.31	7.63	2.73
Table 3:Pre-training throughput of 3B models on H200s
Num	Model	tokens/sec	TFLOPs/	time/	MFU	Speedup
GPUs			GPU	iteration (sec)	(%)	
256	Litespark	471,486.24	393.25	2.23	39.54	4.73
Llama	99,604.57	83.08	10.53	8.43
512	Litespark	508,260.62	215.78	2.07	21.88	6.36
Llama	79,891.47	33.32	13.09	3.38
Table 4:Pre-training throughput of 30B models on H200s
Refer to caption
Figure 1:Pre-training throughput comparison on H200s
3.2Computational efficiency and resource utilization
The dramatic training time improvements in Litespark stem from enhanced computational efficiency and resource utilization. This is manifest from the Tera-FLOPs per GPU and Model FLOPs Utilization (MFU) reported in Tables 3 and 4. Litespark achieves 89.35% MFU compared to Llama’s 44.70% on training with 8 GPUs, indicating our optimizations successfully extract maximum computational value from available hardware. This high utilization rate means that expensive GPU resources operate at near-peak capacity rather than sitting idle due to architectural bottlenecks.

At larger scales, Litespark maintains superior efficiency even as the complexity of distributed training increases. With 128 GPUs, Litespark sustains 17.66% MFU while Llama drops to 4.67%, demonstrating that its optimizations address fundamental bottlenecks in multi-node transformer training. For the 30B model at 256 GPUs, Litespark achieves 39.54% MFU compared to Llama’s 8.43% MFU.

In terms of total computational throughput, as shown in Figure 1, Litespark consistently outperforms the baseline Llama implementation, achieving 2x–6x higher total TFLOPs across all GPU configurations. These throughput metrics indicate that our architectural and algorithmic improvements become increasingly valuable for larger models where memory bandwidth limitations traditionally become more severe.

This consistent high utilization across configurations transforms the economics of GPU usage. By achieving 
17
%
−
40
% MFU compared to Llama’s 
3
%
−
8
% MFU in large-scale configurations, Litespark converts previously wasted computational cycles into productive training progress, maximizing return on infrastructure investment.

3.3Energy efficiency
The throughput improvements directly translate into substantial energy savings. For the 3B model, as shown in Table 5, Litespark reduces energy consumption by 
55
%
−
70
% across different GPU configurations. Training 500B tokens requires only 
0.79
−
3.41
 MWh with Litespark compared to 
1.75
−
8.01
 MWh with baseline Llama. In particular, energy savings increase with scale: while training with only 8 GPUs shows 55% energy reduction, training with 128 GPUs achieves 70% energy savings. Using the standard conversion formula [43],

CO
2
​
eq (tonnes)
=
Energy (MWh)
×
carbon intensity (kg CO
2
​
eq/kWh)
/
1000
with an average US carbon intensity of 0.35 kg CO2eq/kWh [44], these energy savings directly translate into carbon emission reductions. For the 3B model, training 500B tokens produces only 
0.28
−
1.19
 tonnes of CO2eq with Litespark versus 
0.61
−
2.80
 tonnes with Llama, as shown in Figure 2 (left).

The 30B model demonstrates even more dramatic gains in energy efficiency, with a 
75
%
−
83
% reduction in energy, as shown in Table 6. Training 500B tokens on 256 GPUs requires 125.35 MWh with Litespark versus 732.08 MWh with Llama yields an 83% reduction representing over 600 MWh in savings. This corresponds to 43.87 tonnes of CO2eq with Litespark as compared to 256.23 tonnes with Llama – a reduction of over 212 tonnes of CO2eq per 500B tokens. At the largest scale (512 GPUs), Litespark consumes 
189.47
 MWh compared to Llama’s 
751.75
 MWh, maintaining 75% energy savings even with increased communication overhead. The corresponding carbon emissions are 66.31 tonnes versus 263.11 tonnes, as illustrated in Figure 2 (right).

These energy savings have immediate practical implications. For a typical 30B model training run requiring several trillion tokens, our framework could reduce energy consumption from gigawatt-hours to hundreds of megawatt-hours, translating to millions of dollars in electricity cost savings and proportional reductions in carbon emissions. At scale, training a 30B model on 10 trillion tokens would result in approximately 1,478 tonnes of CO2eq with Litespark compared to 5,864 tonnes with Llama — a reduction of over 4,300 tonnes of CO2eq, equivalent to the annual emissions of nearly 860 passenger vehicles [45].

Num	Model	Energy (MWh)/	CO2eq (tonnes)/	Energy savings
GPUs		500B tokens	500B tokens	(%)
8	Litespark	0.79	0.28	54.86
Llama	1.75	0.61
16	Litespark	0.80	0.28	55.56
Llama	1.80	0.63
128	Litespark	1.65	0.58	69.67
Llama	5.44	1.90
256	Litespark	3.41	1.19	57.43
Llama	8.01	2.80
Table 5:Energy consumption of 3B models on H200s
Num	Model	Energy (MWh)/	CO2eq (tonnes)/	Energy savings
GPUs		500B tokens	500B tokens	(%)
256	Litespark	125.35	43.87	82.88
Llama	732.08	256.23
512	Litespark	189.47	66.31	74.80
Llama	751.75	263.11
Table 6:Energy consumption of 30B models on H200s
Refer to caption
Figure 2:
CO
2
 emissions comparison for 3B models (left) and 30B models (right) on H200s
