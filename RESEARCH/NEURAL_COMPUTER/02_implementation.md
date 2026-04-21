# Neural Computers — Implementation (CLIGen + GUIWorld)

← back to [NC.md](NC.md)

3 Implementation of Neural Computers
We build on the Wan2.1 model (Wan et al., 2025), which was a state-of-the-art video generation model at the
time of our experiments. We add NC-specific conditioning and action modules, together with interface-specific
training recipes. Figure 1 illustrates this setup: NCs take a prompt or action stream as input and generate
future interface frames in both CLI and GUI settings. In the present prototypes, these prompts and actions
are logged conditioning streams, so evaluation remains open-loop rather than closed-loop interaction with a
live environment. We refer to these two instantiations as CLIGen (CLI; Section 3.1) and GUIWorld (GUI;
Section 3.2).
In this video-based instantiation, the NC latent runtime state ht is realized by the model’s time-indexed
video latents zt. Under this abstraction, the diffusion transformer acts as the state-update map: it consumes
prior latents together with the current observation and conditioning inputs, and produces the updated state
ht (realized as zt). The decoder Gθ parameterizes a distribution over the next frame xt+1. Auxiliary heads
encode and decode conditioning streams ut, including text prompts and action traces. Structured logs such as
terminal buffers are used for alignment and evaluation where available, not as privileged model-state inputs.
4
CLIGen (General) Dataset CLIGen (Clean) Dataset GUIWorld Dataset
Figure 2 Data types used to learn NC behaviors. Logos denote datasets: CLIGen (General) replays public asciinema
traces spanning diverse real-world terminal workflows. CLIGen (Clean) uses scripted vhs runs to capture
deterministic terminal traces for controlled experiments. GUIWorld captures desktop RGB with synchronized
mouse/keyboard traces to validate action-conditioned rendering and control on GUIs.
3.1 / The CLI Video Generators
CLIGen instantiates the NC abstraction in command-line interfaces. Observations xt are terminal frames
rendered from the underlying text buffer. The conditioning stream ut carries a user prompt and optional
metadata, and the video latent state zt implements the latent runtime state ht by tracking CLI context across
frames. At inference time, the model rolls out from the prompt and first frame, updates zt, and predicts
future terminal frames (Figure 3). We use two CLI datasets: CLIGen (General), which contains diverse,
open-ended terminal traces, and CLIGen (Clean), which contains deterministic Dockerized traces. We
train one NCCLIGen model per dataset under the same architecture.
Dataset Data characteristics Model
CLIGen (General) diverse, open-ended terminal traces NCCLIGen
CLIGen (Clean) deterministic Dockerized scripts with cleaner, better-paced
buffers NCCLIGen
3.1.1 Data pipeline
The CLIGen (General) dataset is built from publicly available asciinema .cast trajectories2
. The asciinema
stack records and replays terminal sessions with synchronized timing and ANSI-faithful decoding. We replay
each session with the official tools and render it into terminal frames, preserving palette transitions, cursor
state, and terminal geometry. Frames, text buffers, and keyboard-event logs share a single monotonic clock.
At render time, we normalize resolution and aspect ratio and apply a filter to remove sensitive strings. We
render sessions to GIF using agg and convert them to video with ffmpeg.
We segment each recording into roughly five-second clips using content-aware splits. We temporally normalize
each clip to a fixed length: shorter clips repeat the final frame, and longer clips are uniformly subsampled.
The resulting 823,989 video streams (approximately 1,100 hours) are resampled to 15 FPS. Underlying buffers
and logs are used to generate aligned textual descriptions with Llama 3.1 70B (Dubey et al., 2024) in three
2https://asciinema.org/
5
Table 1 Data samples for CLIGen (General) and CLIGen (Clean).
Frames from Sample — CLIGen (General)
1 2 3
4 5 6
Semantic
A root terminal session kicks off an AI command to make three 1024x1024 cat shots, shows quick parsing for each one,
then presents pixelated cat previews with numbered links and asks whether to stash them in /root/2023_04_01-02_-
27_11_imgs.
Regular
In a root shell at ~, the user runs ai -i 3 a cute cat, watches a green progress line announcing three 1024x1024 images,
sees sequential parsing messages for images 1 through 3, and ends on a preview pane with three pixelated cat thumbnails,
numbered download links, and a save prompt targeting /root/2023_04_01-02_27_11_imgs.
Detailed
In a dark-background terminal at the root in ~ prompt, the user types ai -i 3 a cute cat. The screen prints Generating
3 1024x1024 images (press CTRL-C to cancel)..., shows parsing messages for images 1–3, and ends on a preview pane
with three numbered thumbnails and a save prompt targeting /root/2023_04_01-02_27_11_imgs.
Frames from Sample — CLIGen (Clean)
1 2 3
4 5 6
Scripted caption
Type python; Enter; Type values = [n*n for n in range(1, 10)]; Enter; Type print(values); Enter; Type exit(); Enter.
6
styles (semantic, regular, and detailed), which serve as prompts. As shown in Figure 2 (left), this split spans
diverse real-world terminal use cases3
.
The CLIGen (Clean) dataset is collected using the open-source vhs toolkit. It enables repeatable terminal
demonstrations and integration tests through scripted execution. Deterministic scripts drive Dockerized
environments to capture cleaner, better-paced traces. We authored roughly 250k scripts. After filtering (51.21%
retained), we keep two subsets. The first contains approximately 78k regular traces (package installation,
log filtering, interactive REPL usage, etc.). The second contains approximately 50k Python math validation
traces. Captions are derived directly from the raw vhs scripts for clarity. We standardize frame rendering
by fixing one monospace font/size, using a consistent palette for success and error highlights, and locking
resolution and theme to remove typography-related confounds. Each episode records its caption type and font
settings for later slicing. Clips longer than five seconds are uniformly subsampled for training, while shorter
clips repeat the final frame to normalize length4
.
3.1.2 Model architecture
We treat CLI generation as text-and-image-to-video: a caption and the first terminal frame condition the rollout.
The first frame is encoded by a VAE into a conditioning latent. In parallel, a CLIP image encoder (Radford
et al., 2021) extracts visual features from the same frame, and a text encoder (e.g., T5 (Raffel et al., 2020))
embeds the caption. Following the Wan2.1 image-to-video (I2V) design, these conditioning features are
concatenated with diffusion noise, projected through a zero-initialized linear layer, and processed by a DiT
stack. Decoupled cross-attention injects the joint caption and first-frame context derived from the CLIP and
text features. The VAE encodes and decodes terminal frames. During generation, the diffusion transformer
advances the latent state zt under the original Wan2.1 I2V sampling schedule, without additional binary
masks or periodic reseeding.
Diffusion Transformer
VAE
Encoder
VAE
Decoder
Type python; Enter; Type 10%7… Tokenizer
Self-Attention
Cross-Attention
FFN
× N
C
CLIP
Figure 3 / NCCLIGen architecture. Terminal frames are observations xt. A prompt and the first frame seed the
conditioning stream. The Wan2.1-based latent state zt rolls forward under the standard I2V sampling scheme.
3.1.3 Implementation Details
Training uses gradient checkpointing and applies dropout 0.1 to the prompt encoder, CLIP, and VAE modules.
Optimization uses AdamW (learning rate 5 × 10−5
, weight decay 10−2
), bfloat16 precision, and gradient
clipping at 1.0. Training NCCLIGen on CLIGen (General) requires ∼15,000 H100 GPU hours at batch size 1.
Training on CLIGen (Clean) across both subsets requires ∼7,000 H100 GPU hours.
3.1.4 Evaluations
Unless otherwise noted, NC in this section refers to the current video-based CLI prototype. We report six
practical takeaways:
3Additional preprocessing details and a .cast example are in Sections B and C.1, with a sample overview in Table 1.
4Additional details are provided in Sections B and C.2, with a representative data sample in Table 1.
7
1 The NC maintains high-fidelity terminal rendering at practical font sizes (e.g., 13 px), preserving readable
interface state.
2 Prompt specificity is an effective control channel: detailed, literal captions improve text-to-pixel alignment.
3 On clean but domain-specific data, global PSNR/SSIM plateau around 25k steps (Figure 5), indicating early
saturation in reconstruction metrics rather than a complete halt in learning.
4 The NC reproduces complex terminal appearances while sustaining coherent short-horizon command rollouts
under fixed conditioning.
5 Symbolic computation remains the main bottleneck: structured arithmetic reveals reliability limits, motivating
stronger symbolic or system-level conditioning.
6 In our setting, without changing the NC backbone or adding RL, reprompting improves symbolic probes
(4%→83%; Figure 6), reinforcing the view that current models are strong renderers and conditionable interfaces
rather than native reasoners (Table 6).
Experiment 1: The NC stays readable at practical font sizes
Table 2 Reconstruction quality.
Metric Value
Average PSNR 40.77
Average SSIM 0.989
Concurrent work (Rivard et al., 2025) argues that generic
natural-image VAEs can perform poorly on structured computer screenshots. We test this claim directly by applying
the Wan2.1 VAE (Wan et al., 2025) to terminal content. In
our setting, reconstruction quality is primarily governed by
font size. At 13 px, it is high (40.77 dB PSNR, 0.989 SSIM). At 6 px, text exhibits noticeable blurring even
when global PSNR/SSIM remain strong, because background regions dominate these metrics.
Figure 4 Wan2.1 VAE reconstructions on CLIGen (General) terminal frames at different font sizes.
However, a sweep over CLIGen (General) frames shows that this effect is confined to extreme cases (Figure 4).
Very small 6 px fonts and ultra-dense text exhibit localized blurring despite high global PSNR. In contrast,
the 13 px terminal font used in CLIGen remains visually sharp across panes and commands. These results
indicate that the VAE is adequate for regular CLIGen usage and highlight that sensible font choices help
ensure stable NC training.
Experiment 2: Performance plateaus early and can degrade with prolonged training
On clean but domain-specific structured interfaces, global reconstruction metrics improve rapidly early and then
show limited additional gains under the current training objective. In CLIGen (Clean), PSNR/SSIM plateau
quickly, suggesting that further optimization becomes bottlenecked less by model capacity than by the
quality and pacing of the available supervision. After the early gains, the remaining errors are often tied to
8
artifact-prone signals (e.g., rendering glitches or rapid screen changes that disrupt temporal alignment), so
additional training on the same objective can yield diminishing or even slightly unstable returns in these
perceptual metrics.
Figure 5 (a–b) Qualitative generations before and after CLIGen training; (c)
CLIGen (Clean) PSNR/SSIM plateau around 25k training steps.
Panels (a–b) illustrate the
effect of training on CLIGen data. Without CLIGen fine-tuning, Wan2.1 produces garbled terminal outputs (a). After 25k steps,
the model generates readable
text with consistent formatting and color cues (b).
Figure 5 plots the corresponding PSNR/SSIM curves
and shows that these global
perceptual metrics flatten
around 25k steps. They improve little with further training up to 460k steps, and extended optimization can even slightly reduce them.
One plausible explanation is that most learnable structured patterns are acquired early, and further gains
require higher-quality, better-paced, or more informative supervision.
Experiment 3: Literal captions drive rendering accuracy
Caption specificity has a strong effect on terminal rendering quality. As shown in Table 3, detailed, literal
descriptions improve reconstruction fidelity. PSNRincreases from 21.90 dB (semantic) to 26.89 dB (detailed),
a gain of nearly 5 dB, compared to less specific, high-level semantic descriptions.
The three caption tiers correspond to the same underlying terminal sequence but differ in length and granularity.
Semantic captions (average 55 words) provide high-level summaries (e.g., “a terminal session generates three
cat images”). Regular captions (average 52 words) include key commands and outputs (e.g., ai -i 3 a cute cat,
status messages). Detailed captions (average 76 words) transcribe screen content more exhaustively, including
exact text, colors, and formatting.
Table 3 Caption styles versus TI2V fidelity.
Prompt style PSNR SSIM Avg.
words
Semantic 21.90 0.813 55
Regular 23.63 0.843 52
Detailed 26.89 0.867 76
This progression helps explain why literal descriptions are
particularly effective for terminal rendering. Unlike natural images, which are dominated by global style patterns,
terminal frames are governed primarily by text placement.
Detailed captions act as scaffolding—explicitly specifying
which tokens appear where—thereby enabling precise textto-pixel alignment.
Experiment 4: Neural computers achieve accurate character-level text generation
Table 4 OCR accuracy versus training.
Steps (k) Char. acc. Exact line
0 0.03 0.01
10 0.18 ↑0.15 0.05 ↑0.04
20 0.33 ↑0.30 0.12 ↑0.11
30 0.41 ↑0.38 0.18 ↑0.17
40 0.52 ↑0.49 0.26 ↑0.25
50 0.52 ↑0.49 0.27 ↑0.26
60 0.54 ↑0.51 0.31 ↑0.30
Beyond PSNR and SSIM, character-level accuracy is a more
direct metric for terminal rendering. Character-level accuracy requires explicit pixel-to-text correspondence. For
CLIGen (Clean), we apply Tesseract to five uniformly sampled (ground-truth, generated) frame pairs per video and
normalize whitespace. We then compute two metrics (full
protocol in Appendix B). Character accuracy uses the Levenshtein distance between concatenated ground-truth and
generated texts. Exact-line accuracy measures the fraction of ground-truth lines whose normalized content exactly
matches the prediction at the same line index.
Table 4 shows that our models achieve substantial text rendering accuracy under this protocol. Character
9
accuracy increases from 0.03 at initialization to 0.54 at 60k steps, with exact-line matches reaching 0.31
(0.26 by 40k). Most gains occur within the first 40k steps, followed by smaller refinements thereafter. These
OCR-based metrics capture properties beyond perceptual similarity. Accurately generating terminal characters
requires modeling text structure, font rendering, and spatial relationships. These are core competencies for
interactive neural computer systems. This level of character-level precision is a step toward usable, not just
plausible, terminal interfaces. At the same time, we interpret this result primarily as evidence of interface
fidelity, while routine reuse and native symbolic computation remain separate questions.
Experiment 5: Does this NC instantiation show native CLI reasoning?
Table 5 Arithmetic probe accuracy (100 problems sampled
from a 1,000-problem held-out pool).
Model Accuracy
Wan2.1 0%
NCCLIGen 4%
Veo3.1 2%
Sora2 71%
We also probe symbolic computation with CLI
arithmetic tasks. These tasks are a sharp stress
test for symbolic reliability: humans answer them
instantly, yet current NC instantiations often fail
on seemingly simple symbolic operations.
Our arithmetic probe presents basic mathematical operations through terminal interactions. We
reserve a held-out pool of 1,000 math problems
and randomly sample 100 problems as the final
evaluation set. Table 5 shows that current video models, including this NC instantiation, struggle on these
symbolic tasks. Wan2.1 achieves 0% accuracy, our NCCLIGen model reaches 4%, and Veo3.1 manages 2%—all
far below human-level performance on these fundamental tasks. These results contrast with common claims of
strong symbolic reasoning in current video models. Sora2’s 71% accuracy is a notable outlier and may reflect
system-level advantages or additional training beyond our current setup. Overall, native symbolic reasoning
remains an open challenge for current video-based NC instantiations. Accordingly, arithmetic probes in this
paper serve as a targeted test of symbolic stability under the current prototype substrate.
The poor arithmetic-probe performance in Table 5 raises a key question. Does this prototype require specialized
reinforcement learning to achieve reliable symbolic computation, or can stronger conditioning substantially
narrow this gap?
Experiment 6: Does this NC instantiation require RL for symbolic probes?
Figure 6 Reprompting boosts performance to 83%.
As shown in Figure 6, NCCLIGen accuracy on CLIGen (Clean) arithmetic tasks rises from 4% to 83%
under reprompting. This suggests that system-level
conditioning can be an effective first lever for improving performance on symbolic probes. It is complementary to (rather than strictly requiring) RL-based
training pipelines. More generally, the success of reprompting highlights how sensitive symbolic-probe
outcomes are to the conditioning interface. Much of the apparent “reasoning” gain can come from better
specification and instruction-following rather than new native computation. For the arithmetic subset, we
include the correct answer explicitly in roughly half of the training captions to encourage reliable rendering of
the output string. Because reprompting can similarly provide stronger hints (or even outsource computation
to an external text system), we interpret the gain primarily as evidence of steerability. It also shows faithful
rendering of conditioned symbolic content. We do not treat it as a clean demonstration that the NC backbone
performs arithmetic internally.
Table 6 Hypotheses for Sora2’s advantage.
Factor Implication
1 Stronger base + similar data Higher intrinsic arithmetic; symbolic capability may be baked in
2 Additional RL training Reward shaping teaches math beyond diffusion; could transfer to CLI
3 System-level reprompt/recaption LLM computes answers; strong conditioning drives generation
10
The evidence supports system-level conditioning as a practical path forward for this NC instantiation. Among
the three hypotheses for improving arithmetic-probe performance—stronger base models, reinforcement
learning, or enhanced conditioning—our results most strongly favor the third approach. The gain from
reprompting (4%→83%), achieved without modifying the underlying NC backbone, is substantial. It shows
that measured “reasoning” on these probes is highly sensitive to specification and conditioning. We therefore
do not treat it as direct evidence of native arithmetic inside the NC backbone.
In our setting, strategic conditioning yields larger symbolic-probe gains than the RL pipeline we tested.
Evaluations should therefore distinguish native computation from conditioning-assisted performance when
assessing reasoning capabilities in current video-based NC instantiations.
3.1.5 Visualizations
(1) CLIGen (General) visualizations. Qualitative samples highlight the breadth of real-world terminal dynamics
captured in CLIGen (General): ANSI escape sequences that repaint regions with changing foreground/background colors, incremental command entry with syntax highlighting and cursor edits, classic shell prompts
and system outputs, long-running jobs with rapidly scrolling and color-coded package logs, full-screen TUIs
such as partition editors, and progress dashboards with updating bars, counts, and ETAs. These traces
emphasize that “looking correct” requires maintaining terminal geometry, palette transitions, and cursor state
frame-by-frame.
(2) CLIGen (Clean) REPL visualizations. In contrast to open-world traces, CLIGen (Clean) REPL samples
are scripted and temporally well-paced (Figures 16–19; additional examples are in Appendix C). Each sample
includes an explicit action trace (e.g., Sleep, Type, Enter, arrow keys, Hide) alongside rendered terminal
frames, making the action-to-pixel link visually unambiguous. The key insight is that these scripted traces
isolate rendering-and-control errors from semantic ambiguity: with explicit actions, failures are dominated
by low-level mechanics (cursor placement, character edits, monospace alignment, line breaks, temporal
consistency).
(3) CLIGen (Clean) math visualizations. Figures 20–22 compare math REPL rollouts, and Figures 23–25 show
reprompting cases. Together they highlight why arithmetic probes should separate native computation from
answer-conditioned rendering. All full-resolution pages are in Appendix E; below we keep clickable thumbnails
at the original location for quick navigation.
11
CLIGen Visualization Thumbnails
Click any thumbnail to jump to its full-resolution page in Appendix
CLIGen (General) Visualizations
The terminal displays a series of ANSI escape code formatted texts with changing background and foreground colors, executing commands like
`\\u001b[48;2;255;128;128;38;2;0;0;0m` which set the background to a shade of pink and text to black, and printing numbered lists with colors. The output includes specific
numbers, such as \"1\", \"5\", \"7\", and \"9\", in different colors, creating a visually dynamic and colorful display, but the exact username, hostname, and path are not
specified in the provided terminal session content.
The user types the command `CREATE TABLE posts (ID INTEGER)`, with the terminal displaying the command in a dark background with colored syntax highlighting,
including green and yellow text, and the cursor moving character-by-character as the user types, with some corrections and backspacing along the way. The output shows
the command being executed, with key words like `CREATE` and `TABLE` in distinct colors, and the filename `posts` appearing in the command line.
Samples A
At the `root@localhost:~#` prompt, the user types the `date` command, which displays the current date and time in a plain text format as \"2021. 10. 11. 22:47:43 KST\", then
begins typing the `cat` command.
The terminal displaying progress bars, package names like `pillow`, `notebook`, and `tzlocal`, and version changes in green and red text. The output shows downloading
and installing statuses, including percentages, for packages like `smmap`, `tomli`, and `protobuf`, with the terminal scrolling through the output rapidly.
Samples B
At the unspecified username@hostname prompt, the terminal displays a partition editor with a disk image file named \"sd.img\" (128 MiB) and the user interacts with it,
creating a new Linux partition from free space, with key output content showing partition details in a table format, including \"sd.img1\" and \"sd.img2\" with their respective
sizes and types, and a new partition \"sd.img3\" with 55M size and Linux type (83). The terminal shows a mix of black and colored text, including blue and red, with a cursor
that blinks and moves to different parts of the screen as the user navigates through the partition editor options, such as \"New\", \"Quit\", and \"Write\", with specific prompts
like \"Partition type: Linux (83)\" and \"Create new partition from free space\".
The terminal displays a progress bar with the command output \"Evaluating\" and percentages from 60% to 85%, showing yellow progress bars with increasing completion,
such as \"│████████████████████▍ │\" to \"│████████████████████████▉ │\", alongside item counts \"24/40\" to \"34/40\" and time estimates \"0:00:20\"
to \"0:00:07\". The output includes specific item completion and estimated time remaining, with the yellow-colored progress bar indicating the evaluation progress.
Samples C
CLIGen (Clean) Visualizations
Sleep 200ms
Type "env | head -n 5"
Enter
Sleep 600ms
Hide
Sleep 200ms
Type "date"
Enter
Sleep 300ms
Type "whoami"
Enter
Sleep 300ms
Sleep 500ms
Type "echo $HOME"
Sleep 90ms
Enter
Sleep 1442ms
Hide
Samples A
Sleep 200ms
Type "top"
Enter
Sleep 2s
Down 3
Sleep 600ms
Up 2
Hide
Sleep 500ms
Type "echo $PATH"
Sleep 90ms
Enter
Sleep 1442ms
Hide
Sleep 400ms
Type "seq 1 28 | paste -
d',' - - - - | column -t -s','
| tee metrics_7x4.txt"
Enter
Sleep 2000ms
Hide
Samples B
Sleep 400ms
Type "echo Home: $HOME"
Enter
Sleep 600ms
Type "date +%H:%M:%S"
Enter
Sleep 600ms
Type "printenv HOME"
Enter
Sleep 600ms
Sleep 400ms
Hide
Sleep 400ms
Type "python - <<'PY'"
Enter
Type "import time"
Enter
Type "for i in range(18):"
Enter
Type " print(f'Frame
{i:02d} ::' + '>' * (i % 20))"
Enter
Type " time.sleep(0.2)"
Enter
Type "PY"
Enter
Sleep 4000ms
Hide
Samples C
Sleep 180ms
Type "echo History size:
$HISTSIZE"
Sleep 120ms
Enter
Sleep 400ms
Type "cal"
Sleep 120ms
Enter
Sleep 400ms
Type "echo Home:
$HOME"
Sleep 120ms
Enter
Sleep 400ms
Sleep 400ms
Hide
Sleep 800ms
Sleep 180ms
Type "echo Learning shell
basics"
Sleep 120ms
Enter
Sleep 400ms
Type "date +%Y-%m-%d"
Sleep 120ms
Enter
Sleep 400ms
Type "echo Login shell: $0"
Sleep 120ms
Enter
Sleep 400ms
Type "uname -r"
Sleep 120ms
Enter
Sleep 400ms
Sleep 400ms
Hide
Samples D
12
CLIGen Visualization Thumbnails
Click any thumbnail to jump to its full-resolution page in Appendix
CLIGen (Clean) Math Comparison
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "5"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "5"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "5"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "5"
Enter
Sleep 800ms
Hide
Samples A
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "10+15"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "10+15"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "10+15"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "10+15"
Enter
Sleep 800ms
Hide
Samples B
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "40/1"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "40/1"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "40/1"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type "40/1"
Enter
Sleep 800ms
Hide
Samples C
CLIGen (Clean) Math Reprompting
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”28-23"
Enter
Sleep 800ms
Hide
The answer is 5
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”28-23"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”28-23"
Enter
Sleep 800ms
Hide
Samples A
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”560-278"
Enter
Sleep 800ms
Hide
The answer is 282
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”560-278"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”560-278"
Enter
Sleep 800ms
Hide
Samples B
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”736*644"
Enter
Sleep 800ms
Hide
The answer is 473984
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”736*644"
Enter
Sleep 800ms
Hide
Sleep 200ms
Type "python"
Enter
Sleep 1s
Type ”736*644"
Enter
Sleep 800ms
Hide
Samples C
13
3.2 The GUI World Models
We also instantiate the NC abstraction in interactive desktop environments with NCGUIWorld. In this setting,
fine-grained action control is essential: GUI interaction requires precise cursor tracking, timely click feedback,
and robustness to rapidly changing interface states. We model each interaction as a synchronized sequence of
RGB frames xt and input events ut (mouse and keyboard). The latent video state maintains interface context
across frames, while temporally aligned action inputs provide control signals designed to preserve pixel-level
correspondence between user actions and visual changes.
3.2.1 Data pipeline
Table 7 Cursor/action statistics.
Split
Avg. cursor
speed
(px/frame)
Actions / sec
Random Slow 1.51 1.58
Random Fast 195.15 4.18
CUA (supervised) 3.79 0.10
The dataset includes two styles of random interaction: “Random Slow” and “Random Fast”, plus a
smaller set of supervised trajectories collected by
Claude CUA (Anthropic). Random Slow (approximately 1,000 hours) contains longer pauses, idle gaps,
and deliberate cursor movements, which can expose
cursor drift after extended inactivity. Random Fast
(approximately 400 hours) features denser cursor motion and typing bursts, stressing acceleration dynamics and hover timing. The supervised trajectories are
approximately 110 hours. These goal-directed traces provide higher-signal action–response pairs without
overwhelming the exploration data. Table 7 summarizes cursor and action statistics across splits; in the
collected CUA trajectories, action density is lower due to latency introduced by Claude’s tool API between
successive steps.
All GUI data is collected inside an Ubuntu 22.04 container running XFCE4 (Arc-Dark theme, Papirus icons)
on a fixed 1024×768 virtual display at 15 FPS. We render the display with Xvfb and interact through a
VNC/noVNC stack. The desktop pins a small open-source app set to launchers. It includes Firefox ESR,
GIMP, VLC, VS Code, Calculator, Terminal, the file manager, and the Mahjongg game, matching the
environment shown in our recordings. Screen capture uses mss and ffmpeg with cursor overlays, and actions
are replayed and logged via xdotool. We keep the recorded discontinuities and interface latency intact rather
than smoothing them. In dataset packaging, we store both raw-action and meta-action views for modeling.
This lets us train either the raw-action or meta-action encoder under the same loss stack5
.
3.2.2 Model architecture
The GUIWorld architecture builds on the Wan2.1 (Wan et al., 2025) by incorporating explicit actionconditioning modules. The central challenge is to align time-stamped user actions with generated frames and
inject this information at the appropriate depth within the transformer.
Action features are encoded on-the-fly from frame-aligned mouse and keyboard signals (Section 3.2.1). We
aggregate them into latent-aligned embeddings that summarize recent action history at each diffusion step.
We evaluate two action encoders. The raw-action encoder (v1) preserves fine-grained mouse/keyboard event
streams. The meta-action encoder (v2) abstracts interactions into coarse API-style categories (clicks, drags,
scrolls, typing, shortcuts). Both encoders use the same temporal alignment and are evaluated as separate
ablations. In our experiments, their differences in rendering fidelity and control behavior are modest6
.
We inject action embeddings into the diffusion backbone in four ways (Figure 7). We study external,
contextual, residual, and internal conditioning. For the injection-scheme ablation, all four modes share the
same meta-action encoder and temporal alignment. They differ only in where the latent action features
interact with the video latents and transformer blocks. We compare raw-action vs. meta-action encoders
separately in Table 11.
External conditioning. In the external mode, action information modulates the latent video sequence before
the diffusion transformer. Action features are applied as a pre-conditioning step at the model input, without
introducing explicit action tokens or cross-attention inside the diffusion backbone. As a result, action
5Conversion details and alignment quality appear in Appendix D.
6Appendix Table 17 summarizes the representational differences.
14
Diffusion Transformer Self-Attention Cross-Attention FFN
× N
Action
Encoder
Visual Tokens Action Tokens
Action
Encoder
Action
Encoder
c
VAE
Encoder
Diffusion Transformer Self-Attention Cross-Attention FFN
× N
Mode Injection point Notes
1 external Conditioned VAE input Actions are injected as an external conditioning stream that
modulates the backbone without sharing the main token sequence.
2 contextual Concat frames, actions Video and action tokens share one sequence with a lag-aware
temporal attention mask, enabling mid-level fusion.
3 residual Hidden state with action delta Actions are added through residual-style modulation branches,
acting as a strong but more indirect baseline.
4 internal After cross-attn, before FFN Actions enter through dedicated cross-attention inside the transformer blocks, fusing near the backbone core.
Figure 7 Four modes for injecting GUI actions into the diffusion transformer. 1 modulates VAE latents before the
transformer; 2 adds action tokens alongside visual tokens; 3 applies block-wise residual updates; and 4 inserts an
action cross-attention module inside transformer blocks.
information enters only through the modified input latents; the diffusion backbone never attends directly to
action tokens, so any action signal must be carried implicitly in z
′
1:T
.
Formally, given VAE latents z1:T and temporally aligned action features u1:T , an external action module
applies a small stack of temporal self-attention and action cross-attention layers. This produces a residual
update ∆z1:T (u1:T ). The modified latents are
z
′
1:T = z1:T + ∆z1:T (u1:T ),
and the diffusion transformer operates solely on z
′
1:T
. The diffusion backbone remains unchanged, and action
features are not exposed as explicit tokens within the transformer.
Contextual conditioning. In the contextual mode, actions are represented as additional tokens and integrated
directly into the transformer’s self-attention. Similar token-based action representations have been explored
in prior world models, including Gato (Reed et al., 2022) and World and Human Action Models (Kanervisto
et al., 2025).
The meta-action encoder produces latent-aligned action tokens A ∈ R
La×D. We concatenate them with
visual tokens V ∈ R
Lv×D to form a joint sequence [V ; A]. Each transformer block applies self-attention over
this combined sequence using a structured temporal mask (Appendix Figure 12). The mask enforces causal
alignment: each frame token attends only to actions within a short past window, and each action token attends
only to frames after a fixed temporal lag. Through this masked joint attention, contextual conditioning fuses
action and visual information within the transformer blocks.
Residual conditioning. In the residual mode, the transformer block structure remains unchanged. A lightweight
action module attaches to a subset of layers as an external residual branch. This follows the residual
conditioning paradigm introduced by ControlNet (Zhang et al., 2023), while remaining modular and additive
to the base diffusion backbone.
At each selected layer l, the transformer applies its standard sequence of self-attention, text or reference
cross-attention, and feed-forward operations to produce hidden states h
(l)
. A separate action module then
takes h
(l)
together with a local temporal window of latent action features and mouse trajectories. It outputs
a residual update ∆h
(l)
(a, mouse). The updated hidden states are given by
h˜(l) = h
(l) + ∆h
(l)
(a, mouse),
15
which are passed to the subsequent transformer block. In this formulation, residual conditioning injects action
information through block-external residual branches. It does not modify the internal computations of the
transformer blocks themselves.
Internal conditioning. In the internal mode, action conditioning is incorporated directly within the transformer
blocks. Related multi-stream world models have explored similar designs, including Matrix-Game-2 (He et al.,
2025). Each selected block augments the standard attention stack with an additional action cross-attention
sub-layer. Specifically, the block applies self-attention, followed by cross-attention over text and reference
features, and then a dedicated action cross-attention layer. Keys and values are derived from latent action
features (and, optionally, mouse inputs).
Given block input h, text or reference context c, and action latents a, the internal block computes
h
′ = FFN
h + CAtext
SA(h), c
+ CAaction(h, a)

,
where SA denotes self-attention and CAtext and CAaction denote the text and action cross-attention modules
applied in sequence. As illustrated in Figure 7, action features are injected directly into the block’s crossattention stage.
In contrast to residual conditioning, internal conditioning integrates action information through a blockinternal attention mechanism rather than an external residual branch. This design mirrors the multi-stream
injection strategy used in Matrix-Game-2 (He et al., 2025) and yields the best SSIM/FVD trade-off for
fine-grained GUI interaction in our ablations. In this setting, precise temporal alignment and spatial locality
are critical. Each conditioning mode (external, contextual, residual, and internal) is trained as a separate
ablation, and no combinations are used.
3.2.3 Implementation Details
We train one model per injection mode (external, contextual, residual, internal), keeping the backbone and
all non-action components fixed. Each run lasts about 64k steps. We tune only the action encoder and
learning-rate schedule. Training optimizes the diffusion loss together with a small temporal contrastive loss
that aligns frame features with action and mouse embeddings (Appendix D). Runs use 64 GPUs for about 15
days, totaling about 23k GPU-hours per full pass.
Preprocessing is implemented in the data loader in two stages. First, we normalize each recording to a
fixed resolution and frame rate. This produces tensors for RGB video, per-frame cursor coordinates, and
mouse/keyboard event traces (in both raw-action and meta-action views). Second, we render an SVG cursor
at each logged position to produce per-frame masks and cursor-only reference frames. The first reference
frame contains the full desktop with a unit mask. Later references paste only the cursor over a neutral
background, with a mask restricted to arrow pixels. After VAE encoding, these references become latent
slots that pin down the static GUI layout at t=0. For t>0, they supervise only a small patch around the
cursor and leave the rest of the frame unconstrained. We drop clips without valid cursor or action traces to
keep supervision consistent.
3.2.4 Evaluation setup
Our ablations target three capabilities: global fidelity, post-action responsiveness, and cursor-control precision.
We use the FVD/LPIPS/SSIM suite as the core metrics. We also add action-driven metrics that focus on
post-interaction frames after clicks, scrolls, and key/type events. For example, we compute SSIM/LPIPS averaged over the k frames after each logged action, and action-driven FVD on post-action clips. Ablations
vary conditioning design and action encoding to measure how these choices affect perceptual quality and
responsiveness when rolled out against ground-truth interfaces7
.
7Full metric definitions and implementation details are provided in Appendix B.3.
16
1 In GUIWorld, a small amount of goal-directed data outperforms much larger random exploration, showing
that alignment quality matters more than nominal scale for action–response learning.
2 Precise cursor control requires explicit visual supervision: SVG mask/reference conditioning raises cursor
accuracy to 98.7%, indicating that local GUI control primitives are learnable in controlled settings.
3 Action injection depth matters: relative to shallow external conditioning, contextual, residual, and especially
internal fusion improve post-action responsiveness and visual consistency.
4 Action representation also matters: under the same injection mode, API-like meta-actions consistently
outperform raw event-stream encoding.
Experiment 7: Data quality dominates performance
Interactive GUI modeling shows that data quality matters more than dataset size for action-driven performance.
We compare slow exploration, fast interaction, and supervised trajectories under contextual conditioning.
This isolates which behaviors best support neural computer training.
Table 8 Overall performance across data sources.
Split FVDall SSIMall LPIPSall
Untrained baseline 149.61 0.496 0.605
Random Fast (train) 48.17 0.695 0.483
Random Slow (train) 20.37 0.830 0.237
Claude CUA (train) 14.72 0.885 0.144
Despite approximately 1,400 hours of random exploration across the slow and fast settings, these datasets
are noisy. They are comparatively sample-inefficient
for learning stable action–response mappings. They
substantially improve global perceptual metrics over
a baseline (Table 8). However, high-frequency cursor
jitter and irregular, non-goal-directed action bursts
make consistent control difficult under dense, stochastic input streams.
In contrast, the substantially smaller high-quality dataset (110 hours from Claude CUA) yields markedly
stronger performance across all metrics. Goal-directed trajectories provide clearer action semantics and more
predictable state transitions. This enables robust action conditioning even with limited data volume. These
results indicate that neural computer development should prioritize curated, purposeful interactions over
large-scale passive data collection. At the current stage, this result primarily indicates that alignment quality
matters more than nominal scale for learning action–response structure in NC prototypes.
Experiment 8: Precise cursor control requires explicit visual supervision
Figure 8 Cursor references in GUIWorld. Left: original desktop frames. Middle: binary cursor masks. Right: cursor-only
reference frames rendered over a neutral background.
Table 9 Cursor conditioning losses versus accuracy.
Loss variant Cursor accuracy
Position (x, y) only 8.7%
Position (x, y) + Fourier 13.5%
Position (x, y) + SVG mask/ref 98.7%
We examine whether the NC internalizes cursor dynamics. A natural baseline is to condition on normalized cursor-coordinate sequences
mouse_trajectories ⊂ [0, 1]T ×2
(details in Appendix D.4). To strengthen this signal, we further
encode the normalized trajectories using a Fourier
mouse encoder. We map coordinates to [−1, 1]2 and
project them through a fixed Gaussian matrix to obtain random Fourier features. A small MLP produces
17
per-frame embeddings, which we aggregate into lag-aware windows aligned with the VAE stride. The resulting
latent action sequence conditions the action modules and participates in the temporal contrastive loss.
However, Table 9 shows that coordinate-based supervision remains insufficient for precise interaction. Positiononly supervision achieves 8.7% accuracy, and even enhanced position features reach only 13.5%. This suggests
that richer coordinate encodings alone do not resolve cursor drift and jitter.
Motivated by the importance of precise cursor placement, we introduce explicit visual cursor supervision. We
render an SVG cursor at each (xt, yt) to produce per-frame cursor masks mt and cursor-only foregrounds ft
(right panel of Figure 8). Following Figure 8, we construct a reference stream. The first frame contains the full
desktop image, while subsequent frames contain only the cursor foreground over a neutral background, masked
to the cursor region. We encode both the video and reference streams with the shared VAE, yielding video
latents z1:T , reference latents z
ref
1:T
, and mask tags τ1:T . The diffusion transformer receives the concatenated
tensor
concat
z1:T , τ1:T , zref
1:T

,
which anchors the static GUI layout at t=0 and provides localized supervision around the cursor for t>0.
Under this explicit visual conditioning, cursor accuracy improves to 98.7%. This suggests that neural computers
benefit from learning the cursor state as a visual object rather than relying solely on abstract coordinates.
Explicit pixel-level supervision helps model cursor acceleration, hover states, and click feedback, which are
essential for reliable GUI interaction. At the same time, this result is best viewed as evidence that local GUI
control primitives are learnable under explicit supervision in controlled settings.
Experiment 9: Action injection under different schemes
Table 10 Action-driven metrics across injection schemes (15 frames after action).
Mode SSIM+15 ↑ LPIPS+15 ↓ FVD+15 ↓
baseline1 (untrained) 0.326 0.649 184.3
baseline2
†
(external) 0.746 0.251 33.4
contextual 0.813 0.190 24.8
residual 0.857 0.138 18.8
internal 0.863 0.141 14.5
† baseline2 (external) was early-stopped at ∼50% of the planned training budget after preliminary rollouts did not warrant further
compute. Included only as a rough reference.
Holding data and the action encoder fixed, we compare injection schemes on clean runs (Table 10). We
compute action-driven metrics over the 15 frames following each click, scroll, or key event. Relative to both
baselines (untrained and external), mid- and deep-level fusion yields consistent improvements in post-action
quality. This includes contextual, residual, and internal injection.
Specifically, moving from input-level conditioning (external) to token-level fusion (contextual) improves SSIM
from 0.746 to 0.813 and reduces FVD from 33.4 to 24.8. Deeper injection sharpens these gains. internal
achieves the highest structural consistency (SSIM 0.863) and the lowest temporal distortion (FVD 14.5), while
residual attains the lowest perceptual distance (LPIPS 0.138). Together, these trends associate deeper action
injection with improved tracking of fine-grained cursor motion and layout changes. 8
Experiment 10: Do action encodings matter?
Table 11 Raw-action vs. API-like action encoding under the same injection mode (15 frames after action).
Mode Encoding SSIM+15 ↑ LPIPS+15 ↓ FVD+15 ↓
internal raw-action (event-stream) 0.847 0.144 16.6
internal meta-action (API-like) 0.863 0.141 14.5
We compare two action encodings under the same injection mode to isolate the effect of representation choice
(Table 11). Under internal conditioning, the meta-action (API-like) encoding yields small but consistent
8Appendix D summarizes the corresponding injection schemes and alignment details.
18
Table 12 Encoding examples for raw-action and meta-action encoders.
Raw-Action Encoder
(13 mouse actions, 169 keyboard actions)
Meta-Action Encoder
(5 meta-actions plus typed parameters)
User intent Raw-action encoder (event
stream)
Meta-action encoder (API-like
slot)
ls -l • Key events per character (e.g.,
l, s, <space>, -, l)
• Activates entries in a 169-d
keyboard multi-hot
• No explicit command semantics; inferred from sequence
• type: KeyboardType
• text: "ls -l"
• Encoded by shared text encoder
ctrl+v • Separate keydown/keyup
events for ctrl and v
• Activated in multi-hot (or
shortcut entry)
• type: Shortcut
• id: ctrl+v
• Embedded via shortcut table
improvements over the raw-action representation. SSIM increases from 0.847 to 0.863, LPIPS drops from 0.144
to 0.141, and FVD drops from 16.6 to 14.5. However, these gains are modest compared to the substantially
larger improvements observed when varying the action injection scheme itself (Table 10). This suggests that
encoding granularity is not the dominant factor governing GUI interaction fidelity.
Table 12 contrasts how short commands and shortcuts (e.g., ls -l, ctrl+v) are represented under the two
encodings. The raw-action encoder treats typing as a stream of individual key events, leaving command or
shortcut semantics to be inferred from the sequence. In contrast, the meta-action encoder collapses each
interaction into a single typed action with associated text or a shortcut identifier. This design aims to model
user actions as structured, tool-like operations rather than fragmented event streams.
In practice, this more structured abstraction does not translate into clear qualitative gains. Rendered text
remains similarly smeared under both encodings, and robustness under theme changes and timing noise is
largely unchanged. Task-level failures such as re-centering, re-acquisition, and multi-step interactions persist
across both representations. We adopt the meta-action encoder as the default for its simplicity and semantic
alignment with system-level conditioning. These results suggest that encoding granularity is secondary to
alignment quality and injection strategy.
3.2.5 Visualizations
Across GUIWorld interactive rollouts, failure modes are dominated by data quality and by where action
information enters the backbone. Goal-directed supervision produces smooth, target-aligned cursor paths and
consistent post-click UI transitions, whereas random exploration yields bursty jitter and spurious actions that
degrade visual coherence (Table 8; Figures 26–30). Consistent with the action-driven metrics in Table 10,
deeper token-level injection (contextual/internal) yields more reliable post-action updates in interactive
elements (hover states, dropdowns, modals) and maintains cursor alignment under rapid motion.
Figures 31–33 emphasize how small low-level deviations compound. Figures 34–36 focus on numeric/UI fidelity
and interaction semantics. Figures 37–39 add stress cases where correctness hinges on precise field edits and
page state. All full-resolution pages are in Appendix E; below we keep clickable thumbnails at the original
location for quick navigation.
19
GUIWorld Visualization Thumbnails
Click any thumbnail to jump to its full-resolution page in Appendix
GUIWorld Samples 1–5
“thinking”: “Now I‘ll click on the Term icon to open the terminal:”
“action”: “left_click”
,
“x”: 82, “y” : 520
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 1
“thinking”: “Now I'll press Enter to execute the command:”
“action”: “type”
,
“text”: “cat ~/Desktop/datetime.txt”
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 2
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
“thinking”: “I see Wikipedia appears in the suggestions. I'll click on it to go to Wikipedia:”
“action”: “left_click”
,
“x”: 240, “y” : 150
Samples 3
“thinking”: “I can see that the theme has changed to \"smooth\" and I can already see the tiles in the background have a different style. Now I need to click the \"Close\" button to
apply these changes and close the preferences dialog:”
“action”: “left_click”
,
“x”: 629, “y” : 423
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 4
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
“thinking”: “I can see the border_photo.png file that we created with ffmpeg. I'll double-click on it to open it in GIMP: ”
“action”: “double_click”
,
“x”: 310, “y”: 143
“thinking”: “Great! Now the image is open in GIMP. I can see it's a peachpuff colored image as requested. Now I'll add a border to it. In GIMP, I'll go to the \"Filters\" menu
and look for a border option:”
“action”: “left_click”
,
“x”: 547, “y”: 43
Samples 5
GUIWorld Samples 6–10
“thinking”: “I see I've clicked the \"2\" button. Now I need to click the \"0\" button twice to make \"200\": ”
“action”: “left_click”
,
“x”: 43, “y” : 465
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 6
“thinking”: “Perfect! GIMP is now open. Let's import an image from the Downloads folder. I'll click on the File menu and select Open:”
“action”: “left_click”
,
“x”: 43, “y” : 465
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 7
“thinking”: “Excellent! Nautilus is open and showing the home directory. Now I need to navigate to the Downloads folder. I can see
the Downloads folder in the current view, so I'll double-click on it:”
“action”: “double_click”
,
“x”: 248, “y” : 408
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 8
“action”: “type”
,
“text”: “512”
“thinking”: “I can see that I've set the width to 512 pixels. Since the \"chain\" link icon between width and height appears to be active, the height should automatically change to
maintain the aspect ratio. I'll click the \"Scale\" button to apply the resize:
“action”: “left_click”
,
“x”: 723, “y”: 616
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
“thinking”: “Now I‘ll type \”512\“ to change the width:”
Samples 9
“thinking”: “Now I'll click and drag to create a selection for the square:”
“action”: “left_click_drag”
,
“x”: 400, “y” : 400
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 10
20
GUIWorld Visualization Thumbnails
Click any thumbnail to jump to its full-resolution page in Appendix
GUIWorld Samples 11–12
“thinking”: “I can see the Firefox icon on the desktop. Now I'll click on it to open Firefox:”
“action”: “double_click”
,
“x”: 224, “y”: 70
“thinking”: “I see Firefox has opened with a welcome page. Now I'll click on the address bar and navigate to Wikipedia:”
“action”: “double_click”
,
“x”: 487, “y”: 81
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 11
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
“thinking”: “"Now I'll type in 768 for the height:”
“action”: “type”
,
“text”: “512”
“thinking”: “Now I'll click the OK button to create the new image:”
“action”: “left_click”
,
“x”: 763, “y”: 514
Samples 12
GUIWorld Samples 13–14
“thinking”: “Now I‘ll click on the Term icon to open the terminal:”
“action”: “left_click”
,
“x”: 82, “y” : 520
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
Samples 13
Ground
Truth
Generation
Collect Ground Truth
Video with CUA
First frame and
actions as input
“thinking”: “I see the page has pagination at the bottom. Let me continue navigating to find the Advanced section. I'll click on page 2 to see more tutorials.”
“action”: “left_click”
,
“x”: 448,
“y” : 539
Samples 14
21
