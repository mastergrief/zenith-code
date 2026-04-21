# Neural Computers

*Mingchen Zhuge, Changsheng Zhao, Haozhe Liu, Zijian Zhou, Shuming Liu, Wenyi Wang, Ernie Chang, Gael Le Lan, Junjie Fei, Wenxuan Zhang, Yasheng Sun, Zhipeng Cai, Zechun Liu, Yunyang Xiong, Yining Yang, Yuandong Tian, Yangyang Shi, Vikas Chandra, Jürgen Schmidhuber*

*Meta AI · KAUST — [metauto.ai/neuralcomputer](https://metauto.ai/neuralcomputer) — April 9, 2026*
*arXiv:2604.06425v1 [cs.LG]*

## Abstract

We propose a new frontier: **Neural Computers (NCs)** — an emerging machine form that unifies computation, memory, and I/O in a learned runtime state. Unlike conventional computers, which execute explicit programs; agents, which act over external execution environments; and world models, which learn environment dynamics — NCs aim to make the model itself the running computer. Our long-term goal is the **Completely Neural Computer (CNC)**: the mature, general-purpose realization of this emerging machine form, with stable execution, explicit reprogramming, and durable capability reuse.

As an initial step, we study whether early NC primitives can be learned solely from collected I/O traces, without instrumented program state. Concretely, we instantiate NCs as video models that roll out screen frames from instructions, pixels, and user actions (when available) in CLI and GUI settings. These implementations show that learned runtimes can acquire early interface primitives — especially I/O alignment and short-horizon control — while routine reuse, controlled updates, and symbolic stability remain open. We outline a roadmap toward CNCs around these challenges. If overcome, CNCs could establish a new computing paradigm beyond today's agents, world models, and conventional computers.

## TL;DR

- **Abstraction:** one set of weights = a computer. Latent runtime state `hₜ` replaces the classical compute/memory/I/O stack; update map `F_θ` integrates observations + conditioning; render map `G_θ` produces the next frame. Auxiliary heads stand in for OS queues, drivers, and UI toolkits.
- **Two prototypes built on Wan2.1 video diffusion:**
  - **CLIGen (§3.1)** — terminal rollouts from asciinema traces. Datasets: CLIGen-General (diverse public) + CLIGen-Clean (scripted Dockerized VHS runs). Learns character-level text rendering, literal-caption driven accuracy, early CLI reasoning primitives. RL not required for symbolic probes.
  - **GUIWorld (§3.2)** — desktop RGB + mouse/keyboard traces. Action-conditioned rendering, cursor control requires explicit visual supervision, data quality dominates over data volume.
- **Findings:** I/O alignment and short-horizon control emerge; routine reuse, controlled updates, symbolic stability remain open problems.
- **Position (§4):** roadmap to CNC across efficiency, computation/reasoning, memory/storage, I/O & control, tool bridges, conditional generalization, programmability, artifact generation. Relates NCs to agents, world models, OS runtimes.

## Sections

- [01_background.md](01_background.md) — §1 Introduction (what is an NC, CNC goal) + §2 Preliminaries (latent runtime formalism, update-and-render loop) + §2.1 Related Work (neuromorphic, NTM/DNC/NPI, world models, Genie/Veo/Sora/NeuralOS).
- [02_implementation.md](02_implementation.md) — §3 Implementation: §3.1 CLIGen (data pipeline, architecture, 6 CLI experiments, visualizations) + §3.2 GUIWorld (data, architecture, 4 GUI experiments, visualizations).
- [03_position.md](03_position.md) — §4 Position: Toward Completely Neural Computers (§4.1 NC→CNC gap, §4.2 Roadmap, §4.3 Relations to other system objects, §4.4 Additional thoughts) + §5 Conclusion.
