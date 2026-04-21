# Neural Computers — Introduction & Preliminaries

← back to [NC.md](NC.md)

1 Introduction
Can a single set of weights act as a “computer”? We term this abstraction a Neural Computer (NC): a neural
system that unifies computation, memory, and I/O in a learned runtime state. This usage is distinct from the
Neural Turing Machine / Differentiable Neural Computer line (Graves et al., 2014, 2016): our concern is not
differentiable external memory, but whether a learning machine can begin to assume the role of the running
computer itself.
To implement this idea, we instantiate NCs as video models. At this stage, video models are the most
practical substrate for this prototype, though we expect the long-term solution to require a fundamentally new
neural architecture (Section 4). This implementation draws on several technical lines. World models (Ha and
Schmidhuber, 2018) show that neural networks can internalize environment dynamics and support predictive
imagination, while high-capacity video generators such as Veo 3.1 (Google, 2025) and Sora 2 (OpenAI,
2025) show that such learned dynamics can be rendered into coherent frame sequences. Frontier interactive
video models such as Genie 3 (Bruce et al., 2024) further extend this trajectory toward action-controllable
generative environments. These lines provide practical machinery for current NC prototypes, but do not by
themselves define the NC abstraction. In parallel, LLM-driven UI systems such as Imagine with Claude1 map
natural-language inputs to structured interface updates. Yet these capabilities remain split across different
systems objects: conventional computers execute explicit programs, agents act through external execution
environments, and world models render or predict environment dynamics, while executable state still resides
outside the model. NCs are motivated by this gap: they are not a smarter layer on top of the existing stack,
but a proposal to make the model itself the running computer. The immediate question in this paper is
whether early runtime primitives can be learned directly from raw interface I/O without privileged access to
program state.
Throughout this paper, NC denotes this proposed machine form, while CNC denotes its mature, generalpurpose realization. We study two interface-specific prototypes of this NC formulation (see Section 2).
NCCLIGen models terminal interaction from text (natural language or command lines) and an initial frame,
while NCGUIWorld models desktop interaction from recent pixels and synchronized mouse/keyboard actions
(Sections 3.1 and 3.2).
Neural Computer (NC) abstraction (Teaser). A neural system (F, G) parameterized by θ that models an interactive
computer interface through a single latent runtime state ht that carries executable interface state and also acts as
working memory (see Eq. (2.1)).
In the NCCLIGen experiments, the NC can render and execute basic command-line workflows. It often stays
aligned with the terminal buffer and captures common “physics” of everyday CLI use (e.g., fast scrollback,
prompt wrapping, window resizing). On carefully scripted data, rollouts can be visually and structurally close
to real sessions, and the model can execute short command chains and render their outputs. Arithmetic-probe
scores improve substantially with stronger system-level conditioning, though symbolic stability remains limited.
In the NCGUIWorld experiments, we evaluate standard world-model designs across action injection, action
encoding, and data quality. Figure 1 summarizes this template across two interface-specific NCs trained
separately without shared parameters. Qualitatively, the model learns coherent pointer dynamics and shorthorizon action responses (e.g., hover/click feedback and window/menu transitions), suggesting that local GUI
control primitives are learnable in controlled settings.
Our experimental insights indicate that current NCs already realize early runtime primitives, most notably
I/O alignment and short-horizon control. The long-term target is a Completely Neural Computer (CNC),
the mature, general-purpose realization of this machine form: a fully learned computer whose compute,
memory, and interfaces are unified in a single learned runtime substrate rather than engineered as separate
modules. These prototypes are an early step toward that CNC vision. Substantial challenges remain in robust
long-horizon reasoning, reliable symbolic processing, stable capability reuse, and explicit runtime governance.
Section 4 outlines these open challenges and a roadmap toward CNCs.
1https://claude.ai/imagine/
2
Completely Neural Computer (CNC) abstraction (Section 4.2). A Neural Computer instance is complete (i.e., a CNC) if
it is (i) Turing complete, (ii) universally programmable, (iii) behavior-consistent unless explicitly reprogrammed, and
(iv) realizes the architectural and programming-language advantages of NCs relative to conventional computers.
Concretely, this work makes the following contributions:
• Define neural computers (NCs) and build video-based prototypes for both CLI and GUI interfaces.
• Provide a data engine and alignment recipe that synchronize text, actions, and frames for the CLI and
GUI environments used in this paper.
• Identify practical design choices for NCs through extensive ablation studies.
• Outline an engineering roadmap toward completely neural computers (CNCs), centered on acceptance
challenges such as reuse, consistency, and runtime governance.
2 Preliminaries
Throughout this paper, we use conventional digital computers as an umbrella term for stored-program machines
(e.g., von Neumann-style architectures): at the theory level they are commonly abstracted as random-access
machines with an instruction set architecture, and at the systems level they are typically realized through
layered operating-system/application stacks. Such systems separate computation, memory, and I/O. Our
motivating question is whether a single set of weights can internalize these roles inside one latent runtime
state, rather than relying on an external execution environment (e.g., OS/simulator) to carry executable state.
We model a video-based neural computer (NC) prototype as a learned latent-state system that folds these
roles into an update-and-render loop.
Figure 1 Neural computers across interfaces. Given a prompt or action stream, an NC rolls out future interface frames
for / NCCLIGen (top) and NCGUIWorld (bottom). Logos denote datasets; NCCLIGen and NCGUIWorld are the
corresponding models trained on those datasets. It models terminal or desktop dynamics.
Specifically, an NC updates a latent runtime state from the current observation and conditioning input, and
then predicts (or samples) the next observation. In this paper, we treat screen frames as observables and
3
define actions as time-indexed conditions. More broadly, the NC framework can accommodate various other
modalities and structural representations for both observables and actions. Given an initial screen frame
x0 and conditioned on user action ut at iteration t, an NC updates its runtime state and samples the next
frame xt+1. Formally, an NC defined by an initial runtime state h0, an update function Fθ, and a decoder Gθ
operates as follows, where Gθ parameterizes a distribution over next frames:
ht = Fθ(ht−1, xt, ut), xt+1 ∼ Gθ(ht). (2.1)
In this formulation, ht provides the persistent runtime memory, Fθ carries the state-update computation, and
(xt, ut, Gθ) define the I/O pathway from observations and actions to the next observable state.
Notation. We use ht for the NC latent runtime state and reserve z for VAE/video latents used in diffusion-style
video models (e.g., Section 3.2).
This update-and-render loop can be described using world-model terminology, where xt are observations and
ut provides conditioning. In that terminology, the input sequence {ut} is referred to as a conditioning stream.
This view supplies practical machinery for the current prototype, but an NC is not merely a predictor of
interface dynamics: it is a learned runtime mechanism in which the latent state ht carries executable context,
Fθ integrates new observations and inputs, and Gθ renders the next frame. Auxiliary heads can encode and
decode prompts, buffers, or action traces, shifting functionality that would traditionally live in OS queues,
device drivers, and UI toolkits into latent-state dynamics.
2.1 Related Work
Early neuromorphic designs (Mead and Ismail, 2012) explored neural computation as a physical substrate.
Differentiable memory and program-execution architectures, including fast weight programmers (Schmidhuber,
1992, 1993b,a), Neural Turing Machines (Graves et al., 2014), Differentiable Neural Computers (Graves et al.,
2016), and Neural Programmer-Interpreters (Reed and De Freitas, 2015), showed that neural controllers with
memory can execute structured procedures. Differentiable world models (Schmidhuber, 1990, 2015) learn
neural representations of environment dynamics, and inspire our update-and-render formulation. Latent video
and world models (Ha and Schmidhuber, 2018; Hafner et al., 2019b,a; Bruce et al., 2024) apply these ideas to
embodied control and interactive environments. Genie 3 (Bruce et al., 2024), in particular, frames such models
as agent-training substrates with improved physical consistency. More recently, high-capacity generators such
as Veo 3 (Google, 2025) and Sora 2 (OpenAI, 2025) emphasize open-ended, photorealistic simulation. In
parallel, systems such as NeuralOS (Rivard et al., 2025) and Imagine with Claude (Anthropic, 2025) bring
model-based conditioning to desktop and DOM-style interfaces. Building on this trajectory, we study two NC
instantiations for CLI and GUI with interface-specific conditioning, supported by a shared data engine and a
staged roadmap toward the CNC vision.
