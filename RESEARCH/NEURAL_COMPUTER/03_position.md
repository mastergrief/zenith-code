# Neural Computers — Position, Roadmap & Conclusion

← back to [NC.md](NC.md)

4 Position: Toward Completely Neural Computers
Section Overview In this section, we ask what current Neural Computer (NC) prototypes have already
shown, what still prevents them from becoming usable or general-purpose runtimes, and why neither current
world models (Ha and Schmidhuber, 2018; OpenAI, 2024; Polyak et al., 2024; Google DeepMind, 2025) nor AI
agents (Hong et al., 2023; Anthropic; Zhuge et al., 2026a) yet amount to this emerging machine form. We then
contrast NCs with conventional computers, clarifying that they are not a smarter layer on top of the existing
stack, and define their mature general-purpose form, namely Completely Neural Computers (CNCs). Finally,
we outline a roadmap toward CNCs, relate NCs to other system objects, and close with several remarks on
NCs.
4.1 From Neural Computers to Completely Neural Computers
Current Status of NCs Our CLI and GUI-based neural computers already show that early runtime primitives
can be learned with measurable interface fidelity. In terminal environments, OCR-based text fidelity is already
measurable (Table 4); in GUI settings, explicit visual supervision yields strong local cursor control (Table 9);
and in GUIWorld, aligned goal-directed data clearly outperforms much larger random exploration (Table 8).
Taken together, these results suggest that current NCs already support early runtime primitives, especially
I/O alignment and short-horizon control, while stable reuse and general-purpose execution remain out of
reach. This does not mean that current prototypes are already close to CNCs; it means that the outline of a
distinct machine form has begun to emerge at prototype scale.
However, the current video-based prototypes are only early NC instantiations: if NCs are to mature into
general-purpose runtimes, they must go well beyond basic I/O and short-term execution. At the formal level,
this ultimately requires Turing completeness (Gödel, 1931; Church, 1935; Turing et al., 1936; Siegelmann
and Sontag, 1992), universal programmability (Von Neumann, 1993; Wilkes, 1981), and behavior consistency
unless explicitly reprogrammed (Queloz, 2025). Before those conditions are met in full, progress is better read
through practical acceptance lenses: routine reuse, execution consistency, and explicit update governance.
These lenses matter because the immediate question is not whether CNCs have already been achieved, but
whether NCs are beginning to behave more like usable runtimes than isolated demonstrations. For example,
once an incident-response routine has been installed, the system should reuse it on later alerts rather than
rediscovering the procedure from scratch each time; and if its behavior changes, that change should be
attributable to an explicit update rather than ordinary execution. In practice, this reduces to three acceptance
lenses: install–reuse, execution consistency, and update governance, which together offer a more useful view of
current NC progress than the full CNC definition alone.
While certain sequential neural architectures are Turing complete (Siegelmann and Sontag, 1992; Pérez et al.,
2021) in principle, turning a trained instance into a reliably programmable runtime remains challenging.
Preliminary attempts, including Neural Virtual Machine (Katz et al., 2019) and NeuroLISP (Davis et al.,
2022), have been explored. Furthermore, ensuring stable behavior over long temporal horizons remains an
open problem in neural systems (Kirkpatrick et al., 2017; Calanzone et al., 2025). Section 4.2 provides a more
detailed discussion of these requirements. To the best of our knowledge, existing works on world models lack
an analysis of the computability class of the learned models. See Section 4.3 for more discussion between NCs
and other system objects, including world models and AI agents.
Fundamental differences between NCs and conventional computers We compare NCs and conventional computers. Here, conventional computers denote random-access machines with instruction set architecture (Hartmanis
and Simon, 1974; Anagnostopoulos et al., 1973) and layered OS/application stacks programmed via humandesigned high-level languages (Backus et al., 1957, 1963). NCs differ fundamentally from conventional
computers in their architectural and programming-language semantics.
At the architectural level, random-access machines instantiate local, compositional symbolic semantics (Newell
and Simon, 1976), yielding exactness and interpretability, but brittleness under noise and model mismatch (Brooks, 1991). Neural computers, by contrast, realize holistic, distributed numerical semantics,
trading precise local semantics for robustness and generalization (Ivakhnenko and Lapa, 1965; Ivakhnenko
et al., 1967; Ivakhnenko, 1968, 1971; Hinton et al., 1986; Bishop, 2006). Empirical evidence indicates that such
22
Table 13 Four system objects compared at a common systems level.
System object Organized around Source of truth Primary role
Conventional computer Explicit programs Explicit programs and explicit machine state
Reliably execute explicit programs
AI agent Tasks External environments, tools,
and workflow state
Accomplish tasks through an
existing software stack
World model Environment dynamics A learned model of state evolution
Predict and roll out how an
environment may evolve
Neural computer Runtime
Installed capabilities and runtime state inside the learned
system
Sustain execution, accumulate capability, and govern
updates within one learned
machine
holistic numerical representations are particularly well suited to domains characterized by high-dimensional
representations (Bengio et al., 2013), soft or statistical constraints (Smolensky, 1988), and globally coupled
structures (Silver et al., 2017; Vaswani et al., 2017), including perception, natural language, planning under
uncertainty, and approximate reasoning. Although conventional computers can, in principle, emulate NCs,
doing so often introduces unnecessary conceptual and engineering complexity when the target tasks are already
well matched to neural architectures.
At the programming-language level, NCs differ from conventional computers because their “language semantics”
are the meanings of user input sequences learned from data rather than explicitly designed by humans. For
example, LLMs can be viewed as programmable computers in which prompts act as programs (Reynolds and
McDonell, 2021). In this case, the programming language is a natural language, which no non-neural system
has historically been able to interpret robustly at scale (Jurafsky and Martin, 2026). More broadly, learned
programming-language semantics are not constrained by a human-specified syntax/semantics boundary and
can, therefore, encode task-relevant conventions implicitly (Wei et al., 2023).
Definition of Completely Neural Computers We use CNC to denote the mature form of an NC. Formally, a
Neural Computer instance is complete if it is (1) Turing complete, (2) universally programmable, (3) behaviorconsistent unless explicitly reprogrammed, and (4) realizes the architectural and programming-language
advantages of NCs relative to conventional computers. The following section unpacks these conditions in
operational terms.
Table 14 Operational reading of the four CNC requirements.
CNC requirement Plain reading What engineering evidence should look like
Turing complete
The system is not restricted to a narrow
family of fixed tasks, but can in principle
express general computation.
As effective memory and context grow, the
same NC should remain able to carry longer
and more structured procedures rather than
failing by a different shortcut each time.
Universally programmable
Inputs should not only trigger one-off behavior, but install routines or internal executors that remain callable later.
Capabilities can be installed, invoked, composed, and retained across tasks rather than
being relearned or outsourced each time.
Behavior-consistent
Ordinary use should not silently change the
machine; behavioral change should come
from explicit updates.
Same-version behavior is reproducible; execution and update traces can be inspected,
replayed, and rolled back; long-horizon drift
is measurable and governable.
Machine-native semantics
The system should not merely imitate conventional computers with neural components, but develop its own machine semantics and programming interfaces.
Composition, routing, continuous state,
and internal executors yield usable systemlevel advantages; prompts, demonstrations,
traces, and constraints begin to function as
programming interfaces rather than mere
logs.
23
4.2 A Roadmap Towards CNC
We frame the path toward CNCs through a set of formal requirements together with the practical challenges
that must be resolved before those requirements become engineerable.
Turing completeness A Neural Computer (NC) instance (a specific architecture with fixed learned weights)
defines a class of computational models in which each model corresponds to at least one memory state
instance. In the formal computability discussion below, “memory state” is used in the classical state-machine
sense; operationally, it corresponds to the NC runtime state introduced earlier. An NC instance is Turing
complete if, for any given Turing machine, there exists an initial memory state that allows the NC to emulate
that machine exactly. Notice that although Recurrent Neural Networks (RNNs), Neural Turing Machines
(NTM) (Graves et al., 2014), and Differentiable Neural Computers (DNC) (Graves et al., 2016) are Turing
complete in the asymptotic sense, a particular RNN, NTM, or DNC instance with finite precision cannot be
Turing complete due to their fixed finite memory size. For an NC instance to achieve universality, unbounded
effective memory is necessary. An NC instance has unbounded effective memory if there are infinitely many
possible memory state instances. Existing works approach such unboundedness by progressively growing
model parameters (Rusu et al., 2016) or context (Vaswani et al., 2017).
Universal programmability An NC is universally programmable if, for each given Turing machine, there
exists an input sequence such that the NC taking this input realizes a new memory state representing
the given machine. Most existing universal programmability results for neural networks are established
by constructing computational primitives and proving that their composition can simulate a universal
computational model (Reed and De Freitas, 2015). Likewise, we believe that universal programmability in
NCs can be achieved through compositional neural programs (Pierrot et al., 2019).
Behavior consistency A CNC must preserve its function unless explicitly reprogrammed. For each memory
state, there must be a non-empty set of inputs that executes the CNC without changing its pure function.
Operationally, this requires a separation between run and update: ordinary inputs should execute installed
capability without silently modifying it, while behavior-changing updates should occur explicitly through
a programming interface. This in turn motivates training and architectural mechanisms that disentangle
function use from function update, so that routines can be installed, executed, and composed without
accidental functional drift. We hypothesize that gating mechanisms, such as those in LSTM (Hochreiter and
Schmidhuber, 1997), are effective in achieving this conditional invariance. In practice, making this separation
reliable requires clear boundaries around what state persists across tasks, what counts as an explicit update,
and what execution evidence can be replayed, compared, or rolled back.
Run / update contract.
• Run: invoke installed capability without silently changing persistent behavior.
• Update: any behavior-changing modification should occur explicitly through a programming interface.
• Required boundaries: state (what persists), update (what counts as reprogramming), and evidence (what can be
replayed, compared, or rolled back).
Architectural semantics Since NC behavior is governed by real-valued parameters, learning can produce
input–output mappings that generalize across variations within the training distribution (Poggio et al., 2019).
For example, after observing many instances of how the visual state of a spreadsheet interface changes
when values are typed into cells, a model may learn the underlying transformation and correctly predict the
screen updates for previously unseen spreadsheets that follow the same interaction rules. Such in-distribution
generalization arises from the smooth function approximation properties of neural networks and their ability
to interpolate across previously observed patterns. Furthermore, learning can also produce novel input–output
mappings that are not explicitly represented in the training data, potentially introducing new computational
primitives (Ha et al., 2016). The combination of such newly formed primitives could enable qualitatively new
functions, yielding out-of-distribution functional generalization (Lake and Baroni, 2018).
Beyond emulating conventional computers, NCs can natively support functions whose semantics are ill-suited
to symbolic APIs (Marcus, 2018), including probabilistic inference over high-dimensional latent states (Kingma
24
and Welling, 2013), representation learning (Bengio et al., 2013), retrieval over dense memories (Graves et al.,
2016), and end-to-end differentiable pipelines that couple perception and control (Silver et al., 2017). These
functions are first-class at the architectural level and operate directly on distributed states. This enables
capabilities such as learned heuristics (Silver et al., 2017), uncertainty-aware decision-making (Clements et al.,
2019), and continual adaptation (Parisi et al., 2019).
Because the memory state of an NC/CNC is numerical, computer configuration and design emerge as
alternatives to application-level programming: the computer itself is configured by optimizing its internal state
to achieve desired computational behaviors under task-defined objectives. Depending on the differentiability
of the loss, methods such as Adam (Adam et al., 2014) and natural evolution strategies(Wierstra et al., 2014)
apply. In a CNC, the memory constitutes a continuous manifold, so realizing a target capability amounts to
synthesizing a machine configuration (a memory state) that minimizes a user-specified loss (e.g., “minimize
proof error”) via direct numerical updates to the computer’s state. This reframes system construction from
discrete code authoring to differentiable configuration of the computer itself (Innes et al., 2019), with progress
evaluated by solver convergence, stability, and reliability relative to combinatorial program search (e.g.,
LLM-based code generation (Hong et al., 2023)).
Programming-language semantics The learned programming-language semantics of NCs enable a shift from
rigid coding to learned specifications, in which user inputs themselves function as programs (Brown et al.,
2020; Wei et al., 2022). Rather than centering development on explicitly authored code, NCs expose a learned
language whose syntax and semantics are acquired from data (Radford et al., 2019; Bommasani, 2021), so
natural-language instructions, examples, and constraints serve as executable specifications (Ouyang et al.,
2022). Consequently, brief user inputs can replace long sequences of low-level actions. Development, therefore,
moves from code authoring to curating, specifying, and verifying inputs under a learned programming-language
semantics, aligning system behavior with human intent via in-context specification rather than forcing users to
conform to rigid, brittle interfaces (Austin et al., 2021). This does not imply that code disappears, but rather
that code becomes one installation medium among several, alongside prompts, demonstrations, trajectories,
and constraints.
Since NCs are programmed via users’ input sequences under learned programming-language semantics, the
training data for programming NCs, i.e., paired user I/O traces (Flener and Schmid, 2008), is far more
abundant and continuously generated than high-quality, human-written code. Every interaction with digital
systems produces structured streams of inputs, interface states, and effects that can be logged at scale (e.g.,
keystrokes, cursor trajectories, screen transitions), yielding orders-of-magnitude more supervision than curated
program corpora (Yao et al., 2022a). These I/O traces constitute executable specifications, revealing user
intentions and computer behavior (Cypher and Halbert, 1993). This enables end-to-end learning of interface
conventions, control policies, and task semantics without requiring explicit program text (Yao et al., 2022b).
This asymmetry in data availability favors NC training regimes that leverage ubiquitous interaction logs and,
by supporting broader task coverage, reduces dependence on brittle, sparsely available code datasets.
4.3 Relations to Other System Objects
Figure 9 summarizes the systems-level shift: conventional computers are used directly, AI agents mediate
existing computers, world models act as a parallel predictive layer, and NCs aim to make the learned runtime
itself the machine.
The comparisons below unpack this shift relative to conventional computers, world models, and AI agents.
Conventional Computers Conventional computers remain the reference system object for reliable execution,
explicit programmability, and mature governance. NCs differ not by adding a smarter application layer on top
of this substrate, but by shifting computation, memory, and I/O into a learned runtime state. In this sense,
NCs are best viewed as a different candidate machine form and computing substrate rather than as a direct
extension of the conventional software stack. This framing does not imply that conventional computers will
disappear soon, but rather that future systems may be built from a different underlying runtime substrate.
25
Figure 9 Changing human-machine relations across conventional computers, the agent era, and neural computers.
Conventional computers are used directly; in today’s agent stack, agents mediate existing computers while world
models serve as a parallel predictive layer; NCs aim to unify these split functions within one learned runtime. In this
sense, NCs are motivated not by replacing the current stack from the outside, but by internalizing its split functions
within one learning machine. A Completely Neural Computer (CNC) is the mature, general-purpose realization of this
machine form.
World Models World models learn environment dynamics by predicting action-conditioned transitions (Ha
and Schmidhuber, 2018). Such target environments range from the most ambitious, where all sensory inputs
form the real world (Schmidhuber, 1990), to much narrower scopes, such as a few control parameters of a robot
arm (Feng et al., 2023). They provide one technical perspective on current NC prototypes, since interactive
computers are an important class of action-conditioned environments, but they do not by themselves define
the NC abstraction. Many current approaches to modeling computational environments, such as the physical
world, also rely heavily on computer-generated data (Richter et al., 2016; Tobin et al., 2017; Dosovitskiy et al.,
2017; Garcia et al., 2023), potentially leading to models that share characteristics with neural computers.
AI Agents Another important comparison point is AI agents built on top of modern AI models and external
software substrates, including computer-use agents, coding and multi-agent systems (OpenAI, 2023; Anthropic;
Hong et al., 2023; Zhuge et al., 2024a; Sager et al., 2025), and recursive self-improvement loops (Zhuge
et al., 2026b). These systems place a learned agent between the user and an external execution substrate,
whether that substrate is a GUI, a codebase, or a broader software toolchain. This provides strong leverage
from existing computers and software stacks, but it also preserves a strict separation between the learned
model and the runtime that actually stores executable state, applies updates, and enforces system contracts.
Computer-use agents operate through low-bandwidth I/O; coding agents typically emit symbolic artifacts that
must be executed elsewhere; and RSI-style loops improve the agent by iterating over external tools, prompts,
or code rather than by turning the runtime itself into the computer. Such systems also increasingly rely on
automated evaluators, including agent-as-a-judge schemes, to rank outputs, validate task completion, and
close iterative improvement loops (Zhuge et al., 2024b). We hypothesize that a sufficiently capable NC can
internalize many of these agentic functions within one persistent neural runtime.
26
4.4 Additional Thoughts
The remarks in this section are intended as hypotheses and design directions motivated by the present results,
rather than as empirical conclusions established by the current prototypes.
ONE ONE (Schmidhuber, 2018) proposed a single neural substrate that incrementally absorbs and reuses
diverse learned skills. While ONE was not instantiated as a computer-like runtime with explicit I/O,
programmability, and update governance, a mature CNC can be viewed as a plausible systems-level realization
of this idea. In this sense, many specialized world-model-like components may ultimately appear not as
separate external systems, but as installable capabilities within one persistent neural runtime.
Video models as a pragmatic prototype substrate We build our prototypes on state-of-the-art video models
because they currently provide the simplest path to an end-to-end learned latent runtime state that jointly
models pixels, dynamics, and action-conditioned control. This choice is pragmatic rather than fundamental.
In our experiments, symbolic and algorithmic reasoning in terminal settings remains inconsistent for most
strong video models, and even simple arithmetic can fail (Table 5). Sora2 is a notable exception in our probe,
achieving 71% arithmetic accuracy, suggesting that some terminal symbolic reasoning is already possible in
modern video generators. At the same time, we do not claim that video models cannot reason more broadly:
recent work reports that video models can act as zero-shot learners and reasoners in naturalistic settings
(Wiedemer et al., 2025). We expect reasoning capabilities to improve quickly with continued progress in video
modeling, but our results suggest that CNC-level reliability will likely require additional architectural and
training ingredients beyond scaling today’s video generators.
A hypothesis: machine-native neural architectures We emphasize that the following is a conjecture rather
than a conclusion drawn from our experiments. Closing the reasoning gap may not require designing neural
networks that more closely mimic animal cognition or the human brain. Many influential architectures,
including convolutional networks (Fukushima, 1980) and linear/quadratic Transformers (Schmidhuber, 1992;
Vaswani et al., 2017), are highly engineered systems, but their core inductive biases remain strongly influenced
by biological perception and attention. These models primarily rely on continuous, distributed representations,
in which reasoning behavior emerges implicitly from large-scale training. We hypothesize that CNCs may
instead benefit from designs that are explicitly machine-native. Developing discrete operations, compositional
structures, and verifiable computation that are harmonious in neural systems may play an essential role in
designing such systems. This approach follows more closely the construction of conventional computers from
well-defined computational primitives and stands in contrast to relying on emergent reasoning in generic video
generation models.
Neural networks generation via NC interaction Neural network generation can be viewed as a form of
programming, i.e., the synthesis of a neural architecture and its corresponding weights. Because NCs’
architectural semantics are already neural and numerical, neural components are first-class, and generation
directly manipulates the memory state rather than translating it into symbolic code. Moreover, NCs can
be programmed through I/O interaction: sequences of inputs, observations, and outcomes act as executable
specifications that shape the internal state and routines of the system (Cypher and Halbert, 1993; Myers,
2002). This suggests a path in which users generate and refine neural modules within NCs through interactive
traces, treating interaction logs as programs that configure and compose neural computation.
Unified hardware requirements and data representation In NCs, tensors and tensor-to-tensor transformations act as primary computational primitives, replacing the heterogeneous mix of data structures and
subsystem-specific abstractions common in conventional computers. Traditional systems span many distinct
domains—scalars, pointers, linked structures, files, sockets, and processes—each with its own memory layout,
invariants, APIs, and failure modes, coordinated by operating systems through largely disjoint subsystems
(virtual memory, filesystems, networking, scheduling, and drivers) (Tanenbaum and Austin, 2013; Silberschatz
et al., 2019). Although this heterogeneity supports broad generality, it also fragments optimization and
tooling because compilers, profilers, and debuggers must reason across incompatible abstractions (Gregg,
2014). By contrast, a tensor-uniform pipeline concentrates representation and execution into a compact set
27
of composable primitives, such as linear algebra and elementwise operations, allowing tooling to target a
shared intermediate representation (Paszke et al., 2019). As a result, optimizations such as operator fusion,
memory planning, and computational-graph rewriting can be applied system-wide (Vasilache et al., 2018);
profiling can focus on throughput and memory bandwidth; and accelerators such as GPUs can be targeted
through common tensor runtimes (Sze et al., 2017). This shared numerical representation also naturally
supports multimodal computation: vision (pixel tensors), language (sequence embeddings), audio (waveforms
or spectrograms), control (state-action tensors), and planning (latent trajectory tensors) all reside in one
representational space and can be jointly reasoned over and optimized in a single graph (Ramachandram and
Taylor, 2017), without repeated type bridging or subsystem translation—steps that are substantially harder
in traditional heterogeneous stacks.
5 Conclusion
Neural computers point toward a machine form in which a single latent runtime state acts as the computer
itself, driving pixels, text, and actions while subsuming what operating systems and interfaces handle today.
In this paper, the main result is that NCs have begun to exhibit early runtime primitives—most notably I/O
alignment and short-horizon control—while stable reuse, symbolic reliability, and runtime governance remain
unresolved. Our CNC capability map remains useful as a longer-horizon view, spanning efficiency, computation
& reasoning, memory & storage, I/O & control, tool bridges, condition-driven generalization, programmability,
and artifact generation. The map is staged and dependency-informed, but the more immediate gap is still
the gap from prototype behavior to usable runtime behavior. Progress toward CNCs will therefore depend
not only on stronger models, but also on whether reuse, consistency, and governance become sustained and
testable. If these gaps continue to close, neural computers will look less like isolated demonstrations and more
