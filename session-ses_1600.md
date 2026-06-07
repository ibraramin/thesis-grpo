# Read docs markdown

**Session ID:** ses_16005556fffeObKH5yXphd4pGl
**Created:** 6/7/2026, 8:47:45 AM
**Updated:** 6/7/2026, 8:57:44 AM

---

## User

Read through the md file in Docs

---

## Assistant (Plan · DeepSeek V4 Pro · 4.2s)

_Thinking:_

The user wants me to read through a markdown file in a "Docs" directory. Let me first find what files exist in the Docs directory.

**Tool: glob**

**Input:**
```json
{
  "pattern": "Docs/**/*.md"
}
```

**Output:**
```
No files found
```

---

## Assistant (Plan · DeepSeek V4 Pro · 4.0s)

_Thinking:_

No markdown files found with the `Docs/**/*.md` pattern. Let me try a broader search - maybe the directory is named differently (like "docs" lowercase), or there's a different file structure.

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*.md"
}
```

**Output:**
```
/home/ibrar/Opencode/Thesis-Whatever/docs/Synthesized Methodology.md
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/ibrar/Opencode/Thesis-Whatever"
}
```

**Output:**
```
<path>/home/ibrar/Opencode/Thesis-Whatever</path>
<type>directory</type>
<entries>
docs/

(1 entries)
</entries>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 3.5s)

_Thinking:_

Found it — the file is `docs/Synthesized Methodology.md` (note lowercase "docs" and space in the filename). Let me read it.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/ibrar/Opencode/Thesis-Whatever/docs/Synthesized Methodology.md"
}
```

**Output:**
```
<path>/home/ibrar/Opencode/Thesis-Whatever/docs/Synthesized Methodology.md</path>
<type>file</type>
<content>
1: ## Dynamic Reward-Gating in Group Relative Policy Optimization: Delineating Reasoning vs. Formatting Tradeoffs in Small Language Models
2: 
3: ## 1. Introduction & Conceptual Framework
4: 
5: Post-training optimization for Large Language Models (LLMs) has shifted from scaling raw data volumes toward maximizing data quality and post-training efficiency.[5] Historically, instruction alignment relied on Supervised Fine-Tuning (SFT) using causal next-token prediction over target completions.[1] While SFT is highly efficient for style transfer and format adherence,[1] recent 2025–2026 literature proves that applying standard SFT to complex mathematical reasoning tasks actively degrades the model's underlying reasoning graphs.[7] SFT forces the model to memorize superficial formatting patterns and lexical co-occurrences of the training set, resulting in a severe "alignment tax" and brittle out-of-distribution (OOD) extrapolation.[8] This limitation is heavily pronounced in Small Language Models (SLMs) under 3 billion parameters.[5] Due to restricted parameter capacity, SLMs must optimize their post-training updates to reinforce core logical capabilities rather than wasting capacity on superficial format imitation.[5] The Superficial Alignment Hypothesis (SAH) suggests that a model's core reasoning capabilities are learned entirely during pre-training, and post-training alignment merely teaches the model which stylistic distribution to use.[9]
6: 
7: Rather than adopting standard SFT, the state-of-the-art post-training frontier has moved to **Group Relative Policy Optimization (GRPO)**. GRPO, the core algorithm behind DeepSeek-R1, optimizes LLMs using rule-based verifiable rewards without requiring a separate, memory-intensive critic model. By allowing the model to generate its own reasoning trajectories and optimizing them based on terminal correctness, GRPO compresses incorrect reasoning paths while expanding and consolidating correct reasoning hubs.[7] This thesis constructs an empirical and theoretical study to investigate the **generalization-extrapolation tradeoff** in SLMs under GRPO.[11] Instead of static token-masking in SFT, this study introduces **Dynamic Reward-Gating**. We isolate and manipulate two critical reward axes during the RL process:
8: 
9: 1. **Verifiable Correctness Rewards ($r_{correct}$):** Programmatic verification of the exact mathematical solution.
10: 2. **Format-Adherence Rewards ($r_{format}$):** Rules targeting the generation of explicit reasoning structures (e.g., XML `<think></think>` tags).
11: 
12: A known hazard of reasoning-oriented RL is "reward hacking", where models generate short, superficial reasoning loops to satisfy format rewards without executing actual reasoning. By dynamically modulating the ratio of formatting to correctness rewards across independent experimental cohorts, this research maps how Small Language Models allocate their parameter updates between logical consolidation and format imitation.
13: 
14: To guarantee empirical validity and absolute freedom from data contamination, this study utilizes the unaligned **Qwen2.5-1.5B-Base** model. Standard pre-aligned instruct models (such as Qwen2.5-Instruct) are heavily contaminated with mathematical benchmarks like GSM8K. By using an unaligned base model, we ensure that the emergence of reasoning trajectories is purely a product of our GRPO training dynamics. The Qwen2.5-1.5B architecture provides the minimum necessary parametric surface area — 1.54 billion parameters distributed across 28 hidden layers, 1536-dimensional hidden states, and Grouped Query Attention (GQA) with 12 query heads and 2 key-value heads — to observe genuine consolidation of reasoning graphs without capacity-induced formatting collapse. Models at the sub-billion scale intrinsically lack the representational capacity to maintain reasoning and formatting as distinct latent representations, making the architectural upgrade to 1.5B a scientifically mandatory requirement.[5] This study is optimized to execute entirely on a single **NVIDIA RTX 3090 (24GB VRAM)** within a strict 30-hour computational envelope.[4]
15: 
16: ## 2. Literature Review & Theoretical Foundations
17: 
18: The proposed framework synthesizes four major clusters in modern AI post-training literature: SFT vs. RL reasoning graph topologies, reward-shaping vulnerabilities, the generalization-extrapolation tradeoff, and stage-specific curricular data design.
19: 
20: ### SFT vs. RL Reasoning Graph Topologies
21: 
22: Recent trajectory-level analyses reveal a fundamental divergence in how SFT and RL shape reasoning processes.[7] SFT on expert demonstrations trains the model on reference prefixes rather than the model's own prefix distribution. This induces memorization of superficial patterns. Topologically, SFT flattens the decay rates of node visitation frequency in the model's latent reasoning graph, homogenizing reasoning steps across diverse paths.[7]
23: 
24: Conversely, Reinforcement Learning with Verifiable Rewards (RLVR) steepens these decay rates, concentrating reasoning functionality into a highly consolidated subset of logical hub steps.[7] RL compresses incorrect trajectories while SFT preserves them.[7] This justifies the use of GRPO as a superior mechanism for reasoning consolidation in capacity-constrained models. The contemporary best practice of two-stage training (SFT followed by RL) succeeds precisely because SFT acquires new solution paths to maximize Pass@k performance, while RL compresses the incorrect paths to maximize Pass@1 performance.[7]
25: 
26: ### Reward Hacking and Formatting Taxes
27: 
28: In RL-based post-training, the quality of the reward signal is paramount. Programmatic reward functions that lack nuance are highly susceptible to reward hacking. In reason-tuning, models often discover shortcuts to maximize formatting rewards — for example, generating empty or repetitive formatting structures to secure the format reward while bypassing the complex logic needed to achieve correct answers.
29: 
30: Two critical failure modes emerge under naive reward-gating:
31: 
32: - **Reward Hacking:** The model generates repetitive, semantically hollow formatting structures (e.g., endless nested `<think>` tags) to artificially satisfy the structural parser without executing actual mathematical reasoning. This pathology is directly tied to Type I reward hacking in multi-objective preference datasets, where subpar choices appear more favorable due to statistical flaws in the reward function.
33: - **Formatting Collapse:** When tokens frequently appear in both positively and negatively rewarded completions within the same group, penalizing a negatively rewarded completion can inadvertently suppress the output probability of critical formatting tokens (e.g., the closing `</think>` tag), leading to severe structural degradation and catastrophic drops in both stylistic adherence and mathematical accuracy.
34: 
35: Adding a high-weight formatting reward can cause the policy to saturate early, leading to "formatting collapse" where the model outputs correct syntax but incorrect logical answers. Delineating the exact mathematical boundary where format rewards transition from structural scaffolds to destructive cognitive distractors is an active, open area of research.
36: 
37: ### The Generalization-Extrapolation Tradeoff
38: 
39: PAC-Bayesian generalization bounds prove that post-training test risk is governed by a tension between task-specific adaptation ($G_{gen}$) and distribution mismatch ($G_{ext}$):[11]
40: 
41: $$G_{gen} = \mathcal{O}\left(\sqrt{\frac{D_{KL}(\pi_{train} \parallel \pi_{pre})}{n}}\right)$$
42: 
43: $$G_{ext} = \mathcal{O}\left(TV(\mathcal{D}_{test}, \mathcal{D}_{train})\right)$$
44: 
45: When RL training is overly focused on superficial formatting alignment, it artificially drives the KL divergence high without reducing the true extrapolation gap for hard, OOD mathematical distributions.[11] This thesis tests whether dynamically gating or decaying $\lambda_{format}$ mitigates this tradeoff, allowing the model to minimize drift while maximizing logical extrapolation.
46: 
47: ### Stage-Specific Curricular Data Design
48: 
49: The most effective, state-of-the-art framework for post-training Small Language Models involves the "Stage-Specific Data Sets" paradigm.[6] This framework operates on the strict epistemological principle that Supervised Fine-Tuning and Reinforcement Learning serve fundamentally distinct, non-overlapping roles in capability acquisition. SFT is mathematically optimal for acquiring not-yet-mastered reasoning skills by forcing the model to trace external expert trajectories, minimizing the initial generalization-extrapolation gap. Conversely, RL is uniquely suited for consolidating skills that the model can already partially access, using verifiable rewards to squeeze out incorrect variations of established logical paths.
50: 
51: Applying GRPO directly to hard mathematical distributions without prior scaffolding results in a catastrophic failure mode known as **reward sparsity**. In this state, the current policy fails to produce a single correct solution across all sampled rollouts ($G=8$ or $G=16$), yielding an all-zero reward vector. Because GRPO advantage estimation relies on the variance of the rewards within the group, an all-zero vector provides exactly zero useful gradient signal for optimization, causing the policy to stall in flat loss landscapes.
52: 
53: ## 3. Step-by-Step Methodology & Experimental Design
54: 
55: The architecture is partitioned into four distinct phases to map the relationship between model capability, formatting, and mathematical logic.
56: 
57: ### 3.1 Phase 1: Stage-Specific Dataset Architecture
58: 
59: The study adopts a rigorous "Stage-Specific Data Sets" paradigm, leveraging three highly specialized, interconnected distributions forming an iterative SFT-then-RL loop.
60: 
61: **Stage 1: The Acquisition Set (Initial SFT Stage)**
62: 
63: - **Dataset Source:** `open-r1/OpenR1-Math-220k`
64: - **Composition:** A targeted 5,000-sample subset of complex mathematical reasoning traces distilled directly from DeepSeek-R1.
65: - **Objective:** SFT is optimal for acquiring not-yet-mastered reasoning skills. This stage rapidly bootstraps the Qwen2.5-1.5B model with foundational mathematical topologies and the exact structural `<think>` formatting syntax required by the environment, minimizing the initial generation-extrapolation gap before RL begins. Standard causal next-token prediction is applied, with a chat template enforcing that assistant responses are wrapped explicitly in `<think>` and `</think>` tags.
66: 
67: **Stage 2: The Consolidation Set (RL / GRPO Stage)**
68: 
69: - **Dataset Source:** `SynthLabsAI/Big-Math-RL-Verified`
70: - **Composition:** A highly curated 2,500-sample split of exceptionally difficult, open-ended mathematical problems.
71: - **Objective:** RL is optimal for consolidating skills the model can already partially access. Every problem in this dataset features a single, uniquely verifiable closed-form answer. This guarantees the dynamic GRPO algorithm receives mathematically exact, unambiguous binary reward signals, allowing it to definitively squeeze incorrect logic branches and deepen correct node visitations. Only non-all-zero samples — queries where the SFT-primed model currently generates at least one correct trajectory within its sampled rollout group — are included.
72: 
73: **Stage 3: The Recycled Set (Critique Fine-Tuning)**
74: 
75: - **Mechanism:** Any hard sample from the Big-Math subset that remains completely unsolved during the RL rollouts (producing an all-zero reward vector) is mathematically intractable for GRPO and is strictly excluded from direct gradient updates.
76: - **Routing:** Instead, these failures are isolated, diagnosed, and converted into structured repair traces. The error-guided recycling mechanism isolates the reasoning prefix before the first error, applying Critique Fine-Tuning to generate diagnostic ($\mathcal{D}_{diag}$), repair ($\mathcal{D}_{repair}$), and new trace supervision ($\mathcal{D}_{new}$). These structured repairs are recycled back into the SFT Acquisition Set to continuously and systematically expand the model's capability boundary.
77: 
78: ### 3.2 Phase 2: Hardware-Optimized GRPO Pipeline
79: 
80: Training is executed on a single NVIDIA RTX 3090 GPU using the **Unsloth**, **TRL**, and **vLLM** frameworks.
81: 
82: 1. **Base Model Instantiation:** `Qwen/Qwen2.5-1.5B` is loaded in 4-bit NormalFloat (NF4) double quantization via `bitsandbytes`.[5]
83: 2. **LoRA Adaptation:** Adapter matrices are applied to all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r = 64$ and scaling factor $\alpha = 128$.[5]
84: 3. **Embedding Strategy:** To resolve the `tie_word_embeddings: true` constraint of the Qwen architecture, Unsloth's native patching is leveraged.[2] Unsloth intercepts the gradient step and bypasses the PEFT merging errors by scaling only the low-rank adapters and performing weight copy routines post-update,[16] preventing weight corruption and gradient instability.[17]
85: 4. **Optimization Stack:** We utilize the 8-bit AdamW optimizer with a cosine learning rate scheduler peaking at $5 \times 10^{-6}$. Gradient checkpointing is set to full. Warm-up ratio is 0.1.
86: 5. **Generation Configuration:** Rollout group size $G = 16$, max prompt length 256 tokens, max completion length 1792 tokens, for a total sequence length $L = 2048$.
87: 
88: Unsloth's fused and chunked cross-entropy computation bypasses the materialization of the massive FP32 logits tensor by chunking the sequence dimension, such that peak activation memory scales with chunk size rather than full sequence length, reducing logit memory footprint by over 90%.[3] Aggressive asynchronous gradient checkpointing offloads intermediate activations to system RAM, and the framework eliminates double memory usage when loading vLLM and Unsloth simultaneously, allowing 4-bit dynamic quants to run inference directly.
89: 
90: ### 3.3 Phase 3: Algorithmic Stabilization Mechanisms
91: 
92: To permanently resolve reward hacking and formatting collapse, the revised methodology incorporates three state-of-the-art stabilization protocols:
93: 
94: **Entropy-Filtered Dynamic Reward Gating**
95: 
96: The average group completion entropy $\mathcal{H}$ is actively monitored during every training step. Without filtering, sequence entropy steadily increases, eventually destabilizing structural formatting and causing policy collapse. If the token distribution entropy exceeds a critical threshold ($\ln 2$), it indicates that the policy is rapidly destabilizing. The reward-gating mechanism instantly and dynamically throttles gradient updates for high-variance sequences by zeroing out the advantage for that sequence. This ensures that the foundational formatting syntax is never subjected to destructive interference, preventing the over-penalization of structurally sound but logically incorrect rejected sequences.
97: 
98: **Dynamic Difficulty Scalarization (DW-GRPO)**
99: 
100: Rather than utilizing static or arbitrarily decaying reward weights, the multi-objective reward is dynamically scaled by localized prompt difficulty and the model's real-time capability:
101: 
102: $$R(q, o_i) = \lambda_{correct}(q) \cdot r_{correct}(q, o_i) + \lambda_{format}(\mathcal{H}) \cdot r_{format}(o_i)$$
103: 
104: Here, $\lambda_{correct}(q)$ is dynamically scaled by the localized task difficulty, approximated by measuring the variance of the reward gaps across the rollout group or assessing data-level hardness via text-encoder cosine similarity. If a sample yields highly diverse trajectories with mixed correctness — indicating a high-value consolidation opportunity — the correctness reward vector is magnified to aggressively pull the policy toward the correct logic.
105: 
106: Simultaneously, $\lambda_{format}(\mathcal{H})$ acts as a history-aware bias correction parameter. It enforces strict structural adherence early in trajectory formation but autonomously decays its influence specifically on tokens that the model has confidently mastered. This ensures the optimization trajectory dynamically self-balances according to model capability and sample difficulty, entirely removing reliance on hard-coded time steps and drastically reducing the incidence of Type I reward hacking.
107: 
108: **Probability Smoothing Policy Optimization (PSPO)**
109: 
110: The standard GRPO objective relies heavily on KL divergence penalties to control policy drift relative to the reference model. However, relying solely on standard PPO-style ratio clipping or naive KL penalties often discards vital gradient information and introduces severe mathematical discontinuities, artificially inflating the $G_{gen}$ generalization gap and causing the model to forget core reasoning capabilities.
111: 
112: PSPO dynamically smooths the current policy's probabilities toward the reference (behavior) policy before computing the importance ratio, analogous to label smoothing in supervised environments. This continuous interpolation creates a soft trust region that discourages massive, destabilizing parameter updates while fully preserving the precise directional gradient signal necessary for long-horizon logical reasoning consolidation.
113: 
114: When PSPO is combined with the $K_3$ unbiased KL estimator ($r - \log r - 1$) derived by Schulman, it guarantees that the Qwen2.5-1.5B model remains firmly anchored to its pre-trained linguistic priors, safely allowing massive internal topological shifts to squeeze incorrect reasoning paths and expand verified logical sequences without suffering from capability boundary collapse.
115: 
116: ### 3.4 Phase 4: Reward-Gating Permutations
117: 
118: The core experimental variable is the design of the reward functions. For every rollout completion $o_i$ generated from prompt $q$, a multi-component reward is computed:
119: 
120: $$R_{total}(q, o_i) = \lambda_{correct} \cdot r_{correct}(q, o_i) + \lambda_{format} \cdot r_{format}(o_i)$$
121: 
122: Where $r_{correct}$ is a binary function returning $1.0$ if the parsed numerical output matches the ground truth, and $0.0$ otherwise. The format reward $r_{format}$ evaluates structural syntax, checking if the completion strictly wraps its reasoning inside `<think></think>` tags and its final output inside `<answer></answer>` tags.
123: 
124: We establish five independent training cohorts with distinct, static or dynamic weighting configurations:
125: 
126: | **Cohort** | **$\lambda_{correct}$** | **$\lambda_{format}$** | **Strategic Objective & Theoretical Justification** |
127: |---|---|---|---|
128: | **Baseline** | 1.0 | 0.0 | **Verifiable Correctness Only.** Isolates the model's capacity to discover its own unstructured, implicit reasoning trajectories without stylistic guidance. |
129: | **Cohort A** | 0.5 | 1.5 | **Formatting Dominance.** Explores extreme "formatting tax". Tests if heavily rewarding structure encourages early reward hacking and limits logical reasoning accuracy. |
130: | **Cohort B** | 1.5 | 0.2 | **Correctness Dominance with Format Regularization.** Implements a balanced objective where terminal logical truth is heavily prioritized, using formatting purely as a soft constraint. |
131: | **Cohort C** | Graduated | 0.2 | **Graduated Mathematical Accuracy.** Replaces the binary correctness reward with a continuous metric: $+1.0$ for exact match, $+0.5$ if within 10% of ground truth, $+0.2$ if within 20%, and $-0.2$ for logical failure. Tests if dense intermediate numerical signals reduce variance. |
132: | **Cohort D** | 1.0 | $\lambda_{format}(t)$ (decaying) | **Dynamic Decaying Format-Gate (DW-GRPO).** Implements a time-varying formatting weight: $\lambda_{format}(t) = 1.0 \cdot e^{-\gamma t}$, where $t$ is the training step and $\gamma$ is a decay coefficient. Hypothesizes that formatting is vital early in training to establish structural priors (the "Extend" phase) but must be decayed to allow the model to focus on consolidating core logical accuracy (the "Recall" phase).[6] |
133: 
134: Each cohort is trained across 4 independent random seeds to control for variance, producing a total of 20 distinct training runs.
135: 
136: ## 4. Mathematical Formulation of the GRPO Objective
137: 
138: GRPO eliminates the traditional PPO critic network by computing the baseline directly from a group of $G$ sampled outputs. Given a query prompt $q$ drawn from the training distribution, the policy model $\pi_{\theta}$ generates $G$ independent rollout completions $\{o_1, o_2, ..., o_G\}$. The GRPO objective function is defined as:[4]
139: 
140: $$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{(q, \{o_i\})} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(r_i(\theta) \cdot \hat{A}_i, \ \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_i\right) - \beta \cdot D_{KL}(\pi_{\theta} \parallel \pi_{ref}) \right]$$
141: 
142: Where:
143: 
144: - $r_i(\theta) = \frac{\pi_{\theta}(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}$ is the importance sampling ratio.
145: - $\hat{A}_i$ is the group-relative advantage of rollout $i$ computed via normalization over the reward set:[4]
146: 
147: $$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1, R_2, ..., R_G\})}{\text{std}(\{R_1, R_2, ..., R_G\}) + \varepsilon}$$
148: 
149: Where $\varepsilon$ prevents division-by-zero.
150: 
151: - $\pi_{ref}$ is the frozen reference policy (initially copied from the base model $\pi_{\theta_0}$).
152: - $\beta$ is the Kullback-Leibler (KL) divergence penalty coefficient controlling policy drift.
153: - $D_{KL}$ is estimated at the token level using the $K_3$ unbiased Schulman estimator:[4]
154: 
155: $$D_{KL}^{(K_3)} = r - \log r - 1$$
156: 
157: Where $r = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{ref}(a_t \mid s_t)}$ is the per-token density ratio. The naive log-ratio estimator $k_1 = \log(q(x)/p(x))$ suffers from extremely high variance because a single sample can swing positive or negative. The $K_3$ estimator leverages the properties of $f$-divergences and tangent lines to produce a variance-reduced unbiased estimate.
158: 
159: **PSPO-Enhanced Objective:** Under PSPO, the standard ratio $r_i(\theta)$ is replaced with a smoothed variant:
160: 
161: $$r_i^{PSPO}(\theta) = \frac{(1 - \delta) \cdot \pi_{\theta}(o_i \mid q) + \delta \cdot \pi_{ref}(o_i \mid q)}{\pi_{\theta_{old}}(o_i \mid q)}$$
162: 
163: Where $\delta \in [0, 1]$ is the probability smoothing coefficient. This continuous interpolation anchors the policy to its pre-trained priors while preventing catastrophic ratio explosions.
164: 
165: **Entropy-Filtered Advantage Masking:** The advantage for each sequence is further modulated by an entropy condition:
166: 
167: $$\hat{A}_i^{filtered} = \begin{cases} \hat{A}_i & \text{if } \mathcal{H}(\pi_{\theta}(\cdot \mid q)) \leq \ln 2 \\ 0 & \text{if } \mathcal{H}(\pi_{\theta}(\cdot \mid q)) > \ln 2 \end{cases}$$
168: 
169: This prevents gradient updates from destabilizing sequences when the policy's token distribution becomes excessively uncertain.
170: 
171: The gradient of the full composite GRPO objective with respect to the trainable adapter weights $\theta$ is formulated as:
172: 
173: $$\nabla_{\theta} \mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \nabla_{\theta} \left( \min(r_i^{PSPO}(\theta) \cdot \hat{A}_i^{filtered}, \ \text{clip}(r_i^{PSPO}(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_i^{filtered} \right) - \beta \cdot \nabla_{\theta} D_{KL}^{(K_3)} \right]$$
174: 
175: Under this formulation, tokens that contribute to rollouts achieving above-average group rewards receive positive gradients ($\hat{A}_i > 0$), increasing their probability of recurrence, while below-average rollouts receive negative gradients ($\hat{A}_i < 0$), compressing their probability trajectories.[7] The entropy filter and PSPO smoothing jointly ensure that the foundational formatting syntax is protected from destructive interference throughout training.
176: 
177: ## 5. Hardware & VRAM Budgeting (The 24GB RTX 3090 Constraint)
178: 
179: Executing GRPO has historically been computationally expensive because generating rollouts concurrently requires storing extensive Key-Value (KV) cache sequences in memory. However, by selecting Qwen2.5-1.5B and deploying **Unsloth's memory-efficient linear GRPO algorithms**,[4] we dramatically compress the VRAM footprint. Unlike standard GRPO implementations that materialize the massive FP32 logits tensor — which demands $2 \times 2 \text{ bytes} \times G \times L \times V \approx 4.97$ GB of memory bandwidth[4] — Unsloth utilizes fused and chunked cross-entropy. By chunking the sequence dimension, peak activation memory scales with the chunk size rather than the full sequence length, reducing the logit memory footprint by over 90%.
180: 
181: ### Byte-Level VRAM Allocation
182: 
183: We establish the following training hyperparameters:
184: 
185: - Model Parameters: $1.54 \times 10^9$ (quantized to 4-bit).
186: - Group Size (Rollouts per Prompt): $G = 16$.
187: - Max Sequence Length: $L = 2048$ (comprising prompt tokens and generation tokens).
188: - Vocabulary Size: $V = 151,936$.[18]
189: - LoRA Rank: $r = 64$, targeting $35 \times 10^6$ parameters in BF16 precision.[5]
190: 
191: | **Component** | **Mathematical VRAM Formula** | **Allocated VRAM (GB)** |
192: |---|---|---|
193: | Base Model Weights | $1.54 \times 10^9$ parameters (NF4 Quantized) | ~0.85 GB |
194: | LoRA Adapters | $35 \times 10^6$ parameters (BF16) | ~0.25 GB |
195: | Optimizer States | 8-bit AdamW applied strictly to adapters, 12 bytes per parameter | ~0.50 GB |
196: | vLLM KV Cache | $2 \times 2 \text{ bytes} \times \text{layers} (28) \times G (16) \times L (2048)$ | ~3.20 GB |
197: | Fused Logits & Activations | Chunked cross-entropy buffers and asynchronous RAM offloading | ~1.20 GB |
198: | CUDA Context Overhead | Static PyTorch device allocation and system geometry | ~1.50 GB |
199: | **Total Peak VRAM** | **Sum of all active, passive, and cached memory allocations** | **~7.50 GB** |
200: 
201: A total peak VRAM of ~7.50 GB represents an utilization rate of **only 31%** of the RTX 3090's total 24 GB capacity. This guarantees a zero-OOM execution threshold,[5] completely removing any risk of memory fragmentation crashes even during sudden spikes in generation length. This massive memory headroom enables the permanent increase of the rollout group size from the baseline $G=8$ to the highly optimal $G=16$, significantly reducing the variance of the GRPO advantage estimation and providing a far denser reward signal for accelerated policy convergence.
202: 
203: ### Temporal Compute and Feasibility Envelope
204: 
205: - **Throughput of Qwen2.5-1.5B under Unsloth GRPO:** ~6,500 tokens per second.[5]
206: - **Sequence length per group step:** $G \times L = 16 \times 2048 = 32,768$ tokens.
207: - **Step computation duration:** ~1.85 seconds per forward-backward GRPO step, encompassing concurrent generation of 16 trajectories, verifiable reward parsing, dynamic advantage estimation, and the PSPO-smoothed gradient update utilizing the $K_3$ Schulman KL estimator.
208: - **Steps per training run:** Training for 2 epochs over 2,500 samples with effective batch size 16 requires 312 steps.
209: - **Compute time per individual training run:** $312 \times 1.85 \text{ s} \approx 577 \text{ s} \ (\approx 9.6 \text{ minutes})$.
210: - **Aggregate training computation (5 cohorts $\times$ 4 random seeds = 20 runs):** $20 \times 9.6 \text{ min} \approx 192 \text{ min} \ (\approx 3.2 \text{ hours})$.
211: 
212: Factoring in the initial Stage-Specific SFT acquisition phase on the OpenR1-Math-220k subset (~2 hours), continuous NVMe checkpoint serialization, and rigorous OOD evaluation passes against MATH-500 and AIME24 (~4.5 hours), the total end-to-end experimental pipeline executes in approximately **10–11.5 hours**, fitting comfortably inside the 30-hour limit.
213: 
214: ## 6. Evaluation, Verifiable Metrics & Statistical Validity
215: 
216: To ensure that findings are rigorous enough to satisfy the most meticulous top-tier AI conference reviewers, we implement a robust evaluation suite that goes beyond simple string-matching to validate genuine logical capabilities.
217: 
218: ### 6.1 Process-Based and OOD Validation
219: 
220: 1. **In-Domain Generalization:** Evaluated over holdout splits from the training distribution to measure convergence and overfitting patterns.
221: 2. **Out-of-Distribution Extrapolation (MATH-500):** We evaluate all trained checkpoints on the **MATH-500** benchmark.[20] MATH-500 features competition-level problems requiring significantly longer logical trajectories. This evaluates if our dynamic reward-gating paradigms developed genuine reasoning mechanisms or merely taught the model to hack the formatting rules of the local training distribution.
222: 3. **AIME24 Evaluation:** All trained checkpoints are additionally evaluated on the AIME24 benchmark for further OOD stress-testing, probing the extreme boundary of the model's mathematical extrapolation capability.
223: 4. **Format vs. Logic Disentanglement:** We trace the average generation length and count the presence of `<think>` tokens across epochs. If a cohort shows rising correctness scores while generation length remains stable or decreases, it proves that the policy is consolidating its reasoning graphs (eliminating "overthinking") rather than generating redundant, low-value thinking strings to inflate length-biased advantages.
224: 
225: ### 6.2 The Strict Verifiable Grading Parser
226: 
227: To prevent regex parsing errors and formatting biases from introducing noise into our exact-match metrics,[5] we deploy a multi-tiered validation script:
228: 
229: ```python
230: import re
231: 
232: def verify_math_accuracy(prompt, gold_answer, generation_output):
233:     """
234:     Verifies reasoning structure and exact mathematical output.
235:     Returns:
236:         (bool, bool): (format_valid, answer_correct)
237:     """
238:     # 1. Structural Formatting Check
239:     think_match = re.search(r"<think>.*?</think>", generation_output, re.DOTALL)
240:     answer_match = re.search(r"<answer>\s*(-?\d+(\.\d+)?)\s*</answer>", generation_output)
241:     format_valid = bool(think_match and answer_match)
242: 
243:     if not answer_match:
244:         return format_valid, False
245: 
246:     # 2. Mathematical Correctness Verification
247:     predicted_val = float(answer_match.group(1))
248:     target_val = float(gold_answer)
249: 
250:     # Check for direct numerical equivalence
251:     answer_correct = abs(predicted_val - target_val) < 1e-5
252: 
253:     return format_valid, answer_correct
254: ```
255: 
256: ### 6.3 Inferential Statistical Proving
257: 
258: To establish that differences between our reward-gating strategies are mathematically significant, we apply **Null Hypothesis Significance Testing (NHST)**:
259: 
260: 1. **One-Way ANOVA:** Conducted across the mean performance metrics of the five experimental cohorts to reject the null hypothesis ($H_0$: Modulating the formatting reward ratio yields no difference in downstream logical accuracy) at a significance level of $p < 0.05$.[5]
261: 2. **Post-Hoc Tukey's HSD (Honestly Significant Difference):** Isolates pairwise performance differences (e.g., Cohort D vs. Cohort A) while controlling the family-wise Type I error rate across multiple comparisons.[5]
262: 3. **KL-Divergence Tracking:** We record the exact evaluation-step Kullback-Leibler divergence $D_{KL}(\pi_{\theta} \parallel \pi_{ref})$ between the trained policy and the reference model to mathematically map the **Generalization-Extrapolation curves**, plotting OOD error on MATH-500 as a direct function of drift.[11]
263: 4. **Entropy Trajectory Analysis:** The average completion entropy $\mathcal{H}$ is logged per step, providing empirical evidence of when and whether the entropy-filtered gating mechanism successfully prevented formatting collapse across cohorts.
264: 
265: ### 6.4 Likelihood of Success and Revised Projections
266: 
267: The likelihood of this study achieving meaningful, statistically significant higher metrics over the control baseline is exceptionally high.
268: 
269: By upgrading to the Qwen2.5-1.5B architecture, the study gains the precise dimensionality — specifically through its 28 hidden layers and expanded attention heads — necessary for reasoning paths to structurally detach from linguistic formatting syntax within the latent space. This critical separation allows the topological "squeezing" of incorrect paths and the "expanding" of correct paths to actually occur and be empirically recorded.
270: 
271: By adopting the Stage-Specific Curricular Data framework (OpenR1-Math-220k + Big-Math-RL-Verified), the study entirely escapes the mathematical trap of reward sparsity. Priming the model with bridge-adjusted hard samples during the SFT phase ensures that the GRPO algorithm subsequently has a dense, non-zero gradient landscape to optimize against during the consolidation phase.
272: 
273: Most importantly, by implementing Entropy-Filtered Reward Gating, Dynamic Difficulty Scalarization (DW-GRPO), and Probability Smoothing Policy Optimization (PSPO), the study mathematically insulates itself against the two most lethal pathologies of reason-tuning: reward hacking and formatting collapse. The policy will be permitted to autonomously discover optimized, highly efficient logical trajectories without destroying its structural formatting adherence or forgetting its pre-trained capabilities.
274: 
275: Quantitative projections based on current literature indicate that the baseline Qwen2.5-1.5B model, which typically scores around 23.20% to 31.58% on the out-of-distribution MATH-500 benchmark under standard, unmodified SFT settings, will experience a dramatic performance paradigm shift. Pure, unoptimized online GRPO typically elevates this metric marginally to approximately 32.00%, while specialized offline Dynamic Fine-Tuning (DFT) can push it to 35.43%.
276: 
277: However, by synthesizing the stabilized, dynamically weighted PSPO-GRPO framework with a Stage-Specific Consolidation dataset, the model is projected to establish highly resilient logical hubs. Based on identical hardware and architectural constraints utilizing PSPO and robust KL divergence estimation, the fully optimized experimental cohorts are expected to shatter these baselines, pushing the MATH-500 Pass@1 accuracy into the **55% to 62% range**. This represents an unassailable, mathematically verifiable improvement over standard SFT and naive GRPO controls, standing as a computationally feasible, methodologically watertight investigation into the deep mechanics of post-training alignment.
278: 
279: ## 7. Technical Implementation Specification
280: 
281: The following specifications are provided for reproducible code generation.
282: 
283: ### 7.1 Environment & Base Architecture
284: 
285: - **Hardware Setup:** 1x NVIDIA RTX 3090 (24GB VRAM).
286: - **Core Frameworks:** `unsloth` (VRAM-optimized kernel execution), `trl` (GRPOTrainer), `vllm` (accelerated generation). Unsloth's fused chunked cross-entropy and asynchronous gradient checkpointing reduce VRAM consumption by up to 90%, allowing group sizes of 16 to fit securely within 24GB limits.
287: - **Base Model:** `Qwen/Qwen2.5-1.5B`.
288: - **Quantization:** Load the model in 4-bit NormalFloat (NF4) using `unsloth.FastLanguageModel.from_pretrained`.
289: - **LoRA Configuration:** Target all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with rank $r = 64$ and alpha $\alpha = 128$.
290: - **Optimizer:** 8-bit AdamW to reduce optimizer state footprint.
291: 
292: ### 7.2 Stage 1: Supervised Fine-Tuning (SFT)
293: 
294: - **Objective:** Standard causal next-token prediction to instill the base reasoning topologies and teach the structural formatting tags.
295: - **Dataset:** 5,000-sample slice of `open-r1/OpenR1-Math-220k`.
296: - **Formatting:** Apply a standard chat template, enforcing that the assistant's responses are wrapped explicitly in `<think>` and `</think>` tags prior to the final answer.
297: 
298: ### 7.3 Stage 2: GRPO Reinforcement Learning
299: 
300: - **Objective:** Compress incorrect logical trajectories and consolidate correct ones using verifiable, rule-based rewards.
301: - **Dataset:** 2,500-sample subset of `SynthLabsAI/Big-Math-RL-Verified` (uniquely verifiable, closed-form mathematical answers).
302: - **Trainer Configuration (`trl.GRPOConfig`):**
303:   - `learning_rate`: $5 \times 10^{-6}$
304:   - `lr_scheduler_type`: `cosine`
305:   - `warmup_ratio`: 0.1
306:   - `num_generations` (Group Size $G$): 16
307:   - `max_prompt_length`: 256
308:   - `max_completion_length`: 1792
309: 
310: ### 7.4 Reward Function Implementation
311: 
312: **Correctness Reward:**
313: 
314: ```python
315: def correctness_reward(prompt, completion, ground_truth):
316:     match = re.search(r"<answer>\s*(-?\d+(\.\d+)?)\s*</answer>", completion)
317:     if not match:
318:         return 0.0
319:     predicted = float(match.group(1))
320:     return 1.0 if abs(predicted - float(ground_truth)) < 1e-5 else 0.0
321: ```
322: 
323: **Format Reward:**
324: 
325: ```python
326: def format_reward(completion):
327:     has_think = re.search(r"<think>.*?</think>", completion, re.DOTALL) is not None
328:     has_answer = re.search(r"<answer>.*?</answer>", completion, re.DOTALL) is not None
329:     return 1.0 if (has_think and has_answer) else 0.0
330: ```
331: 
332: **Dynamic Reward Weighting (DW-GRPO):**
333: 
334: The format reward weight decays as training steps progress, preventing reward hacking and forcing the policy to transition from formatting imitation to actual mathematical problem solving: $\lambda_{format}(t) = 1.0 \cdot e^{-\gamma t}$, where $\gamma$ is the decay coefficient and $t$ is the current training step.
335: 
336: ### 7.5 Algorithmic Stabilizations
337: 
338: To prevent training collapse, the generated code must override or modify the standard GRPO loss components with these mathematical constraints:
339: 
340: - **KL Divergence Estimator:** Replace the naive log-ratio KL penalty with the Schulman $K_3$ unbiased estimator: $D_{KL}^{(K_3)} = r - \log r - 1$, where $r = \frac{\pi_{\theta}}{\pi_{ref}}$.
341: - **Probability Smoothing Policy Optimization (PSPO):** Instead of utilizing hard PPO-style ratio clipping, dynamically smooth the current policy probabilities toward the reference policy before computing the importance ratio: $r_i^{PSPO} = \frac{(1 - \delta) \cdot \pi_{\theta} + \delta \cdot \pi_{ref}}{\pi_{\theta_{old}}}$.
342: - **Entropy-Filtered Gating:** Monitor the average sequence entropy. If the completion entropy exceeds $\ln 2$, dynamically throttle the gradient update by zeroing out the advantage for that sequence, preventing formatting collapse when rejected sequences are over-penalized.
343: 
344: ## Works Cited
345: 
346: 1. I trained Qwen2.5-1.5b with RLVR (GRPO) vs SFT and compared ..., accessed June 5, 2026, https://www.reddit.com/r/LocalLLaMA/comments/1rk2kcn/i_trained_qwen2515b_with_rlvr_grpo_vs_sf_and/
347: 
348: 2. [Bug] [Qwen3 4-bit] GGUF Conversion in v2025.6.12 Produces Garbage Output (GGGG...) · Issue #2860 · unslothai/unsloth — GitHub, accessed June 5, 2026, https://github.com/unslothai/unsloth/issues/2860
349: 
350: 3. 500K Context Length Fine-tuning | Unsloth Documentation, accessed June 5, 2026, https://unsloth.ai/docs/blog/500k-context-length-fine-tuning
351: 
352: 4. Long-context GRPO (R1 Reasoning) — Unsloth, accessed June 5, 2026, https://unsloth.ai/blog/grpo
353: 
354: 5. Small Language Model Fine-Tuning Strategies.pdf
355: 
356: 6. Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning — arXiv, accessed June 5, 2026, https://arxiv.org/html/2606.04466v1
357: 
358: 7. RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs — arXiv, accessed June 5, 2026, https://arxiv.org/html/2509.21128v2
359: 
360: 8. On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification — arXiv, accessed June 5, 2026, https://arxiv.org/html/2508.05629v3
361: 
362: 9. Revisiting the Superficial Alignment Hypothesis — arXiv, accessed June 5, 2026, https://arxiv.org/html/2410.03717v1
363: 
364: 10. LIMA: Less Is More for Alignment — arXiv, accessed June 5, 2026, https://arxiv.org/pdf/2305.11206
365: 
366: 11. Data Difficulty and the Generalization–Extrapolation Tradeoff in LLM Fine-Tuning — arXiv, accessed June 5, 2026, https://arxiv.org/html/2605.12906v1
367: 
368: 12. Fine-Tuning Qwen2.5-1.5B for Text-to-SQL Generation — Ready Tensor, accessed June 5, 2026, https://app.readytensor.ai/publications/fine-tuning-qwen25-15b-for-text-to-sql-generation-kaa6DwgRemd5
369: 
370: 13. config.json · unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit at 7ba24e424c1087371db2a9cab6d803e4c0912d0c — Hugging Face, accessed June 5, 2026, https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit/blob/7ba24e424c1087371db2a9cab6d803e4c0912d0c/config.json
371: 
372: 14. LLM Training Series — Part 3. A practical SFT walkthrough showing why... | by Vivek Vedant | May, 2026 | GoPenAI, accessed June 5, 2026, https://blog.gopenai.com/fine-tuning-qwen2-5-with-lora-more-structured-not-more-correct-3eea922cefda
373: 
374: 15. adity12345/qwen2.5-1.5b-custom — Hugging Face, accessed June 5, 2026, https://huggingface.co/adity12345/qwen2.5-1.5b-custom
375: 
376: 16. TOKENIZER DOESN'T WORK! PLEASE HELP! · Issue #1628 · unslothai/unsloth — GitHub, accessed June 5, 2026, https://github.com/unslothai/unsloth/issues/1628
377: 
378: 17. Quality of inference severely changed after merge_and_unload · Issue #1035 · huggingface/peft — GitHub, accessed June 5, 2026, https://github.com/huggingface/peft/issues/1035
379: 
380: 18. config.json · Qwen/Qwen2.5-1.5B-Instruct at main — Hugging Face, accessed June 5, 2026, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
381: 
382: 19. FastLanguageModel load_in_4bit not working, when used with fast_inference=True · Issue #2008 · unslothai/unsloth — GitHub, accessed June 5, 2026, https://github.com/unslothai/unsloth/issues/2008
383: 
384: 20. Built-in Benchmarks — NeMo Evaluator, accessed June 5, 2026, https://docs.nvidia.com/nemo/evaluator/0.3.0/evaluation/benchmarks.html

(End of file - total 384 lines)
</content>
```

---

## Assistant (Plan · DeepSeek V4 Pro · 15.0s)

_Thinking:_

The user asked me to read through the md file in Docs. I've found and read the file `docs/Synthesized Methodology.md`. It's a comprehensive research methodology document for a thesis titled "Dynamic Reward-Gating in Group Relative Policy Optimization: Delineating Reasoning vs. Formatting Tradeoffs in Small Language Models."

Let me summarize what this document contains for the user since they asked me to read it. But the user said "Read through the md file in Docs" - they might just want me to read it so I'm aware of it, or they might want me to summarize it. Given the tone instructions say to be concise, I'll provide a brief summary of what the document is about.

Actually, the user is in plan mode (read-only), and they simply asked me to read the file. I should acknowledge that I've read it and give a concise summary.

I've read through the full document. Here's what it contains:

---

**Title:** *Dynamic Reward-Gating in Group Relative Policy Optimization: Delineating Reasoning vs. Formatting Tradeoffs in Small Language Models*

This is a detailed thesis methodology document (~384 lines) for a research project on post-training optimization of **Qwen2.5-1.5B** (an SLM under 3B params) on a single **RTX 3090 (24GB)**.

### Core structure:
1. **Introduction** — Frames the problem: SFT degrades reasoning (alignment tax), GRPO is superior for consolidation, and the thesis explores dynamically gating format vs. correctness rewards to prevent reward hacking and formatting collapse.
2. **Literature Review** — Covers SFT vs. RL reasoning topologies, reward hacking, Generalization-Extrapolation tradeoff, and Stage-Specific Curricular Data Design.
3. **Methodology (4 Phases)**:
   - *Phase 1*: 3-stage dataset pipeline (SFT on OpenR1-Math-220k → GRPO on Big-Math-RL-Verified → Critique Fine-Tuning for failed samples)
   - *Phase 2*: Hardware-optimized GRPO with Unsloth/TRL/vLLM, 4-bit NF4 quant, LoRA r=64, G=16 group size
   - *Phase 3*: Three stabilization mechanisms — Entropy-Filtered Reward Gating, Dynamic Difficulty Scalarization (DW-GRPO), and Probability Smoothing Policy Optimization (PSPO)
   - *Phase 4*: 5 experimental cohorts with different λ_correct / λ_format ratios, including a decaying format gate
4. **Mathematical Formulation** — Full GRPO objective with PSPO-smoothed ratios, entropy-filtered advantages, and K₃ KL estimator
5. **VRAM Budgeting** — Byte-level allocation table showing ~7.5GB peak (31% of 24GB)
6. **Evaluation** — MATH-500, AIME24, ANOVA, Tukey's HSD, entropy trajectory analysis
7. **Implementation Specs** — Reward function code, LoRA targets, training hyperparameters

The doc also includes a full Works Cited section with 20 references (arXiv papers, GitHub issues, blog posts).

---

Is there something specific you'd like to do with this document? (e.g., implement the methodology, create a plan for building it, etc.)

---

## User

Based on the md, what would be our control model ?

---

## Assistant (Plan · DeepSeek V4 Pro · 11.3s)

_Thinking:_

The user is asking about the "control model" based on the methodology document. Let me look at the document to identify what serves as the control.

Looking at the methodology, specifically Phase 4 (Section 3.4), there's a table of 5 cohorts:

1. **Baseline** — λ_correct = 1.0, λ_format = 0.0 — "Verifiable Correctness Only. Isolates the model's capacity to discover its own unstructured, implicit reasoning trajectories without stylistic guidance."
2. Cohort A — Formatting Dominance
3. Cohort B — Correctness Dominance with Format Regularization
4. Cohort C — Graduated Mathematical Accuracy
5. Cohort D — Dynamic Decaying Format-Gate (DW-GRPO)

The Baseline cohort with λ_correct = 1.0 and λ_format = 0.0 is clearly the control model. It's the one where only verifiable correctness is rewarded, with no formatting reward at all. This serves as the control against which all other cohorts (which use various combinations of correctness + format rewards) are compared.

The document also mentions in Section 6.3: "Conducted across the mean performance metrics of the five experimental cohorts to reject the null hypothesis (H₀: Modulating the formatting reward ratio yields no difference in downstream logical accuracy)" and "Likelihood of Success" section mentions "the baseline Qwen2.5-1.5B model, which typically scores around 23.20% to 31.58% on the out-of-distribution MATH-500 benchmark under standard, unmodified SFT settings."

So the control is the **Baseline** cohort with λ_correct = 1.0, λ_format = 0.0 — pure correctness reward only, no format guidance.

The **Baseline** cohort is the control model:

| Cohort | λ_correct | λ_format | Description |
|---|---|---|---|
| **Baseline** | 1.0 | 0.0 | Verifiable Correctness Only |

It receives no format reward whatsoever — only binary correctness reward (1.0 for exact numerical match, 0.0 otherwise). This isolates the model's raw ability to discover its own unstructured reasoning trajectories without any stylistic guidance or formatting incentives, serving as the reference point against which all other reward-gating strategies (Cohorts A–D) are compared.

---

## User

Construct a multistage plan to implement it.

---

## Assistant (Plan · DeepSeek V4 Pro · 11.6s)

_Thinking:_

The user wants me to construct a multistage plan to implement the methodology described in the document. Let me think about this carefully.

The document describes a complex ML research project with multiple phases:

1. Phase 1: Stage-Specific Dataset Architecture (SFT → GRPO → Critique Fine-Tuning loop)
2. Phase 2: Hardware-Optimized GRPO Pipeline
3. Phase 3: Algorithmic Stabilization Mechanisms
4. Phase 4: Reward-Gating Permutations (5 cohorts, 4 seeds each = 20 runs)

Then evaluation with MATH-500 and AIME24.

Let me think about what needs to happen for this implementation:

**Code Structure:**
1. Environment setup (dependencies, CUDA, etc.)
2. Data loading and preprocessing
3. SFT training stage
4. GRPO core loop with all the stabilization mechanisms
5. Multi-cohort experiment orchestration
6. Evaluation suite
7. Statistical analysis

But I need to think about what language/framework this would be in. The document mentions Python with Unsloth, TRL, vLLM, bitsandbytes.

Let me ask the user some clarifying questions before building the plan. Actually, the user said "construct a multistage plan to implement it" - they want the methodology implemented as code. 

Key questions I should ask:
1. What does the current repo look like? Is it empty?
2. Do they want just the plan written out, or should I eventually implement it?
3. What's the expected structure - Python scripts? Jupyter notebooks? A full project with configs?

Let me first check what's in the repo, then present a plan.

Actually, the user is in plan mode - I should NOT execute anything. I should just present the plan. Let me explore what's in the repo first (read-only) and then construct the plan.

Let me check the repo structure first.

Let me first understand what we're working with in the repo.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/ibrar/Opencode/Thesis-Whatever"
}
```

**Output:**
```
<path>/home/ibrar/Opencode/Thesis-Whatever</path>
<type>directory</type>
<entries>
docs/

(1 entries)
</entries>
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/*",
  "path": "/home/ibrar/Opencode/Thesis-Whatever"
}
```

**Output:**
```
/home/ibrar/Opencode/Thesis-Whatever/docs/Synthesized Methodology.md
```

---

## Assistant (Plan · DeepSeek V4 Pro · 98.9s)

_Thinking:_

The repo is essentially empty except for the methodology doc. So we're starting from scratch.

Before I build the plan, I should ask the user a few clarifying questions to make sure we're aligned:

1. Is this intended to be a **runnable codebase** (Python scripts/project) or are we writing the **thesis text/paper**?
2. Given the hardware constraints (single RTX 3090, 30-hour envelope), do they actually have access to this GPU? Or is this purely a theoretical implementation?
3. What's the scope — do they want the full multi-cohort experiment orchestration, or just enough code to run a single cohort?

Let me think about the plan more carefully now...

Actually, I should present a plan and ask clarifying questions at the same time. Let me think about the core components:

**Implementation Plan:**

### Overall Architecture
A Python project using:
- `unsloth` for 4-bit loading and LoRA
- `trl` (GRPOTrainer) for the RL loop
- `vllm` for accelerated generation
- `datasets` (HuggingFace) for data loading
- `bitsandbytes` for quantization

### Key Modules:
1. **`config/`** — YAML/JSON configs for each cohort
2. **`data/`** — Dataset loading and preprocessing
3. **`training/`** — SFT and GRPO training scripts
4. **`rewards/`** — Custom reward functions (correctness, format, graduated, dynamic)
5. **`stabilization/`** — PSPO, entropy filtering, K3 KL estimator
6. **`evaluation/`** — MATH-500 and AIME24 evaluation
7. **`analysis/`** — Statistical analysis (ANOVA, Tukey's HSD, plots)
8. **`orchestration/`** — Multi-cohort, multi-seed run management

### Pipeline Flow:
1. **Setup** → Install deps, download model, prepare datasets
2. **SFT Stage** → Train on OpenR1-Math-220k (5k samples)
3. **GRPO Stage** → For each cohort × 4 seeds:
   - Load SFT checkpoint
   - Run GRPO on Big-Math-RL-Verified
   - Apply cohort-specific reward function
   - Save checkpoints
4. **Evaluation** → Run all checkpoints on MATH-500 + AIME24
5. **Analysis** → ANOVA, Tukey's HSD, KL divergence plots

Let me present this as a structured plan to the user.

I should also ask: do we want this as a single script or a full project structure? Also, are we actually running this, or building the codebase for someone to run later?

Let me present the plan.

The repo is currently empty except for the methodology doc. Before I lay out the full plan, a few clarifying questions:

**Tool: question**

**Input:**
```json
{
  "questions": [
    {
      "question": "What form should the implementation take?",
      "header": "Output Format",
      "options": [
        {
          "label": "Full Python project",
          "description": "Modular codebase with configs, training scripts, evaluation, and orchestration — ready to run on a machine with the required GPU"
        },
        {
          "label": "Single end-to-end script",
          "description": "One consolidated Python script that chains all stages together"
        },
        {
          "label": "Just the implementation plan",
          "description": "Write out the detailed plan without creating any code yet"
        }
      ]
    },
    {
      "question": "Do you have access to an RTX 3090 to actually run this, or is this a codebase being built for future execution?",
      "header": "Hardware Availability",
      "options": [
        {
          "label": "Yes, I can run it",
          "description": "I have the GPU and will execute the code"
        },
        {
          "label": "No, build-only",
          "description": "Just build the codebase; execution happens later on different hardware"
        }
      ]
    },
    {
      "question": "How should the 5 cohorts and 4 seeds be managed?",
      "header": "Experiment Orchestration",
      "options": [
        {
          "label": "Sequential bash orchestration",
          "description": "A shell script that loops over cohorts and seeds, calling the training script with different args for each run"
        },
        {
          "label": "Python orchestration",
          "description": "A Python driver script that manages the multi-run execution, checkpointing, and logging programmatically"
        },
        {
          "label": "Manual per-run",
          "description": "Just provide the required commands; I'll launch them myself"
        }
      ]
    }
  ]
}
```

**Output:**
```
User has answered your questions: "What form should the implementation take?"="Just the implementation plan", "Do you have access to an RTX 3090 to actually run this, or is this a codebase being built for future execution?"="I can't run it, I only have a local 3050 (the machine you are on) but run the code on a pyenv enviroment to snuff out any dependency issue or other stuff.", "How should the 5 cohorts and 4 seeds be managed?"="Python orchestration". You can now continue with the user's answers in mind.
```

---

## Assistant (Plan · DeepSeek V4 Pro · 96.3s)

_Thinking:_

The user wants:
1. Just the implementation plan (not code yet)
2. They can't fully run it (only have a 3050 locally) but want to verify the code works in a pyenv environment to catch dependency issues
3. Python orchestration for managing cohorts/seeds

So I should construct a detailed, actionable plan for implementing this methodology as a Python project. The plan should be comprehensive enough that someone can execute it step-by-step.

Let me think about the architecture, modules, data flow, and dependencies carefully.

Key considerations:
- RTX 3050 can't run the full 1.5B model + vLLM in 24GB, but we can at least verify imports, data loading, reward functions, and orchestration logic
- pyenv environment means we need to manage dependencies cleanly
- The plan should include a stage for "smoke testing" on the 3050 to validate code correctness without needing full training

Let me build a comprehensive implementation plan now.

**Architecture:**

Project root: `Thesis-Whatever/`
```
├── pyproject.toml          # Dependencies and project config
├── .python-version          # pyenv version pinning
├── configs/
│   ├── base.yaml            # Shared config (model, LoRA, training params)
│   ├── cohorts.yaml         # 5 cohort definitions with λ values
│   └── evaluation.yaml      # Evaluation config
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py        # HuggingFace dataset loading + preprocessing
│   │   └── preprocessing.py # Chat template, formatting, filtering
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract reward interface
│   │   ├── correctness.py   # Binary correctness reward
│   │   ├── format.py        # Format adherence reward
│   │   ├── graduated.py     # Graduated correctness (Cohort C)
│   │   └── dynamic.py       # DW-GRPO dynamic weighting
│   ├── stabilization/
│   │   ├── __init__.py
│   │   ├── pspo.py          # Probability Smoothing Policy Optimization
│   │   ├── entropy_filter.py # Entropy-filtered advantage masking
│   │   └── kl_estimator.py  # K3 Schulman KL estimator
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py   # SFT training wrapper
│   │   └── grpo_trainer.py  # GRPO training wrapper with customizations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── math500.py       # MATH-500 evaluation
│   │   ├── aime24.py        # AIME24 evaluation
│   │   └── parser.py        # Verifiable grading parser
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py    # ANOVA, Tukey's HSD
│   │   └── plotting.py      # KL divergence, entropy, correctness curves
│   └── orchestration/
│       ├── __init__.py
│       ├── runner.py        # Multi-cohort, multi-seed orchestration
│       └── logger.py        # Experiment logging and checkpoint tracking
├── scripts/
│   ├── download_model.py    # Model acquisition
│   ├── download_datasets.py # Dataset acquisition
│   ├── run_sft.py           # SFT entry point
│   ├── run_grpo.py          # GRPO entry point for a single cohort/seed
│   ├── run_evaluation.py    # Evaluation entry point
│   ├── run_analysis.py      # Statistical analysis entry point
│   └── run_all.py           # Full pipeline orchestration
├── tests/
│   ├── test_rewards.py
│   ├── test_stabilization.py
│   └── test_evaluation.py
└── outputs/                  # Training outputs, checkpoints, logs
```

Wait, I need to think about this more carefully. The GRPOTrainer from TRL is the standard path, but this research requires custom modifications:
- Custom reward functions per cohort
- PSPO probability smoothing (this modifies the PPO objective)
- Entropy-filtered advantage masking
- K3 KL estimator (may already be in TRL)

TRL's GRPOTrainer might not support all of these out of the box. We may need to:
1. Subclass GRPOTrainer
2. Override the loss computation
3. Or implement a custom training loop

Let me think about how TRL's GRPOTrainer works:
- It takes reward functions as callables
- It computes advantages using group normalization
- It applies PPO-style clipping with KL penalty
- The KL penalty might be configurable

For PSPO, we'd need to modify how the importance ratio is computed. This requires overriding the loss computation.

For entropy filtering, we'd need to compute sequence entropy and zero out advantages. This also requires loss computation override.

So the approach would be:
1. Subclass `GRPOTrainer` from TRL
2. Override `compute_loss` or the relevant method
3. Inject PSPO smoothing, entropy filtering, and K3 KL

Let me now think about the plan stages...

Actually, I should present a staged implementation plan, where each stage builds on the previous one and can be independently tested/validated.

Let me structure this:

**Stage 0: Environment & Project Scaffold**
**Stage 1: Data Pipeline**
**Stage 2: Reward Functions**
**Stage 3: SFT Training**
**Stage 4: GRPO Core (Baseline)**
**Stage 5: Stabilization Mechanisms (PSPO, Entropy, K3)**
**Stage 6: Multi-Cohort Orchestration**
**Stage 7: Evaluation Suite**
**Stage 8: Statistical Analysis**
**Stage 9: Integration & Smoke Testing**

Each stage should have:
- Inputs
- Outputs  
- Implementation details
- Validation criteria
- Dependencies on other stages

The validation criteria are especially important since the user wants to verify on their 3050.

Let me think about what can be validated on the 3050 vs what requires the 3090:
- 3050 can verify: imports, data loading, reward functions, config parsing, orchestration logic, evaluation logic (on small models if needed), unit tests
- 3050 CANNOT: Full 1.5B model training with vLLM, GRPO with G=16
- But we could smoke-test with a tiny model (e.g., Qwen2.5-0.5B or smaller, or just verify the SFT/GRPO trainer classes instantiate correctly without running full training)

Let me also think about the dependency stack:
- Python 3.10+ (for TRL, Unsloth compatibility)
- PyTorch 2.x with CUDA
- unsloth
- trl >= 0.9 (GRPOTrainer)
- vllm
- bitsandbytes
- transformers
- datasets
- accelerate
- peft
- numpy, scipy, pandas (for analysis)
- matplotlib, seaborn (for plotting)
- pyyaml (for configs)
- wandb or tensorboard (for logging, optional)

Actually, Unsloth has specific version requirements. Let me think about this...

Unsloth recommends using their installation instructions. They often have version pins for transformers, etc.

Let me now formulate the complete plan. I'll be thorough but concise.

For the GRPO trainer customization, I should think about what TRL's GRPOTrainer actually exposes:
- In TRL, GRPOTrainer.train() handles the full loop
- Reward functions are passed as a list of callables
- The KL penalty type can be configured (kl, abs, mse, low_var_kl, full)
- But customizing the ratio computation or advantage masking requires subclassing

Looking at the TRL source (from memory):
- `_generate_and_score_completions` generates and scores
- `compute_loss` computes the GRPO loss
- The loss computation uses the standard PPO clipped objective

So to implement PSPO + entropy filtering:
1. Subclass GRPOTrainer
2. Override `compute_loss` to:
   a. Apply PSPO smoothing to the log probabilities
   b. Compute K3 KL estimator
   c. Apply entropy-based advantage masking
   d. Return the modified loss

This is feasible. Let me also think about whether we need to modify the generation loop for entropy computation... Entropy would be computed from the logits during generation, which may require accessing the model's output distribution during rollout generation. This could be tricky with vLLM's batched generation.

Alternative: compute approximate entropy from the token-level log probabilities that are already available in the GRPO pipeline (per-token log probs from the policy model). We can approximate sequence entropy as the negative mean log probability.

Actually, the document says "average group completion entropy H is actively monitored during every training step." This could be approximated as the negative mean of log probabilities: H ≈ -mean(log π_θ(token | context)). This is already computable from the log probs we get during training.

OK, I think I have enough to construct a solid plan. Let me write it out.

One more thing: I should consider the "Critique Fine-Tuning" recycle loop from Phase 1. The document mentions routing failed samples back to the SFT set. This is a more complex feedback loop. For a practical implementation, we could:
- Option A: Implement the full recycle loop (complex)
- Option B: Mark it as a future enhancement and note where the hook point would be
- Option C: Implement a simplified version

I'll include it in the plan but note it's advanced.

Also, for the "non-all-zero samples" filtering during the Consolidation set — this requires running generation before GRPO to filter. This should be part of the data pipeline.

Let me finalize the plan now.

Good. Here's the comprehensive implementation plan:

---

# Multistage Implementation Plan

## Project: Dynamic Reward-Gating in GRPO for SLMs

---

### Stage 0 — Environment & Project Scaffold

**Goal:** Reproducible Python environment with all dependencies resolving cleanly.

| # | Task | Details |
|---|---|---|
| 0.1 | Pyenv + venv setup | Pin Python 3.10 (TRL/Unsloth compatibility), create `pyproject.toml` |
| 0.2 | Dependency manifest | `torch>=2.1`, `unsloth`, `trl>=0.9`, `vllm`, `bitsandbytes`, `transformers`, `datasets`, `peft`, `accelerate`, `pyyaml`, `scipy`, `matplotlib`, `seaborn` |
| 0.3 | Project tree | `src/` (data, rewards, stabilization, training, evaluation, analysis, orchestration), `configs/`, `scripts/`, `tests/`, `outputs/` |
| 0.4 | Config schema | YAML-based configs: `base.yaml` (shared hyperparams), `cohorts.yaml` (λ values per cohort), `evaluation.yaml` |
| 0.5 | Smoke test | `python -c "import torch, unsloth, trl, vllm; print('OK')"` — validates all imports on the 3050 |

**Validation:** All imports pass. `pyproject.toml` installs without conflicts.

---

### Stage 1 — Data Pipeline

**Goal:** Load, preprocess, and filter the three datasets.

| # | Task | Details |
|---|---|---|
| 1.1 | Dataset downloader | `scripts/download_datasets.py` — fetch `open-r1/OpenR1-Math-220k` (5k subset) and `SynthLabsAI/Big-Math-RL-Verified` (2.5k subset) via `datasets` |
| 1.2 | SFT preprocessing | Apply chat template, wrap assistant responses in `<think>...</think>`, format as `{"prompt": ..., "completion": ...}` |
| 1.3 | RL preprocessing | Extract `prompt` and `answer` (ground truth) fields for GRPO — no completion needed |
| 1.4 | Non-all-zero filter | Run a single generation pass with the SFT-primed model to identify which Big-Math samples produce at least one correct rollout; exclude zero-reward samples from GRPO training set |
| 1.5 | Tokenization | Tokenize with Qwen2.5 tokenizer, max lengths: prompt=256, completion=1792 |

**Validation:** Datasets load and preprocess without errors. Filter produces non-empty subset.

---

### Stage 2 — Reward Functions

**Goal:** Implement all reward functions exactly as specified.

| # | Task | Details |
|---|---|---|
| 2.1 | `correctness_reward(prompt, completion, answer)` | Parse `<answer>` tag, extract numeric value, compare with ground truth via `abs(pred - target) < 1e-5` → 0.0 or 1.0 |
| 2.2 | `format_reward(completion)` | Check for `<think>...</think>` AND `<answer>...</answer>` → 0.0 or 1.0 |
| 2.3 | `graduated_reward(prompt, completion, answer)` | Cohort C: exact match → 1.0, within 10% → 0.5, within 20% → 0.2, failure → -0.2 |
| 2.4 | `dynamic_format_weight(t, gamma)` | Cohort D: λ_format(t) = 1.0 * exp(-γ * t), where t = training step |
| 2.5 | Composite reward factory | `make_reward_fn(cohort_cfg)` → returns a callable that combines λ_correct * r_correct + λ_format * r_format with cohort-specific logic |
| 2.6 | Unit tests | Test parsing edge cases: no tags, malformed tags, negative numbers, decimals |

**Validation:** All unit tests pass. Manual spot-check with hand-crafted completions.

---

### Stage 3 — SFT Training

**Goal:** Train the base Qwen2.5-1.5B on OpenR1-Math-220k to bootstrap reasoning structure.

| # | Task | Details |
|---|---|---|
| 3.1 | Model loader | `src/training/sft_trainer.py` — load `Qwen/Qwen2.5-1.5B` in 4-bit NF4 via `unsloth.FastLanguageModel.from_pretrained` |
| 3.2 | LoRA config | Target all 7 projection layers (`q,k,v,o,gate,up,down`), r=64, α=128 |
| 3.3 | Training args | LR=5e-6 cosine, warmup_ratio=0.1, 1 epoch over 5k samples, batch_size=4, gradient_accumulation=4 |
| 3.4 | SFT run script | `scripts/run_sft.py` — loads config, runs training, saves LoRA adapters + merged model checkpoint |
| 3.5 | Embedding tie fix | Apply Unsloth patching to handle `tie_word_embeddings: true` for the Qwen architecture |

**Validation:** SFT runs to completion without OOM. Checkpoint saves correctly.

---

### Stage 4 — GRPO Core (Baseline Cohort)

**Goal:** Implement the core GRPO training loop with the Baseline reward (correctness only, λ=1.0).

| # | Task | Details |
|---|---|---|
| 4.1 | GRPO config | `GRPOConfig`: G=16, max_prompt=256, max_completion=1792, LR=5e-6 cosine, warmup=0.1, 2 epochs, 312 steps |
| 4.2 | Base GRPO trainer | `src/training/grpo_trainer.py` — wrap `trl.GRPOTrainer`, load SFT checkpoint, pass Baseline reward fn |
| 4.3 | vLLM integration | Configure Unsloth's optimized vLLM co-loading for fast generation within GRPO rollout collection |
| 4.4 | Checkpointing | Save adapter weights every N steps + at training end |
| 4.5 | Logging | Log per-step: mean reward, format adherence %, average completion length, KL divergence |
| 4.6 | Baseline run script | `scripts/run_grpo.py --cohort baseline --seed 42` |

**Validation:** Baseline GRPO completes without OOM. Rewards and logs are sensible.

---

### Stage 5 — Stabilization Mechanisms

**Goal:** Inject PSPO, entropy filtering, and K3 KL estimator into the GRPO loss computation.

| # | Task | Details |
|---|---|---|
| 5.1 | K3 KL estimator | `src/stabilization/kl_estimator.py` — `k3_kl(log_ratio) = exp(log_ratio) - log_ratio - 1` |
| 5.2 | PSPO ratio smoothing | `src/stabilization/pspo.py` — `r_pspo = ((1-δ) * π_θ + δ * π_ref) / π_old` with configurable δ (default ~0.1) |
| 5.3 | Entropy filter | `src/stabilization/entropy_filter.py` — compute sequence-level entropy from log probs; if H > ln(2), zero the advantage |
| 5.4 | Customized GRPO trainer | Subclass `GRPOTrainer`, override `compute_loss` to: (a) smooth log probs via PSPO, (b) compute K3 KL penalty, (c) apply entropy mask to advantages, (d) compute standard clipped GRPO loss with modifications |
| 5.5 | Stabilization config | Toggle switches in `configs/base.yaml`: `pspo.enabled`, `pspo.delta`, `entropy_filter.enabled`, `kl_estimator: "k3"` |
| 5.6 | Unit tests | Test K3 properties (≥0, zero at r=1), PSPO smoothing bounds, entropy filter threshold |

**Validation:** Unit tests pass. Modified trainer instantiates without errors.

---

### Stage 6 — Multi-Cohort Orchestration

**Goal:** Python driver that manages all 5 cohorts × 4 seeds = 20 runs.

| # | Task | Details |
|---|---|---|
| 6.1 | Cohort configs | `configs/cohorts.yaml` — define all 5 cohorts with their λ values, graduated thresholds, decay γ |
| 6.2 | Run manager | `src/orchestration/runner.py` — `ExperimentRunner` class that: iterates cohorts × seeds, constructs reward fn per cohort, launches GRPO training, logs to structured directories (`outputs/{cohort}/{seed}/`) |
| 6.3 | Resume support | Check for existing checkpoints; skip completed runs |
| 6.4 | Dry-run mode | `--dry-run` flag that instantiates everything but exits before training — for validation on the 3050 |
| 6.5 | Progress tracking | CSV log of all runs: cohort, seed, status, best reward, timestamp |
| 6.6 | Entry point | `scripts/run_all.py` — parses args, initializes runner, executes full pipeline or subset |

**Validation:** Dry run on 3050: all 20 configurations instantiate correctly, reward functions are wired properly.

---

### Stage 7 — Evaluation Suite

**Goal:** Evaluate all trained checkpoints on MATH-500 and AIME24.

| # | Task | Details |
|---|---|---|
| 7.1 | Verifiable parser | `src/evaluation/parser.py` — the `verify_math_accuracy()` function from the doc (structural format check + numeric equivalence) |
| 7.2 | MATH-500 loader | Load MATH-500 benchmark from HuggingFace (`hendrycks/competition_math` or equivalent) |
| 7.3 | AIME24 loader | Load AIME24 problems |
| 7.4 | Batch evaluator | `src/evaluation/evaluator.py` — for each checkpoint: generate completions with greedy decoding, parse answers, compute Pass@1 accuracy |
| 7.5 | Evaluation report | Per-cohort, per-seed CSV of MATH-500 and AIME24 scores; aggregated mean ± std across seeds |
| 7.6 | Metrics tracking | Also log: average generation length, `<think>` token presence rate, format adherence % |

**Validation:** Evaluation runs on a single checkpoint without errors.

---

### Stage 8 — Statistical Analysis

**Goal:** ANOVA, Tukey's HSD, and visualization of results.

| # | Task | Details |
|---|---|---|
| 8.1 | Inferential stats | `src/analysis/statistics.py` — one-way ANOVA on MATH-500 scores across cohorts; post-hoc Tukey's HSD for pairwise comparisons; report p-values and effect sizes |
| 8.2 | KL divergence curves | Plot $D_{KL}(\pi_\theta \parallel \pi_{ref})$ vs. OOD error on MATH-500 (Generalization-Extrapolation curve) |
| 8.3 | Entropy trajectories | Plot per-step average completion entropy for each cohort to validate entropy filter behavior |
| 8.4 | Reward/format curves | Plot correctness % and format adherence % over training steps for all cohorts |
| 8.5 | Output artifacts | Save all plots as PNG/PDF in `outputs/analysis/` |

**Validation:** Analysis script runs on synthetic or minimal real data without errors.

---

### Stage 9 — Integration & Smoke Testing

**Goal:** Validate the full pipeline end-to-end on the RTX 3050.

| # | Task | Details |
|---|---|---|
| 9.1 | Tiny model smoke test | Run the SFT + GRPO pipeline with a tiny model (e.g., Qwen2.5-0.5B or mock) for 5 steps to catch integration bugs |
| 9.2 | Dry-run full pipeline | `run_all.py --dry-run` — validates configs, reward wiring, orchestration logic, evaluation loading |
| 9.3 | Dependency freeze | Lock `requirements.txt` with exact versions that work |
| 9.4 | README | Document: setup instructions, config guide, how to run each stage, expected outputs |

**Validation:** Dry-run completes. All scripts are importable and callable.

---

### Dependency Graph

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5 ──► Stage 6
                                                                              │
                                              Stage 7 ◄───────────────────────┘
                                                │
                                              Stage 8
                                                │
                                              Stage 9
```

Stages 2, 5 can be developed in parallel with Stage 1 since they're standalone modules with clear interfaces. Stages 7-8 can begin once Stage 2 (reward parsing shared with evaluation) is done.

---

### Key Design Decisions to Lock Down

1. **GRPOTrainer subclassing:** TRL's `GRPOTrainer.compute_loss` is the hook point. Need to confirm the exact method signature in TRL ≥0.9 before implementing Stage 5.

2. **PSPO integration:** The smoothing requires access to both π_θ and π_ref log probabilities during loss computation. TRL already tracks both, but we need to verify the variable names.

3. **Critique Fine-Tuning recycle loop:** The doc describes routing unsolved samples back to SFT. I recommend implementing this as an optional post-GRPO hook rather than inline during training — simpler to debug and validate.

4. **vLLM + Unsloth co-loading:** Unsloth's blog post ([4]) describes specific techniques to avoid double memory allocation. This may require exact version pins.

---

Does this plan align with what you need? Any sections you'd like to expand, reorder, or scope differently before we begin implementation?

---

## User

All good, but remember that I will be running the actual production code in a vast.ai instance, and vast ai instanced can be a headache in case of cuda version, so what would be the easiest way to run the production code on there ?Also, would it be possible to run on a 12GB (3060) device compared to a 3090 ? If so then how much slower would it be ? 

---

## Assistant (Plan · DeepSeek V4 Pro · 81.5s)

_Thinking:_

The user has two key questions:

1. **vast.ai compatibility**: How to handle CUDA version headaches on vast.ai instances. vast.ai instances can have varying CUDA versions, and mismatches between the host CUDA and the container's PyTorch CUDA can cause issues.

2. **RTX 3060 (12GB) feasibility**: Can the methodology run on a 12GB card instead of 24GB, and if so, how much slower?

Let me think through both carefully.

### vast.ai CUDA Strategy

The cleanest approach for vast.ai would be:
- **Use a Docker container image** with everything pre-configured. vast.ai supports custom Docker images.
- **Alternative: Use a setup script** that auto-detects the CUDA version and installs the right PyTorch/Unsloth versions.
- **Best approach: Pre-built Docker image** — Create a Dockerfile that pins exact CUDA version, PyTorch, Unsloth, vLLM, TRL versions. Push to Docker Hub or use vast.ai's built-in template options.

Actually, the real headache with vast.ai is:
1. The host machine has a specific NVIDIA driver version
2. That driver determines the max CUDA version you can use
3. You need PyTorch compiled for that CUDA version (or lower, since CUDA is backward compatible)
4. Unsloth and vLLM also have CUDA version requirements

The safest approach:
- **Option A: Docker image with CUDA 12.1** — Most compatible across vast.ai GPUs. Use `nvidia/cuda:12.1.0-runtime-ubuntu22.04` as base, install everything via pip.
- **Option B: vast.ai's recommended approach** — Use their template system, install via a startup script that auto-detects CUDA version.
- **Option C: conda/pip with flexible install** — Have a setup script that queries `nvidia-smi` to get the driver version, then installs the appropriate PyTorch CUDA build.

For vast.ai specifically, the Docker approach is most reliable because:
- You control the entire environment
- No CUDA version mismatch
- Reproducible across instances
- vast.ai has good Docker support

For the Startup Script approach:
```bash
# Auto-detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
# Install PyTorch with matching CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION/./}
```

But actually, the simplest approach for vast.ai is to use their "PyTorch" template which comes with a CUDA version already configured, and then just install the additional dependencies.

Let me think about what CUDA versions are commonly available on vast.ai:
- Most instances have NVIDIA drivers supporting CUDA 11.8, 12.1, or 12.4
- Unsloth recommends CUDA 11.8+ and PyTorch 2.1+
- vLLM needs CUDA 11.8+

The easiest production approach:
1. **vast.ai recommended Docker flow**: Build a Docker image locally, push to Docker Hub, use on vast.ai
2. **Alternative: vast.ai setup script**: Use their built-in PyTorch image, then run a setup script

I think the Docker approach is best and I should include Docker support in the plan.

### RTX 3060 12GB Feasibility

Let me calculate the VRAM breakdown for a 12GB card:

The document's VRAM table for 24GB:
- Base Model Weights (NF4): ~0.85 GB
- LoRA Adapters (BF16): ~0.25 GB
- Optimizer States (8-bit AdamW): ~0.50 GB
- vLLM KV Cache: ~3.20 GB
- Fused Logits & Activations: ~1.20 GB
- CUDA Context Overhead: ~1.50 GB
- Total: ~7.50 GB

For a 12GB 3060, the main issue is the **vLLM KV Cache**. Let me recalculate:

The KV cache formula is: 2 × 2 bytes × layers × G × L 
= 2 (KV) × 2 bytes (FP16) × 28 (layers) × G × 2048 (L)
= 4 × 28 × G × 2048 bytes
= 229,376 × G bytes
= 0.229 × G MB per... wait let me recalculate.

Actually, the KV cache size per token per layer in FP16:
- For each layer: key_states (num_kv_heads × head_dim) + value_states (num_kv_heads × head_dim)
- Qwen2.5-1.5B has 2 KV heads, head_dim = 128 (1536/12 = 128)
- Per token per layer: 2 × 2 heads × 128 dim × 2 bytes (FP16) = 1024 bytes
- For 28 layers: 28 × 1024 = 28,672 bytes per token
- For G=16, L=2048: 16 × 2048 × 28,672 = 939,524,096 bytes ≈ 0.94 GB

Hmm, that's much less than 3.2 GB. Let me reconsider. The document says:
- vLLM KV Cache: 2 × 2 bytes × layers (28) × G (16) × L (2048) = 3.20 GB

OK so their formula seems simplified. Let me trust their number: ~3.2 GB for G=16, L=2048.

So for a 12GB card:
- If we keep G=16: Total ≈ 7.5 GB — this should fit in 12GB! But leaves only ~4.5 GB headroom which is tight.
- vLLM overhead could spike with longer generations.
- The "CUDA Context Overhead" could be larger in practice.

Actually wait, the issue might be more subtle. The 3060 has:
- 12 GB VRAM (vs 24 GB on 3090)
- Much lower memory bandwidth: 360 GB/s (3060 12GB) vs 936 GB/s (3090)
- Fewer CUDA cores: 3584 vs 10496
- Lower FP16 compute: 12.74 TFLOPS vs 35.58 TFLOPS

The primary bottlenecks would be:
1. **Memory bandwidth** — inference/generation is memory-bandwidth bound, so the 3060 would be ~2.6x slower for generation
2. **Compute** — training (forward/backward) would be ~2.8x slower
3. **VRAM** — might fit but with less headroom for vLLM KV cache peaks

For the full pipeline timing:
- The doc estimates ~3.2 hours for pure training (20 runs × 9.6 min)
- On a 3060: maybe 2.5-3x slower → ~8-10 hours for training
- Plus SFT (~2h → ~5-6h), evaluation (~4.5h → maybe similar since evaluation is mostly generation)
- Total: maybe 15-20 hours vs 10-11.5 hours

But the VRAM question is critical. 7.5 GB in 12 GB leaves only 4.5 GB headroom. This might be tight because:
- vLLM KV cache can spike with longer-than-expected generations
- CUDA context overhead might be higher on 3060
- The chunked cross-entropy buffers might need more room

Mitigation for 12GB:
- Reduce G from 16 to 8 (halves KV cache → saves ~1.6 GB, total ~5.9 GB)
- Reduce max_completion_length from 1792 to 1024
- Use gradient checkpointing more aggressively

But reducing G from 16 to 8 means higher variance in GRPO advantage estimation, which could affect convergence. The doc specifically argues that G=16 is "highly optimal" for variance reduction.

For the plan, I should:
1. Recommend Docker for vast.ai
2. Provide VRAM calculations for 12GB and speed estimates
3. Note that G may need to be reduced to 8 on 12GB

Let me construct my response.

Actually, I realized I should also address: what's the simplest vast.ai deployment strategy?

For vast.ai:
- They have "Instance Templates" / "Docker Images"
- The easiest approach: Use their Ubuntu + CUDA template, then run a setup script
- Even easier: Build a Docker image locally, push to Docker Hub or GitHub Container Registry, then reference it on vast.ai

The Docker approach:
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3.10 python3-pip git
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install unsloth trl vllm bitsandbytes transformers datasets peft accelerate
# etc.
```

But Unsloth often has specific install instructions (they recommend their own install script). Let me think...

Actually, for vast.ai the even simpler approach:
1. Use vast.ai's default "PyTorch" template (comes with CUDA + PyTorch pre-installed)
2. Run a `setup.sh` script on instance start that installs the remaining packages
3. This avoids Docker entirely and is more flexible

The user can just select a PyTorch template on vast.ai, paste the setup script, and it works.

Let me give a balanced answer covering both approaches and the 3060 feasibility.

For the 3060 question — let me be precise about speed estimates. The key bottleneck for GRPO is the generation rollout phase (inference), which is memory-bandwidth bound:
- 3090: 936 GB/s
- 3060 12GB: 360 GB/s
- Ratio: 2.6x

For the backward pass (training), it's compute-bound:
- 3090: 35.6 TFLOPS FP16
- 3060: 12.7 TFLOPS FP16  
- Ratio: 2.8x

So overall, the 3060 would be roughly 2.5-3x slower per step. But also, do we have enough VRAM?

Let me recalculate the 3060 VRAM more carefully:

Component | VRAM
Base Model NF4 | 0.85 GB
LoRA BF16 | 0.25 GB
Optimizer 8-bit | 0.50 GB
KV Cache (G=16) | ~3.20 GB
Activations + Buffers | ~1.20 GB
CUDA Overhead | ~1.50 GB
**Total** | **~7.50 GB**

This fits in 12 GB with ~4.5 GB to spare. Actually, wait — the vLLM KV cache might be larger in practice. Let me reconsider.

For Qwen2.5-1.5B:
- 28 layers, 2 KV heads, head_dim = 128
- Per-layer KV per token (FP16): 2 × 2 heads × 128 = 512 elements × 2 bytes = 1,024 bytes
- Per-layer per token: 1,024 bytes
- For 28 layers: 28 × 1,024 = 28,672 bytes per token
- For G=16, L=2048: 16 × 2048 × 28,672 = 939,524,096 bytes ≈ 0.94 GB

So the document's 3.2 GB seems high. Maybe they're using FP32 for KV cache? Let me re-read:
"vLLM KV Cache: 2 × 2 bytes × layers (28) × G (16) × L (2048)"

2 × 2 bytes = 4 bytes per element (FP32?). And they're not accounting for head_dim? Let me try their formula directly:
2 × 2 × 28 × 16 × 2048 = 3,670,016 bytes... that's only

---

