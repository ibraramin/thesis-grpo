## **Advanced Methodological Optimization for Post-Training Small Language Models via Supervised Fine-Tuning and Group Relative Policy Optimization** 

## **1. Introduction and Architectural Context** 

The adaptation of compact neural architectures—specifically within the 1.5-billion parameter regime—for complex, multi-step mathematical computation represents one of the most demanding frontiers in contemporary natural language processing. The objective of yielding a statistically significant cohort that provides measurable benchmark improvements over baseline control models necessitates an orchestration of precise data curation, topological initialization, and stabilized policy gradients. Group Relative Policy Optimization (GRPO) has recently emerged as a highly efficient alternative to Proximal Policy Optimization (PPO), primarily because it eliminates the massive memory overhead associated with maintaining a distinct value network for advantage estimation. Instead, GRPO computes baselines directly from a group of rollout trajectories generated from the same prompt, making it exceptionally well-suited for localized hardware environments and open-source models like Qwen2.5-1.5B. 

However, the application of GRPO and Reinforcement Learning with Verifiable Rewards (RLVR) to small language models is frequently bottlenecked by reward sparsity, dataset capability mismatches, and representational collapse. This comprehensive report investigates the pathological failure modes observed within the proposed experimental methodology—most notably an extreme suppression of correct outputs (a 2% pass rate) during the zero-filter evaluation stage. By rigorously evaluating the structural interventions proposed, including strategic dataset substitution, sample volume scaling, and the theoretical necessity of Supervised Fine-Tuning (SFT) initialization, this document establishes a robust, highly optimized methodological framework. Furthermore, advanced algorithmic stabilizers, including Probability Smoothing Policy Optimization (PSPO) and Dynamic Difficulty Scalarization (DW-GRPO), are introduced to ensure sustainable, monotonic convergence in small-scale models. 

## **2. Diagnostic Analysis of the Zero-Filter Pathology and Gradient Starvation** 

## **2.1 Capability Mismatch and the 2% Pass-Rate Phenomenon** 

In the preliminary evaluation stages of the specified methodology, the observation that the Qwen2.5-1.5B model successfully resolves only 2% of the queries during the zero-filter step requires rigorous diagnostic separation. The zero-filter phase isolates non-all-zero samples—defined as queries where the model generates at least one correct mathematical trajectory within its sampled rollout group before the reinforcement learning updates begin. A 2% yield strongly indicates that out of a 2,500-sample corpus, only 50 prompts elicit a usable, non-zero gradient signal for subsequent policy optimization. 

The immediate diagnostic question is whether this extreme suppression is indicative of a fundamental flaw in the coding pipeline, such as a malfunctioning reward parser or a misaligned chat template, or if it is a natural statistical outcome of the experimental parameters. Assuming that the generation sequence syntax and the regular expression (Regex) reward parsing algorithms are executing correctly, a 2% 

baseline pass rate is rarely a programmatic error; rather, it is a textbook manifestation of a severe capability mismatch between the parameter scale of the model and the mathematical difficulty distribution of the chosen dataset. 

The baseline pipeline utilized the SynthLabsAI/Big-Math-RL-Verified dataset. This corpus comprises over 250,000 highly verified, intensely difficult mathematical open-ended problems, specifically engineered to push the reasoning boundaries of frontier-class large language models. The lineage of Big-Math-RL-Verified incorporates advanced competitive mathematical derivations sourced from benchmarks like Omni-MATH and heavily filtered Olympiad-level problems. A 1.5-billion parameter model, constrained by its limited latent dimensional space, possesses insufficient representational capacity to spontaneously deduce complex algebraic, combinatorial, or topological proofs without extensive prior optimization and localized domain adaptation. 

When exposed to the extreme complexity of Big-Math-RL-Verified, the unaligned model's policy distribution fails to randomly generate the correct terminal states. The resulting high-entropy token sampling wanders off-target, yielding an incorrect final string and consequently producing an all-zero reward vector for the entire rollout group. 

## **2.2 The Mechanics of Advantage Computation in All-Zero Scenarios** 

To understand why this 2% pass rate is catastrophic for the proposed research objective, one must examine the fundamental mathematics of Group Relative Policy Optimization. In standard PPO, a value network estimates the expected return of a state, and the advantage is calculated as the difference between the actual reward and this baseline. GRPO bypasses the value network entirely. Instead, for a given prompt $q$, the policy generates a group of $G$ completions $\{o_1, o_2, \dots, o_G\}$. The advantage $A_i$ for each completion $o_i$ is computed by normalizing the rewards within that specific rollout group: 

$$A_i = \frac{r_i - \mu_R}{\sigma_R}$$ where $\mu_R$ is the mean of the rewards in the group, and $\sigma_R$ is the standard deviation. 

If the base model is completely incapable of solving the problem, every single response in a rollout group of size $G=16$ scores a baseline format reward or a pure $0.0$ for correctness. Because there is no variance within the group ($\sigma_R = 0$), the advantage estimation flattens to zero. When the advantage is zero, the policy gradient update is zero. The optimizer is effectively starved of directional signals, and no topological adaptation occurs within the neural weights. Thus, the 2% metric indicates that for 98% of the dataset, the training loop is expending valuable computational cycles executing forward and backward passes that result in absolutely no learning. This is a mathematically expected outcome when applying a compact, under-initialized model to an advanced frontier dataset. 

## **2.4 The Principle of Optimal Difficulty and Learnable Percentage** 

Recent empirical literature on reinforcement learning for language models, specifically the findings regarding "Hard Examples Are All You Need," demonstrates that training on highly difficult examples yields substantial performance gains on reasoning tasks, sometimes upwards of 47%. The underlying mechanism is that complex examples provide dense, multi-step learning opportunities that force the policy to squeeze incorrect logic branches and deepen correct node visitations. 

However, this paradigm carries a critical, non-negotiable prerequisite: the model must be capable of generating the correct answer at least once during its exploratory rollouts. The "learnable percentage" metric defines the subset of hard examples where the base model struggles significantly but occasionally succeeds. If a problem is so opaque that the empirical pass rate is zero, the example falls outside the model's learnable boundary, and the model derives no actionable gradient. The optimal optimization 

boundary for a 1.5B model lies in the intermediate difficulty zone—where the base model gets the answer wrong frequently, but occasionally stumbles upon the correct logic, allowing the GRPO mechanism to compute a positive advantage and reinforce the successful logic pathway against the rejected sequences. The 2% pass rate on Big-Math-RL-Verified proves the dataset exists entirely outside the model's learnable boundary. 

## **3. Evaluation of Strategic Dataset Substitution** 

The proposed structural modification to pivot away from Big-Math-RL-Verified toward a computationally easier dataset is theoretically sound, empirically supported, and strictly necessary for the convergence of a 1.5B parameter architecture. The initial suggestion to utilize GSM8k (Grade School Math 8K) is an excellent starting point, but the accompanying hesitation regarding its limited size reveals a deep understanding of post-training data requirements. 

## **3.1 The Role of GSM8k in Small Language Model Alignment** 

GSM8k represents a foundational corpus of high-quality, grade-school-level mathematical word problems. Each problem requires multi-step reasoning, basic arithmetic operations, and clear logical transitions. While the difficulty level is trivial for human adults or frontier models, it serves as an optimal structural substrate for compact models. Small models inherently face competing demands during fine-tuning between mathematical accuracy and format adherence. Complex datasets force the model to simultaneously learn advanced domain knowledge and output structuring, creating competing gradient pressures that often exceed the model's representational capacities, leading to representational collapse. GSM8k allows the model to map basic sequential logic trajectories without overwhelming its attention heads with advanced calculus or combinatorial theorems. Empirical studies confirm that fine-tuning a 1.5B model specifically on GSM8k using GRPO can elevate its exact-match accuracy from a baseline of 13% to over 64%, and occasionally up to 80% with extensive reinforcement and verifiable reward integration. Therefore, shifting the difficulty distribution down to the GSM8k level is the correct tactical decision to solve the zero-filter gradient starvation problem. 

## **3.2 Overcoming Volume Limitations via Synthetic Expansion** 

The apprehension that GSM8k is "too small" is highly accurate. The base GSM8k training split contains approximately 7,500 to 8,000 problems. While this volume was historically sufficient for Supervised Fine-Tuning (SFT), modern reinforcement learning algorithms, particularly GRPO, require massive exposure to diverse problem permutations to prevent catastrophic overfitting. If a model optimizes over the same 8,000 problems for hundreds of epochs, it simply memorizes the specific numerical answers rather than learning the generalized logic paths, resulting in a collapse of out-of-distribution validation performance. 

To resolve the volume limitations of the standard GSM8k distribution without sacrificing the optimal difficulty level, the methodology must ingest massively expanded, synthetically derived counterparts. The open-source community has recently generated extensive mathematical datasets by distilling the reasoning traces of frontier models (such as GPT-4, DeepSeek-R1, and Qwen-Max) onto the structural foundation of basic math problems. Table 1 details the optimal dataset matrices for small-scale mathematical policy optimization. 

|**Dataset Identifier**|**Volume**||**Composition and**<br>**Origin**|**Optimization**<br>**Utility**|**Source**|
|---|---|---|---|---|---|
|GSM8k**(Base)**|8,000||Human-annotated<br>grade-school<br>word problems.|Baseline<br>validation<br>and<br>foundational<br>logic anchoring.||
|TinyGSM|12.3<br>Million||Synthetic variants<br>of<br>GSM8k<br>generated entirely<br>by GPT-3.5.|Massive-scale<br>supervised<br>initialization for<br>1B-class models.||
|OpenMathInstruct-2|14<br>Million||Problem-solution<br>pairs derived from<br>augmenting<br>GSM8k<br>and<br>MATH<br>datasets<br>via<br>Nemotron<br>models.|High-diversity<br>SFT<br>and<br>high-volume<br>GRPO rollouts.||
|OpenR1-Math-220k|220,000||Distilled<br>step-by-step logic<br>traces specifically<br>generated<br>by<br>DeepSeek-R1.|Advanced<br>structural<br>logic<br>mapping<br>and<br>CoT formatting.||



Transitioning the pipeline's dataset from Big-Math-RL-Verified to a curated 50,000-sample subset of OpenMathInstruct-2 or OpenR1-Math-220k will immediately resolve the 2% accuracy bottleneck. The model's baseline capabilities will align with the difficulty of the synthetic grade-school logic, resulting in a substantially higher zero-filter pass rate (likely between 25% and 60%). This ensures the GRPO algorithm has a continuously populated landscape of diverse reward vectors to compute normalized advantages. 

## **4. The Statistical Necessity of Sample Size Scaling** 

The second structural intervention proposed in the original inquiry suggests increasing the available dataset samples from the initial 2,500 up to 30,000 or even 50,000 instances. The rationale that 2,500 

samples yielding roughly 600 usable zero-filter samples is "barely enough" is a severe understatement. Operating a full-scale policy optimization algorithm on 600 samples is mathematically unviable and guarantees methodological failure. 

## **4.1 Batch Dynamics and Policy Gradient Variance** 

Reinforcement learning algorithms operate through stochastic gradient descent over vast multidimensional policy landscapes. Unlike Supervised Fine-Tuning, where a model is presented with a deterministic cross-entropy target, GRPO requires the model to probabilistically explore token space, evaluate the generated string against a non-differentiable reward function, and then adjust its token probabilities accordingly. This process is inherently noisy and suffers from high variance. 

To stabilize this variance, policy gradient methods rely on the law of large numbers, which necessitates processing large batches of continuous, novel experiences. In a standard hardware-optimized GRPO configuration for a 1.5B model, the global rollout batch size frequently ranges from 128 to 512, with a micro-batch size of 4 to 8 per GPU. If the pipeline utilizes an effective batch size of 128, a dataset containing only 600 usable samples would be exhausted in fewer than 5 optimization steps per epoch. Because learning rates in GRPO are kept intentionally microscopic (e.g., $1 \times 10^{-6}$ to $5 \times 10^{-6}$) to prevent the Kullback-Leibler (KL) divergence from exploding and destroying the pre-trained linguistic priors, 5 optimization steps are entirely insufficient to induce a meaningful topological shift in the model's weights. The Unsloth GRPO framework documentation explicitly warns that models require an absolute minimum of 300 to 1,000 continuous training steps before the reward curves begin to show any statistical separation from baseline noise. 

## **4.2 Scaling to 50,000 Instances for Continuous Episodic Training** 

Increasing the sample volume to 30,000 or 50,000 instances is technically sound and strictly necessary to achieve the stated objective. By processing 50,000 instances, the optimizer can run for thousands of episodes without forcing the model to repeatedly overfit to the same narrow set of problem states. 

When processing a 50,000-sample corpus, the model will generate multiple rollouts per prompt (e.g., $G=8$ or $G=16$). This means the reward parser will evaluate between 400,000 and 800,000 unique reasoning trajectories over the course of training. This massive volume of trial-and-error allows the gradients to organically smooth out the noise of anomalous token selections, slowly shaping the Low-Rank Adaptation (LoRA) matrices toward generalized mathematical deduction rather than memorized pattern matching. 

Furthermore, larger sample sizes enable the implementation of curriculum learning protocols. The data can be dynamically queued based on difficulty, gradually transitioning from simple GSM8k style questions into slightly more complex subsets of OpenMathInstruct-2 as the model's rolling average reward increases, ensuring the optimization remains continually within the learnable boundary. 

## **5. The Ontological Imperative of Supervised Fine-Tuning (SFT)** 

A pivotal misconception presented in the original inquiry is the feeling that executing Supervised Fine-Tuning (SFT) on a different dataset prior to the GRPO step is "awfully redundant" and adds unnecessary complexity. Eliminating the SFT phase based on this assumption represents a catastrophic structural risk to the pipeline and directly contravenes the established ontological progression of modern language model alignment. 

The transition of language models from mere statistical text predictors to advanced deductive reasoners relies on a strict sequential paradigm: pre-training establishes vast linguistic memorization, Supervised 

Fine-Tuning instills behavioral imitation and syntax boundaries, and Reinforcement Learning cultivates robust generalization and exact-match logic extraction. 

## **5.1 The "Cold Start" Pathology in Pure RL Environments** 

The theoretical hypothesis that one can skip SFT and apply policy gradients directly to a base model was aggressively tested by DeepSeek during the development of their frontier reasoning models. The experimental architecture known as DeepSeek-R1-Zero was trained by applying massive-scale reinforcement learning directly to a base model without any SFT data. The objective was to see if Chain-of-Thought (CoT) reasoning and self-verification could emerge purely from reward maximization. While DeepSeek-R1-Zero eventually forced the emergence of step-by-step extrapolation, the resulting model suffered from severe, crippling pathologies. Because the base model had no prior imitation-based anchor for _how_ to format its thoughts, it exhibited endless repetitive loops, severe language mixing (e.g., swapping between English, Chinese, and Python syntax mid-sentence), and exceptionally poor readability. The RL algorithm effectively hacked the correctness reward without any regard for human interpretability. 

To correct these terminal issues, the finalized and widely successful DeepSeek-R1 architecture explicitly incorporated a preliminary "Cold Start" phase. Unlike the zero-version, DeepSeek-R1 utilized a small, highly curated dataset of thousands of long Chain-of-Thought SFT examples to fine-tune the model as the initial RL actor. This cold start data prevented the early, unstable phase of RL training, teaching the model the exact structural template it was expected to follow before the multi-objective RL rewards began altering its probabilistic distributions. 

## **5.2 Formatting Collapse in Small Parameter Regimes** 

For a 1.5B parameter architecture like Qwen2.5-1.5B, this preliminary SFT initialization is exponentially more critical than it is for a 671B frontier model. Modern reasoning frameworks enforce strict XML-like structural formatting—specifically requiring the model to wrap its internal derivations within <think> and </think> tags, and its final extracted answer within <answer> and </answer> tags. 

The SFT phase is solely responsible for teaching the model these specific syntactic boundaries. If GRPO is applied to a completely unaligned base model without this topological anchor, the model generates unstructured prose. The multi-objective reward function in GRPO evaluates both mathematical correctness ($R_{correct}$) and formatting compliance ($R_{format}$). When the unaligned model fails to use the XML tags, the formatting reward penalizes the output heavily. This introduces chaotic, high-variance gradient noise. The model becomes trapped in a destabilized state where it cannot successfully explore logic paths because it is constantly being heavily penalized for syntactic failures it does not understand. 

Therefore, utilizing a highly structured dataset like OpenR1-Math-220k for an initial SFT pass is not redundant; it is the foundational prerequisite that anchors the model's linguistic geometry. It allows the subsequent GRPO phase to focus exclusively on optimizing the latent mathematical reasoning paths rather than struggling to teach the model basic output formatting simultaneously. 

## **6. Advanced Algorithmic Stabilizers for Small Parameter Regimes** 

To maximize the efficacy of the 1.5B parameter architecture during the GRPO phase and guarantee benchmark improvement, the underlying optimization algorithms must be augmented to handle the specific vulnerabilities of small models, namely reward sparsity, competing objective collapse, and 

gradient instability. Simply running standard GRPO on a 1.5B model is highly prone to stalling; advanced stabilizers are strictly required. 

## **6.1 Probability Smoothing Policy Optimization (PSPO)** 

Mainstream policy gradient methods, including PPO and standard GRPO, utilize empirical ratio clipping mechanisms to bound the trust region of the policy update. This prevents the new policy from diverging too radically from the reference policy in a single step. However, in mathematical reasoning environments defined by long-horizon generation and sparse, binary rewards, hard clipping can aggressively discard valuable gradient directionality, leading to stalled convergence and plateauing accuracy. 

Probability Smoothing Policy Optimization (PSPO) serves as a critical, state-of-the-art enhancement to the traditional GRPO framework. Instead of relying exclusively on rigid, hard clipping of the probability ratios, PSPO introduces an experience-weighted reward smoothing mechanism. It tracks group-level reward statistics using exponential moving averages across historical training steps, allowing the model to capture historical reward trends in a lightweight, memory-efficient manner without storing entire rollout trajectories. 

By replacing hard clipping with a soft trust region, PSPO preserves the subtle gradient signals in environments where correct mathematical derivations are sparse. The mathematical formulation transitions the objective function toward a smoothed, potential-shaped optimization space. This allows the compact network to ingest inexpensive internal signals—such as token embeddings, attention entropy, and policy entropy—producing adaptive shaping terms without requiring the massive memory overhead of an auxiliary value network. Empirical evaluations demonstrate that deploying PSPO on mathematical benchmarks consistently accelerates convergence and has been shown to elevate metric accuracy on datasets like GSM8k by upwards of 20 percentage points under matched RL budgets when compared to baseline GRPO. 

## **6.2 Dynamic Weighting and Difficulty Scalarization (DW-GRPO)** 

A standard multi-objective reward function utilizes static coefficients, treating all prompts equally regardless of their inherent computational complexity. In small-scale models, this creates static biases in token and group-level advantage assignments, often over-penalizing the model for failing on exceptionally hard prompts or over-rewarding it for trivial successes. Difficulty-Aware Reweighting Policy Optimization (DARO), an implementation of the broader DW-GRPO (Dynamic Weighting GRPO) paradigm, dynamically adjusts these reward contributions during training to correct static biases. 

The DARO objective function is generalized to incorporate trainable group weights for each empirical pass-rate bucket $\mu$, resulting in the regularized multitask objective: 

$$\mathcal{L}_{DARO} = \sum_{\mu} \left[ w_\mu \mathcal{L}_\mu - \ln w_\mu \right]$$ where $w_\mu$ represents the dynamic, self-adjusting difficulty weight. 

Furthermore, DW-GRPO actively scales the correctness and formatting rewards based on localized prompt difficulty. The reward parameterization transitions from static to dynamic: 

$$R(q, o_i) = \lambda_{correct}(q) \cdot r_{correct}(q, o_i) + \lambda_{format}(\mathcal{H}) \cdot r_{format}(o_i)$$ 

Here, $\lambda_{correct}(q)$ is dynamically magnified when a prompt yields diverse, high-variance trajectories within a rollout group. This forces the policy to aggressively exploit the rare correct logic paths on borderline-difficult queries. Simultaneously, $\lambda_{format}(\mathcal{H})$ acts as an entropy-guided credit shaping tool. It enforces strict structural compliance early in the training trajectory, but automatically decays its influence as the token distribution entropy ($\mathcal{H}$) drops, indicating that the model has successfully internalized the <think> and <answer> syntax. This dynamic reweighting 

substantially mitigates the competing pressures that typically cause representational collapse in 1.5B architectures, allowing the model to transition seamlessly from learning formatting to mastering mathematics. 

## **6.3 Reward Function Engineering and Precision Extraction** 

The structural integrity of the GRPO phase is entirely dependent on the flawless execution of the computational reward functions. Relying on simple string matching to determine if an answer is correct is highly susceptible to false negatives due to minor formatting variations, such as trailing commas, floating-point representations, currency symbols, or unit inclusions. 

The correctness reward ($r_{correct}$) must execute a rigorous mathematical normalization protocol. This involves symbolic mathematical parsing utilizing extended libraries (such as customized sympy distributions) capable of managing scientific notation and algebraic equivalencies without arbitrary failure. The extraction layer leverages highly specific regular expressions (Regex) to isolate the terminal output bounded by syntax markers. Once extracted, the numerical values must be subjected to precision comparisons using an acceptable floating-point tolerance threshold (e.g., $1 \times 10^{-3}$) to accommodate the inherent rounding derivations generated by language models. A binary scoring system awards $1.0$ for precision matches and $0.0$ for failures. 

Concurrently, the soft format reward ($r_{format}$) must verify structural topology. Rather than a binary pass/fail, it computes fractional credit for the presence of required delimiters. For example, the function scans the generated text and adds fractional points (e.g., $+0.125$) for the presence of <reasoning>, </reasoning>, and <answer> tags, while subtracting fractional penalties for malformed or duplicated tags. Setting the global formatting weight to an optimal scalar relative to correctness (e.g., $0.2 \cdot R_{format} + 1.0 \cdot R_{correct}$) ensures that the model is continuously guided toward readability without overriding the primary objective of logical accuracy. 

## **7. Parameter-Efficient Fine-Tuning (PEFT) and Hardware Architecture** 

Executing this pipeline on consumer-grade hardware (such as a single NVIDIA RTX 3090 with 24GB VRAM) necessitates extreme memory optimizations. The baseline weights of Qwen2.5-1.5B must be instantiated in 4-bit NormalFloat (NF4) double quantization via bitsandbytes, while parameter updates are routed exclusively through Low-Rank Adaptation (LoRA) matrices. 

A critical vulnerability unique to fine-tuning specialized mathematical models lies in adapter rank parameterization. While generalized open-source models often exhibit monotonic reward growth with massive high-rank adapters (e.g., $r=256$), specialized foundational models, specifically Qwen2.5-Math-1.5B, suffer from acute "specialist instability" at high ranks. RL updates processed at $r=256$ actively conflict with the highly rigid, pre-optimized manifold of these specialized models, resulting in catastrophic performance collapse where the model actively loses its mathematical capabilities. 

Therefore, a targeted, highly constrained hyperparameter configuration is required. Table 2 outlines the optimal bounds for the 1.5B architecture to ensure stable gradient flow without triggering VRAM limits or topological collapse. 

||**Hyperparameter**<br>**Optimal Target**<br>**Value**<br>**Theoretical Justification and**<br>**System Impact**<br>**Source**<br>**LoRA Rank ($r$)**<br>32 to 64<br>Provides<br>sufficient<br>representational capacity for new<br>reasoning paths without causing<br>catastrophic specialist instability.<br>**LoRA**<br>**Alpha**<br>**($\alpha$)**<br>$2 \times r$ (64<br>to 128)<br>Scales the adapter activation<br>appropriately to maintain stable<br>weight<br>magnitudes<br>during<br>gradient descent.<br>**Target Modules**<br>All<br>Linear<br>Projections<br>Ensuringq_proj,k_proj,v_proj,<br>gate_proj,<br>and<br>up_proj<br>are<br>targeted<br>maximizes<br>the<br>cross-attention<br>influence<br>on<br>reasoning traces.<br>**Learning Rate**<br>$1<br>\times<br>10^{-6}$ to $5<br>\times 10^{-6}$ Utilizing a microscopic rate with<br>cosine decay mitigates sudden<br>KL divergence explosions that<br>destroy base model coherence.<br>**Group Size ($G$)**<br>8 to 16<br>Ensures<br>sufficient<br>trajectory<br>variance for relative advantage<br>computation within strict 24GB<br>memory limits.|
|---|---|



|**Context Length**|2048 to 3584|Accommodates<br>lengthy,<br>step-by-step logic trajectories and<br>zero-shot<br>scratchpads<br>without<br>triggering<br>Out-Of-Memory<br>(OOM) errors.||
|---|---|---|---|



To achieve these parameters within localized hardware, frameworks like Unsloth are mandatory. Unsloth natively chunks the sequence dimension during fused cross-entropy computations, bypassing the materialization of massive FP32 logits tensors and reducing the memory footprint of the backward pass by over 90%. Integrating this optimization with vLLM generation offloads allows the simultaneous, high-speed execution of policy generation and gradient updates, ensuring the 50,000-sample dataset can be processed efficiently. 

## **8. Revised Methodological Framework and Execution Pipeline** 

Synthesizing the extensive diagnostic findings, the validation of dataset scaling, and the integration of advanced algorithmic stabilizers, the following structured blueprint replaces the highly vulnerable baseline pipeline. This revised architecture aligns precisely with the objective of yielding significant cohort improvements while remaining mathematically sound and hardware-efficient. 

## **Phase I: Foundational Topology Initialization (The SFT Cold Start)** 

The proposal to eradicate the SFT phase is discarded. Instead, the SFT phase is heavily optimized to establish a dense, reliable behavioral imitation matrix. 

1. **Corpus Selection and Pre-Processing:** The initialization completely abandons the frontier-level Big-Math-RL-Verified dataset. Instead, it utilizes a curated 10,000-sample subset of OpenR1-Math-220k combined with data from OpenMathInstruct-2. This exposes the untrained model to thousands of robust, correct, and completely resolved step-by-step logic traces. 

2. **Formatting Imprint:** The instruction tuning templates rigorously enforce the exact XML syntax required for subsequent evaluation: wrapping internal thoughts within <think> and </think>, and terminal solutions within <answer> and </answer>. 

3. **Optimization Protocol:** Utilizing standard causal next-token prediction, the model is trained with an effective batch size of 64 to 128 for 2 complete epochs. This phase effectively cures the "cold start" sparsity problem. The model's baseline probability distribution is permanently altered to favor multi-step deductive structures over conversational responses. 

## **Phase II: Difficulty-Calibrated Policy Optimization (GRPO with PSPO)** 

With the model now intrinsically capable of generating structured reasoning formats, the reinforcement learning algorithm is deployed to optimize accuracy. The original 2,500 sample limit is discarded to prevent high-variance gradient starvation. 

1. **Scaling the Trajectory Pool:** A secondary, massive subset of 50,000 problems from synthetic grade-school corpora (TinyGSM or OpenMathInstruct-2) is deployed. This establishes an environment where the zero-filter pass rate will natively sit between 30% and 50%, guaranteeing continuous gradient flow. 

2. **Algorithmic Execution:** 

- **Generation Constraints:** For each mathematical prompt, the model generates a rollout group of $G=16$ sequences. Generation temperatures are kept relatively high (e.g., $T=0.9$) to encourage logic path exploration. 

- **Reward Extraction and Normalization:** The symbolic Regex parser processes all 16 terminal outputs. Correctness is scored strictly via precision floating-point matching ($r=1.0$ or $r=0.0$). Formatting compliance is awarded fractional structural scores. 

- **Advantage Computation (PSPO):** The standard GRPO clipping is overridden by Probability Smoothing Policy Optimization (PSPO). PSPO calculates the group-relative advantage utilizing an exponential moving average, discarding rigid bounds to capture long-term reward trends and preserve subtle gradient directionality in sparse spaces. 

- **Dynamic Scalarization (DW-GRPO):** The system continuously monitors the sequence entropy. The dynamic weights ($w_\mu$) automatically scale up the advantage signal for prompts that trigger high variance, squeezing maximum optimization value from borderline-hard examples without starving the gradients. 

- **Gradient Application:** The LoRA adapters ($r=64$, $\alpha=128$) are updated via the 8-bit AdamW optimizer with a learning rate peaked at $2 \times 10^{-6}$ using a cosine decay schedule. The KL divergence penalty remains active to restrict the policy from collapsing back into linguistic gibberish. 

## **Phase III: Automated Error Routing and Iterative Recycling** 

To further maximize data efficiency and ensure continuous improvement beyond the initial GRPO plateau, a feedback mechanism is implemented for unresolved sequences. 

1. **Hard Failure Isolation:** During Phase II rollouts, prompts that completely fail to generate any reward (an all-zero vector across all $G=16$ trajectories) are actively intercepted and cached. Because they offer zero relative advantage, updating the policy on these sequences introduces pure stochastic noise, which is discarded from the active gradient step. 

2. **Teacher-Model Critique:** The isolated failures are routed to a frontier teacher model (e.g., a 70B parameter network via API). The teacher diagnoses the logical breakdown, rectifies the mathematics, and generates a structured, formatted repair trace. 

3. **SFT Re-insertion:** These newly synthesized correct traces are injected back into the Phase I SFT Acquisition dataset. The 1.5B model is periodically refreshed with this recycled data via a brief SFT epoch, allowing it to rapidly internalize the specific mathematical topologies it was fundamentally incapable of exploring during the reinforcement phase. This creates a highly resilient, closed-loop iterative learning environment. 

## **9. Synthesis and Conclusion** 

The endeavor to optimize compact 1.5-billion parameter language models for rigorous, step-by-step mathematical computation exposes the fragile and highly complex interplay between neural capacity, dataset difficulty, and algorithmic stability. The severe suppression of accuracy observed during early baseline filtering is not a technical failure of execution, but an expected symptom of profound capability mismatch, critically exacerbated by an insufficient empirical sample size that starves the gradient estimator of variance. 

The analytical findings definitively validate the proposed structural changes: shifting the corpus distribution downward toward environments like GSM8k and its synthetic variants (OpenMathInstruct-2), and aggressively scaling the sample size beyond 50,000 instances to secure dense, 

continuous gradient diversity. Furthermore, eradicating the Supervised Fine-Tuning phase is strongly contraindicated; the SFT "cold start" is an absolute ontological necessity to prevent structural generation collapse and endless repetition loops during policy optimization. 

Because the application of standard GRPO is often too rigid for the high-variance, sparse-reward environment of small models, integrating Probability Smoothing Policy Optimization (PSPO) and Dynamic Difficulty Scalarization (DW-GRPO) provides the necessary algorithmic elasticity. By executing this multi-stage, hardware-optimized paradigm, the latent deductive capabilities of a 1.5-billion parameter architecture can be systematically mapped, structured, and refined, yielding highly verifiable improvements across evaluation benchmarks without succumbing to gradient starvation or format instability. 

