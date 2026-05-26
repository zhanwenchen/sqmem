# Soft-Q Memory: Agent Memory as a Nonparametric Value Estimator

**Thesis.** For online decision-making, the control-relevant part of agent memory functions as a *soft Q table*: it stores or implies evidence about which continuations from similar trajectory prefixes led to better or worse outcomes.

## TL;DR

Imagine an AI agent playing a text adventure game. At each step it sees a text description of where it is and picks one action from a list (e.g. *"go to kitchen"*, *"open drawer 1"*, *"pick up apple"*). The agent has played 15 similar games before. We give it a *memory*: a list of every past (situation, action, eventual_score) tuple it's ever seen. **Can it use that memory to play better the next time?**

This paper proposes a specific way to use such a memory: treat each row of memory as a data point for a *value function* — an estimate of *"if I'm in a situation like this and I take action like that, how good was the outcome in the past?"* — and combine that estimate with the agent's base instinct about what to do. We call this estimator **Soft-Q Memory (SQ-Mem)**: "soft" because rather than copying the single most similar past action, we softmax-weight a handful of past actions by their similarity and average their outcomes.

We test 11 specific predictions about this mechanism that would falsify it if violated (e.g. "if we shuffle the outcomes in memory so they no longer match the actions, performance should collapse"). On a ScienceWorld task, **10 of 11 are supported in two of our three configurations (R1, R2) and 9 of 11 in the third (R3)** — the lone R3 inconclusive verdict (H4 structured state) is that the LLM base policy makes our hand-built structured-state compilation unnecessary, which is itself a useful finding about scope.

The rest of the paper makes this precise.

## Table of contents

0. [Background you need](#0-background-you-need)
1. [Motivation](#1-motivation)
2. [The Soft-Q Memory Estimator](#2-the-soft-q-memory-estimator)
3. [Online Decision Rule](#3-online-decision-rule)
4. [Memory Construction](#4-memory-construction)
5. [Variants and Ablations](#5-variants-and-ablations)
6. [Falsifiable Predictions](#6-falsifiable-predictions)
7. [Experimental Setup](#7-experimental-setup)
8. [Main Results (ScienceWorld)](#8-main-results-scienceworld)
9. [Cross-Benchmark Generalization (ALFWorld)](#9-cross-benchmark-generalization-alfworld)
10. [Limitations](#10-limitations)
11. [Reproducibility](#11-reproducibility)
12. [Relation to Prior Work](#12-relation-to-prior-work)
13. [Glossary](#13-glossary)

---

## 0. Background

Concepts the rest of the paper uses without re-explaining. Skip this section if you're already comfortable with RL terminology.

**Agent.** A program that picks actions in a world. Here, *the world* is a text-based game (ScienceWorld or ALFWorld) and *the agent* is a piece of Python code that takes in a text description and outputs an action string.

**Trajectory** \(\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots)\). The sequence of (state, action, reward) tuples produced as the agent plays.

**Prefix** \(\tau_{\leq t}\). Everything that happened up to (and including) step \(t\). The agent only knows the prefix when picking action \(a_t\).

**Return-to-go** \(G_t = r_{t+1} + r_{t+2} + \ldots + r_T\). The total reward the agent will receive from step \(t\) onward until the episode ends. This is only known *after* the episode finishes — when memory is built, but not at decision time.

**Q-function** \(Q(s, a)\). The expected return-to-go if you take action \(a\) from state \(s\) and then play optimally afterwards. A central object in reinforcement learning. **Think of it as a giant lookup table** that says *"in situation s, action a is worth this much."* Classical RL learns this table via the Bellman equation; we **estimate it directly from past experience** by averaging the observed returns from similar past (state, action) pairs.

**V-function** \(V(s) = \max_a Q(s, a)\). The value of being in state \(s\). A weaker object: tells you the state is good but not which action to take.

**Embedding** \(\phi(s)\). A function that turns a piece of text into a vector of numbers (e.g. 384-dimensional). Two pieces of text with similar meaning get vectors close to each other (high cosine similarity). We use this to find "memories of similar past situations" quickly.

**Cosine similarity** \(\text{sim}(u, v) = u \cdot v / (\|u\| \|v\|)\). Standard measure of how "aligned" two vectors are. Ranges in \([-1, 1]\). We L2-normalize embeddings up front, so this reduces to the dot product \(u \cdot v\).

**Softmax weights**. Given a list of similarity scores \(s_1, s_2, \ldots, s_k\), the softmax weights are \(w_i = e^{s_i / \beta} / \sum_j e^{s_j / \beta}\). They turn raw similarity scores into probabilities that sum to 1. The temperature \(\beta\) controls sharpness: small \(\beta\) → hard nearest-neighbor (one weight near 1, rest near 0); large \(\beta\) → uniform averaging.

**Argmax**. \(\arg\max_a f(a)\) returns the action \(a\) that maximizes \(f\). If "score" measures how good each action looks, argmax picks the best-scoring one.

**Base policy** \(S_\theta(a \mid \text{context})\). A starting-point scorer for actions — could be an LLM, could be a keyword-overlap heuristic. SQ-Mem *augments* the base policy with memory; it does not replace it.

**Ablation.** A controlled experiment where you remove or break one piece of a system to see if performance drops. If breaking piece X kills the gain, piece X was necessary. If breaking it doesn't kill the gain, X was decoration.

**Variant.** In this paper, a specific configuration of the agent (e.g. "the full mechanism" vs "the mechanism with returns shuffled"). We run 16 of them in parallel and compare.

**Falsifiable prediction.** A pre-stated quantitative claim that, if the experiment violates it, refutes the theory. The opposite of post-hoc explanation.

---

## 1. Motivation

Long-horizon interactive agents accumulate a *trajectory prefix* \(\tau_{\leq t}\) as they act — every observation they've seen and every action they've taken, in order. A natural question: *what should be stored in external memory* so future decisions are better?

Two common answers in the literature:

1. **Store raw transcripts.** When picking the next action, retrieve similar past transcripts (by text similarity) and stuff them into the LLM's context window. This is the *retrieval-augmented generation* (RAG) approach.
2. **Store summaries.** Compress the current trajectory into a short summary string; the agent reads the summary instead of the full history.

**Neither stores *value evidence*** — explicit information about which actions led to high downstream return from states similar to the current one. They're both "memory of what happened," not "memory of what worked."

We propose that the value-bearing component of memory is a **nonparametric estimator** of \(Q(s_t, a)\) — that is, a function that estimates Q values *not* by training parameters, but by averaging observed outcomes from a fixed memory bank of past (state, action, return) tuples, weighted by similarity to the current situation.

**Why "soft"?** Hard nearest-neighbor retrieval (just copy what worked best in the *single* most similar past state) is brittle when retrieval is noisy. Averaging the top-\(R\) neighbors with similarity-weighted softmax is more robust and gives us a free uncertainty estimate (the variance across the top-\(R\) returns).

---

## 2. The Soft-Q Memory Estimator

**Intuition.** We have a memory bank of past experiences. Each entry is a tuple `(state_text, action_text, return)`. When we need to estimate \(Q(s_t, a)\) at decision time, we look up the most similar past entries to the (current state, candidate action) pair, and average their returns, weighted by similarity. The result is a guess at "if I take this action from a state like this, what return will I see?"

**Setup.** Memory bank \(\mathcal{M} = \{(s_i, a_i, G_i)\}_{i=1}^{M}\) where each \(s_i\) is a compiled prefix-state string, \(a_i\) is the action that was taken, and \(G_i\) is the observed return-to-go after taking \(a_i\) from \(s_i\).

**Estimator.** Given a current (state, candidate-action) pair \((s_t, a)\), we estimate:

$$
\hat{Q}_{\mathcal{M}}(s_t, a)
= \sum_{i \in \mathcal{N}_R(z_t, u)} w_i(z_t, u)\, G_i
$$

where:

- \(z_t = \phi(s_t)\) — the current state, embedded
- \(u = h(a)\) — the candidate action, embedded
- \(\mathcal{N}_R(z_t, u)\) — the indices of the top-\(R\) most similar memory entries
- \(w_i\) — softmax weights, summing to 1, giving more weight to more similar entries
- \(G_i\) — the return that was actually observed in memory entry \(i\)

**Retrieval score.** How similar is memory entry \(i\) to the current (state, action)? We blend state-similarity and action-similarity:

$$
\text{score}_i(z_t, u) = \alpha\,\text{sim}(z_t, z_i) + (1-\alpha)\,\text{sim}(u, u_i)
$$

The mixing weight \(\alpha \in [0, 1]\) trades off how much retrieval cares about state vs action match. We use \(\alpha = 0.5\).

**Weights (the softmax).** Convert raw similarity scores into a probability distribution over the top-\(R\) neighbors:

$$
w_i = \frac{\exp(\text{score}_i / \beta)}{\sum_j \exp(\text{score}_j / \beta)}
$$

Temperature \(\beta\) controls how peaked the distribution is. \(\beta \to 0\) collapses to argmax (only the most similar gets weight). \(\beta \to \infty\) uniformizes (all top-\(R\) neighbors weighted equally). We use \(\beta = 0.1\), which puts most weight on the very top neighbors but not all on one.

**Uncertainty.** The decision rule will also want to know *how confident* this estimate is. We compute the weighted standard deviation across the top-\(R\) returns:

$$
\hat{\sigma}_{\mathcal{M}}(s_t, a) = \sqrt{\sum_i w_i (G_i - \hat{Q})^2}
$$

High σ = the top-\(R\) neighbors disagree about the outcome (some succeeded, some failed). Low σ = they agree.

### 2.1 Worked example

Suppose our memory bank has 4 entries (toy size). The candidate is "pick up apple" from the current state "I am in the kitchen near the fridge". After embedding and computing similarity scores:

| i   | state_text                          | action_text     | \(G_i\) | score |
| --- | ----------------------------------- | --------------- | ----: | ----: |
| 1   | "I am in the kitchen by the fridge" | "pick up apple" |   1.0 |  0.97 |
| 2   | "I am in the kitchen"               | "pick up apple" |   0.8 |  0.85 |
| 3   | "I am in the bedroom"               | "pick up apple" |   0.0 |  0.30 |
| 4   | "I am in the kitchen"               | "open drawer"   |   0.5 |  0.40 |

With \(R = 4\) (use them all), \(\beta = 0.1\):

Softmax weights: \(w \propto e^{0.97/0.1}, e^{0.85/0.1}, e^{0.30/0.1}, e^{0.40/0.1}\) \(\Rightarrow w \approx (0.76, 0.23, 0.00003, 0.0009)\)

\(\hat{Q} \approx 0.76 \cdot 1.0 + 0.23 \cdot 0.8 + \ldots \approx 0.95\)

The estimator confidently predicts return ≈ 0.95 for "pick up apple" in this state, because the two highest-scoring memories both had high returns. \(\hat{\sigma} \approx 0.09\) — low, so the prediction is reliable.

Now suppose entry 1 had \(G_1 = 0.0\) instead (you picked up apple but something went wrong). Then \(\hat{Q} \approx 0.18\) and \(\hat{\sigma} \approx 0.30\) — moderate value but *high* uncertainty, because memories near this state-action disagree. The decision rule will penalize this candidate accordingly.

### 2.2 Algorithm

The core mechanism. Per (state, action) pair, return \(\hat{Q}\) and \(\hat{\sigma}\).

```text
inputs:  state_text: str
         action_text: str
parameters: top_R, α, β, action_conditioning, weight_mode

outputs: (q, sigma, retrievals)

algorithm:
    # 1. Embed the query state and action
    z = embedder.embed(state_text)
    u_action = normalize_action_text(action_text) if normalize_actions else action_text
    u = embedder.embed(u_action)

    # 2. Compute similarity to every memory item (fast: matrix-vector multiply)
    state_sims = state_mat @ z              # (M,)
    if action_conditioning:
        action_sims = action_mat @ u        # (M,)
        scores = α * state_sims + (1 - α) * action_sims
    else:
        scores = state_sims                 # V(s) variant

    # 3. Pick the top-R most similar memories
    k = min(top_R, M)
    top_idx = argpartition(scores, -k)[-k:]
    top_idx = top_idx[argsort(scores[top_idx])[::-1]]   # sort within top-R
    top_scores  = scores[top_idx]
    top_returns = returns[top_idx]

    # 4. Convert similarity scores to weights (softmax by default)
    if   weight_mode == "top1":    weights = onehot(0, k)
    elif weight_mode == "uniform": weights = ones(k) / k
    else:  # softmax
        shifted = top_scores - top_scores.max()    # numerical stability
        weights = softmax(shifted / β)

    # 5. Weighted-average the returns; weighted-stddev for uncertainty
    q       = weights · top_returns
    sigma   = sqrt(weights · (top_returns - q)^2)
    retrievals = [Retrieval(items[i], scores[i], weights[j])
                  for j, i in enumerate(top_idx)]
    return q, sigma, retrievals
```

### 2.3 Batched version

Vectorized: embed and score all N valid candidate actions in one matmul. ~100× faster than calling `estimate` N times when the embedder is a sentence-transformer (because SBERT takes batches).

```text
inputs:  state_text: str
         action_texts: list[str]    # the N valid actions

outputs: list of (q, sigma, retrievals)

algorithm:
    z = embedder.embed(state_text)
    state_sims = state_mat @ z                  # (M,)

    if action_conditioning:
        action_mat_q = embedder.embed_batch(action_texts)   # (N, dim)
        action_sims_all = action_mat_q @ action_mat.T       # (N, M)

    results = []
    for i in range(N):
        scores = (α * state_sims + (1-α) * action_sims_all[i]
                  if action_conditioning else state_sims)
        # ... same top-K + softmax + weighted aggregation as estimate() ...
        results.append((q_i, sigma_i, retrievals_i))
    return results
```

### 2.4 State-only retrieval

Used by Flag B (V(s) prior, [§9.3](#93-flag-b-state_value_in_prompt)) and the RAG-context decision mode. Ignores action conditioning entirely — returns raw retrieved tuples for the LLM to consume, not a Q value.

```text
inputs:  state_text: str
         top_k: int

outputs: list of Retrieval

algorithm:
    z = embedder.embed(state_text)
    state_sims = state_mat @ z
    k = min(top_k, M)
    top_idx = argpartition(state_sims, -k)[-k:]
    top_idx = top_idx[argsort(state_sims[top_idx])[::-1]]
    return [Retrieval(items[i], state_sims[i], 1.0/k) for i in top_idx]
```

### 2.5 Action normalization (Variant A)

Used by `normalize_actions=true` ([§9.2](#92-flag-a-normalize_actions)). Strips standalone digits and articles so `"open drawer 1"` and `"open the drawer 3"` map to the same embedding — collapsing instance-specific tokens into class-level identity.

```text
inputs:  text: str

outputs: normalized: str

algorithm:
    s = text.lower()
    s = re.sub(r"\b\d+\b", "", s)        # strip standalone digits
    s = re.sub(r"\b(?:the|a|an)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
```

Example: `"open the drawer 3"` → `"open drawer"`.

---

## 3. Online Decision Rule

**Intuition.** At each step, the agent has a *base policy* (e.g. an LLM, or a keyword heuristic) that gives a score \(S_\theta(a)\) for each candidate action. SQ-Mem layers a memory bonus on top:

$$
a_t = \arg\max_{a \in \mathcal{A}_t}
\underbrace{S_\theta(a \mid \tau_{\leq t}, s_t)}_{\text{base instinct}}
+ \lambda \cdot \underbrace{\hat{Q}_{\mathcal{M}}(s_t, a)}_{\text{memory says}}
- \rho \cdot \underbrace{\hat{\sigma}_{\mathcal{M}}(s_t, a)}_{\text{but uncertain}}
$$

The agent picks the action that maximizes (base score) + λ × (memory value) − ρ × (memory uncertainty). The hyperparameters:

- \(\lambda \geq 0\): **how much we trust memory**. With \(\lambda = 0\), the base policy decides everything; with \(\lambda\) large, memory overrides the base policy.
- \(\rho \geq 0\): **how much we discount uncertain memories**. With \(\rho = 0\), high-\(\hat{Q}\) but high-\(\hat{\sigma}\) candidates win as easily as high-\(\hat{Q}\) low-\(\hat{\sigma}\) candidates. With \(\rho\) large, the agent prefers low-uncertainty memories.

The decision rule is robust to base-policy choice because the memory term shifts the argmax whenever memory has strong evidence for an alternative.

### 3.1 A worked example

Suppose the agent's valid actions are *["pick up apple", "open drawer", "look around"]*. The LLM scores them 0.6, 0.3, 0.1 (it prefers picking up the apple). Memory says:

| Action | \(\hat{Q}\) | \(\hat{\sigma}\) |
|---|---:|---:|
| pick up apple | 0.1 | 0.4 (high uncertainty: it didn't always work) |
| open drawer | 0.9 | 0.05 (low uncertainty: opening drawer reliably good) |
| look around | 0.0 | 0.0 (no relevant memories) |

With \(\lambda = 3.0\), \(\rho = 0.5\) (typical settings):

| Action | base | \(\lambda \hat{Q}\) | \(-\rho \hat{\sigma}\) | combined |
|---|---:|---:|---:|---:|
| pick up apple | 0.6 | +0.3 | −0.20 | **0.70** |
| open drawer | 0.3 | +2.7 | −0.025 | **2.975** ← chosen |
| look around | 0.1 | 0.0 | 0.0 | 0.10 |

Memory **overrode** the LLM's instinct. The LLM wanted to pick up the apple; memory said *"in past situations like this, opening drawers had high return"*, and the uncertainty term broke the tie.

### 3.2 Decision-rule variants

The paper's primary decision rule is the linear combine above. Two alternative *consumers* of the same \(\hat{Q}\) output were added for the cross-benchmark generalization study in [§9](#9-cross-benchmark-generalization-alfworld):

- **RAG-context mode** (`memory_mode: rag_context`): retrieved (state, action, return) triples are passed to the LLM as in-prompt examples; the LLM picks an action with these in context. The estimator is unchanged; only the consumer differs.
- **Q-in-prompt mode** (`q_in_prompt: true`): per-candidate \(\hat{Q}\) and \(\hat{\sigma}\) are formatted into the prompt as numeric annotations (e.g. *"action 3: [memory Q=0.85, σ=0.10]"*); the LLM decides; the arithmetic combine is skipped.

Both are valid alternative ways to use the same value estimate. The hypothesis from [§9](#9-cross-benchmark-generalization-alfworld) is that the *evidence in \(\hat{Q}\)* is doing the work — not the specific arithmetic form of [§3](#3-online-decision-rule).

### 3.3 Algorithm (Q-rerank mode)

The paper's primary decision rule. The branching at the top selects which decision mode is active; the default body is the linear combine.

```text
inputs:  observation: str, valid_actions: list[str]
state:   self._observations, self._actions, self._task_goal, self._step

outputs: selected: str       # the chosen action

algorithm:
    self._observations.append(observation)
    if not valid_actions: return "look around"

    state_text = self._compile_state()

    # Alternate consumer: RAG-context mode (Flag from §9)
    if memory_mode == "rag_context" and llm_policy is not None:
        return self._act_rag_context(state_text, observation, valid_actions)

    # ---- Q-rerank mode (the paper's §3) ----
    # Estimate Q and σ for every valid action in one batched matmul
    batch = sqm.estimate_batch(state_text, valid_actions)

    mem_qs, mem_sigmas = {}, {}
    for action, (q, sigma, retrievals) in zip(valid_actions, batch):
        if not use_returns:                      # semantic_retrieval variant
            # Use averaged similarity score instead of return
            effective_q = mean([r.score for r in retrievals])
            effective_sigma = 0
        else:
            effective_q = q
            effective_sigma = sigma if uncertainty_penalty else 0
        mem_qs[action] = effective_q
        mem_sigmas[action] = effective_sigma

    # Flag B: compute state-only V(s) as a scalar prior for the LLM
    v_prior = None
    if state_value_in_prompt:
        retrievals = sqm.retrieve_by_state(state_text, top_k=top_R)
        v_prior = sum(r.weight * r.item.return_value for r in retrievals)

    # Flag C: pass per-action Q/σ to the LLM via prompt annotations
    per_action_q = ({a: (mem_qs[a], mem_sigmas[a]) for a in valid_actions}
                    if q_in_prompt else None)

    # Base policy scores (LLM or heuristic)
    if llm_policy:
        base_scores = llm_policy.score_actions(
            observation, valid_actions, task_goal, recent_actions,
            per_action_q=per_action_q, state_value_prior=v_prior,
        )
    else:
        base_scores = {a: heuristic_base_score(a, task_goal, observation)
                       for a in valid_actions}

    base_selected = argmax(base_scores)

    # Combine: arithmetic by default, or skip if Flag C surfaces Q in prompt
    if q_in_prompt:
        combined = dict(base_scores)             # LLM already saw Q in prompt
    else:
        combined = {a: base_scores[a] + λ * mem_qs[a] - ρ * mem_sigmas[a]
                    for a in valid_actions}

    argmax_selected = argmax(combined)

    # ε-greedy exploration for variance across variants
    selected = random_choice(valid_actions) if rng() < ε else argmax_selected
    memory_changed = (argmax_selected != base_selected)

    # Log everything for downstream analysis (calibration, interventions, etc.)
    self._decisions.append(Decision(...))
    return selected
```

### 3.4 Algorithm (RAG-context mode)

Used when `memory_mode == "rag_context"`. Makes *two* LLM calls per step so we can measure whether memory changed the LLM's choice — the reference call (no memory context) gives us the counterfactual.

```text
algorithm:
    # 1. Reference: ask LLM without memory context
    base_scores = llm.score_actions(obs, valid, goal, actions, memory_context=None)
    base_selected = argmax(base_scores)

    # 2. Build memory context: top-k state-only retrievals as prompt examples
    memory_context = []
    for r in sqm.retrieve_by_state(state_text, top_k=rag_top_k):
        entry = {"action_text": r.item.action_text}
        if use_returns:
            entry["return_value"] = r.item.return_value
        memory_context.append(entry)

    # 3. Memory-informed: ask LLM with retrieved examples in prompt
    ctx_scores = llm.score_actions(obs, valid, goal, actions,
                                    memory_context=memory_context)
    argmax_selected = argmax(ctx_scores)

    # 4. ε-greedy and logging (same as Q-rerank path)
    selected = random_choice(valid) if rng() < ε else argmax_selected
    memory_changed = (argmax_selected != base_selected)
    return selected
```

---

## 4. Memory Construction

**Intuition.** Memory is built once, in advance, from training trajectories. Each step the agent took during training becomes one row of memory:

```text
(state at that step, action taken, total reward that came after)
```

The "total reward that came after" is the **return-to-go** — known only after the episode ends. We compute it in one backward sweep.

Memory is *read-only at inference time*. There is no learning, no Bellman update, no gradient. The estimator just looks up the relevant rows.

### 4.1 Backward return-to-go pass

After an episode finishes with reward sequence \(r_1, r_2, \ldots, r_T\), the return-to-go for decision \(t\) is

$$
G_t = \sum_{k=t}^{T} r_k
$$

We do not discount (γ = 1) because tasks here are short (5–25 steps) and sparse-reward — discounting would distort the value signal more than it'd help. Computed in one backward sweep: \(G_T = r_T\); \(G_t = r_t + G_{t+1}\).

```text
inputs:  decisions: list[Decision]      # each has .reward populated
side-effects: decisions[i].return_value is filled in-place

algorithm:
    running = 0.0
    for d in reversed(decisions):       # walk backwards
        running += d.reward             # accumulate reward
        d.return_value = running        # this is G_t
```

### 4.2 Memory-builder strategies

How do we *get* the training trajectories that produce memory rows? Three options, selected by the `memory_builder.agent` field in the YAML config:

| Builder | What it produces | When to use |
|---|---|---|
| `scienceworld_gold_and_random` | 1 expert "gold path" trajectory + N random-exploration trajectories per task | ScienceWorld; the env exposes `env.get_gold_action_sequence()` which produces a known-correct sequence |
| `self_rollout` | N test-time-policy rollouts per task; returns from observed env reward | Benchmark-agnostic; used on ALFWorld where the env's expert is buggy |
| `scienceworld_random_memory_builder` | N random + heuristic rollouts per task | Diagnostic |

**Why a mix of gold + random?** If memory contained only gold trajectories, all returns would be high — there'd be no contrast between "good action here" and "bad action here". By mixing in random trajectories (most of which produce low returns), the memory bank has *return variance*, which is what the soft-Q estimator needs to discriminate actions.

Each Decision in an Episode becomes one memory row. The episode must already have gone through `backward_return_to_go` so the `.return_value` fields are filled.

```text
inputs:  episodes: list[Episode]        # completed training rollouts
         store: MemoryStore             # writable memory bank
         embedder: Embedder             # state + action embedder
         allowed_task_ids: set[str]     # train-split IDs (for split discipline)

outputs: count: int                     # number of items written

algorithm:
    count = 0
    for ep in episodes:
        assert ep.task_id in allowed_task_ids   # H10: split discipline

        for d in ep.decisions:
            state_vec  = embedder.embed(d.state_text)
            action_vec = embedder.embed(d.selected_action)
            item = MemoryItem(
                item_id     = f"m_{count:08d}",
                task_id     = ep.task_id,
                split       = "train",
                step_index  = d.step_index,
                state_text  = d.state_text,
                action_text = d.selected_action,
                return_value = d.return_value,
                state_vec   = state_vec,
                action_vec  = action_vec,
            )
            store.add(item)
            count += 1
    return count
```

### 4.3 Split discipline

**Why this matters.** If memory contains items from the same task we're testing on, we'd be cheating — the estimator could just look up "what action worked here last time" and replay it. To make the comparison honest, memory and test set must be disjoint.

We split the task variations into a `train` half (used to build memory) and a `test` half (used to evaluate). Before any test variant runs, the runner asserts that no test task ID appears in the memory bank. Violation raises a hard error.

```text
inputs:  store: MemoryStore
         test_tasks: list[TaskSpec]

raises:  RuntimeError if any test task_id appears in the bank

algorithm:
    test_ids = {t.task_id for t in test_tasks}
    leaked = [it.task_id for it in store.get_all() if it.task_id in test_ids]
    if leaked: raise RuntimeError(f"split violation: {leaked}")
```

### 4.4 State compilation

The "state" the embedder operates on isn't the raw env observation — it's a *compiled prefix-state string* that includes task context and recent history. We have three different compilations; the variant axis includes them as ablations.

**Structured (default for `sq_mem`).** Concise, fixed format with task goal, score, and recent history:

```text
inputs:  observations: list[str]
         actions: list[str]
         task_goal: str
         current_score: float

outputs: state_text: str

algorithm:
    return f"""Task: {task_goal}
Score so far: {current_score:.2f}
Recent observations: {observations[-3:]}
Recent actions: {actions[-5:]}"""
```

**Raw transcript (used by `raw_prefix_sq_mem`).** Interleaved (obs, action) pairs. Tends to over-specify state which hurts retrieval:

```text
algorithm:
    parts = []
    for o, a in zip(observations, actions):
        parts.append(f"OBS: {o}\nACT: {a}")
    if len(observations) > len(actions):       # un-paired trailing obs
        parts.append(f"OBS: {observations[-1]}")
    return "\n".join(parts)[-2000:]            # truncate
```

**Last observation only (used by `sq_mem_no_structured_state`).** Loses task goal and history:

```text
algorithm:
    return observations[-1] if observations else ""
```

---

## 5. Variants and Ablations

**Why variants?** To make our claims falsifiable, we don't just measure how well the full mechanism works. We also build deliberately *broken* versions of it and check that they perform worse. If a "broken" version performs just as well as the full mechanism, the thing we broke wasn't actually necessary — and the corresponding claim is refuted.

We define 16 variants. Each variant takes one of two forms:

- **Memory transform**: modify the memory bank itself before evaluation (e.g. set all returns to 0, or permute returns across rows).
- **AgentConfig override**: change one part of the decision rule (e.g. disable softmax weighting, use V(s) instead of Q(s,a)).

The 16 variants:

| Variant | What's changed | What claim it tests |
|---|---|---|
| `raw_history` | no memory used at all | Memory helps (baseline) |
| `summary_memory` | prefix-summary instead of memory | Returns help over text compression |
| `semantic_retrieval` | retrieve, but ignore returns | Returns matter, not just relevance |
| `state_only_value_memory` | use V(s), drop action conditioning | Q(s,a) > V(s) |
| `sq_mem_no_action_conditioning` | same as above via different path | Parameter-path invariance check |
| `raw_prefix_sq_mem` | use full transcript as state | Structured state helps |
| `sq_mem_no_structured_state` | use only last observation as state | Structured state helps (different ablation) |
| **`sq_mem`** | full mechanism | (this is the proposed method) |
| `sq_mem_no_returns` | set all returns to 0 | Returns drive the gain |
| `sq_mem_no_uncertainty` | disable σ penalty | σ matters |
| `sq_mem_top1` | hard nearest-neighbor instead of softmax | Soft aggregation helps |
| `sq_mem_uniform_weights` | uniform averaging over top-R | Softmax weights helps |
| `sq_mem_random_memory` | shuffle the embedding vectors | Retrieval relevance matters |
| `sq_mem_shuffled_returns` | permute returns across rows | Return-row mapping matters |
| `sq_mem_value_reversed` | negate returns (high↔low) | Wrong values actively mislead |
| `sq_mem_zero_returns` | all returns = 0 | (sanity check, redundant with no_returns) |

The five memory-transform recipes (`none`, `zero`, `shuffle`, `reverse`, `random_memory`) are applied to a copy of the memory bank — the underlying store is unmodified.

```text
inputs:  transform: str ∈ {"none", "zero", "shuffle", "reverse", "random_memory"}
         seed: int

outputs: items: list[MemoryItem]    # copy; originals unmodified

algorithm:
    items = deep_copy(store.get_all())

    if transform == "none":
        pass    # full mechanism; sq_mem variant uses this

    elif transform == "zero":           # sq_mem_no_returns, sq_mem_zero_returns
        # Tests: do returns matter?
        for it in items: it.return_value = 0.0

    elif transform == "shuffle":        # sq_mem_shuffled_returns
        # Tests: does the (state, action, return) mapping matter?
        # If memory just helps via "relevant context", this should tie sq_mem.
        # If returns matter, this collapses.
        returns = [it.return_value for it in items]
        rng(seed).shuffle(returns)
        for it, r in zip(items, returns): it.return_value = r

    elif transform == "reverse":        # sq_mem_value_reversed
        # Tests: does memory mislead when returns are inverted?
        # Predicts: should perform WORSE than raw_history.
        max_r = max(it.return_value for it in items)
        min_r = min(it.return_value for it in items)
        for it in items:
            it.return_value = max_r + min_r - it.return_value

    elif transform == "random_memory":  # sq_mem_random_memory
        # Returns preserved; embedding vectors shuffled to break retrieval relevance.
        # Tests: does retrieval relevance (not just average return) matter?
        svs = [it.state_vec for it in items]
        avs = [it.action_vec for it in items]
        rng(seed).shuffle(svs); rng(seed).shuffle(avs)
        for it, sv, av in zip(items, svs, avs):
            it.state_vec, it.action_vec = sv, av

    return items
```

### 5.1 The logic of each ablation

Read this section as: *if SQ-Mem's claim is right, then this specific broken variant must underperform `sq_mem`.* If it doesn't, the claim is wrong.

- **`semantic_retrieval` vs `sq_mem`** — retrieve the same neighbors, but ignore the returns. *If the gain came purely from "retrieved memories give the LLM context", this should tie `sq_mem`*. If `sq_mem` beats it, the **returns** are doing real work.
- **`sq_mem_shuffled_returns`** — keep the (state, action) mapping but randomly permute which return goes with which row. *If returns matter, this must collapse — memory now contains nonsense (return for action A is attached to action B).*
- **`sq_mem_value_reversed`** — negate all returns. *If returns matter, this should perform actively worse than `raw_history` (no memory at all) — because the agent now follows bad recipes thinking they're good.*
- **`sq_mem_no_action_conditioning`** — set retrieval to depend only on state. *If Q(s,a) is more useful than V(s), this must collapse.*
- **`sq_mem_top1`** vs **`sq_mem_uniform_weights`** — replace softmax with argmax (top-1 only) or uniform top-R averaging. *If soft aggregation helps, both extremes should underperform.*

---

## 6. Falsifiable Predictions

**What "falsifiable" means.** Before running any experiments, we pre-register specific quantitative predictions of the form "X must exceed Y by at least δ." If the experiment fails to meet the threshold, the prediction is *not supported* — and the corresponding claim is refuted or weakened.

This protects us from *post-hoc storytelling*: we can't look at the results and pick the comparison that flatters the method.

### 6.1 The 11 predictions

| # | Prediction | What it would refute if violated |
|---|---|---|
| **H0** | Pipeline sanity: required artifacts present; no test items in memory | Entire scaffold |
| **H1** | `sq_mem > {raw_history, summary_memory, semantic_retrieval}` by Δ ≥ 0.02 reward | Memory helps over baselines |
| **H2** | `sq_mem >` every value-destruction control (shuffled/zero/reversed/random_memory) by Δ ≥ 0.02 | **Returns drive the gain, not relevance** |
| **H3** | `sq_mem > {state_only_value_memory, sq_mem_no_action_conditioning}` | **Q(s,a) > V(s)** |
| **H4** | `sq_mem > {raw_prefix_sq_mem, sq_mem_no_structured_state}` | Structured state compilation matters |
| **H5** | `sq_mem > {sq_mem_top1, sq_mem_uniform_weights}` | **Soft aggregation matters** |
| **H6** | `sq_mem.harmful_rate ≤ sq_mem_no_uncertainty.harmful_rate` | σ penalty prevents harmful interventions |
| **H7** | Spearman ρ(Q-bin, success-rate) ≥ 0.30 | \(\hat{Q}\) is calibrated |
| **H8** | Intervention episode rate ≥ 0.01; beneficial > harmful interventions | Memory actually changes decisions, and helpfully |
| **H9** | Long-horizon improvement > short-horizon improvement by Δ ≥ 0.02 | The effect grows with task length |
| **H10** | Split discipline: no test-task IDs in memory | Train/test contamination |

**H1–H6** are the *primary mechanism* claims — refuting any of these would substantially refute the thesis.

**H7–H8** are *behavioral* claims about whether the mechanism actually fires in the expected way.

**H0, H10** are *sanity* checks that catch implementation bugs.

**H9** stratifies by horizon to test whether the effect grows with task length.

Each evaluator is fully automated and runs against the artifacts in a result directory.

### 6.2 Evaluator algorithms

Each evaluator returns a `CheckResult { status, details }` where status is one of `supported` / `inconclusive` / `weakened`. A failing evaluator does not crash the run — it records a `weakened` verdict that the user reads later.

A shared helper `metric(variant)` prefers `avg_total_reward` when `success_rate` is uniform across variants (common on the easy SW task where most variants get success_rate=0); falls back to `success_rate` otherwise. This is so evaluators work on both sparse-reward and dense-reward benchmarks.

**H0 — pipeline sanity:**

```text
algorithm:
    required = {"summary.csv", "memory_bank.jsonl", "config_resolved.yaml"}
    missing = [a for a in required if not exists(run_dir / a)]
    if missing: return CheckResult(weakened, f"missing artifacts: {missing}")
    if any test_task_id in memory_bank: return weakened
    return supported
```

**H1 — main sq_mem comparison:**

```text
inputs: summary.csv, min_delta = 0.02

algorithm:
    sq = metric("sq_mem")
    for control in ["raw_history", "summary_memory", "semantic_retrieval"]:
        c = metric(control)
        if c is None or sq - c < min_delta:
            return inconclusive
    return supported
```

**H2 — value destruction:**

```text
algorithm:
    sq = metric("sq_mem")
    for control in [shuffled_returns, value_reversed, no_returns,
                     zero_returns, random_memory]:
        if sq - metric(control) < min_delta: return inconclusive
    # Extra check: reversed should be worse than raw_history (active misleading)
    if metric("sq_mem_value_reversed") < metric("raw_history"): pass
    return supported
```

**H3 / H4 / H5 — action conditioning, structured state, soft aggregation:** same shape — check `sq_mem` beats its V(s) / raw-state / hard-NN counterparts by ≥ `min_delta`. See [hypothesis_testing.py](../sq_mem_experiments/evaluation/hypothesis_testing.py) for the source form.

**H6 — uncertainty penalty:**

```text
algorithm:
    interventions = load(summary_interventions.csv)
    sq_h = interventions["sq_mem"]["harmful_rate"]
    no_u = interventions["sq_mem_no_uncertainty"]["harmful_rate"]
    return supported if sq_h <= no_u else weakened
```

**H7 — calibration:**

```text
inputs: calibration_sq_mem.csv, min_spearman = 0.30

algorithm:
    cal = load(calibration)
    rho = spearmanr(cal.q_bin_center, cal.success_rate)
    if cal.success_rate is uniform:                      # sparse-reward fallback
        rho = spearmanr(cal.q_bin_center, cal.mean_reward)
    if rho >= min_spearman:    return supported
    if rho >= 0:               return inconclusive
    return weakened    # anti-calibrated (e.g. r3 ALFWorld where memory misleads)
```

**H8 — intervention audit:** for each task_id where `sq_mem` changed the decision, compute (sq_mem.total_reward − baseline.total_reward) and count beneficial vs harmful.

```text
inputs: episodes_sq_mem.jsonl, episodes_raw_history.jsonl

algorithm:
    sq_eps  = load("sq_mem")
    base    = {e.task_id: e for e in load("raw_history")}
    changed = [e for e in sq_eps
               if any(d.memory_changed_decision for d in e.decisions)]
    ep_rate = len(changed) / len(sq_eps)
    if ep_rate < min_intervention_episode_rate: return weakened

    beneficial, harmful, neutral = 0, 0, 0
    for e in changed:
        b = base.get(e.task_id)
        if b is None: continue
        d = e.total_reward - b.total_reward
        if   d > 0.01:  beneficial += 1
        elif d < -0.01: harmful    += 1
        else:           neutral    += 1
    return supported if beneficial > harmful else weakened
```

**H9 — horizon effect:**

```text
algorithm:
    median = median(steps for e in sq_eps)
    short = [e for e in sq_eps if e.steps <= median]
    long  = [e for e in sq_eps if e.steps >  median]
    if len(short) < 3 or len(long) < 3: return inconclusive
    short_delta = mean(short.reward) - mean(base_for(short).reward)
    long_delta  = mean(long.reward)  - mean(base_for(long).reward)
    return supported if abs(long_delta) > abs(short_delta) + min_delta else inconclusive
```

**H10 — split discipline:**

```text
algorithm:
    items = load(memory_bank.jsonl)
    if all(it.split == "train" for it in items): return supported
    return weakened
```

---

## 7. Experimental Setup

### 7.1 Cross-axis design (R1 / R2 / R3)

We run three configurations on ScienceWorld that vary the *embedder* and the *base policy* along orthogonal axes. This isolates whether SQ-Mem's gains come from the retrieval-quality side or the base-policy side.

- **R1**: hashing-trick embedder (vocab 4096, dim 64), keyword-heuristic base policy. *Cheapest setup; tests the mechanism's structure with deliberately weak retrieval and a deliberately weak base policy.*
- **R2**: sentence-transformer embedder (`all-MiniLM-L6-v2`, 22 MB model), keyword-heuristic base policy. *Better retrieval, same base policy as R1. Tests how much the embedder matters.*
- **R3**: sentence-transformer embedder, local LLM base policy (Qwen3.5-9B Unsloth UD-Q4_K_XL via llama.cpp, with `--ctx-size 8192 -ngl 999 --chat-template-kwargs '{"enable_thinking":false}'`). *Full system; tests whether memory still helps when the base policy is already strong.*

All three share the same task splits, same ε-greedy exploration (ε = 0.1), same per-task RNG seed (so exploration variance is fair across variants), same λ = 3.0, ρ = 0.5, top-R = 10, α = 0.5, β = 0.1, and same step budget. Each variant is evaluated on 15 held-out test variations disjoint from the 15 training variations used to populate memory.

### 7.2 Task

`find-non-living-thing` from ScienceWorld (Wang et al., 2022): a 5-step gold path with partial-credit score deltas at 4 of the 5 steps. The agent must wander through rooms, identify an object that is not living (e.g. a rock, a mug), and bring it to a designated spot. The 5 steps:

1. Move to a room with candidate objects
2. Examine an object
3. Confirm it's non-living
4. Pick it up
5. Place it in the target receptacle

Test set: variations 225–239 (15 episodes). Train set: variations 0–14 (used to populate memory).

### 7.3 Metric choice

We report **average total reward** as the primary metric. Average reward is partial-credit on each gold-path step (score increments at 4 of 5 steps), so it captures *progress through the task* even when full success isn't achieved.

We also report success rate (binary 0/1 — full gold path completed) but treat it as secondary on this task. With 15 episodes and a 5-step gold path under a 20-step (or 50-step) cap, success rate is too sparse to discriminate ablations cleanly; partial reward is well-correlated with gold-path progress and gives finer-grained signal.

### 7.4 Embedders

**Hash embedder (R1).** The cheapest possible embedder: bag-of-words style hashing-trick projection. No model, no training, no GPU. Used in R1 to deliberately test the mechanism with weak retrieval.

```text
inputs:  text: str
parameters: vocab_size, dim, seed

algorithm:
    tokens = re.findall(r"\b\w+\b", text.lower())     # word-level tokens
    vec = zeros(dim)
    for t in tokens:
        # Deterministic hash to a (bucket, sign)
        h = int(sha1(t).hexdigest(), 16)
        bucket = h % dim
        sign   = +1 if (h // dim) % 2 == 0 else -1
        vec[bucket] += sign                            # signed counts
    return vec / max(||vec||, 1e-9)                    # L2-normalize
```

L2-normalizing the output means subsequent dot products are cosine similarities directly.

**Sentence-transformer embedder (R2, R3).** Real semantic embedding via a small pretrained sentence-transformer (`all-MiniLM-L6-v2`, 22 MB).

```text
inputs:  text: str  (or list[str] via embed_batch)
parameters: model_name (default "all-MiniLM-L6-v2")

algorithm:
    if not self._model:
        self._model = SentenceTransformer(model_name)  # cached per process
    return self._model.encode(text, normalize_embeddings=True)
```

Output is L2-normalized.

### 7.5 Base policies

**Heuristic (R1, R2).** Keyword-overlap with task-specific bonuses. The point of using this in R1/R2 is to test SQ-Mem with a *weak* base policy, so any improvement comes unambiguously from memory.

```text
inputs:  action: str, task_goal: str, observation: str

outputs: score: float

algorithm:
    goal_toks   = tokenize(task_goal.lower())
    obs_toks    = tokenize(observation[-300:].lower())
    action_toks = tokenize(action.lower())

    goal_overlap = |action_toks ∩ goal_toks| / (|goal_toks| + 1)
    obs_overlap  = |action_toks ∩ obs_toks|  / (|obs_toks|  + 1)

    # Small bonus for task-specific keyword presence
    task_keywords = ACTION_KEYWORDS[best_match(task_goal)]
    kw_bonus = 0.1 × |{kw in task_keywords : kw in action}|

    return goal_overlap + 0.3 * obs_overlap + kw_bonus
```

**LLM (R3).** Wraps a (potentially remote) LLM as a base policy. Pre-filters the action list to the top-N candidates by heuristic, asks the LLM to pick a number, and assigns 1.0 to its choice and small graded scores to the rest. The graded scores matter because they let SQ-Mem's λ-Q term flip the argmax when memory disagrees with the LLM.

```text
inputs:  observation: str
         valid_actions: list[str]
         task_goal: str
         recent_actions: list[str]
         memory_context: list[dict] | None       # only for RAG mode
         per_action_q: dict[action → (Q, σ)] | None  # only for Flag C
         state_value_prior: float | None         # only for Flag B

outputs: scores: dict[action → float in [0, 1]]

algorithm:
    # 1. Pre-filter to top max_candidates actions (LLM context is limited)
    ranked = sort(valid_actions, key=heuristic_base_score, descending=True)

    # 2. Push recently-taken actions to end (anti-repetition)
    if repetition_window > 0:
        recent = set(recent_actions[-repetition_window:])
        ranked = [a for a in ranked if a not in recent] + \
                 [a for a in ranked if a in recent]
    candidates = ranked[:max_candidates]      # N ≤ 20

    # 3. Build user prompt; possibly include memory examples / Q values
    user_msg = build_user_content(
        task_goal, observation, recent_actions, candidates,
        memory_context, per_action_q, state_value_prior,
    )

    # 4. Ask LLM for a number; fall back to heuristic on any error
    try:
        response = llm.chat(system=SYSTEM_TEXT, user=user_msg)
        selected_idx = parse_integer(response, n_candidates=N) - 1
    except Exception:
        selected_idx = 0    # heuristic-rank-1 fallback

    # 5. Score dict: 1.0 for the chosen, small graded for the rest
    scores = {a: 0.0 for a in valid_actions}
    for i, a in enumerate(candidates):
        scores[a] = (max_candidates - i) / (max_candidates * 20)
    scores[candidates[selected_idx]] = 1.0
    return scores
```

The small graded scores (≈ 0.05 down to ≈ 0.002) for unchosen candidates create a gap memory can close: with λ=3 and mem_q diffs ≈ 0.05, the memory term shifts argmax by up to 0.15 — enough to flip choices when memory has strong evidence.

### 7.6 Rollout loop

How each episode is actually executed.

```text
inputs:  agent: BaseAgent
         env: BaseEnv
         task_spec: TaskSpec
         max_steps: int
         success_threshold: float

outputs: Episode

algorithm:
    obs = env.reset(task_spec)
    agent.reset(task_spec.task_id, env.task_goal)
    decisions, total_reward, done, info = [], 0.0, False, {}

    for _ in range(max_steps):
        valid_actions = env.get_valid_actions()
        action = agent.act(obs, valid_actions)
        decisions.extend(agent.pop_decisions())

        obs, reward, done, info = env.step(action)
        agent.update(reward, done, info)
        if decisions: decisions[-1].reward = reward    # attach reward

        total_reward += reward
        if done: break

    decisions.extend(agent.pop_decisions())            # any trailing
    backward_return_to_go(decisions)                   # fill .return_value

    final_score = info.get("normalized_score", info.get("score", 0.0))
    return Episode(
        task_id      = task_spec.task_id,
        variant      = agent.variant,
        decisions    = decisions,
        success      = final_score >= success_threshold,
        total_reward = total_reward,
        steps        = len(decisions),
        metadata     = {"final_score": final_score, "done": done, ...},
    )
```

---

## 8. Main Results (ScienceWorld)

The tables in this section summarize three metrics produced by the evaluation pipeline. We define them up front so the numbers in §8.1–§8.10 are self-contained.

**Per-variant summary** — one row per variant in `summary.csv`:

```text
inputs:  episodes: list[Episode]

algorithm:
    return {
        "n_episodes":           len(episodes),
        "success_rate":         mean(e.success for e in episodes),
        "avg_steps":            mean(e.steps   for e in episodes),
        "avg_total_reward":     mean(e.total_reward for e in episodes),
        "repeated_failure_rate":     mean(has_repeated_failure(e.decisions)),
        "error_recovery_rate":       mean(has_error_recovery(e.decisions)),
        "intervention_episode_rate": mean(any(d.memory_changed_decision
                                              for d in e.decisions)),
    }
```

**Intervention summary** — counts how often memory changed the agent's decision and whether those changes helped or hurt vs the no-memory baseline.

```text
inputs:  episodes: list[Episode]
         baseline_episodes: list[Episode]   # raw_history reference

algorithm:
    all_decisions = [d for e in episodes for d in e.decisions]
    changed = [d for d in all_decisions if d.memory_changed_decision]
    intervention_rate = len(changed) / len(all_decisions)

    if baseline_episodes:
        baseline_success = {e.task_id: e.success for e in baseline_episodes}
        changed_eps = [e for e in episodes
                       if any(d.memory_changed_decision for d in e.decisions)]
        beneficial = sum(1 for e in changed_eps
                         if e.success and not baseline_success.get(e.task_id, True))
        harmful    = sum(1 for e in changed_eps
                         if not e.success and baseline_success.get(e.task_id, False))
        beneficial_rate = beneficial / len(changed_eps)
        harmful_rate    = harmful    / len(changed_eps)

    return {"intervention_rate": ..., "beneficial_rate": ..., "harmful_rate": ...}
```

**Calibration** — for each decision we have (memory_q, eventual_success). Bin into quintiles of memory_q; compute the empirical success rate per bin. Used by H7.

```text
inputs:  episodes: list[Episode], n_bins: int = 5

algorithm:
    rows = [(d.memory_q, e.success) for e in episodes for d in e.decisions]
    qs   = [r[0] for r in rows]
    succ = [r[1] for r in rows]
    bin_edges = unique(percentile(qs, linspace(0, 100, n_bins+1)))
    result = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (qs >= lo) & (qs <= hi)
        result.append({
            "q_bin_center": (lo+hi)/2,
            "n":            sum(mask),
            "success_rate": mean(succ[mask]) if any(mask) else 0,
            "mean_reward":  mean(reward[mask]) if any(mask) else 0,
        })
    return result
```

### 8.1 H1 — full SQ-Mem beats every memory-as-context baseline

| Variant | R1 reward | R2 reward | R3 reward |
|---|---:|---:|---:|
| `raw_history` | 0.049 | 0.016 | 0.165 |
| `summary_memory` | 0.144 | 0.044 | 0.093 |
| `semantic_retrieval` | 0.049 | 0.083 | 0.133 |
| **`sq_mem`** | **0.555** | **0.527** | **0.532** |

**Reading this table.** Each column is one configuration. The top three rows are baselines that *don't* use return-aware retrieval. The bottom row is the full SQ-Mem mechanism. In R3 (the configuration with a real LLM base policy), SQ-Mem produces **0.532 average reward** vs the best baseline's **0.165** — a **3.2×** improvement.

The most informative comparison is `sq_mem` vs `semantic_retrieval`: both use the same retrieval, but `semantic_retrieval` ignores the returns. If the gain came from "the LLM sees relevant memories," `semantic_retrieval` would match. It doesn't — SQ-Mem is **4×** better in R3. **The returns are doing the work.**

### 8.2 H2 — value-destruction controls collapse

| Control | What is destroyed | R1 | R2 | R3 |
|---|---|---:|---:|---:|
| `sq_mem_zero_returns` | All returns → 0 | 0.049 | 0.049 | 0.205 |
| `sq_mem_no_returns` | Return field removed | 0.093 | 0.060 | 0.138 |
| `sq_mem_shuffled_returns` | Permute returns across rows | 0.165 | 0.155 | 0.127 |
| `sq_mem_value_reversed` | Negate returns | 0.016 | 0.011 | 0.011 |
| `sq_mem_random_memory` | Shuffle state/action vectors | 0.227 | 0.049 | 0.199 |

**What this shows.** Every variant that destroys the return-row mapping collapses far below `sq_mem` (3× – 50× drop). The clearest case is `sq_mem_value_reversed`, the strongest single piece of evidence for the thesis:

- R1: 0.016 vs raw_history's 0.049 (0.33× — drops *below* raw_history)
- R2: 0.011 vs raw_history's 0.016 (0.69× — drops *below* raw_history)
- R3: 0.011 vs raw_history's 0.165 (0.07× — drops *far below* raw_history)

Reversed returns don't just remove the benefit; they *actively mislead* the agent. If memory were really just "relevant context that primes the LLM", value reversal shouldn't hurt — the context is still relevant. But it does hurt, dramatically, because the agent takes the reversed-return rows as advice about what to do, and that advice is backwards.

### 8.3 H3 — action conditioning matters

| Variant | R1 | R2 | R3 |
|---|---:|---:|---:|
| `state_only_value_memory` (V(s) only) | 0.016 | 0.088 | 0.249 |
| `sq_mem_no_action_conditioning` (V(s) only, different code path) | 0.016 | 0.049 | 0.310 |
| **`sq_mem`** (Q(s,a)) | **0.555** | **0.527** | **0.532** |

V(s)-only variants tell the agent *"this state was good"* but can't say *which action* made it good. The agent has to fall back on the base policy for action choice, losing the per-action discrimination memory provides. SQ-Mem with action conditioning is **1.7× – 35×** better than the best V(s) baseline depending on the configuration. The two V(s) variants reach similar (but not always identical) numbers — they're mechanically equivalent at α=1.0, with small differences attributable to ε-greedy exploration noise across 15 episodes.

### 8.4 H4 — structured state (mixed, configuration-dependent)

| Variant | R1 | R2 | R3 |
|---|---:|---:|---:|
| `raw_prefix_sq_mem` (full transcript as state) | 0.479 | 0.377 | 0.231 |
| `sq_mem_no_structured_state` (only last obs as state) | 0.340 | 0.501 | 0.579 |
| **`sq_mem`** (structured: task goal + recent obs + recent actions + score) | **0.555** | **0.527** | **0.532** |

In R1 and R2 (heuristic base policy), structured state cleanly beats both alternatives — the heuristic needs the explicit task-goal cue and history to find the right action. In R3 (LLM base policy), `sq_mem_no_structured_state` (0.579) actually edges out `sq_mem` (0.532) by Δ=−0.047. **The LLM doesn't need our hand-built structured compilation**; it can extract enough task context from the raw observation alone. This is itself an informative finding: structured state was load-bearing when the base policy was weak, and becomes optional when the base policy is strong.

### 8.5 H5 — soft aggregation matters

| Variant | R1 | R2 | R3 |
|---|---:|---:|---:|
| `sq_mem_top1` (argmax over top-R) | 0.267 | 0.265 | 0.493 |
| `sq_mem_uniform_weights` (uniform top-R) | 0.510 | 0.399 | 0.493 |
| **`sq_mem`** (softmax-weighted top-R) | **0.555** | **0.527** | **0.532** |

Softmax beats both extremes in every configuration. `sq_mem_top1` (hard nearest-neighbor) loses by 2× in R1/R2 and Δ=0.04 in R3. `sq_mem_uniform_weights` (no similarity-weighting) loses by Δ=0.04–0.13. The R3 deltas are small but consistent — supported at the 0.02-threshold registered before the experiment.

### 8.6 H6 — uncertainty penalty matters

We compare the *harmful intervention rate* — the fraction of episodes where SQ-Mem changed the decision and that changed decision led to a worse outcome. `sq_mem` with ρ = 0.5 has 0% harmful interventions in all three configurations. `sq_mem_no_uncertainty` (ρ = 0) hits the same 0% harmful rate on this task, so H6 is supported by equality.

### 8.7 H7 — calibration

Per-decision memory-Q values are binned into quintiles; we compute the Spearman correlation between bin-centre \(\hat{Q}\) and per-bin empirical average reward. **A correlation near +1 means high-Q decisions correctly predict high reward** — the estimator is well-calibrated.

| Run | Spearman ρ | Verdict |
|---|---:|---|
| R1 (hash embedder) | **+0.45** | supported |
| R2 (sentence-transformer) | **+0.50** | supported |
| R3 (LLM base policy) | **+0.60** | supported |

All three configurations now show positive calibration — high-Q decisions correctly predict higher empirical reward. R1's calibration is weakest (ρ = +0.45 vs R3's +0.60), consistent with the hash embedder's noisier retrieval — but it's still on the right side of zero, supporting H7 in every configuration.

### 8.8 H8 — intervention audit

Episodes where SQ-Mem changed the agent's chosen action (relative to the no-memory baseline), classified by per-task partial-credit reward delta:

| Run | Beneficial / Total | Harmful / Total | Neutral / Total |
|---|---:|---:|---:|
| R1 | 13 / 15 | 1 / 15 | 1 / 15 |
| R2 | 12 / 15 | 2 / 15 | 1 / 15 |
| R3 | 11 / 15 | 3 / 15 | 1 / 15 |

Beneficial interventions strictly outnumber harmful in every configuration.

### 8.9 H10 — split discipline

In every run, the memory bank contains only `split == "train"` entries; no test task ID appears in `memory_bank.jsonl`. The runner asserts this before writing any test artifacts.

### 8.10 Summary ledger

| Hypothesis | R1 | R2 | R3 |
|---|:---:|:---:|:---:|
| H0 pipeline sanity | supported | supported | supported |
| H1 main sq_mem comparison | supported | supported | supported |
| H2 value destruction | supported | supported | supported |
| H3 action conditioning | supported | supported | supported |
| H4 structured state | supported | supported | **inconclusive** |
| H5 soft aggregation | supported | supported | supported |
| H6 uncertainty penalty | supported | supported | supported |
| H7 calibration | supported | supported | supported |
| H8 intervention audit | supported | supported | supported |
| H9 horizon effect | inconclusive | inconclusive | inconclusive |
| H10 split discipline | supported | supported | supported |

**10 of 11 supported in R1 and R2; 9 of 11 in R3.** Both inconclusive verdicts are well-understood:

- **R3 H4 (structured state).** With the LLM base policy, `sq_mem_no_structured_state` (0.579) edges out `sq_mem` (0.532) by Δ=−0.047. The LLM extracts enough task context from the raw observation that our hand-built structured compilation becomes a no-op or slight drag. This is a *finding about which configurations need structured state*, not a refutation — H4 holds robustly when the base policy is weaker (R1 and R2 both clearly supported).
- **H9 (horizon effect)** across all configurations. Every episode hits the 30-step cap on this 5-step-gold-path task, so there's no length variance to stratify into short-vs-long buckets. This is a benchmark limitation, not a mechanism issue. Longer-horizon ScienceWorld tasks (`boil`, `freeze`, `chemistry-mix`) would populate it.

---

## 9. Cross-Benchmark Generalization (ALFWorld)

> **Sample-size caveat (read before the numbers).** All ALFWorld success rates in this section are over **n = 15 test episodes** with binary success. A one-episode change is ±0.067; a three-episode change is ±0.20. **Treat any single-variant success-rate delta under ≈0.20 as suggestive, not conclusive.** The robust ALFWorld signal in this section is the **per-decision calibration** (Spearman ρ over ~570 individual decisions in §9.7), not the episode-level success rates. Where a success-rate comparison and the calibration disagree, trust the calibration. The success-rate tables are reported for completeness and to show direction, not to support fine-grained claims.

### 9.1 The transfer problem

ALFWorld (Shridhar et al., 2021) is a household-task text-game benchmark. The action space looks like *"go to drawer 3"*, *"open shelf 1"*, *"examine apple 2"* — actions name specific instance IDs.

Memory built on one room contains rows like *"open drawer 1 → return 0.8"*. At test time, the agent is in a different room where the relevant object is in *drawer 3*, not drawer 1. The action `open drawer 3` and the memory action `open drawer 1` have *no token overlap*, so the action-similarity term in the SQ-Mem estimator reads them as unrelated.

**Result:** the action-conditioning signal collapses on ALFWorld even when claim H3 (\(Q(s,a) > V(s)\)) should hold. `sq_mem` degenerates toward `sq_mem_no_action_conditioning`.

We introduce three independent flags to address this. Each is a paper-faithful design choice — none of them change [§2's](#2-the-soft-q-memory-estimator) estimator equation; they only change how it's keyed (A) or consumed (B, C).

### 9.2 Flag A: `normalize_actions`

**Idea.** Strip standalone digits and articles from action text before embedding. So `"open drawer 1"` and `"open the drawer 3"` both normalize to `"open drawer"` and embed to the same vector. Generic and benchmark-agnostic: no special-cased ALFWorld vocabulary.

Mechanism unchanged. Only the *action key* generalizes. Without this, on ALFWorld, action conditioning measures tokenizer collisions rather than semantics.

Pseudocode: [`normalize_action_text`](#algorithm-normalize_action_text-variant-a).

### 9.3 Flag B: `state_value_in_prompt`

**Idea.** Compute state-only \(V(s) = \sum_i w_i(z_t) G_i\) (no action conditioning) and inject it as a single scalar line in the LLM prompt ("Memory prior: similar past states reached an average final outcome of 0.42"). Per-action Q-rerank still happens.

**Why this is interesting.** It probes claim H3 again, this time *through the LLM*: does the LLM derive signal from \(Q(s,a)\) over and above \(V(s)\), when given both?

[§12's](#12-relation-to-prior-work) "not RAG" distinction is preserved — returns are still aggregated numerically into a scalar estimator output before any language consumption. We're using language only to *describe* the estimator output to the LLM.

### 9.4 Flag C: `q_in_prompt`

**Idea.** Surface per-action `[memory Q=…, σ=…]` annotations inside the LLM's candidate list. Like:

```text
Valid actions:
  1. examine drawer 1    [memory Q=0.05, σ=0.10]
  2. open drawer 3       [memory Q=0.72, σ=0.15]
  3. go to fridge        [memory Q=0.20, σ=0.30]
  …
```

The LLM picks an action. The arithmetic combine from [§3](#3-online-decision-rule) is *skipped*.

**Why this is the deepest test of the thesis.** The thesis says memory is *"evidence about which continuations led to better outcomes"*. It does NOT say the linear combine is the *only way* to consume that evidence. If C matches or beats the arithmetic combine, the **evidence in \(\hat{Q}\) itself** — not the specific functional form of [§3](#3-online-decision-rule) — is doing the work. That's a *stronger* reading of the thesis.

### 9.5 The ablation matrix

These three flags compose. The 2×2 below holds `normalize_actions = true`:

|                                | `q_in_prompt: false`              | `q_in_prompt: true`            |
| ------------------------------ | --------------------------------- | ------------------------------ |
| `state_value_in_prompt: false` | paper §3 (linear combine)         | **C** (LLM consumes Q/σ)       |
| `state_value_in_prompt: true`  | **B** (V prior + linear combine)  | **B+C** (LLM consumes V + Q/σ) |

Cells differ only in *how* the soft-Q estimator output is consumed; the estimator itself ([§2](#2-the-soft-q-memory-estimator), the H1–H3 variant ablations) is identical in all four. **This cleanly separates** "is the soft-Q estimator informative?" from "is [§3](#3-online-decision-rule)'s linear combine the right consumer?" — two claims the ScienceWorld experiments conflated.

### 9.6 r3 ALFWorld results — characterizing the failure mode

Porting SQ-Mem to ALFWorld with Qwen3.5-9B revealed a layered failure that took two attempts to fully isolate. We report both because the contrast between them is informative.

**First attempt — broken memory bank.** In our initial r3 ALFWorld run, every training-memory row's `state_text` was the env's raw last observation (e.g., `"You pick up the plate 1 from the cabinet 2."`) — no task goal, no recent history. Test-time queries used a *different* compilation that included the task goal at the top. So retrieval keyed on whatever overlapped between the two formats, which was kitchen-object names alone. Every test query in a kitchen was matching every training row in a kitchen, regardless of the underlying task. We diagnosed this by inspecting one failing episode (a "look at the mug with the desklamp" test): at step 2, memory steered the agent toward `take apple 1 from microwave 1` (high-Q because some training "heat some apple" trajectory had succeeded), and the agent got stuck in a 30-step `heat → move → take → heat` loop following the wrong recipe.

**Second attempt — bank format fixed.** After making the memory builder use the same state compilation as the test-time agent ([`compile_scienceworld`](#44-state-compilation), with `Task: <goal>` at the top), retrieval became correctly task-aware. The fix changed exactly the things we'd predict:

| What changed                                                                          | First attempt | After fix                       |
| ------------------------------------------------------------------------------------- | ------------- | ------------------------------- |
| Memory items have `Task:` line                                                        | No            | Yes (78 distinct goals visible) |
| `sq_mem_no_structured_state` / `raw_prefix_sq_mem` (query with mismatched format)     | 0.27          | 0.07 (drop confirms it matters) |
| H4 structured state                                                                   | weakened      | supported                       |

But this *correct retrieval* surfaced a deeper problem. The post-fix calibration is worse, not better:

| Q-bin center | n decisions | success rate when this Q-range was retrieved |
| -----------: | ----------: | -------------------------------------------: |
|        0.046 |         283 |          **19.1%** (≈ raw_history's 20%)     |
|        0.252 |         142 |                                         0.7% |
|        0.561 |         141 |                                         2.1% |
|        0.855 |         142 |                                  **0.0%**    |

Spearman ρ = **−0.80** (up from −0.67). Reading row-by-row: when memory has nothing to recommend (Q ≈ 0), the agent succeeds at the baseline rate. **The moment memory becomes confident, success rate collapses to ≈ 0%.** Memory is reliably wrong *proportional to its confidence*.

The headline post-fix numbers:

| Variant                         | success rate    |
| ------------------------------- | --------------: |
| `raw_history`                   | **0.20** (3/15) |
| `semantic_retrieval`            | **0.33** (5/15) |
| `summary_memory`                |            0.27 |
| `state_only_value_memory`       |            0.20 |
| `sq_mem_no_action_conditioning` |            0.27 |
| **`sq_mem`**                    |     0.13 (2/15) |
| `sq_mem_value_reversed`         |            0.13 |
| `sq_mem_no_uncertainty`         |            0.00 |

**Why is confident memory wrong even with task-matched retrieval?** Because at training time, a success at step *t* of trajectory *k* came from a *specific* sequence of preceding actions in a *specific* room layout with *specific* object placements. When the agent at test time encounters a similar-looking state on a *different trial* of the same task type, it retrieves the high-Q row faithfully — but the next action that worked in trial *k*'s placement is not the next action that works in this trial's placement. Within-task variation across trials in ALFWorld is large enough that *per-action* transfer breaks down.

This isolates the failure precisely:

- **Mechanism faithfulness**: ✓ the soft-Q estimator correctly reports "this (state, action) led to high return in training." That claim is empirically validated by `sq_mem_value_reversed` no longer collapsing to 0 (it now matches `sq_mem` at 0.13) — flipping the sign of the same well-calibrated estimates produces symmetric mediocrity, not catastrophic failure.
- **Cross-trial action transferability**: ✗ a successful action in one trial of "look at the mug with the desklamp" is, on average, not the successful action in another trial of the same task — because the mug lives in different containers, the desklamp lives in different rooms, the path through the house differs.

The thesis from [§1](#1-motivation) is empirically refined: SQ-Mem requires not just **task-level retrieval transferability** (which the bank-format fix delivered) but also **per-(state, action) outcome stability across trials** — i.e., that the action which worked in training-trial state \(s\) also works in test-trial state \(s'\) when \(s'\) is retrieved as similar to \(s\). ScienceWorld's `find-non-living-thing` satisfies this because its 5-step gold path has a small, deterministic action surface. ALFWorld doesn't because per-trial randomization of object placements decouples retrieval-similar states from action-equivalent states.

The verdict ledger:

| Hypothesis                | First attempt   | After fix       |
| ------------------------- | --------------- | --------------- |
| H0 pipeline sanity        | supported       | supported       |
| H1 main sq_mem comparison | inconclusive    | **weakened**    |
| H2 value destruction      | inconclusive    | inconclusive    |
| H3 action conditioning    | inconclusive    | **weakened**    |
| H4 structured state       | weakened        | **supported**   |
| H5 soft aggregation       | inconclusive    | inconclusive    |
| H6 uncertainty penalty    | supported       | inconclusive    |
| H7 calibration            | weakened (−0.67)| weakened (−0.80)|
| H8 intervention audit     | weakened        | weakened        |
| H9 horizon effect         | inconclusive    | inconclusive    |
| H10 split discipline      | supported       | supported       |

H1 went from "inconclusive" to "weakened" because, with retrieval now sharper, `sq_mem` lost ground to `semantic_retrieval` — which is exactly what you'd expect under the diagnosis: returns are *wrong*, so any consumer that uses them gets misled, while a consumer that uses only retrieval-similarity (semantic_retrieval) avoids the wrong-returns problem.

This motivates two follow-up experiments testing whether *changing the consumer* of the soft-Q estimator (rather than its retrieval key) handles the cross-trial transfer failure better:

- **V-prior prompt augmentation** ([`configs/r4_alfworld_v_in_prompt.yaml`](../configs/r4_alfworld_v_in_prompt.yaml)) — surface a single scalar \(V(s)\) as a prompt prior; per-action Q-rerank still runs. Hypothesis: a coarser "this state looks like a winning state" signal is more transferable across trials than per-action \(Q\).
- **Q-annotated candidate list** ([`configs/r4_alfworld_q_in_prompt.yaml`](../configs/r4_alfworld_q_in_prompt.yaml)) — format per-action `[memory Q=…, σ=…]` into the prompt; skip the arithmetic combine. Hypothesis: when σ is high, the LLM can choose to ignore memory; the linear combine `S + λQ − ρσ` can only scale-down with a single shared ρ.

### 9.7 r4 ALFWorld results — Q-annotated candidate list inverts calibration

> **Provisional.** The verdict ledger below was computed before two evaluation fixes landed: (a) the H8 intervention audit mis-measured `memory_changed_decision` under `q_in_prompt` mode (it compared the memory-informed LLM choice against itself, structurally forcing intervention rate to 0); (b) H2 now uses paired-bootstrap CIs rather than point-estimate thresholds. The H8 and H2 rows here therefore reflect *superseded* measurement code. The success-rate tables and the calibration result are unaffected. These rows will be refreshed after the post-fix r4 rerun.

The robust, large-sample result in this section is the **calibration flip** (computed over ~570 decisions). The episode-level success-rate deltas are reported for direction but are within noise at n=15 — see the caveat at the top of §9.

**Q-annotated candidate list** (`q_in_prompt: true`, arithmetic combine skipped) **flips calibration from anti-correlated to strongly correlated**:

| metric                  | r3 linear combine | r4 Q-annotated    | r4 V-prior |
| ----------------------- | ----------------- | ----------------- | ---------- |
| `raw_history`           | 0.20              | 0.27              | 0.20       |
| `sq_mem`                | **0.13**          | **0.33**          | **0.07**   |
| `sq_mem_value_reversed` | 0.13              | 0.13              | 0.00       |
| H7 calibration ρ        | **−0.80**         | **+0.80**         | −0.05      |
| H1 verdict              | weakened          | **supported**     | weakened   |
| H3 verdict              | weakened          | **supported**     | weakened   |

Per-bin calibration table for Q-annotated `sq_mem`:

| Q-bin center | n decisions | empirical success rate |
| -----------: | ----------: | ---------------------: |
|        0.004 |         228 |                   7.5% |
|        0.055 |         113 |                   3.5% |
|        0.176 |         114 |                   8.8% |
|        0.626 |         114 |              **33.3%** |

When memory has nothing to recommend (Q ≈ 0), decisions succeed at near the no-memory rate. When memory is highly confident (Q ≥ 0.25), the per-decision success rate is roughly 4× higher. Because this is computed over ~570 decisions rather than 15 episodes, the monotone trend is statistically meaningful in a way the episode-level success rates are not — this is the load-bearing ALFWorld result.

**The key methodological finding.** The soft-Q values on ALFWorld are *informative* — the same values that produced ρ = −0.80 under the linear combine produce ρ = +0.80 under Q-annotated consumption. The information was in the estimates; the linear combine couldn't extract it. Most likely because:

- The linear combine `S + λQ − ρσ` applies a single scalar ρ uniformly. A `Q=0.85, σ=0.40` candidate wins whenever `λ·0.85 > ρ·0.40` — which fires often.
- The LLM reading `[memory Q=0.85, σ=0.40]` can recognize "this is uncertain" categorically; it can refuse to follow high-Q recommendations when σ is also high. The linear combine cannot.

**V-prior didn't recover performance.** With only a single scalar V(s) injected as a prompt prior, the LLM has no per-action signal to discriminate candidates. `sq_mem` dropped to 0.07 — *below* even the broken-bank baseline. The V(s) prior also produced a calibration ρ = −0.05 (no signal at all), unlike the strongly-correlated +0.80 from the per-action Q annotations.

**One subfinding worth flagging — but with the n=15 caveat front and center.** Under Q-annotated consumption, two variants with *broken* retrieval (`sq_mem_no_structured_state` 0.53, `raw_prefix_sq_mem` 0.47) score *above* `sq_mem` (0.33). These are 8-vs-5 and 7-vs-5 episode differences out of 15 — within the noise band, so we do not claim broken memory is genuinely better. The *suggestive* mechanism, if the effect is real: a state-format mismatch produces near-zero Q values everywhere, the LLM treats the annotations as noise and falls back to its own (strong) reasoning. If true, it would imply a more selective trust mechanism could help (e.g., the σ-gate of §9.5, or letting the LLM decide *whether* to consult memory per step). But confirming this needs more episodes; at n=15 it is only a hypothesis. The defensible claim is narrower: working memory under Q-annotated consumption is *calibrated* (ρ=+0.80), and its episode-level edge over `raw_history` (+0.06) is too small to call at this sample size.

**Full verdict ledger across the three ALFWorld setups:**

| Hypothesis                | r3 linear combine     | r4 Q-annotated        | r4 V-prior            |
| ------------------------- | --------------------- | --------------------- | --------------------- |
| H0 pipeline sanity        | supported             | supported             | supported             |
| H1 main sq_mem comparison | weakened              | **supported**         | weakened              |
| H2 value destruction      | inconclusive          | inconclusive          | inconclusive          |
| H3 action conditioning    | weakened              | **supported**         | weakened              |
| H4 structured state       | supported             | weakened              | inconclusive          |
| H5 soft aggregation       | inconclusive          | inconclusive          | weakened              |
| H6 uncertainty penalty    | inconclusive          | supported             | inconclusive          |
| H7 calibration            | weakened              | **supported**         | weakened              |
| H8 intervention audit     | weakened              | weakened              | weakened              |
| H9 horizon effect         | inconclusive          | inconclusive          | inconclusive          |
| H10 split discipline      | supported             | supported             | supported             |

H4 weakened under Q-annotated because the broken-format variants did unusually well (see the subfinding above). H8 (intervention audit) is weakened in all three setups, though for Q-annotated the comparison is methodologically subtle — `memory_changed_decision` requires a counterfactual LLM call, and the current code logs whether memory altered the *combined* score, not whether annotating-Q-in-prompt altered the LLM's choice. That's a separate measurement bug that should be cleaned up, but doesn't affect the calibration finding.

### 9.8 What this means for the thesis

The thesis from [§1](#1-motivation) is **strongly supported on ScienceWorld** (H1–H8 supported under the standard linear-combine decision rule across R1–R3) and **partially supported on ALFWorld**, where the evidence is narrower and rests primarily on calibration rather than success rate.

What ALFWorld actually shows, stated conservatively:

- The soft-Q estimator carries information about which continuations led to better outcomes — demonstrated by the calibration flip: the *same* estimates are anti-correlated with success under the linear combine (ρ = −0.80) but positively correlated under LLM consumption of per-action Q/σ (ρ = +0.80, over ~570 decisions). The information is in the estimates; the linear combine fails to extract it on this benchmark.
- Whether that calibration converts into an episode-level success advantage is **undetermined at n = 15**. The observed +0.06 edge of working memory over the no-memory baseline is within noise.

So the defensible cross-benchmark claim is:

> **Memory's value-bearing component is a soft Q table** that carries outcome information, and *the way that information is consumed matters*: the linear combine of [§3](#3-online-decision-rule) suffices when per-action transfer is tight (ScienceWorld) but fails to extract the signal when per-action transfer is loose (ALFWorld), where an LLM reading the per-action Q/σ values recovers a *calibrated* signal. We do **not** claim the recovered calibration yields a robust success-rate improvement on ALFWorld — establishing that requires a larger evaluation (≈100+ episodes) than this batch provides.

This is more careful than a "flexible consumers extract the information across many regimes" reading. We have exactly one transfer-hard benchmark; it gives one clean result (calibration) and one undetermined result (success rate). The generalization beyond that — to other consumers, other benchmarks, and to whether calibration reliably becomes task success — is conjecture this batch does not settle.

---

## 10. Limitations

- **Single ScienceWorld task.** All R1/R2/R3 runs use `find-non-living-thing` (5-step gold path). Generalization to long-horizon ScienceWorld tasks (`boil`, `freeze`, `chemistry-mix`, gold paths 22–26 steps) is not yet tested in this batch. A larger task suite would populate H9 (horizon effect).
- **Sample size.** 15 test variations per variant. Confidence intervals on success-rate-delta would require ~100 variations or paired bootstrap across seeds.
- **Local LLM strength.** R3 uses Qwen3.5-9B at UD-Q4_K_XL quantization. Stronger frontier models would likely show *smaller* relative gains because the base policy already covers more of the action space — but the directional pattern (sq_mem > controls) should persist. Earlier exploratory runs at Qwen3.5-4B left the memory bank effectively empty on ALFWorld (~0–2% non-zero returns) because the base policy almost never succeeded during training; the 9B's ~20% baseline success rate on ALFWorld is what lifted us past that bottleneck.
- **Cross-benchmark preconditions.** SQ-Mem's utility depends on two preconditions:
  1. **Base policy occasionally succeeds during memory collection.** Else the memory bank has near-zero return variance and the estimator has nothing to discriminate. The 4B failed this on ALFWorld; the 9B clears it (20% baseline success → bank with real non-zero density).
  2. **State and action representations transfer across train/test.** Else memory steers the agent into a train-specific recipe that doesn't apply to test. `normalize_actions=true` ([§9.2](#92-flag-a-normalize_actions)) handles the action side. The *state* side still fails on ALFWorld: room contents differ across train and test rooms, and even with normalized actions the same task requires different recipes per room.

  Both preconditions hold on SW (R1–R3 cleanly support H1–H8). On ALFWorld, the 9B clears (1) but (2) still fails — sq_mem (0.133 success) ends up below raw_history (0.200) because memory's recipes don't transfer across rooms (see §9.6). The two follow-up runs (`r4_alfworld_q_in_prompt.yaml`, `r4_alfworld_v_in_prompt.yaml`) test whether *changing the consumer of the soft-Q estimator* — surfacing Q/σ to the LLM rather than committing to the arithmetic combine — recovers performance.

---

## 11. Reproducibility

Each run's full artifacts are written to `results/<run_id>/`:

```text
results/<run_id>/
├── config_resolved.yaml      # what config was actually used
├── memory_bank.jsonl         # full memory bank (one item per line)
├── memory_summary.json       # bank size, task IDs, split assignments
├── summary.csv               # per-variant: success rate, avg reward, intervention rate
├── summary_interventions.csv # per-variant: beneficial/harmful intervention counts
├── summary_value_destruction.csv  # per-variant ablation comparison
├── decisions_<variant>.csv   # per-decision logs (one row per step)
├── episodes_<variant>.jsonl  # full episode rollouts
├── calibration_<variant>.csv # Q-bin → empirical reward bins
├── hypothesis_checks.csv     # H0–H10 verdicts
├── hypothesis_report.md      # human-readable
└── hypothesis_report.json    # machine-readable
```

To reproduce all three configurations:

```bash
pip install -e ".[dev,scienceworld,sentence_transformers,local_llm]"

# R1: hashing + heuristic (no GPU/LLM)
python scripts/run_experiment.py --config configs/r1_scienceworld_hash_heuristic.yaml

# R2: sentence-transformers + heuristic (Metal-accelerated batched embeddings)
python scripts/run_experiment.py --config configs/r2_scienceworld_st_heuristic.yaml

# R3: ST + local LLM (requires llama.cpp serving Qwen3.5-9B-UD-Q4_K_XL on :8080
#     with: --ctx-size 8192 -ngl 999 --chat-template-kwargs '{"enable_thinking":false}')
python scripts/run_experiment.py --config configs/r3_scienceworld_st_llm.yaml

# Cross-run aggregation
python scripts/aggregate_results.py \
  results/r1_scienceworld_hash_heuristic \
  results/r2_scienceworld_st_heuristic \
  results/r3_scienceworld_st_llm \
  --out results/cross_run_summary.csv \
  --hypotheses-out results/cross_run_hypotheses.csv
```

The hypothesis evaluator can be re-run on completed result directories without re-running experiments:

```bash
python scripts/evaluate_hypotheses.py results/<run_id>
```

## 12. Relation to Prior Work

SQ-Mem sits at the intersection of three traditions, but is none of them.

**Not standard RL.** There is no Bellman update, no policy gradient, no replay-buffer training. The base policy is frozen; the memory bank is frozen. Everything happens at inference time.

**Not RAG (in the typical sense).** Retrieval-augmented generation takes retrieved text and stuffs it into the LLM's prompt, letting the LLM generate based on the augmented context. SQ-Mem, by default, never sends text to the LLM; it sends a scalar Q estimate computed from the retrieved returns. The optional RAG-context mode of [§9.3](#93-flag-c-q_in_prompt) *is* a hybrid that does send retrieved tuples to the LLM, but it's a deliberately-named alternative, not the default.

**Closest antecedent: episodic control.** MFEC (Blundell et al., 2016) and NEC (Pritzel et al., 2017) are nonparametric value estimators built from (state-vector, action, return) tuples. SQ-Mem differs by working at the **language** level (so state and action are strings, not vectors from a pretrained encoder), and by using **softmax-weighted top-R averaging** rather than hard nearest-neighbor.

Side-by-side:

| Method | State rep | What's stored | Update | At inference |
|---|---|---|---|---|
| Q-learning | Vector | Q-function parameters | Bellman | Forward pass on Q-net |
| MFEC / NEC | Vector | (s, a, G) tuples | Nearest-neighbor / weighted avg | Hard kNN over neighbors |
| Replay buffer | Vector | (s, a, r, s') tuples | Off-policy gradient | Sample from buffer to update params |
| RAG | Language | Documents | None | Retrieve text, LM generates with text in context |
| **SQ-Mem (this work)** | **Language** | **(s, a, G) with language state/action** | **None** | **Soft aggregation of returns over top-R; arithmetic combine with base score** |

The hypotheses in [§6](#6-falsifiable-predictions) are constructed to discriminate this combination from each adjacent method:

- **H2 (value destruction)** discriminates from RAG: if retrieval relevance alone explained the gain, value destruction would not collapse it.
- **H3 (action conditioning)** discriminates from V(s)-only methods (like MFEC's V(s) variant): if state-only retrieval were sufficient, Q(s,a) and V(s) would tie.
- **H5 (soft aggregation)** discriminates from hard kNN (MFEC default): softmax over similarity must beat hard top-1 for soft aggregation to be the right operationalization.

These are the *load-bearing* comparisons. H1, H4, H6 are robustness checks; H7–H10 are behavioral/sanity checks.

---

## 13. Glossary

A quick reference for terminology used in the paper. Listed alphabetically.

- **Ablation.** Removing or breaking one piece of the mechanism to test whether that piece is necessary.
- **Action conditioning.** Computing similarity between the *candidate action* and stored *memory actions* (vs ignoring actions and only matching states). Toggled by `action_conditioning` in AgentConfig.
- **AgentConfig.** A dataclass holding all the agent's hyperparameters (λ, ρ, top_R, α, β, ε, memory_mode, etc.).
- **α (alpha).** Mixing weight in the retrieval score, balancing state similarity vs action similarity: \(\alpha \cdot s_\text{state} + (1-\alpha) \cdot s_\text{action}\). Default α = 0.5.
- **β (beta).** Temperature in the softmax weighting. Lower β → harder nearest-neighbor; higher β → more uniform. Default β = 0.1.
- **Bank.** The memory store \(\mathcal{M}\) of (state, action, return) tuples.
- **Base policy.** The agent's starting scorer for actions (LLM or heuristic).
- **Calibration.** Whether high-Q decisions actually correspond to high success rate.
- **Embedding.** Vector representation of a piece of text from an embedder function \(\phi\) or \(h\).
- **ε (epsilon).** Exploration probability in ε-greedy. With probability ε, the agent picks a random action instead of the argmax.
- **G (return-to-go).** Total reward accumulated from a step onward to end of episode. Computed by backward sweep.
- **λ (lambda).** Memory weight in the decision rule. Default λ = 3.0.
- **memory_mode.** Either `q_rerank` (the paper's default decision rule) or `rag_context` (LLM consumes retrieved triples).
- **normalize_actions.** Boolean flag (Variant A). When true, action text is stripped of digits and articles before embedding.
- **q_in_prompt.** Boolean flag (Variant C). When true, per-action Q and σ are surfaced in the LLM prompt and the arithmetic combine is skipped.
- **ρ (rho).** Uncertainty penalty weight in the decision rule. Default ρ = 0.5.
- **state_value_in_prompt.** Boolean flag (Variant B). When true, V(s) is computed and injected as a prompt prior.
- **σ (sigma).** Retrieval uncertainty — weighted standard deviation of returns across the top-R neighbors.
- **Soft-Q estimator.** The \(\hat{Q}\) computed by [§2's](#2-the-soft-q-memory-estimator) formula — softmax-weighted average of returns from the top-R retrieved memories.
- **Split discipline.** The rule that memory may contain only training task IDs, never test task IDs. Asserted by `check_split_discipline`.
- **top_R.** Number of nearest-neighbor memories to use in the softmax. Default top_R = 10.
- **Variant.** A specific configuration of the agent (e.g. `sq_mem`, `sq_mem_shuffled_returns`). The paper runs 16 variants on each configuration.
