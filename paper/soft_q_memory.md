# Soft-Q Memory: Agent Memory as a Nonparametric Value Estimator

**Thesis.** For online decision-making, the control-relevant part of agent memory functions as a soft Q table: it stores or implies evidence about which continuations from similar trajectory prefixes led to better or worse outcomes.

## 1. Motivation

Long-horizon interactive agents accumulate a trajectory prefix \(\tau_{\leq t}\) as they act. A natural question is: what should be stored in external memory to make future decisions better?

Existing approaches store either raw transcripts (retrieved by similarity) or summaries of the current trajectory. Neither explicitly stores *value evidence* — information about which actions led to high downstream return from states similar to the current one.

We propose that the value-bearing component of memory is a nonparametric estimator of $Q(s_t, a)$, the expected return after taking action $a$ from state $s_t$.

## 2. The Soft-Q Memory Estimator

Given a memory bank $\mathcal{M}$ of tuples $(s_i, a_i, G_i)$ where $G_i$ is the observed return-to-go after executing $a_i$ from prefix state $s_i$, we estimate:

$$
\hat{Q}_{\mathcal{M}}(s_t, a)
= \sum_{i \in \mathcal{N}_R(z_t, u)} w_i(z_t, u)\, G_i
$$

where:

- $z_t = \phi(s_t)$ is a dense embedding of the compiled prefix state
- $u = h(a)$ is a dense embedding of the candidate action
- $\mathcal{N}_R(z_t, u)$ is the top-$R$ retrieved neighborhood
- $w_i$ are softmax weights over a combined state-action similarity score

**Retrieval score:**

$$
\text{score}_i(z_t, u) = \alpha\,\text{sim}(z_t, z_i) + (1-\alpha)\,\text{sim}(u, u_i)
$$

**Weights:**

$$
w_i = \frac{\exp(\text{score}_i / \beta)}{\sum_j \exp(\text{score}_j / \beta)}
$$

## 3. Online Decision Rule

At each step, the agent combines a base policy score with the memory Q-value:

$$
a_t = \arg\max_{a \in \mathcal{A}_t}
\left[
S_\theta(a \mid \tau_{\leq t}, s_t)
+ \lambda\,\hat{Q}_{\mathcal{M}}(s_t, a)
- \rho\,\hat{\sigma}_{\mathcal{M}}(s_t, a)
\right]
$$

where $\hat{\sigma}_{\mathcal{M}}(s_t, a) = \sqrt{\sum_i w_i (G_i - \hat{Q})^2}$ is the retrieval uncertainty.

## 4. Memory Construction

Memory is built from training trajectories only (train/test split enforced). Each decision in a training episode becomes one memory row after the backward return-to-go pass:

```
compiled prefix state | action | downstream return
```

## 5. Key Predictions (Falsifiable)

1. `sq_mem` > `semantic_retrieval`: returns, not just relevance, drive the gain.
2. `sq_mem` > `sq_mem_shuffled_returns`: correct return-row mapping matters.
3. `sq_mem_value_reversed` < `raw_history`: wrong values actively mislead.
4. `sq_mem` > `sq_mem_no_action_conditioning`: Q(s,a) beats V(s).
5. Calibration of $\hat{Q}_{\mathcal{M}}$ is monotone: higher predicted value → higher empirical success rate.
6. Memory changes decisions at a non-trivial rate, and beneficial interventions dominate.

## 6. Relation to Prior Work

SQ-Mem is not RL in the standard sense: there is no Bellman update, no policy gradient, and no replay buffer training. It is inference-time, nonparametric, and requires only a fixed memory bank built from prior trajectories.

Unlike retrieval-augmented generation (RAG), the retrieved content is not used as language context for a generator. The retrieved *returns* are aggregated numerically as value estimates.

The closest antecedents are episodic control methods (MFEC, NEC) and experience replay, but applied at the level of language-action sequences rather than state vectors.
