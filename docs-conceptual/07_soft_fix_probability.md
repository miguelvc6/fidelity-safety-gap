## Soft Fix Probability Loss

The soft fix probability augments the standard six-slot cross-entropy loss with a constraint-aware term that rewards the model for *any* prediction that would repair the current violation, even when it does not exactly reproduce the gold edit. The idea is borrowed from the repair evaluation heuristics and translated into a differentiable signal that can guide training.

### How the probability is computed

1. **Heuristic candidate generation** – For every violation row we use `ConstraintRepairHeuristics` to enumerate candidate repair patterns for both the `add` and `delete` actions. Each pattern lists allowed ids for the subject, predicate, and object slots; some slots may accept literals, placeholder tokens (e.g. `subject`), or values derived from the constraint definition (allowed classes, inverse properties, conflicting triples, etc.).

2. **Pattern semantics** – A pattern represents "if the model predicts any triple that matches these value sets, the constraint would be fixed". The heuristics are tailored per constraint type, so the candidate pools reflect domain knowledge (single-value conflicts, type/value-type restrictions, requires-statement rules, inverse relationships, and so on). More information is available in [constraint_types.md](constraint_types.md).

3. **Soft matching** – During training we take the model's logits for the six slots, apply `softmax`, and sum the probability mass each slot assigns to the ids allowed by a pattern. The probability for one candidate triple is the product of its per-slot masses (subject x predicate x object). Because we use the dense distributions rather than the argmax, gradients can flow through the whole computation.

4. **Union into a fix probability** – A violation is considered repaired if *any* candidate triple is produced. For each action we sum (and clamp) the candidate triple probabilities to obtain the probability that the `add` action fixes the violation, and likewise for `delete`. Treating both actions as independent attempts gives the final probability:
 ``` fix_prob = 1 - (1 - add_prob) * (1 - del_prob) ```
 This scalar is near one when the model concentrates probability mass on constraint-satisfying triples, and near zero when it spreads probability over irrelevant ids.

### Training loss

The training objective becomes:

```
Loss = CrossEntropy(six slots) + λ(t) * (1 - fix_prob)
```

`λ(t)` is a time-dependent weight (see below). Large fix probabilities shrink the penalty, encouraging the model to search the candidate space early in training and eventually commit to the precise repairs that satisfy the constraint.

### Data requirements and when it runs

- The loss term is enabled only when `training_config.fix_probability_loss.enabled` is true **and** the dataset split is in-memory (list of `Data` objects).
- During training setup the script loads `violation_contexts` from the interim folder and attaches a `context_index` to every graph in the list so batches can look up the right context.
- If the dataset is streamed (`GraphStreamDataset`) or context lengths do not match the graph list, the fix-probability loss is disabled for that split.

### Configuration and scheduling

The new block in `training_config` toggles the term and defines its decay schedule:

```json
"fix_probability_loss": {
  "enabled": true,
  "initial_weight": 0.5,
  "final_weight": 0.05,
  "decay_epochs": 40,
  "warmup_epochs": 0,
  "schedule": "exponential"
}
```

- `enabled`: master switch for the fix-aware term.
- `initial_weight`: λ at the start of training (before decay / after warmup).
- `final_weight`: asymptotic λ once the decay is complete.
- `decay_epochs`: how quickly the weight decays (interpreted as the time
  constant for the exponential schedule or the full span for a linear schedule).
- `warmup_epochs`: optional number of epochs to keep the initial weight before
  the decay begins.
- `schedule`: currently `exponential` (default) or `linear`.

Together these parameters let you start with a strong emphasis on
constraint-level behaviour (large λ), then gradually decay the coefficient so
late epochs focus on matching the exact gold repairs.
