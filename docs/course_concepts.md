# Course concepts in this project

How each lecture of 15.071 shows up in the repo. When the slides or README use a concept, this is where it lives in the code.

## The shape we copied from the case studies

Every lecture runs the same arc, and so does this project:

1. **A business problem and a stakeholder.** An infrastructure fund deciding which new queue entries to back.
2. **The data snapshot.** Outcome, features, n, and a train/test split that matches how the model will be used.
3. **A benchmark ladder.**
   - Start from the rule a manager can apply with no model, which is what everything must beat (the Lecture 6 demo uses "largest fire first").
   - Then simple models, then complex ones.
4. **Model selection on the training data only**, then a single "showtime" on the test set.
5. **From predictions to decisions.** Turn a probability into an action under a constraint, measure the result in dollars, and sweep the assumptions.
6. **Takeaways and caveats.** Spell out what the model cannot claim.

## Lecture by lecture

| Lecture | Concept | Where it's used here |
|---|---|---|
| 1 (agents) | Self-tests that exercise the whole path cheaply | [tests/selftest.py](../tests/selftest.py): the `check()` harness is copied from the Lecture 1 demo's `selftest.py` |
| 2 (linear regression) | Train/test split; one-hot encoding; predictions to operating profit | [split.py](../src/queue_risk/split.py); one-hot encoding inside the pipelines in [models.py](../src/queue_risk/models.py); [impact.py](../src/queue_risk/impact.py) |
| 3 (logistic regression) | Classification; cross-entropy; ROC/AUC vs. the threshold-dependent confusion matrix; a threshold set by business value | Logistic regression in [models.py](../src/queue_risk/models.py); AUC in [evaluate.py](../src/queue_risk/evaluate.py); top-K as the threshold in [impact.py](../src/queue_risk/impact.py) |
| 4 (model selection) | Regularization (LASSO); scaling fit on training data only; cross-validation; retrain on the full training set; test set used once | LASSO logistic; `Pipeline` scalers; `rolling_folds()` (cross-validation made temporal); the `--final` flag in [pipeline.py](../src/queue_risk/pipeline.py) |
| 5 (CART) | Interpretable tree; `ccp_alpha` pruning; historical baseline; a capacity-limited worklist; trees can't rank within a leaf (ties); break-even analysis; calibration ("predicted total out of actual total") | CART in [models.py](../src/queue_risk/models.py); `HistoricalRate`; `precision_at_k()` with pro-rata ties; `calibration_ratio()`; the Deal Desk's K dial |
| 6 (ensembles) | Random forest; gradient boosting; **temporal split**; SHAP; "correlation vs. causation" on policy features | RF and boosting in [models.py](../src/queue_risk/models.py); `temporal_split()`; SHAP driver chart (to do); the [Deal Desk](../demo/README.md) is modeled on the Lecture 6 demo |
| 7 (recommendations) | Stacking uses **out-of-fold** upstream predictions so the meta-model doesn't see leaked, too-good features | The opposition score is an upstream model's output feeding the tabular model. It is built from the year *before* entry and from articles the text model never trained on |
| 8 (neural nets, CTR) | **Lag variables built from past data only**; cold start; tree ensembles often beat neural nets on tabular data | County and region track-record features in [features.py](../src/queue_risk/features.py); `county_has_history` flag; no neural net on the tabular side |
| 9 (training NNs) | Validation set (not cross-validation) for deep models; early stopping; input normalization | Text model training plan in [text.py](../src/queue_risk/text.py) |
| 10 (NLP) | Tokenize, encode, TF-IDF; "make vendors beat readily-accessible baselines"; confidence thresholds | TF-IDF + logistic regression is the text baseline |
| 11 (transformers) | Contextual embeddings; class imbalance makes some classes harder to learn | Opposition vs. support is context-dependent ("residents *rallied for* / *against*") |
| 12 (pre-train / fine-tune) | Three readers: counting words, frozen BERT, fine-tuned BERT; small learning rate, few epochs; **per-class recall** on the rare class; automation vs. capability expansion | [text.py](../src/queue_risk/text.py) follows the Lecture 12 demo exactly. Capability expansion is the pitch: screen *every* queue entry, not the few an analyst can read |
| 13 (computer vision) | From pilot to rollout; validation and test can differ; start small and monitor | Caveats slide: queue rules change (e.g. FERC Order 2023), so the model needs retraining |
| 14 (ViT) | Data augmentation grouped by item; embeddings support retrieval | Possible Project 2 extension: retrieve similar past projects for a due-diligence memo |

## Habits from the demo READMEs worth copying

- **"Where the numbers come from."** A checkpoint table listing every number the slides quote. The export script refuses to write if any of them move.
- **"Two things worth knowing about the data."** Name the dataset's quirks plainly, such as right-censoring, the 135 duplicate IDs and the one garbled FIPS code.
- **Say what the model is not.** It ranks how likely a project is to be built; it doesn't claim that backing a project makes it more likely to be built (Lecture 5 demo: "Nothing here says enrolling a patient reduces their cost").
- **Assumptions in one block.** Cost per project and K live at the bottom of [config.py](../src/queue_risk/config.py), are printed in the page footer, and are swept, never quoted as a single number.
