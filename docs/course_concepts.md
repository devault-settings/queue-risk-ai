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
| 7 (recommendations) | Stacking uses **out-of-fold** upstream predictions so the meta-model doesn't see leaked, too-good features | The reader scores all 572 Sabin paragraphs **out of fold** (`crossfit` in [02_paul_text_models.ipynb](../notebooks/02_paul_text_models.ipynb)) before stage 2 uses them; stage 2's own scores are rolling out-of-fold, ready for late fusion with the tabular model |
| 8 (neural nets, CTR) | **Lag variables built from past data only**; cold start; tree ensembles often beat neural nets on tabular data | County and region track-record features in [features.py](../src/queue_risk/features.py); `county_has_history` flag; no neural net on the tabular side |
| 9 (training NNs) | Validation set (not cross-validation) for deep models; early stopping; input normalization | Fine-tuned DistilBERT in [02_paul_text_models.ipynb](../notebooks/02_paul_text_models.ipynb): a 15% validation split per fit, early stopping on validation loss |
| 10 (NLP) | Tokenize, encode, TF-IDF; "make vendors beat readily-accessible baselines"; confidence thresholds | Keyword rules (`sabin.rule_outcome`) are the baseline to beat; TF-IDF + logistic regression is the first learned reader ([02_paul_text_models.ipynb](../notebooks/02_paul_text_models.ipynb)) |
| 11 (transformers) | Contextual embeddings; class imbalance makes some classes harder to learn | Frozen DistilBERT and MiniLM embeddings; class-weighted loss for the 2:1 blocked/proceeded imbalance; paragraphs cut from the left so the last event survives |
| 8 + 12, applied to tables | A pre-trained transformer for tabular data (TabPFN) tests both Lecture 8's "tree ensembles usually win on tables" and Lecture 12's "pre-training wins when data is scarce" | Optional TabPFN in [models.py](../src/queue_risk/models.py); run in [Colab](../notebooks/tabpfn_colab.ipynb) |
| 12 (pre-train / fine-tune) | Three readers: counting words, frozen BERT, fine-tuned BERT; small learning rate, few epochs; **per-class recall** on the rare class; automation vs. capability expansion | [02_paul_text_models.ipynb](../notebooks/02_paul_text_models.ipynb) runs the ladder: keywords → TF-IDF → frozen DistilBERT / MiniLM → fine-tuned DistilBERT, trained on Claude's labels and tested once on the human ones, with recall on blocked/delayed. Capability expansion is the pitch: screen *every* queue entry, not the few an analyst can read |
| 13 (computer vision) | From pilot to rollout; validation and test can differ; start small and monitor | Caveats slide: queue rules change (e.g. FERC Order 2023), so the model needs retraining |
| 14 (ViT) | Data augmentation grouped by item; embeddings support retrieval | The mean MiniLM embedding of a county's earlier paragraphs is a stage-2 feature. Possible Project 2 extension: retrieve similar past projects for a due-diligence memo |

## Habits from the demo READMEs worth copying

- **"Where the numbers come from."** A checkpoint table listing every number the slides quote. The export script refuses to write if any of them move.
- **"Two things worth knowing about the data."** Name the dataset's quirks plainly, such as right-censoring, the 135 duplicate IDs and the one garbled FIPS code.
- **Say what the model is not.** It ranks how likely a project is to be built; it doesn't claim that backing a project makes it more likely to be built (Lecture 5 demo: "Nothing here says enrolling a patient reduces their cost").
- **Assumptions in one block.** Cost per project and K live at the bottom of [config.py](../src/queue_risk/config.py), are printed in the page footer, and are swept, never quoted as a single number.
