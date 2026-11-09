# Evaluation protocol and change log

The protocol below was fixed before any Stage 1 or Stage 2 result on this data existed. Every change made after results existed is listed in the change log with its reason.

## Data
- The 27 attacked events (9 per split) are 9 attack runs reused across the training, validation and test splits with small changes. Results therefore show that known attack scenarios are recognized again, not that the method generalizes to new runs or scenarios.
- Each dynamic attack type has one fixed target.

## Stage 1 (detection)
- Training follows `docs/previous_work.md`: Adam at 1e-3, 20 epochs, one sequence per step, normal sequences only.
- Five training runs with random seeds 42, 1, 2, 3 and 4. Results are reported as mean and standard deviation, with counts.
- Threshold: best validation F1 over the documented quantile candidates, with ties resolved by precision and then recall. A sequence is flagged when its score exceeds the threshold. The threshold is applied unchanged to test.
- The threshold is selected on normal and protocol-compliant (D2, D3, G2, G4) validation sequences.
- Coverage is reported per sequence (detected attack sequences over attack sequences) and per event (an event counts as detected if a flagged sequence contains one of its windows).
- Settings not fixed by the earlier notes: PyG NNConv defaults (root weight and bias on), target not detached, epoch order shuffled, and the epoch-20 model is used without model selection.
- The edge network is a single Linear layer, as in the original notebook, and the root weight W_root is included.

## Stage 2 (diagnosis)
- Scored on true alerts of the protocol-compliant classes D2, D3, G2 and G4, reported per training run and per event, with counts beside every percentage and the chance level.
- All tuning uses training and validation events only. The test split is scored once after code and settings are fixed, with no retuning afterwards.

## Ground truth for diagnosis
| Attack | Protocol | Device | Flow |
|---|---|---|---|
| D2 | DNP3 or TCP | SEL_451_72 | SCADA <-> SEL_451_72 |
| D3 | DNP3 or TCP | SEL_451_73 | SCADA <-> SEL_451_73 |
| G2 | GOOSE | not mapped | publisher stream |
| G4 | GOOSE | not mapped | publisher stream |

The D2 and D3 devices come from the testbed records. GOOSE streams are not mapped to devices, so G2 and G4 are scored on time and protocol only. No mapping is inferred from the traffic.

## Change log
1. The encoder weight layout was corrected to match PyG NNConv (checked by `tests/test_model.py`), and all training runs were repeated. No test data were involved.
2. Device localization changed from the whitened embedding deviation to exact node Shapley values, based on training and validation events only. Normal embeddings are nearly constant and some units are inactive (seed 4 has no varying dimension at SEL_451_72), so the whitened distance is unstable and ranked SCADA first for every G2 alert although SCADA publishes no GOOSE. The whitened and L2 versions are kept as ablations. The test split had not been scored at this point.
3. The reference search and the raw z-score baseline use z-scores from normal training windows, while the model keeps the repository scaling.
4. Disclosure: feature values of the repeated events were compared across all splits, including test, to check for copies. No score or result was computed on test. Repeated events are close (88-100% of values equal, against 87-97% for adjacent normal windows), so events of one type are not fully independent. G4 in the test split differs most.
5. One additional round of validation tuning was run. The selection rule was written before the candidates were compared: the highest mean of device top-1, protocol and flow accuracy on validation alerts, averaged over training runs, with ties within 0.02 going to the lower standard deviation and then to the simpler method. Training events served as a check only, and baselines received the same validation tuning.
6. Outcome of item 5 on validation alerts. Candidates: node Shapley with destination ownership, node Shapley with outstation ownership, L2 and whitened deviation for the device; group Shapley over all windows, at t*, and at the first affected window, mean |z| at t*, and max |z| over the sequence for the protocol; and Stage 1 with a detached target. Selected: node Shapley with outstation ownership, max |z| protocol, target not detached. Detaching lowered protocol accuracy by Shapley from 0.71 to 0.54. Group Shapley is kept as an ablation. Consequences reported in the paper: outstation ownership gives SCADA no link entries, so the D2 and D3 device answer is a choice between two links and the same ownership is applied to every baseline; max |z| does not use the model, so a model-free baseline with the same rule is included.
7. The mask baseline initially scored near zero because its size penalty was too weak. A validation grid over training runs 42 and 1 (initialization 1, 0, -1 and size weight 0.05, 0.5, 2) selected initialization 1 and size weight 0.5.
8. Settings were fixed after items 1-7. File hashes are in `results/stage2/frozen_hashes.txt`. The test split was then scored once.
9. After the test run, figures and coverage counts were generated from the fixed models (`src/make_paper_figures.py`) without changing any setting.

## Rule-based detector (supplementary)
10. A window-level rule-based detector (`src/rules.py`) was written from DNP3 and IEC 61850 GOOSE semantics on the unscaled window features. Rule design and refinement used training and validation windows only, and the test split was scored once after the rules were fixed. Disclosure: one exploratory count of the GOOSE stNum/sqNum rule was run over all splits, including test labels, before the check was restricted to training and validation. It showed the rule firing on every G1 window and on two normal windows. Nothing was tuned on it.
11. A first version of the GOOSE rule with an adjacent-window tolerance raised its alarm one window late, so two G1 events were missed and the following normal windows were flagged. It was replaced by the same-window invariant (stNum change XOR sqNum reset), which flags all four G1 windows in training and validation and two normal training windows (3310 and 3610). The rules were fixed after this change.
12. Test results with the fixed rules (`results/rules/test.json`): 15/15 deterministic sequences, 5/5 deterministic events, 0/370 normal sequences flagged, and the correct device for 3/3 D1 events. Combining these alerts with the fixed Stage 1 models (`src/hybrid.py`, `results/rules/hybrid_test.json`) gives P .904 +- .024, R .936 +- .014, F1 .919 +- .009, and 9/9 events in every training run.
