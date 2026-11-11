# Validation record

Prepared November 11, 2026. All new numerical checks use synthetic inputs; no real EEG training or SHAP attribution was run.

## Completed checks

- Twenty-four protocol combinations: three-/seven-class labels; imagery/video/pooled contexts; all three within-subject test folds and a LOSO split. Training, validation and test groups are disjoint and every class is represented.
- Held-out-data perturbation: adding a large constant to validation/test inputs leaves training normalization statistics unchanged.
- Majority and logistic-regression command-line runs for both class settings: successful metric/prediction exports and correct test sample counts.
- Stochastic-depth mask check: values are zero or inverse keep probability after the one-line correction.
- Full inherited Swin configuration: finite forward logits and successful gradient backpropagation on synthetic `2 x 4 x 32 x 32` inputs.
- Provider-schema preprocessing integration: synthetic `(64,630,5)` DE becomes `(147,4,32,32)`, with label counts 63/21/63 and all 21 trial identities preserved.
- Full one-epoch synthetic Swin LOSO run: validation-based checkpoint selection, one held-out test evaluation, output files and safe checkpoint reload complete successfully.

The integration check used 21 synthetic samples for each of one training, one validation and one test subject. This is a software exercise; its classification metrics are deliberately not presented as scientific results. Temporary data and checkpoints were removed automatically.

## Tested environment

Python 3.12.3; NumPy 1.26.4; SciPy 1.15.1; scikit-learn 1.6.1; PyTorch 2.7.0+cu118. Requirements list compatible direct dependencies, not an exact cross-platform environment lock. GPU portability and complete real-data runs remain unverified.

Run `python tests/check_protocol.py` for the protocol/model checks and `python tests/check_integration.py` for the preprocessing and full one-epoch synthetic training exercise. The integration check can take longer and uses more memory than the protocol checks.
