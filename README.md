# EEG Emotion Recognition with Swin Transformer

A course-project adaptation of **ST-SHAP** to the **EmoEEG-MC** dataset, exploring emotion classification across frequency bands, recording contexts and individuals.

## Overview

How well do spatial EEG representations transfer to a different person? This project explores that question using a published Swin Transformer-based framework and differential-entropy (DE) features from 64-channel EEG.

The original work was a Spring 2025 BMEB215 course project at Southern University of Science and Technology. The team comprised **Zhenghan Zhang, Hongjia Liu, Shuoshuo Yang and Jiandong Liu**. Hongjia Liu was the primary contributor to implementation and experiments. The project adapted the dataset interface, electrode-grid representation, label settings and band/context comparisons. The Swin backbone and ST-SHAP method originate from the cited authors.

The repository separates **historical course-project notebooks** from a **subsequent evaluation revision**. The revised pipeline and documentation were prepared with AI assistance in November 2026. They are follow-up work, separate from the original course submission.

## Method

1. Read provider-supplied DE features in delta, theta, alpha, beta and gamma bands.
2. Map 64 electrodes onto a 9 x 9 grid and interpolate each one-second feature map to 32 x 32.
3. Stack four consecutive maps from one frequency band into a `4 x 32 x 32` input.
4. Classify using the inherited Swin backbone: patch size 4, embedding width 96, stage depths 2/2/6/2 and attention heads 3/6/12/24.
5. Explore three-class and seven-class labels under imagery, video and pooled-context conditions. Pooled-context training does not establish transfer from one context to another.

The three-class task merges sadness/disgust/fear into negative, retains neutral, and merges joy/tenderness/inspiration into positive. The resulting label ratio is **3:1:3**. An always-majority prediction therefore gives **42.9% accuracy**; balanced accuracy and macro-F1 complement accuracy.

## Experimental status

The historical notebooks contain selected within-subject peaks up to 96.5% and a printed held-out-subject observation of approximately 51.7%. These are exploratory values: random window splits, normalization before splitting, and test-based checkpoint/fold selection prevent interpreting them as independent generalization scores. They are documented in the [short project report](docs/PROJECT_REPORT.md), without being promoted to benchmark results.

The revised pipeline uses training-only normalization, trial-grouped within-subject splits, a separate validation subject for LOSO, majority/logistic-regression baselines, and saved predictions and split indices. **It has passed synthetic software checks; a full real-EEG rerun is pending.** The original poster's SHAP illustration has not been independently reproduced from the available project files.

## Repository layout

```text
src/                    Revised preprocessing and evaluation pipeline
historical/             Two representative original notebooks, outputs cleared
third_party/st_shap/    Inherited backbone and MIT license
docs/PROJECT_REPORT.md  Methods, observations and evaluation limitations
docs/VALIDATION.md      Software checks and tested environment
tests/                  Synthetic protocol and integration checks
```

The historical notebooks preserve exploratory code, including known issues and external cache dependencies. Use `src/pipeline.py` for revised experiments; see [provenance](docs/PROVENANCE.md) for the relationship between versions.

## Setup and data

Use Python 3.10 or later:

```sh
python -m pip install -r requirements.txt
python tests/check_protocol.py
```

Download the **DE derivatives** from the [EmoEEG-MC dataset](https://openneuro.org/datasets/ds005540/versions/1.0.7). Place them beneath `data/`; files are searched recursively:

```text
data/sub-01/sub-01_ses-ima_task-emotion_de.npy
data/sub-01/sub-01_ses-vid_task-emotion_de.npy
```

Supported shapes are `(64, 630, 5)` and `(64, 21, 30, 5)`, in the provider's reordered trial order: sadness, disgust, fear, neutral, joy, tenderness, inspiration, with three trials each. Only complete subjects 01-54 are supported. Subjects with missing trials require event-based labeling. Record the exact dataset version with each experiment; the historical version is unconfirmed.

## Run experiments

```sh
python src/pipeline.py prepare --data-root data --output cache/gamma3.npz --band 4 --classes 3
python src/pipeline.py run --cache cache/gamma3.npz --model majority --protocol loso --subject 15 --output runs/majority_s15
python src/pipeline.py run --cache cache/gamma3.npz --model logreg --protocol loso --subject 15 --output runs/logreg_s15
python src/pipeline.py run --cache cache/gamma3.npz --model swin --protocol loso --subject 15 --epochs 40 --output runs/swin_s15
```

For within-subject evaluation, use `--protocol within --subject 15 --fold 0`, then run folds 1 and 2 into different output directories. All windows sharing a trial index, including corresponding indices across pooled contexts, stay together. LOSO reserves a development subject for validation and evaluates the held-out subject after checkpoint selection. This fixed validation-subject design is not fully nested cross-validation.

Runs export accuracy, balanced accuracy, macro-F1, a confusion matrix, per-sample predictions, split indices, dependency versions and data hashes. Normalization-based models save fitted training statistics; Swin also saves its selected checkpoint. Existing output directories are not overwritten. Summarize every prespecified test subject/fold; select hyperparameters, bands and contexts using development data only.

## References and attribution

- Miao et al. *ST-SHAP: A hierarchical and explainable attention network for emotional EEG representation learning and decoding.* Journal of Neuroscience Methods 414, 110317 (2025). [Paper](https://doi.org/10.1016/j.jneumeth.2024.110317) · [Author code](https://github.com/llljinjinjin/ST_SHAP_code).
- Xu et al. *A Multi-Context Emotional EEG Dataset for Cross-Context Emotion Decoding.* Scientific Data (2025). [Paper](https://doi.org/10.1038/s41597-025-05349-2) · [Provider code](https://github.com/ncclab-sustech/EmoEEG-MC).
- Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV (2021).
- Lundberg and Lee. *A Unified Approach to Interpreting Model Predictions.* NeurIPS (2017).

The inherited backbone retains its MIT license. A one-line stochastic-depth correction is documented in [THIRD_PARTY.md](THIRD_PARTY.md). No blanket license is newly assigned to the team notebooks. Participant data, pretrained weights and third-party journal PDFs are not distributed here.
