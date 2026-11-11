# Adapting ST-SHAP to EmoEEG-MC

**Course project:** BMEB215, SUSTech, Spring 2025.  
**Team:** Zhenghan Zhang, Hongjia Liu, Shuoshuo Yang, Jiandong Liu.  
**Primary implementation and experimental contributor:** Hongjia Liu.

This report and the revised pipeline were prepared with AI assistance in November 2026. The original project is an adaptation of published work; the subsequent revision has been checked using synthetic inputs and awaits real-data evaluation.

## Research question

EEG classification can benefit from patterns specific to a participant or trial. This project explores how a spatial EEG representation behaves across frequency bands, imagery/video conditions and held-out participants. It builds on ST-SHAP rather than proposing a new attention architecture.

## Data and representation

The notebooks read EmoEEG-MC's provider-supplied differential-entropy derivatives, covering 64 electrodes and five frequency bands. Complete records contain 21 trials of 30 seconds per context. Seven emotions are evaluated either separately or merged into negative, neutral and positive labels.

For one band, the principal within-subject implementation divides each trial into seven nonoverlapping four-second windows, leaving the final two seconds unused. Each second's 64 electrode values is placed on a 9 x 9 grid, with unoccupied positions set to zero, then interpolated to 32 x 32. Four consecutive maps form the model's four input channels. The grid is an engineering representation rather than measured anatomical coordinates; padding and interpolation may affect predictions.

## Model and project contribution

The Swin model uses 4 x 4 patch embedding, width 96, four stages with 2/2/6/2 blocks and 3/6/12/24 attention heads. Its backbone file in the original project is identical to the bundled ST-SHAP author implementation.

The project-level contribution is the EmoEEG-MC interface, electrode mapping, three-/seven-class task settings and band/context experiment organization. Hongjia Liu carried out most of the implementation and experiments within the four-person team. No first-publication claim is made for these adaptations, and the individual ownership of every module is not separately recorded.

## Historical observations

| Observation | Supported interpretation |
| --- | --- |
| Subject 15 selected within-subject maximum: 96.5% | Best epoch and best fold, rather than average CV accuracy |
| Subject 15, pooled-context alpha: printed 51.7% at epoch 20 | A rounded exploratory LOSO observation, with the test subject repeatedly evaluated during selection |
| Seven-class task code | Evidence of exploration; a completed validated aggregate is not established |
| SHAP frame ranking in the poster | A historical illustration; corresponding EmoEEG-MC execution artifacts were not recovered |

The within-subject notebook fits normalization before splitting and randomly splits windows, allowing windows from the same trial into training and evaluation. It then selects evaluation epochs and folds. The LOSO notebook uses the held-out subject to select training epochs. These issues prevent treating the historical numbers as independent test scores or using their difference to quantify a generalization penalty. Saved notebook outputs can also survive later edits to the source.

The three-class label ratio is 3:1:3, so a constant majority predictor attains 3/7 accuracy (42.9%), whereas uniform random guessing gives 1/3. A selected accuracy exceeding 1/3 alone does not demonstrate useful or statistically reliable classification. Balanced accuracy, macro-F1 and per-class performance are necessary complements.

The bundled upstream explainer targets SEED; its presence does not establish a project-specific EmoEEG-MC SHAP result. A frame's position within a four-second input window also cannot identify its position within a full 30-second trial. No causal or clinical interpretation is established.

## Revised evaluation and next steps

The revised pipeline keeps trial groups together and fits normalization using training data only. Within-subject evaluation uses three outer trial-grouped folds and a separate inner validation split. LOSO holds one subject out for test and reserves another development subject for validation. Swin weights are selected by validation balanced accuracy; the held-out test is evaluated once afterward. Majority and logistic-regression models provide comparison baselines.

Software checks cover split isolation, normalization, baseline runs, preprocessing and a full one-epoch synthetic Swin run. **No revised real-EEG result is reported.** Recovering the dataset and experiment inputs is necessary to run a prespecified set of subjects/folds, report participant-level variability, and compare models on identical splits. Band/context selection must remain within development data. Pooled-context training does not establish cross-context transfer, and LOSO does not establish unseen-stimulus generalization.

For a reproducible SHAP analysis, retain checkpoint provenance, training-only background sample IDs, explained sample IDs, target classes and saved attribution arrays. Attribution stability across seeds and meaningful perturbations would provide stronger evidence than a single illustration.

## References

1. Miao et al. ST-SHAP. Journal of Neuroscience Methods 414, 110317 (2025). [DOI](https://doi.org/10.1016/j.jneumeth.2024.110317).
2. Xu et al. A Multi-Context Emotional EEG Dataset for Cross-Context Emotion Decoding. Scientific Data (2025). [DOI](https://doi.org/10.1038/s41597-025-05349-2).
3. [EmoEEG-MC derivatives](https://openneuro.org/datasets/ds005540/versions/1.0.7). The historical poster cites version 1.0.3; the exact historical input version is unconfirmed.
