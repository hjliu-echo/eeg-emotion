# Code provenance

| Location | Origin and role |
| --- | --- |
| `historical/ST-SHAP_3k_final.ipynb` | Original within-subject band/context exploration; code preserved, outputs/counters cleared |
| `historical/ST-SHAP_3single_final.ipynb` | Original held-out-subject exploration; code preserved, outputs/counters cleared |
| `src/electrodes.py` | Electrode grid extracted from the original within-subject notebook |
| `third_party/st_shap/SWIN_trans.py` | Inherited ST-SHAP backbone with one documented stochastic-depth correction |
| `src/pipeline.py` and `tests/` | AI-assisted evaluation revision prepared in November 2026, separate from the original course work |
| README and project report | Summaries of the source project and its current evidence limits, prepared with AI assistance |

The original notebook filenames are retained for traceability. New introductory cells distinguish historical exploration from the revised implementation. The notebooks require external data/caches and contain known preprocessing, evaluation and plotting issues; they are included as source examples, not as verified clean-run entry points. Their original `model.SWIN_trans` imports refer to the historical folder layout. Use the repository-root command-line pipeline for revised experiments.

[source_manifest.json](source_manifest.json) records SHA-256 hashes of the original notebook and backbone files before curation. They describe local source provenance; an upstream commit was not established. The source files are retained unchanged in the applicant's local project folder.

No raw EEG, cached participant features, pretrained checkpoints, private contact material or third-party paper PDFs are included. No new blanket license is assigned to the team notebooks; the inherited backbone's MIT notice is retained in `third_party/st_shap/LICENSE`.
