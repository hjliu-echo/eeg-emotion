# Third-party code and modifications

`third_party/st_shap/SWIN_trans.py` is adapted from the copy included in the supplied `ST_SHAP_code-main` directory, attributed to [llljinjinjin/ST_SHAP_code](https://github.com/llljinjinjin/ST_SHAP_code). Its original MIT license is retained alongside it. The original author repository is linked for reference; its benchmark results belong to the original authors.

The project-level `model/SWIN_trans.py` and the bundled author's model file are byte-identical, SHA-256 `d95d2b515f485e1cab7cc9f28231fb90f713c2be7311e1d5a66087969816b278`. No original architecture contribution should be inferred from that file. The original upstream commit is not recoverable from the supplied folder; provenance is limited to the local source hash and named author repository.

One follow-up correction is applied in the curated copy:

```diff
-    random_tensor.floor()  # binarize
+    random_tensor.floor_()  # binarize
```

Without assigning the returned tensor or modifying in place, `floor()` leaves the stochastic-depth mask continuous rather than Bernoulli. The revised pipeline therefore does not reproduce the historical training behavior exactly. This correction was made during the November 2026 AI-assisted preparation, not during the original course project. No other backbone changes were made.

The 64-channel grid in `src/electrodes.py` is extracted from the local course notebook and is a pragmatic grid representation, not measured electrode coordinates. Legacy notebooks are a four-person team artifact. No new license is asserted for them. Original journal PDFs, dataset derivatives, upstream sample attribution arrays and model weights are excluded.
