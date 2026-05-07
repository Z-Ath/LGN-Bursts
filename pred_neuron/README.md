## Files

- `classification.py`: standalone CIFAR10 classifier comparison plus Grad-CAM and ViT attention-rollout visualization.
- `early_layer_emb.py`: builds ViT-B/16 embeddings from a selected encoder block for the low-layer analysis.
- `embedding_ridge_baseline.py`: predicts neural responses from familiar/deviant embeddings with PCA features and RidgeCV.
- `embedding_xgboost_baseline.py`: baseline XGBoost prediction pipeline using PCA features from familiar/deviant embeddings.
- `embedding_xgboost_optuna.py`: XGBoost prediction pipeline with optional Optuna hyperparameter search.
- `image_stat_control.py`: tests brightness or contrast image-stat controls before embedding and neuron-level importance analysis.
- `importance_control.py`: shared raw-data loaders, ViT embedding cache builder, control modes, SHAP importance, and familiar-vs-deviant plotting.
- `low_layer_neuron_importance.py`: repeats the neuron-level prediction and importance pipeline using an early ViT layer.
- `method_comparison_plots.py`: compares saved prediction-result files across baseline, Optuna, and random-target methods.
- `multiblock_neuron_importance.py`: uses separate familiar, deviant, diff, hadamard, abs-diff, sum, and support feature blocks for neuron-level importance.
- `neuron_control_importance.py`: neuron-level SHAP analysis with familiar/deviant control modes and paper-data export.
- `neuron_filtering.py`: ranks sessions and filters neurons by prediction quality.
- `random_image_control.py`: replaces familiar/deviant embeddings with embeddings from other images as a control.
- `random_target_control.py`: predicts randomized neural targets sampled from other sessions as a control.
- `sanity_check.py`: checks Burst-B trial counts, single-session prediction behavior, and saved per-dimension metrics.
- `single_session_neuron_plots.py`: creates single-session neuron diagnostics, bar plots, and prediction-vs-truth scatter plots.
- `unique_image_count_analysis.py`: plots neuron-level differences between unique familiar and deviant image counts.
