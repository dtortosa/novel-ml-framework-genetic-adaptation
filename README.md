# Machine Learning & Model Optimization Framework

A high-performance machine learning framework for processing, modeling, and interpreting a large, high-dimensional dataset (~180 GB compressed).

It runs parallelized feature extraction across multiple cores and benchmarks several model classes—deep learning, XGBoost, random forest, and ElasticNet—using nested cross-validation and hyperparameter optimization.

## 🧠 System Architecture & Pipeline Execution

The workload is split into separate scripts, each handling one stage of the pipeline:

### 1. Parallel Feature Engineering
- **Implementation:** `00a_pressure_distance_calc_v1.py` & `00b_bat_distance_calc_v1.sh`
- Uses a Python `multiprocessing` pool to compute spatial distances over coordinate indices in parallel. Includes exception handling, percentage-based data-loss checks, and fallback handlers.

### 2. Multi-Algorithm Benchmarking
- **Implementation:** `01_models_benchmark.py` & `01_models_benchmark.slurm`
- Distributes model selection across Slurm cluster nodes using 400 parallel tasks.
- Trains several model types: deep neural networks (TensorFlow/Keras), gradient-boosted trees (XGBoost), and regularized linear models (ElasticNet).
- Uses **nested cross-validation** to prevent data leakage during hyperparameter tuning.

### 3. Results Post-Processing
- **Implementation:** `02_process_results_models_comparison.py`
- Parses and flattens the cross-validation results from `.tsv` files, decoding the embedded JSON into structured tables.

### 4. Stacking Ensemble
- **Implementation:** `03_explore_selected_model_class.py` & `03_explore_selected_model_class.slurm`
- Combines the best-performing models into a cross-validated stacking regressor (meta-learner), running over multiple random seeds to track variance.
- Saves the final models to disk as Gzip-compressed `joblib` files.

## 📊 Key Engineering & Performance Outcomes

- **Data scale:** Built pipelines to load, clean, and transform ~180 GB of high-dimensional tabular features.
- **Scaling:** Configured Slurm to distribute training across hundreds of CPU cores.
- **Model performance:** Found neural network architectures reaching an $R^2$ of **80%** on held-out test sets.
- **Explainability:** Used Permutation Feature Importance and Accumulated Local Effects (ALE) plots to interpret the non-linear deep models.

## 🛠️ Technical Stack

- **HPC orchestration:** Slurm, Singularity, Bash
- **Machine learning & math:** TensorFlow, Keras, XGBoost, Scikit-Learn, Joblib
- **Parallel computing:** Python multiprocessing (Pool and Starmap)
- **Data engineering:** Python, R, Pandas, NumPy, JSON
