# High-Performance Distributed Machine Learning & Model Optimization Framework

An high-performance machine learning framework designed to process, model, and interpret massive, high-dimensional target datasets (~180 GB compressed). 

The platform implements multi-core, parallelized feature extraction alongside a model benchmarking engine that evaluates multiple model classes (Deep Learning, XGBoost, Random Forest, ElasticNet) through rigid statistical frameworks like nested cross-validation and hyperparameter optimization.

## 🧠 System Architecture & Pipeline Execution

The workload is decoupled into isolated, production-grade scripts managed via an asynchronous infrastructure layout:

### 1. High-Throughput Parallel Feature Engineering
- **Implementation:** `00a_pressure_distance_calc_v1.py` & `00b_bat_distance_calc_v1.sh`
- Implements an optimized, multi-threaded worker pipeline using Python's `multiprocessing` pool to calculate spatial matrix proximities over coordinate indices. Features robust exception mapping, strict percentage-based data-loss checks, and dynamic fallback handlers.

### 2. Distributed Multi-Algorithm Benchmarking Engine
- **Implementation:** `01_models_benchmark.py` & `01_models_benchmark.slurm`
- Scales out model selection matrices across high-priority Slurm supercomputing nodes utilizing 400 parallel compute tasks. 
- Orchestrates multi-model training sequences containing deep neural networks (TensorFlow/Keras), gradient boosted trees (XGBoost), and regularized linear frameworks (ElasticNet).
- Executes strict **Nested Cross-Validation** boundaries to protect predictive targets against hyperparameter data leakage.

### 3. Metric Post-Processing & Telemetry Extraction
- **Implementation:** `02_process_results_models_comparison.py`
- Parses, normalizes, and flattens cross-validated output records from raw serialization targets (`.tsv`) into structural data formats using programmatic JSON decoding abstractions.

### 4. Production Stacking Ensemble Serialization
- **Implementation:** `03_explore_selected_model_class.py` & `03_explore_selected_model_class.slurm`
- Pools the top-performing model architectures into a cross-validated Meta-Learner Stacking Regressor (combining multi-seeded variance tracking loops).
- Serializes the optimized champion models to persistent disk storage using high-efficiency Gzip-compressed binary payloads via `joblib`.

## 📊 Key Engineering & Performance Outcomes

- **Big Data Scale:** Engineered robust processing pipelines to ingest, clean, and transform ~180 GB of high-dimensional tabular features.
- **Infrastructure Scaling:** Configured SLURM task distribution matrices to scale out parallel multi-process training operations across hundreds of remote CPU cores.
- **Model Efficiency:** Discovered neural network topologies capable of achieving an $R^2$ score of **80%** on unexposed evaluation sets.
- **Explainable AI (XAI):** Integrated model interpretability vectors including Permutation Feature Importance and Accumulated Local Effects (ALE) plots to extract deterministic insights from highly non-linear, deep architectures.

## 🛠️ Technical Stack

- **HPC Cluster Orchestration:** SLURM, Singularity Containerization, Linux Bash
- **Machine Learning & Core Math:** TensorFlow, Keras, XGBoost, Scikit-Learn, Joblib
- **Parallel Computing:** Multiprocessing (Pool & Starmap worker abstraction routines)
- **Data Engineering:** Python, R Workspace, Pandas, NumPy, JSON Serialization
