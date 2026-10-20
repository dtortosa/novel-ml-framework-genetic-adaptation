# Modeling Signals of Adaptation Across the Human Genome with Machine Learning

Machine learning framework to model signals of recent positive selection (iHS) across the human genome while controlling for genomic confounders such as recombination rate, GC content and the density of functional elements. It tests whether signals of adaptation increase near specific gene sets, such as genes involved in thermogenesis or virus-interacting proteins (VIPs).

The pipeline processes ~180 GB of compressed genomic data (1000 Genomes Project) into gene-level features. It then benchmarks elastic net, random forests, XGBoost and deep neural networks using nested cross-validation and Bayesian hyperparameter optimization, and interprets the selected models with explainable AI.

Results featured by the European Commission [CORDIS Results in Brief, 2025](https://cordis.europa.eu/article/id/459619-ancient-climate-adaptation-links-to-current-metabolic-health)

## Pipeline

Each stage of the pipeline is a separate script:

### 1. Parallel Feature Engineering
- **Implementation:** `00a_pressure_distance_calc_v1.py` & `00b_bat_distance_calc_v1.sh`
- Computes, for every protein-coding gene, the genomic distance to the nearest gene in each set of interest (e.g., VIPs, thermogenic genes), using a Python `multiprocessing` pool. Includes exception handling and checks on the percentage of data lost at each step.

### 2. Multi-Algorithm Benchmarking
- **Implementation:** `01_models_benchmark.py` & `01_models_benchmark.slurm`
- Distributes model selection across Slurm cluster nodes using 400 parallel tasks.
- Trains elastic net, random forests, gradient-boosted trees (XGBoost) and deep neural networks (multilayer perceptrons, TensorFlow/Keras).
- Uses **nested cross-validation**, so performance is evaluated on data never used during hyperparameter tuning, and Bayesian optimization (Optuna) to search the neural network hyperparameter space ( hyperparameters).

### 3. Results Post-Processing
- **Implementation:** `02_process_results_models_comparison.py`
- Parses and flattens the cross-validation results from `.tsv` files, decoding the embedded JSON into structured tables.

### 4. Stacking Ensemble
- **Implementation:** `03_explore_selected_model_class.py` & `03_explore_selected_model_class.slurm`
- Combines the best-performing models into a cross-validated stacking regressor, run over multiple random seeds to quantify variance.
- Saves the final models as Gzip-compressed `joblib` files.

## Results

- **Model performance:** the best model, a deep neural network (5 layers, 1,000 neurons), reached a median R² of 0.83 on held-out data in the Yoruba population.
- **Transfer:** without re-tuning, the same network kept R² ≈ 0.80 in other 25 populations across 4 continents.
- **Explainability:** permutation feature importance and accumulated local effects (ALE) plots showed that iHS increases near thermogenic genes, independently of the genomic confounders in the model.
- **Scaling:** model selection distributed over hundreds of CPU cores with Slurm.

## Technical Stack

- **HPC:** Slurm, Singularity/Apptainer, Bash
- **Machine learning:** TensorFlow/Keras, XGBoost, scikit-learn, Optuna, joblib
- **Parallel computing:** Python multiprocessing (Pool, starmap)
- **Data processing:** Python, R, Pandas, NumPy
