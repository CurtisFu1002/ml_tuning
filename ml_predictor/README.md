# MI Kernel Performance Predictor

This project uses an XGBoost-based machine learning pipeline to predict the performance of AMD GPU MI kernels and accelerate the Tensile tuning process.

##  1. Environment Setup

### Prerequisites
* **Hardware**: AMD GPU (MI210/MI300 series) for data collecting.
* **Software**: ROCm environment with `hipBLASLt` and `TensileLite` installed and builded.

### Installation
We recommend using the official ROCm Docker container. Detailed instructions for Docker usage and building Tensile with `mi_tune` can be found in the [Data Collection README](./data_collection/README.md).


##  2. Project Structure

The project is organized into two primary stages:

[Data Collection](./data_collection/): Tools to generate GEMM problems and run parallel benchmarking on multiple GPUs to collect GFLOPS data.

[Model Training (train_mi)](./train_mi/): Scripts for data analysis, training the XGBoost model, and evaluating prediction accuracy.


## 3. Results And Analysis 

All experimental data, logs, and visualizations are stored in the train_mi/ directory. You can find more details in [Final Report](./train_mi/README.md)


### Quick Stats Summary
- Top-1 Accuracy: 45.6%

- Recall@20: 97.2%

- Mean Regret: 3.67%

- Avg Inference Time: 0.37s per problem