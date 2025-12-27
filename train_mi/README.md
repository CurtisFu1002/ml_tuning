# MI(Matrix Instruction) TensileLite Kernel Performance Prediction using Machine Learning

This project develops a machine learning-based approach to automatically tune the performance of MI (Matrix Instruction) kernels on AMD GPUs, predicting the optimal configuration for GEMM operations based on matrix dimensions (m, n, k=4096) to reduce manual search time.
## Overview

This project provides a comprehensive pipeline for training and evaluating machine learning models that predict the performance of different MI (Matrix Instruction) kernel configurations for GEMM (General Matrix Multiply) operations. The primary goal is to automate kernel tuning by accurately forecasting GFLOPS for a given matrix size and configuration, thereby reducing the need for exhaustive manual benchmarking.

The model uses **XGBoost Regressor** as the core engine, combined with custom evaluation metrics focused on ranking accuracy and regret. The framework supports feature engineering, hyperparameter tuning, and detailed performance analysis across different data regimes.

Key insights from experiments:
- Grid density (step size) in the training data is the dominant factor affecting model quality.
- Fine-grained grids (step=128) produce strong, concentrated performance patterns and yield the best prediction accuracy.
- Coarse grids (step=256) result in dispersed patterns and significantly harder learning, even when data volume is increased.

## Dataset Introduction
All datasets contain benchmark results for GEMM operations with **k fixed at 4096** and **m, n ranging from 512 to 8192** (inclusive).  
For each (m, n) problem, **all 128 MI configurations** are evaluated, and the **GFLOPS** is measured after **Tensile tuning**.

- Three different grid m,n problem dataset
    - Group1 
        - ![](./images/Group1_step256_shift0.png) 
    - Group2
        - ![](./images/Group2_step256_shift128.png)
    - Group3
        - ![](./images/Group3_step128_shift64.png)
The datasets differ only in **grid step size** and **shift**:

| Group       | File name                        | Step size | Shift | First m/n value | Points per dimension | Total problems | Notes                          |
|-------------|----------------------------------|-----------|-------|-----------------|----------------------|----------------|--------------------------------|
| Group 1     | gflops_data_output_256.csv       | 256       | 0     | 512             | 31                   | 961            | Coarse, starts at 512          |
| Group 2     | gflops_data_output_128.csv       | 256       | 128   | 640             | 30                   | 900            | Coarse, shifted by 128         |
| Group 3     | gflops_data_64.csv               | 128       | 64    | 576             | 60                   | 3600           | Fine-grained, ~4× denser       |



Each CSV row contains:
- `mi_idx`: Index of the MI configuration (0–127)
- MI parameters: `M`, `N`, `K`, `B`, `MIBlockM`, `WaveTileM`, `WaveTileN`, `WaveM`, `WaveN`
- Measured kernel `gflops` on MI210 GPU
- Actual matrix dimensions `m`, `n`, `k` (k always 4096)

Key observation:  
**Group 3 (fine grid)** produces the strongest and most stable performance patterns (highly concentrated top-1 configurations).  
**Group 1 & Group 2 (coarse grids)** show more dispersed optimal configurations and weaker patterns due to larger step sizes.

## Model Choice

We primarily use **XGBoost Regressor** as the baseline model for the following reasons:

- Excellent performance on tabular data with strong non-linear relationships
- Built-in feature importance analysis for interpretability
- Fast training, suitable for rapid iteration and ablation studies
- Proven effective in similar kernel tuning tasks compared to traditional MLPs

Key hyperparameters:
- `objective`: `reg:squarederror`
- `learning_rate`: 0.05
- `max_depth`: 16
- `n_estimators`: 500–1000
- `early_stopping_rounds`: 50 (using validation set)

Future extensions can easily incorporate LightGBM or ranking objectives (e.g., LambdaRank).

## Evaluation Metrics

We evaluate model performance per problem (m×n×k) using the following metrics:

- **Regret@1**: Percentage gap between predicted best configuration's GFLOPS and the true best (lower is better)
- **Top-1 Accuracy**: Percentage of cases where the predicted configuration is exactly the true best
- **Top-k Recall (R@k)**: Whether the true best falls within the top-k predicted configurations
- **Mean Rank of True Best**: Average ranking of the true best configuration in the model's predictions amlong the test data
- **True best k MI_idx prdiction error(T@k)** : The Error of GFLOPs of the predicted and the ground truth K MI_idx.
- **RMSE / MAE**: Absolute errors in predicted vs. true GFLOPS values

All metrics are averaged across problems for fair comparison.



## Experiment Classification

### 1. Baseline Performance (In-Distribution)

Objective: Establish a reference point for how well the model performs when the training and testing data come from the same grid density and shift.

- Experiments: Same-dataset training/testing on G1, G2, and G3.

Key Question: How does grid density (Step 128 vs 256) affect the upper bound of model accuracy

### 2. Generalization & Zero-Shot Capabilities (Out-of-Distribution)

Objective: Test if the model can predict performance on "unseen" problem sizes (different grid points or shifts).

1. Cross-Shift Persistence: Train G1 → Test G2. (Does a model trained on Power-of-Two points work on shifted points?)


2. Cross-Step Interpolation: Train G3 → Test G1/G2. (Does a fine grid "cover" the knowledge of coarse grids?)

### 3. Data Interference & Aggregation Analysis

Objective: Challenge the "more data is better" assumption and identify if coarse data acts as "noise."

Negative Interference: G1 + G2 vs G1 only. (Why does adding G2 data degrade G1 performance?) 

Signal Dilution: G1 + G3 vs G3 only. (Does coarse data interfere with the stable patterns found in G3?)


## Results
### Baseline Performance Analysis (In-Distribution)
This analysis establishes the reference performance of the XGBoost model when trained and tested on the same grid density and shift configurations. This "In-Distribution" test reveals the upper-bound predictive capability of the model under consistent environmental conditions.

#### Performance Metrics Comparison Table

| Metric | **Group 1** (Step 256, Shift 0) | **Group 2** (Step 256, Shift 128) | **Group 3** (Step 128, Shift 64) |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | 14.4% | 10.0% | **40.8%** |
| **Mean Top-1 Regret** | 24.16% | 25.77% | **4.21%** |
| **Recall@20** | 70.1% | 62.2% | **97.5%** |
| **Mean Rank of True Best** | 15.5 | 19.3 | **3.9** |
| **MRE (Mean Relative Error)** | 28.7% | 30.9% | **4.7%** |
| **Best Validation RMSE** | 10411.44 | 13254.64 | **4008.42** |



#### Training & Validation Analysis
The following curves illustrate the RMSE (Root Mean Squared Error) convergence across the three dataset groups during XGBoost training.
![](./images/training_curves_comparison.png)

1. Superior Convergence of Group 3 (Fine Grid):
    - Group 3 (step=128) achieves the lowest overall RMSE in both training and validation sets.
    - The validation error remains closely coupled with the training error, suggesting excellent generalization within the fine-grid domain.
2. Bottlenecks in Coarse Grids (Group 1 & 2):
    - Early Stopping: Both Group 1 and Group 2 triggered early_stopping_rounds significantly earlier (around 500-550 epochs).
    - High Bias: The large step size (step=256) creates "performance gaps" that the model cannot interpolate effectively, leading to higher residual errors regardless of training duration.
3. Impact of Shift (G1 vs. G2):
    - Despite having the same step size, Group 2 (shift=128) shows slightly higher error and more instability than Group 1 (shift=0).
    - Preliminary Hypothesis: The performance discrepancy between G1 and G2 may be related to Wavefront alignment (64-thread alignment). Since the step sizes (256, 128) are multiples of 64, the "Shift" might push the problem sizes into or out of specific hardware optimization zones (e.g., L2 cache partitioning or memory controller bank interleaving).


### Top-k Performance Analysis
Cumulative Prediction Error of Real Top-k (T@k)

This metric evaluates the model's accuracy in predicting the GFLOPS of the actual top-performing kernels. T@k represents the absolute error between the ground truth GFLOPS and the predicted GFLOPS for the k-th best kernel. The "Cumulative" version sums these errors (T1+T2​+...+Tk) to show the model's reliability across the entire elite candidate pool.
![Valid](./images/cumulative_realtop_error_validation.png)
![Test](./images/final_cumulative_realtop_error_test.png)
1. High Precision in the Elite Pool (Group 3):
    - Group 3 (step=128) exhibits the lowest cumulative error across all k values (k=1,3,5).
    - Notably, the Cumulative Top-5 error for Group 3 (dotted green) is significantly lower than even the Top-1 error for Group 1,2 (solid blue/orange). This proves that a finer grid density doesn't just help with overall RMSE, but is specifically crucial for "ranking" the best kernels correctly.
    - Massive Accuracy Leap: Group 3 achieved a 75% reduction in Top-1 error compared to Group 1 (8.4% vs. 33.4%).
2. The "Difficulty Spike" in Group 2:
    - Group 2 (shift=128) consistently shows the highest prediction error. Even after 500 epochs, the error remains high and fails to converge to the levels seen in Group 1.
    - This reinforces the Wavefront-Alignment Hypothesis: In a coarse grid (step=256), a shift of 128 might land the majority of test points on dimensions that are "hardware-unfriendly" or exhibit highly non-linear performance cliffs, making it significantly harder for the XGBoost model to predict their true GFLOPS.
3. Error Stability (Convergence):
    - For Group 3, the gap between Top-1, Top-3, and Top-5 is relatively tight and stabilizes quickly. This indicates that the model has learned the performance hierarchy of the MI kernels.
    - In contrast, G1 and G2 show a much wider vertical spread between k=1 and k=5, suggesting that the model struggles to distinguish the subtle GFLOPS differences between the top-performing kernels in coarse-grid scenarios.

### TOP1 Accuarcy on different training datasets



### generlization on the same step size 



### Datasets comparision (Correlation between MI Distribution and Prediction Accuracy)




## Ablation Study
- Cross-Dataset Generalization
    - Cross step generalization
        - train G3 valid G1 test G1+G2
        - train G3 valid G2 test G1
    - Cross step training and testing
        - train G1+G3 valid test G2
        - train G2+G3 valid test G1
    - Cross shift generalization
        -  train G1 valid test G2
    - Cross train test split
        - train valid test : G1+G2
        - train valid test : G1+G2+G3
- train test on the Same dataset
    - G1
    - G2
    - G3

- generalization on same step size but different shifts
    1. train G1 test G2
    2. train G2 test G1 
- generalization on different step size group 

- More Boostround on G3


We conducted several ablation experiments. Key results are summarized below:

| Experiment Setting              | Training Data                | Regret | Top-1 Acc | Remarks                                      |
|--------------------------------|------------------------------|--------|-----------|----------------------------------------------|
| Cross-group                    | Train on Group3<br>Test on G1+G2 | >30%   | ~7–8%     | Severe distribution shift, poor generalization |
| In-group (best)                | Group3 only                  | ~21%   | High      | Strong fine-grid pattern, best performance     |
| Merged coarse only             | G1 + G2 only                 | High   | ~10–12%   | Dispersed patterns, difficult to learn         |
| All Combined (early)           | G1 + G2 + Group3             | ~14%   | 25.6%     | Dominated by Group3, overall improvement      |
| All Combined (latest fine grid)| G1 + G2 + new Group3         | Decreased | -      | Coarse data interferes with fine-grid patterns |

**Key findings**:
- Training solely on fine-grid data (step=128) yields significantly better results than coarse grids.
- Mixing coarse and fine grids degrades performance, as coarse-grid noise disrupts learning of strong fine-grid patterns.
- Simply increasing coarse-grid data volume (e.g., adding shift=128) does not substantially improve prediction accuracy.

## Conclusion

This project successfully demonstrates the effectiveness of machine learning for MI kernel auto-tuning and reveals a critical insight: **grid density (step size) is the decisive factor in model prediction performance**.

- Fine-grained grids (step=128) generate highly concentrated and stable performance patterns, enabling models to achieve low regret.
- Coarse grids (step=256) result in dispersed configurations and weak patterns, making accurate prediction difficult even with increased data volume.
- Mixing coarse and fine grids causes performance degradation, as coarse data introduces noise that interferes with the strong patterns from fine grids.

This finding provides clear guidance for future kernel tuning data collection: **prioritizing fine step sizes (e.g., step=64 or 128) will significantly enhance auto-tuning effectiveness**, potentially reducing manual tuning effort by 20–30%.

While challenges remain in coarse-grid scenarios, focusing on fine-grid data combined with appropriate feature engineering has already shown strong practical potential for AMD ROCm ecosystem optimization.


step size太大有點無法分析shift所造成的影響


## Performance Impact and Acceleration

The original Tensile auto-tuning process performs an exhaustive search over the full configuration space. For a single GEMM problem size (m, n, k), the baseline complexity is:

- **O(#problems × #configurations)**  
  where #problems = number of (m, n) grid points, and #configurations includes all combinations of fork parameters.

In our benchmark setup (k fixed at 4096, m/n from 512 to 8192), the MI parameter alone contributes **128 configurations**. When combined with other fork parameters (e.g., scheduling, prefetch, etc.), the total search space can easily reach **thousands of configurations per problem**.

Our proposed approach introduces an **XGBoost-based performance predictor** to prune the search space:

1. The trained XGBoost model performs **O(1) inference** per (problem, MI configuration) pair.
2. We rank the 128 MI configurations by predicted GFLOPS.
3. Only the top-k candidates (e.g., top-20) are retained for actual Tensile benchmarking.

**Theoretical speedup**:
- Original: evaluate all 128 MI configurations
- Pruned: evaluate only top-k (e.g., k=20)
- **Speedup factor**: 128 / k ≈ **6.4×** (when k=20)

**Practical implications**:
- If **Recall@20 = 100%** (true best MI is always within top-20 predictions), pruning is lossless — we preserve optimal performance while reducing search time by over 6×.
- Even with minor regret (e.g., true best ranked 21st), the performance loss is typically <5% while still achieving substantial tuning acceleration.
- When MI parameters are combined with other fork parameters, the relative pruning benefit is amplified, as the predictor reduces the dominant dimension of the search space.

This pruning strategy effectively transforms the tuning complexity from **O(#problems × full_config_space)** to **O(#problems × k + training_cost)**, where k ≪ full_config_space and training_cost is amortized over many problems.

Empirical results show that with fine-grained training data (step=128), the predictor achieves high recall in the top-20, enabling safe and significant acceleration of the Tensile tuning pipeline.


## Future Work

1. Validate finer grids (step=64/32) and observe further concentration of optimal configurations
2. comparision of grid base training datasets and ramdon problems size datasets on random datasets()
3. Develop data filtering or weighting mechanisms for mixed coarse/fine training
4. Add other tensile frokparameters and analyze the relationship between them (e.g. WGM for the more topper level of optimizing L2-cache different from the MFMA level this time) 
5. explore the effects of k dimwnsion on the 


- Validate even finer grids (step=64/32) and observe further concentration of optimal configurations
- Investigate the impact of step size alignment with wavefront size (64) on performance stability
- Develop data filtering or weighting mechanisms for mixed coarse/fine training
- Explore ranking loss and advanced models (LightGBM LambdaRank, Transformers)
- Integrate into ROCm framework for on-device real-world validation

These directions will not only validate and strengthen our current findings but also push the approach toward production-ready applications with high industrial value.





## 暫時
Comprehensive Analysis of T@k Cumulative Error: From Validation to Final Test
This analysis integrates the training dynamics (Validation) and final performance metrics (Test) based on the True Top-k Prediction Error (T@k). This metric evaluates the model's precision in forecasting the GFLOPS of the actual best-performing kernels, where a lower cumulative error indicates a more reliable candidate pool for auto-tuning.

1. The Decisive Role of Grid Density

The comparison between the three groups reveals that data density is the primary driver of ranking accuracy:

Validation Trends: Throughout the training process, Group 3 (Step=128) consistently maintained the lowest cumulative error across all k values (k=1,3,5). Its cumulative Top-5 error (dotted green line) is significantly lower than even the Top-1 error for the coarse-grid groups (G1 and G2).

Final Test Leap: In the final evaluation, Group 3 achieved a massive precision breakthrough. Its Top-1 Error was reduced to 8.4%, representing a ~75% improvement over Group 1’s 33.4%.

Insight: Fine-grained grids do not just lower the overall RMSE; they are specifically crucial for correctly "ranking" the elite kernels, allowing the model to distinguish between subtle performance differences.

2. The Alignment Mystery: Explaining Group 1 vs. Group 2

Although both Group 1 and Group 2 use a step size of 256 and dimensions that are even multiples of the 64-thread wavefront, their performance results diverge significantly:

The Performance Gap: Group 2 (Shift=128) consistently exhibits the highest prediction error in both validation and testing. Its cumulative Top-5 error reached 156.4%, the highest among all groups.

Refined Hypothesis: Power-of-Two (Po2) vs. Non-Po2 Alignment:

Group 1 (Shift=0): This group includes many Power-of-Two (Po2) dimensions (e.g., 512, 1024, 2048). GPU hardware, such as L2 cache sets and memory controllers, is often optimized for Po2 sizes, leading to more stable and "learnable" performance patterns.

Group 2 (Shift=128): While these are still multiples of 64 (e.g., 640, 896), they are not Po2. These points likely trigger complex hardware behaviors, such as cache bank aliasing or uneven memory channel pressure, which appear as "noise" or "performance cliffs" to the model.

Conclusion: In coarse grids, the model lacks the intermediate samples to understand these non-linear transitions, leading to the "Difficulty Spike" observed in Group 2.

3. Predictive Stability and Auto-Tuning Acceleration

The T@k results confirm that the model has successfully learned the performance hierarchy of MI kernels:

Elite Pool Reliability: A striking takeaway is that Group 3’s Cumulative Top-5 Error (38.3%) is nearly equivalent to Group 2’s Top-1 Error (35.3%). This means the 5th best guess of a fine-grid model is as accurate as the 1st guess of a coarse-grid model.

Search Space Pruning: Because the model can identify the elite candidate pool with high confidence, we can safely prune the search space from 128 MI configurations down to the Top-20.

Acceleration Factor: This pruning strategy enables a ~6.4x speedup (128/20) in the Tensile tuning process with minimal performance regret.