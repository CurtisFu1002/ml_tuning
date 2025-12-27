#!/bin/bash

echo "=========================================="
echo "  Tensile MI Configuration Optimizer"
echo "=========================================="

# Configuration parameters
MODEL_PATH="../model/G3_1500round.xgb"
YAML_PATH="./speedup_test_logic.yaml"
OUTPUT_DIR="./optimal_configs"
TOP_K=20

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if YAML file exists
if [ ! -f "$YAML_PATH" ]; then
    echo "Error: YAML file not found: $YAML_PATH"
    exit 1
fi

# Run optimization
echo "Starting optimization..."
echo "  Model: $MODEL_PATH"
echo "  YAML: $YAML_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Top-K: $TOP_K"
echo ""

python3 predict_mi_and_config_gen.py \
    --model "$MODEL_PATH" \
    --yaml "$YAML_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --top-k "$TOP_K"

# Check if optimization succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo " Optimization completed successfully"
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Summary Report: $OUTPUT_DIR/optimization_summary_top${TOP_K}.txt"
    echo ""
    echo "Generated optimized YAML files:"
    ls -lh "$OUTPUT_DIR"/*.yaml 2>/dev/null || echo "  (No YAML files generated)"
else
    echo ""
    echo "Optimization failed"
    exit 1
fi