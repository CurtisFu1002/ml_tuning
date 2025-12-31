#!/bin/bash
# ======================================================================
#  Optimal MI Configuration Benchmark Runner
#  Purpose: Run Tensile benchmarks for optimized YAML configurations
#           and measure execution time for each problem size
# ======================================================================

set -e

# -----------------------------
# Configuration
# -----------------------------
OPTIMAL_DIR=${1:-./optimal_configs}
BUILD_DIR=${2:-/workspace/rocm-libraries/projects/hipblaslt/tensilelite/mi_tune}
RESULT_DIR=${3:-./benchmark_results}
GPU_ID=${4:-0}

TENSILE_BIN=${BUILD_DIR}/Tensile.sh
LOG_DIR=${RESULT_DIR}/logs
SUMMARY_FILE=${RESULT_DIR}/timing_summary.txt

# Create directories
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

echo "=============================================================="
echo " Optimal MI Configuration Benchmark"
echo "--------------------------------------------------------------"
echo " Optimal YAMLs  : ${OPTIMAL_DIR}"
echo " Build directory: ${BUILD_DIR}"
echo " Result directory: ${RESULT_DIR}"
echo " Tensile binary : ${TENSILE_BIN}"
echo " Target GPU     : ${GPU_ID}"
echo "=============================================================="

# Check if Tensile binary exists
if [ ! -f "${TENSILE_BIN}" ]; then
    echo "Error: Tensile binary not found at ${TENSILE_BIN}"
    exit 1
fi

# Check if optimal configs directory exists
if [ ! -d "${OPTIMAL_DIR}" ]; then
    echo "Error: Optimal configs directory not found: ${OPTIMAL_DIR}"
    exit 1
fi

# Set GPU device
export ROCM_VISIBLE_DEVICES=${GPU_ID}

# Ensure Tensile.sh has execute permission
cd "${BUILD_DIR}"
chmod +x Tensile.sh
dos2unix Tensile.sh 2>/dev/null || true

# Initialize summary file
echo "=============================================================" > "${SUMMARY_FILE}"
echo " Optimal MI Configuration Benchmark Summary" >> "${SUMMARY_FILE}"
echo "=============================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Benchmark Date: $(date '+%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_FILE}"
echo "GPU Device: ${GPU_ID}" >> "${SUMMARY_FILE}"
echo "Build Directory: ${BUILD_DIR}" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "-------------------------------------------------------------" >> "${SUMMARY_FILE}"
printf "%-50s %15s %10s\n" "YAML File" "Time (s)" "Status" >> "${SUMMARY_FILE}"
echo "-------------------------------------------------------------" >> "${SUMMARY_FILE}"

# Counter for statistics
total_yamls=0
successful_runs=0
failed_runs=0
total_time=0

# Find all optimal YAML files
YAML_FILES=$(find "${OPTIMAL_DIR}" -name "optimal_*.yaml" | sort)

if [ -z "$YAML_FILES" ]; then
    echo "Error: No optimal YAML files found in ${OPTIMAL_DIR}"
    exit 1
fi

echo ""
echo "Found $(echo "$YAML_FILES" | wc -l) optimal YAML files to benchmark"
echo ""

# Process each YAML file
for yaml_file in ${YAML_FILES}; do
    yaml_name=$(basename "$yaml_file" .yaml)
    out_dir="${RESULT_DIR}/${yaml_name}"
    log_file="${LOG_DIR}/${yaml_name}.log"
    
    mkdir -p "${out_dir}"
    
    total_yamls=$((total_yamls + 1))
    
    echo "[${total_yamls}] Processing: ${yaml_name}"
    echo "    Output: ${out_dir}"
    
    # Run Tensile benchmark and measure time
    start_time=$(date +%s)
    
    if ./Tensile.sh "${yaml_file}" "${out_dir}" > "${log_file}" 2>&1; then
        status="SUCCESS"
        successful_runs=$((successful_runs + 1))
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        total_time=$((total_time + elapsed))
        
        # Fix tensile_client permissions
        find "${out_dir}" -type f -name "tensile_client" -exec chmod +x {} \; 2>/dev/null || true
        
        echo "     Completed in ${elapsed}s"
        printf "%-50s %15d %10s\n" "${yaml_name}" "${elapsed}" "${status}" >> "${SUMMARY_FILE}"
        
    else
        status="FAILED"
        failed_runs=$((failed_runs + 1))
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        
        echo "     Failed after ${elapsed}s (check log: ${log_file})"
        printf "%-50s %15d %10s\n" "${yaml_name}" "${elapsed}" "${status}" >> "${SUMMARY_FILE}"
    fi
    
    echo ""
done

# Calculate statistics
avg_time=0
if [ ${successful_runs} -gt 0 ]; then
    avg_time=$((total_time / successful_runs))
fi

# Write summary statistics
echo "-------------------------------------------------------------" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "Summary Statistics:" >> "${SUMMARY_FILE}"
echo "  Total YAMLs processed : ${total_yamls}" >> "${SUMMARY_FILE}"
echo "  Successful runs       : ${successful_runs}" >> "${SUMMARY_FILE}"
echo "  Failed runs           : ${failed_runs}" >> "${SUMMARY_FILE}"
echo "  Total execution time  : ${total_time}s" >> "${SUMMARY_FILE}"
echo "  Average time per YAML : ${avg_time}s" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"
echo "=============================================================" >> "${SUMMARY_FILE}"

# Print final summary to console
echo "=============================================================="
echo " Benchmark Completed"
echo "--------------------------------------------------------------"
echo " Total YAMLs processed : ${total_yamls}"
echo " Successful runs       : ${successful_runs}"
echo " Failed runs           : ${failed_runs}"
echo " Total execution time  : ${total_time}s"
echo " Average time per YAML : ${avg_time}s"
echo "--------------------------------------------------------------"
echo " Results saved to      : ${RESULT_DIR}"
echo " Summary report        : ${SUMMARY_FILE}"
echo " Individual logs       : ${LOG_DIR}"
echo "=============================================================="

# Exit with error if any runs failed
if [ ${failed_runs} -gt 0 ]; then
    echo ""
    echo "Warning: ${failed_runs} benchmark(s) failed. Check logs for details."
    exit 1
fi

exit 0