#!/bin/bash
# ======================================================================
#  Parallel Tensile Batch Runner  (v2.0)
#  Purpose: Automatically distribute multiple tuning YAMLs across GPUs
#            and run Tensile tuning jobs in parallel with permission auto-fix.
# chmod +x 2_run_all_tensile_yaml.sh

# ./2_run_all_tensile_yaml.sh \
#   /workspace/ml_tuning/baseline_data_infra/tuning_batches_104CU_test \
#   /workspace/rocm-libraries/projects/hipblaslt/tensilelite/mi_tune \
#   /workspace/ml_tuning/baseline_data_infra/tuning_results_104CU \
#   "1 2 3"
# ======================================================================

# -----------------------------
# Command-line arguments
# -----------------------------
YAML_DIR=${1:-/workspace/ml_tuning/baseline_data_infra/tuning_batches_104CU_test}
BUILD_DIR=${2:-/workspace/rocm-libraries/projects/hipblaslt/tensilelite/mi_tune}
OUT_ROOT=${3:-/workspace/ml_tuning/baseline_data_infra/tuning_results_104CU}
GPU_LIST=${4:-"1 2 3"}  # can be changed to "0 1 2 3" later

TENSILE_BIN=${BUILD_DIR}/Tensile.sh
LOG_ROOT=${OUT_ROOT}/logs_parallel

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

echo "=============================================================="
echo " Parallel Tensile Tuning Launcher"
echo "--------------------------------------------------------------"
echo " YAML directory : ${YAML_DIR}"
echo " Build directory: ${BUILD_DIR}"
echo " Output root    : ${OUT_ROOT}"
echo " Tensile binary : ${TENSILE_BIN}"
echo " Target GPUs    : ${GPU_LIST}"
echo "=============================================================="

# -----------------------------
# Parallel GPU job launcher
# -----------------------------
for dev in ${GPU_LIST}; do
  YAML_LIST=$(ls ${YAML_DIR}/tuning_batch_*_dev${dev}.yaml 2>/dev/null)
  [ -z "$YAML_LIST" ] && { echo "No YAML found for GPU${dev}."; continue; }

  (
    export ROCM_VISIBLE_DEVICES=${dev}
    LOG_DIR=${LOG_ROOT}/dev${dev}
    mkdir -p "${LOG_DIR}"
    echo "[GPU${dev}] Starting tuning jobs..."

    for yaml_file in ${YAML_LIST}; do
      yaml_name=$(basename "$yaml_file" .yaml)
      out_dir="${OUT_ROOT}/${yaml_name}"
      mkdir -p "${out_dir}"

      echo "[GPU${dev}] Running ${yaml_name}"

      # Ensure permission before every run (safe even if already executable)
      (
        cd "${BUILD_DIR}"
        chmod +x Tensile.sh
        dos2unix Tensile.sh 2>/dev/null

        start_time=$(date +%s)
        ./Tensile.sh "${yaml_file}" "${out_dir}"
        status=$?
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "[GPU${dev}] ${yaml_name} finished in ${elapsed}s (status=${status})"

        # tensile_client permission
        find "${out_dir}" -type f -name "tensile_client" -exec chmod +x {} \; 2>/dev/null
        ) > "${LOG_DIR}/${yaml_name}.log" 2>&1

      if [ $? -ne 0 ]; then
        echo "[GPU${dev}] Failed: ${yaml_name}"
      else
        echo "[GPU${dev}] Done: ${yaml_name}"
      fi
    done

    echo "[GPU${dev}] Completed all batches."
  ) &
done

wait
echo "=============================================================="
echo " All GPUs completed their assigned YAMLs."
echo " Logs saved under ${LOG_ROOT}"
echo "=============================================================="
