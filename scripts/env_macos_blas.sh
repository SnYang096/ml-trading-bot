# macOS + NumPy/OpenBLAS: limit BLAS threads to avoid SIGSEGV in gemm_thread_n.
# Source after activating .venv:
#   source scripts/env_macos_blas.sh
#
# Also applied automatically when using scripts/activate_mlbot.sh

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENBLAS_MAIN_FREE="${OPENBLAS_MAIN_FREE:-1}"
