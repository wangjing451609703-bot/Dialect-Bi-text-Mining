#!/usr/bin/env bash
#SBATCH -J env_setup
#SBATCH --partition=kisski
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -C inet

set -eo pipefail
set -x
mkdir -p logs


export CONDAROOT="${WORK:-${SCRATCH:-$HOME}}"
export CONDA_PKGS_DIRS="$CONDAROOT/.conda/pkgs"
export CONDA_ENVS_PATH="$CONDAROOT/.conda/envs"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"

module purge
module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_DIR="$CONDA_ENVS_PATH/llmdir"
if [ ! -d "$ENV_DIR" ]; then
  conda create -y -p "$ENV_DIR" python=3.10
fi


set +u
conda activate "$ENV_DIR"


conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 \
  || conda install -y -c pytorch pytorch torchvision torchaudio cpuonly

pip install -U pip
pip install sentence-transformers faiss-cpu numpy pandas tqdm pytrec_eval rank-bm25 pyyaml



export HF_HOME="${CONDAROOT}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

mkdir -p runs outputs processed_from_release_tar data_llmdir
echo "[OK] environment ready at $ENV_DIR"
