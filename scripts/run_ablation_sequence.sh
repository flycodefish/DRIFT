#!/bin/bash
# 串行跑完 5 个消融实验
# no_election_no_cluster + JSD 0.1/0.2/0.3/0.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
LOG_DIR="outputs/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "configs/cifar10/config_de_asfl_no_election_no_cluster.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.1.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.2.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.3.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.5.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "============================================"
echo "  消融实验序列：共 ${TOTAL} 个实验"
echo "  开始时间：$(date)"
echo "  日志目录：${LOG_DIR}"
echo "============================================"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [$idx/${TOTAL}] 开始: ${cfg} ====="
    echo "  开始时间: $(date)"

    # 确保上次 Ray 完全清理
    ray stop --force 2>/dev/null
    sleep 2
    # 杀掉残留 MPS server
    pkill -f nvidia-cuda-mps-server 2>/dev/null || true
    sleep 1

    python scripts/test_30_clients.py --config "${cfg}" \
        2>&1 | tee "${LOG_DIR}/run_${idx}_$(basename ${cfg} .yaml).log"

    echo "===== [$idx/${TOTAL}] 完成: ${cfg} ====="
    echo "  完成时间: $(date)"

    # 实验间清理
    ray stop --force 2>/dev/null
    sleep 2
done

echo ""
echo "============================================"
echo "  全部 ${TOTAL} 个消融实验完成！"
echo "  结束时间：$(date)"
echo "============================================"
