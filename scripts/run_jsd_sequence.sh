#!/bin/bash
# JSD 阈值消融实验：0.1/0.2/0.3/0.5
cd /home/yjq/De-ASFL-V8-3-加入训练-基本实现
LOG_DIR="outputs/ablation_jsd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
    "configs/cifar10/config_de_asfl_jsd_0.1.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.2.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.3.yaml"
    "configs/cifar10/config_de_asfl_jsd_0.5.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "============================================"
echo "  JSD阈值消融：共 ${TOTAL} 个实验"
echo "  开始时间：$(date)"
echo "  日志目录：${LOG_DIR}"
echo "============================================"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [$idx/${TOTAL}] 开始: ${cfg} ====="
    echo "  开始时间: $(date)"
    ray stop --force 2>/dev/null
    sleep 3

    conda run -n fl_env python test_30_clients.py --config "${cfg}" \
        2>&1 | tee "${LOG_DIR}/run_${idx}_$(basename ${cfg} .yaml).log"

    echo "===== [$idx/${TOTAL}] 完成: ${cfg} ====="
    echo "  完成时间: $(date)"
    ray stop --force 2>/dev/null
    sleep 2
done

echo ""
echo "============================================"
echo "  全部 ${TOTAL} 个JSD消融实验完成！"
echo "  结束时间：$(date)"
echo "============================================"
