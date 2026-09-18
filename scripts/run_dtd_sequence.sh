#!/bin/bash
# DTD 对比基线实验：α频谱 + CIFAR-100 + NEU + HAR
# 共8组，50客户端CIFAR-100放最后（显存压力最大）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
LOG_DIR="outputs/dtd_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
    # CIFAR-10 α频谱 (30客户端, 100轮) — 5组
    "configs/cifar10/config_dtd_cifar10_alpha005.yaml"
    "configs/cifar10/config_dtd_cifar10_alpha01.yaml"
    "configs/cifar10/config_dtd_cifar10_alpha05.yaml"
    "configs/cifar10/config_dtd_cifar10_alpha1.0.yaml"
    "configs/cifar10/config_dtd_cifar10_alpha10.yaml"
    # 工业数据集 (12客户端, 80轮) — 2组
    "configs/neu/config_dtd_neu.yaml"
    "configs/har/config_dtd_har.yaml"
    # CIFAR-100可扩展性 (50客户端, 150轮) — 1组, 放最后
    "configs/cifar100/config_dtd_cifar100.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "============================================"
echo "  DTD基线实验：共 ${TOTAL} 组"
echo "  开始时间：$(date)"
echo "  日志目录：${LOG_DIR}"
echo "============================================"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "===== [${idx}/${TOTAL}] 开始: ${cfg} ====="
    echo "  开始时间: $(date)"
    ray stop --force 2>/dev/null
    sleep 3

    python scripts/run_fl.py --config "${cfg}" \
        2>&1 | tee "${LOG_DIR}/run_${idx}_$(basename ${cfg} .yaml).log"

    echo "===== [${idx}/${TOTAL}] 完成: ${cfg} ====="
    echo "  完成时间: $(date)"
    ray stop --force 2>/dev/null
    sleep 2
done

echo ""
echo "============================================"
echo "  全部 ${TOTAL} 组DTD实验完成！"
echo "  结束时间：$(date)"
echo "============================================"
