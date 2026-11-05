#!/usr/bin/env bash
# Commands for the experiments reported in the paper. Each run selects its checkpoint on the
# validation split and evaluates the test split once (results.json in the run folder).
# Set the dataset roots below. IP102: folder with images/ and train.txt, val.txt, test.txt.
# CPB: folder with the images and pests_{train,validation,test}_original.csv.
set -e
IP102=${IP102:-/data/IP102/Classification/ip102_v1.1}
CPB=${CPB:-/data/CPB}
RUNS=${RUNS:-runs}
COMMON="--batch-size 32 --workers 4 --epochs1 30 --epochs2 5 --warmup-epochs 2 --cutmix 0.5 --smoothing 0.1"
HIP="--arch hipest --fusion skip --loss ce --skip-stage2"

export HIPEST_DATASET=ip102
for s in 0 1 2; do
  # Final recipe (v2), ImageNet-1k weights: baseline and HiPest-Net
  python train.py --data-root $IP102 --out $RUNS/v2_convnext_s$s --seed $s --arch convnext_t $COMMON
  python train.py --data-root $IP102 --out $RUNS/v2_hipest_s$s  --seed $s $HIP $COMMON
done
for s in 0 1; do
  # ImageNet-22k weights
  python train.py --data-root $IP102 --out $RUNS/in22k_convnext_s$s --seed $s --weights in22k --arch convnext_t $COMMON
  python train.py --data-root $IP102 --out $RUNS/in22k_hipest_s$s  --seed $s --weights in22k $HIP $COMMON
done

# Component study (seed 0)
python train.py --data-root $IP102 --out $RUNS/v2_flathead_s0 --seed 0 $HIP $COMMON --flat-head
python train.py --data-root $IP102 --out $RUNS/v2_noattn_s0   --seed 0 $HIP $COMMON --no-coord --no-cra --no-lka
python train.py --data-root $IP102 --out $RUNS/v2_resnext50_s0 --seed 0 --arch resnext50 $COMMON

# Earlier recipe (v1): class-balanced focal loss and decoupled stage II (seed 0)
V1="--batch-size 32 --workers 4 --epochs1 30 --epochs2 5 --warmup-epochs 2"
python train.py --data-root $IP102 --out $RUNS/base_convnext_s0      --seed 0 --arch convnext_t $V1
python train.py --data-root $IP102 --out $RUNS/hipest_s0             --seed 0 --arch hipest $V1
python train.py --data-root $IP102 --out $RUNS/abl_loss_only_s0      --seed 0 --arch hipest $V1 --no-coord --no-cra --no-lka --no-bifpn --flat-head --loss cbfocal --skip-stage2
python train.py --data-root $IP102 --out $RUNS/abl_arch_ce_s0        --seed 0 --arch hipest $V1 --loss ce --skip-stage2
python train.py --data-root $IP102 --out $RUNS/abl_arch_ce_s2q05_s0  --seed 0 --arch hipest $V1 --loss ce --sampling-q-stage2 0.5
python train.py --data-root $IP102 --out $RUNS/abl_arch_ce_s2q0_s0   --seed 0 --arch hipest $V1 --loss ce --sampling-q-stage2 0

# Post-hoc evaluation: test-time size/flip chosen on validation, weight soups, saved log-probabilities
python evalx.py --data-root $IP102 --runs $RUNS --out $RUNS/evalx --workers 4 --bs 32

# DINOv2 ViT-S/14 (seed 0)
python train.py --data-root $IP102 --out $RUNS/dinov2s_s0 --seed 0 --arch timm:vit_small_patch14_dinov2.lvd142m \
  --batch-size 32 --workers 4 --epochs1 12 --epochs2 5 --warmup-epochs 1 --loss ce --skip-stage2 --cutmix 0.5 \
  --smoothing 0.1 --lr-backbone 5e-5 --layer-decay 0.75 --drop-path 0.1

# 320 px fine-tune of the first ImageNet-22k baseline
python train.py --data-root $IP102 --out $RUNS/hr320_in22k_convnext_s0 --seed 0 --init $RUNS/in22k_convnext_s0/final.pt \
  --arch convnext_t --weights in22k --img-size 320 --batch-size 16 --workers 4 --epochs1 6 --epochs2 1 --warmup-epochs 1 \
  --cutmix 0.5 --smoothing 0.1 --lr-backbone 2e-5 --lr-new 2e-5

# Citrus Pest Benchmark (binary, official split), seeds 0-2
export HIPEST_DATASET=cpb
for s in 0 1 2; do
  python train.py --data-root $CPB --out $RUNS/cpb_convnext_s$s --seed $s --arch convnext_t $COMMON
  python train.py --data-root $CPB --out $RUNS/cpb_hipest_s$s  --seed $s $HIP $COMMON --flat-head
done
