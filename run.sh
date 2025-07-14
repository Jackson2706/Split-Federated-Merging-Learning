#!/bin/bash

# Tạo thư mục logs nếu chưa tồn tại
mkdir -p logs

# Hàm chạy lệnh và ghi log
run_and_log() {
    CMD=$1
    LOG_FILE=$2

    echo "Running: $CMD" > "$LOG_FILE"
    echo "Start time: $(date)" >> "$LOG_FILE"
    eval "$CMD" >> "$LOG_FILE" 2>&1
    echo "End time: $(date)" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
}

### AlexNet - CIFAR10
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedavg_alexnet.yaml"   "logs/cifar10_fedavg_alexnet.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fednova_alexnet.yaml"  "logs/cifar10_fednova_alexnet.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedprox_alexnet.yaml"  "logs/cifar10_fedprox_alexnet.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedsgd_alexnet.yaml"   "logs/cifar10_fedsgd_alexnet.txt"
run_and_log "python3 HierFL/main.py --cfg HierFL/config/cifar_hierfl_alexnet.yaml"         "logs/cifar10_hierfl_alexnet.txt"
# run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/cifar_splitfed_alexnet.yaml"     "logs/cifar10_splitfed_alexnet.txt"
run_and_log "python3 HSPL/main.py --cfg HSPL/config/cifar_our_alexnet_5_10.yaml"           "logs/cifar10_hspl_alexnet_5_10.txt"
run_and_log "python3 HSPL/main.py --cfg HSPL/config/cifar_our_alexnet_10_20.yaml"          "logs/cifar10_hspl_alexnet_10_20.txt"
run_and_log "python3 HSPL/main.py --cfg HSPL/config/cifar_our_alexnet_25_50.yaml"          "logs/cifar10_hspl_alexnet_25_50.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_alexnet_5_10.yaml"     "logs/cifar10_our_alexnet_5_10.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_alexnet_10_20.yaml"    "logs/cifar10_our_alexnet_10_20.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_alexnet_25_50.yaml"    "logs/cifar10_our_alexnet_25_50.txt"
### ResNet50 - CIFAR100
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedavg_resnet50.yaml"  "logs/cifar100_fedavg_resnet50.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fednova_resnet50.yaml" "logs/cifar100_fednova_resnet50.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedprox_resnet50.yaml" "logs/cifar100_fedprox_resnet50.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedsgd_resnet50.yaml"  "logs/cifar100_fedsgd_resnet50.txt"
run_and_log "python3 HierFL/main.py --cfg HierFL/config/cifar_hierfl_resnet50.yaml"        "logs/cifar100_hierfl_resnet50.txt"
run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/cifar_splitfed_resnet50.yaml"    "logs/cifar100_splitfed_resnet50.txt"
run_and_log "python3 HSPL/main.py --cfg HSPL/config/cifar_our_resnet50_5_10.yaml"          "logs/cifar100_hspl_resnet50_5_10.txt"
run_and_log "python3 HSPL/main.py --cfg HSPL/config/cifar_our_resnet50_10_20.yaml"         "logs/cifar100_hspl_resnet50_10_20.txt"
run_and_log "python3 HSPL/main.py --cfg HSPL/config/cifar_our_resnet50_25_50.yaml"         "logs/cifar100_hspl_resnet50_25_50.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_resnet50_5_10.yaml"    "logs/cifar100_ours_v1_resnet50_5_10.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_resnet50_10_20.yaml"   "logs/cifar100_ours_v1_resnet50_10_20.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_resnet50_25_50.yaml"   "logs/cifar100_ours_v1_resnet50_25_50.txt"


# ######### HAM10000
# ### ResNet50
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedavg_resnet50.yaml"   "logs/ham10000_fedavg_resnet50.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fednova_resnet50.yaml"  "logs/ham10000_fednova_resnet50.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedprox_resnet50.yaml"  "logs/ham10000_fedprox_resnet50.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedsgd_resnet50.yaml"   "logs/ham10000_fedsgd_resnet50.txt"

# run_and_log "python3 HierFL/main.py --cfg HierFL/config/ham10000_hierfl_resnet50.yaml"         "logs/ham10000_hierfl_resnet50.txt"

# run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/ham10000_splitfed_resnet50.yaml"       "logs/ham10000_splitfed_resnet50.txt"

# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_resnet50_5_10.yaml"       "logs/ham10000_our_resnet50_5_10.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_resnet50_10_20.yaml"       "logs/ham10000_our_resnet50_10_20.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_resnet50_25_50.yaml"       "logs/ham10000_our_resnet50_25_50.txt"

# ### VGG
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedavg_vgg.yaml"   "logs/ham10000_fedavg_vgg.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fednova_vgg.yaml"  "logs/ham10000_fednova_vgg.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedprox_vgg.yaml"  "logs/ham10000_fedprox_vgg.txt"
# run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedsgd_vgg.yaml"   "logs/ham10000_fedsgd_vgg.txt"

# run_and_log "python3 HierFL/main.py --cfg HierFL/config/ham10000_hierfl_vgg.yaml"         "logs/ham10000_hierfl_vgg.txt"

# run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/ham10000_splitfed_vgg.yaml"       "logs/ham10000_splitfed_vgg.txt"

# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_vgg_5_10.yaml"       "logs/ham10000_our_vgg_5_10.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_vgg_10_20.yaml"       "logs/ham10000_our_vgg_10_20.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_vgg_25_50.yaml"       "logs/ham10000_our_vgg_25_50.txt"