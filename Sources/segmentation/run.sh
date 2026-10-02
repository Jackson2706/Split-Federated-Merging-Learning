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

######### ISIC-2018
### ResNet50
## IID Scenario
run_and_log "python3 Federated/main.py --cfg Federated/config/isic_fedavg_resnet50.yaml"  "logs/isic_fedavg_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/isic_fednova_resnet50.yaml" "logs/isic_fednova_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/isic_fedprox_resnet50.yaml" "logs/isic_fedprox_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/isic_fedsgd_resnet50.yaml"  "logs/isic_fedsgd_resnet50.txt"
run_and_log "python3 HierFL/main.py --cfg HierFL/config/isic_hierfl_resnet50.yaml"        "logs/isic_hierfl_resnet50.txt"
# run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/isic_splitfed_resnet50.yaml"    "logs/isic_splitfed_resnet50.txt"
# run_and_log "python3 HSPL/main.py --cfg HSPL/config/isic_our_resnet50_5_10.yaml"          "logs/isic_hspl_resnet50_5_10.txt"
# run_and_log "python3 HSPL/main.py --cfg HSPL/config/isic_our_resnet50_10_20.yaml"         "logs/isic_hspl_resnet50_10_20.txt"
# run_and_log "python3 HSPL/main.py --cfg HSPL/config/isic_our_resnet50_25_50.yaml"         "logs/isic_hspl_resnet50_25_50.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/isic_our_resnet50_5_10.yaml"    "logs/isic_ours_v1_resnet50_5_10.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/isic_our_resnet50_10_20.yaml"   "logs/isic_ours_v1_resnet50_10_20.txt"
# run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/isic_our_resnet50_25_50.yaml"   "logs/isic_ours_v1_resnet50_25_50.txt"
