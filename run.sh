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

####### CIFAR10
### AlexNet
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedavg_alexnet.yaml"   "logs/cifar_fedavg_alexnet.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fednova_alexnet.yaml"  "logs/cifar_fednova_alexnet.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedprox_alexnet.yaml"  "logs/cifar_fedprox_alexnet.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedsgd_alexnet.yaml"   "logs/cifar_fedsgd_alexnet.txt"

run_and_log "python3 HierFL/main.py --cfg HierFL/config/cifar_hierfl_alexnet.yaml"         "logs/cifar_hierfl_alexnet.txt"

run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/cifar_splitfed_alexnet.yaml"       "logs/cifar_splitfed_alexnet.txt"

run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_alexnet_5_10.yaml"       "logs/cifar_our_alexnet_5_10.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_alexnet_10_20.yaml"       "logs/cifar_our_alexnet_10_20.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_alexnet_25_50.yaml"       "logs/cifar_our_alexnet_25_50.txt"

### ResNet50
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedavg_resnet50.yaml"   "logs/cifar_fedavg_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fednova_resnet50.yaml"  "logs/cifar_fednova_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedprox_resnet50.yaml"  "logs/cifar_fedprox_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/cifar_fedsgd_resnet50.yaml"   "logs/cifar_fedsgd_resnet50.txt"

run_and_log "python3 HierFL/main.py --cfg HierFL/config/cifar_hierfl_resnet50.yaml"         "logs/cifar_hierfl_resnet50.txt"

run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/cifar_splitfed_resnet50.yaml"       "logs/cifar_splitfed_resnet50.txt"

run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_resnet50_5_10.yaml"       "logs/cifar_our_resnet50_5_10.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_resnet50_10_20.yaml"       "logs/cifar_our_resnet50_10_20.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/cifar_our_resnet50_25_50.yaml"       "logs/cifar_our_resnet50_25_50.txt"

run_and_log "python3 Ours_v2/main.py --cfg Ours_v2/config/cifar_ourv2_a1.yaml"       "logs/cifar_fedavg_ours_v2_a1.txt"
run_and_log "python3 Ours_v2/main.py --cfg Ours_v2/config/cifar_ourv2_a2.yaml"       "logs/cifar_fedavg_ours_v2_a2.txt"
run_and_log "python3 Ours_v2/main.py --cfg Ours_v2/config/cifar_ourv2_a3.yaml"       "logs/cifar_fedavg_ours_v2_a3.txt"
run_and_log "python3 Ours_v2/main.py --cfg Ours_v2/config/cifar_ourv2_a4.yaml"       "logs/cifar_fedavg_ours_v2_a4.txt"
run_and_log "python3 Ours_v2/main.py --cfg Ours_v2/config/cifar_ourv2_a5.yaml"       "logs/cifar_fedavg_ours_v2_a5.txt"

######### HAM10000
### ResNet50
run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedavg_resnet50.yaml"   "logs/ham10000_fedavg_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fednova_resnet50.yaml"  "logs/ham10000_fednova_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedprox_resnet50.yaml"  "logs/ham10000_fedprox_resnet50.txt"
run_and_log "python3 Federated/main.py --cfg Federated/config/ham10000_fedsgd_resnet50.yaml"   "logs/ham10000_fedsgd_resnet50.txt"

run_and_log "python3 HierFL/main.py --cfg HierFL/config/ham10000_hierfl_resnet50.yaml"         "logs/ham10000_hierfl_resnet50.txt"

run_and_log "python3 SplitFL/main.py --cfg SplitFL/config/ham10000_splitfed_resnet50.yaml"       "logs/ham10000_splitfed_resnet50.txt"

run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_resnet50_5_10.yaml"       "logs/ham10000_our_resnet50_5_10.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_resnet50_10_20.yaml"       "logs/ham10000_our_resnet50_10_20.txt"
run_and_log "python3 Ours_v1/main.py --cfg Ours_v1/config/ham10000_our_resnet50_25_50.yaml"       "logs/ham10000_our_resnet50_25_50.txt"