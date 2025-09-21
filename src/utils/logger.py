import os
import time

import psutil
import torch


class Logger:
    def __init__(self, log_dir="logs", name="cloud"):
        os.makedirs(log_dir, exist_ok=True)
        self.name = name
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.start_time = time.time()

        with open(self.log_path, "w") as f:
            f.write("Step, Loss, Accuracy, CPU%, RAM%, GPU%, Time(s)\n")

    def _get_system_usage(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        gpu = 0.0
        if torch.cuda.is_available():
            try:
                gpu = torch.cuda.utilization()
            except Exception:
                gpu = 0.0
        return cpu, ram, gpu

    def log(self, step, loss, acc, step_start):
        cpu, ram, gpu = self._get_system_usage()
        elapsed = time.time() - step_start

        log_line = f"{step}, Loss: {loss:.4f}, Acc: {acc:.2f}, CPU: {cpu:.1f}%, RAM: {ram:.1f} MB, GPU: {gpu:.1f} RAM, Time: {elapsed:.2f}"

        # ✅ print to console with prefix
        print(f"[{self.name.upper()}] {log_line}")

        # ✅ write to file
        with open(self.log_path, "a") as f:
            f.write(log_line + "\n")

    def log_final(self):
        total_time = time.time() - self.start_time
        msg = f"[{self.name.upper()}] Finished. Total time: {total_time:.2f}s"
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")
