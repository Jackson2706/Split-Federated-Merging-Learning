# src/monitoring_utils.py

import os
import psutil
import torch

# Pynvml chỉ hoạt động trên hệ thống có GPU NVIDIA
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_SUPPORT = True
except:
    GPU_SUPPORT = False

class ResourceMonitor:
    """
    Một lớp để theo dõi việc sử dụng tài nguyên (RAM, GPU) cho một khối mã.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_ram = 0
        self.peak_ram = 0
        self.start_gpu_mem = 0
        self.peak_gpu_mem = 0
        self.gpu_handle = None

        if GPU_SUPPORT and torch.cuda.is_available():
            gpu_index = torch.cuda.current_device()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        
    def _get_ram_usage_mb(self):
        """Lấy mức sử dụng RAM hiện tại của tiến trình (tính bằng MB)."""
        return self.process.memory_info().rss / (1024 ** 2)

    def _get_gpu_usage_mb(self):
        """Lấy mức sử dụng bộ nhớ GPU hiện tại (tính bằng MB)."""
        if self.gpu_handle:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return info.used / (1024 ** 2)
        return 0

    def start(self):
        """Bắt đầu theo dõi."""
        self.start_ram = self._get_ram_usage_mb()
        self.peak_ram = self.start_ram
        
        self.start_gpu_mem = self._get_gpu_usage_mb()
        self.peak_gpu_mem = self.start_gpu_mem

    def stop(self):
        """Dừng theo dõi và trả về mức sử dụng đỉnh."""
        self.peak_ram = max(self.peak_ram, self._get_ram_usage_mb())
        self.peak_gpu_mem = max(self.peak_gpu_mem, self._get_gpu_usage_mb())
        
        # Chúng ta quan tâm đến mức sử dụng *thêm* trong quá trình chạy
        ram_used = self.peak_ram - self.start_ram
        gpu_mem_used = self.peak_gpu_mem - self.start_gpu_mem
        
        return {"ram_peak_mb": ram_used, "gpu_peak_mb": gpu_mem_used}

    def track(self):
        """Cập nhật giá trị đỉnh trong khi đang chạy."""
        self.peak_ram = max(self.peak_ram, self._get_ram_usage_mb())
        self.peak_gpu_mem = max(self.peak_gpu_mem, self._get_gpu_usage_mb())

# Đừng quên tắt pynvml khi không cần nữa (thường là khi chương trình kết thúc)
def shutdown_pynvml():
    if GPU_SUPPORT:
        pynvml.nvmlShutdown()