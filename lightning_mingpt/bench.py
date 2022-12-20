import gc
import time

import torch

import lightning as L


class Bench(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_benchmark(self, fn, args=[], kwargs={}, num_runs=10, device_type="auto"):
        """Returns an array with the last loss from each epoch for each run."""
        hist_losses = []
        hist_durations = []
        hist_memory = []
    
        if device_type == "auto":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.deterministic = True
        for i in range(self.num_runs):
            gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.reset_peak_memory_stats()
            time.sleep(1)
    
            time_start = time.perf_counter()
            final_loss, used_memory = fn(*args, **kwargs)
            time_end = time.perf_counter()
    
            hist_losses.append(final_loss)
            hist_durations.append(time_end - time_start)
            hist_memory.append(used_memory)
    
        return {"losses": hist_losses, "durations": hist_durations, "memory": hist_memory}
