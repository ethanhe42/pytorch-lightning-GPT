import gc
import time
from typing import Type

import torch

import lightning as L
from lightning.app.components import LightningTrainerMultiNode


def _hook_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
    else:
        used_memory = -1
    return used_memory


class BenchRun(L.LightningFlow):
    def __init__(
        self,
        work_cls: Type[L.LightningWork],
        num_nodes: int,
        cloud_compute: L.CloudCompute,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.results = []

        if num_nodes > 1:
            self.multinode = LightningTrainerMultiNode(
                work_cls,
                num_nodes=num_nodes,
                cloud_compute=cloud_compute,
            )
        else:
            self.w = work_cls(cloud_compute=cloud_compute)
    
    def run(self, *args, **kwargs):
        results = []
        if self.num_nodes > 1:
            self.multinode.run(*args, **kwargs)
            # wait until finished
            if all(w.has_succeeded for w in self.multinode.ws):
                results = [w.results for w in self.multinode.ws]
        else:
            self.w.run()
            results = self.w.results
        self.results = results
        import pprint
        pprint.pprint(self.results)

    # def configure_layout(self):
    #     # return [{"name": "Training Logs", "content": self.tensorboard_work.url}]


class Bench(L.LightningWork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {}

    def run_benchmark(self, name, fn, args=[], kwargs={}, num_runs=10, device_type="auto"):
        """Returns an array with the last loss from each epoch for each run."""
        hist_losses = []
        hist_durations = []
        hist_memory = []
    
        if device_type == "auto":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
        torch.backends.cudnn.deterministic = True
        for i in range(self.num_runs):
            print(f"Run {i+1}/{self.num_runs}")
            gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.reset_peak_memory_stats()
            time.sleep(1)
    
            time_start = time.perf_counter()
            final_loss = fn(*args, **kwargs)
            used_memory = _hook_memory()
            time_end = time.perf_counter()
    
            hist_losses.append(final_loss)
            hist_durations.append(time_end - time_start)
            hist_memory.append(used_memory)
    
        self.results[name] = dict(
            losses=hist_losses,
            durations=hist_durations,
            memory=hist_memory
        )

        import pprint
        pprint.pprint(self.results)
