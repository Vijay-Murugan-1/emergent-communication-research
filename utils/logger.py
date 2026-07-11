import os
import json
import csv
import time
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

class ResearchLogger:
    """Unified logger that writes to TensorBoard, CSV, and JSON simultaneously."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize outputs
        self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self.json_path = os.path.join(log_dir, "metrics.json")
        
        self.metrics_history = defaultdict(list)
        self.csv_file = None
        self.csv_writer = None
        self.current_step = 0

    def log_scalars(self, metrics_dict: dict, step: int):
        self.current_step = step
        
        # Add system metrics
        if torch.cuda.is_available():
            metrics_dict["system/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics_dict["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            
        metrics_dict["system/timestamp"] = time.time()
        metrics_dict["step"] = step

        # Write to TensorBoard
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)):
                self.tb_writer.add_scalar(k, v, step)
                
            # Keep history for JSON
            self.metrics_history[k].append(v)
            
        # Write to CSV
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics_dict.keys()))
            self.csv_writer.writeheader()
        
        # Ensure all fields are present for CSV (fill missing with None)
        csv_row = {k: metrics_dict.get(k, None) for k in self.csv_writer.fieldnames}
        self.csv_writer.writerow(csv_row)
        self.csv_file.flush()
        
    def log_json(self):
        """Dumps full history to JSON."""
        with open(self.json_path, "w") as f:
            json.dump(self.metrics_history, f, indent=4)

    def close(self):
        self.log_json()
        self.tb_writer.close()
        if self.csv_file:
            self.csv_file.close()
