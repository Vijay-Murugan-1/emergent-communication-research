import matplotlib.pyplot as plt
import os
import pandas as pd

class PlotGenerator:
    """Generates standard research plots from CSV logs."""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def generate_all(self):
        csv_path = os.path.join(self.log_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            return
            
        df = pd.read_csv(csv_path)
        
        # Loss Curves
        self._plot_series(df, ["loss/total", "loss/reconstruction", "loss/policy"], "Training Losses", "losses.png")
        
        # Reward Curves
        reward_cols = [c for c in df.columns if c.startswith("reward/")]
        if reward_cols:
            self._plot_series(df, reward_cols, "Reward Components", "rewards.png")
            
        # Communication Stats
        comm_cols = [c for c in df.columns if c.startswith("comm/")]
        if comm_cols:
            self._plot_series(df, comm_cols, "Communication Diagnostics", "communication.png")
            
    def _plot_series(self, df, columns, title, filename):
        plt.figure(figsize=(10, 6))
        for col in columns:
            if col in df.columns:
                # Drop NAs to plot cleanly if some metrics are validation-only
                valid_data = df[['step', col]].dropna()
                plt.plot(valid_data['step'], valid_data[col], label=col)
                
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
