import os
import json
import torch
import shutil
from omegaconf import OmegaConf

class CheckpointManager:
    """Handles saving full state required for reproducibility."""
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save(self, epoch: int, step: int, model_state, optimizer_state, scheduler_state, config, metrics, seed, git_hash, is_best: bool = False):
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save PyTorch states
        torch.save({
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state
        }, os.path.join(checkpoint_path, "state.pt"))
            
        # Save Metadata
        with open(os.path.join(checkpoint_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))
            
        with open(os.path.join(checkpoint_path, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        with open(os.path.join(checkpoint_path, "seed.json"), "w") as f:
            json.dump({"seed": seed}, f)
            
        with open(os.path.join(checkpoint_path, "git_hash.txt"), "w") as f:
            f.write(git_hash)
            
        # Update latest
        latest_path = os.path.join(self.checkpoint_dir, "latest")
        if os.path.exists(latest_path):
            shutil.rmtree(latest_path)
        shutil.copytree(checkpoint_path, latest_path)
        
        # Update best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best")
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(checkpoint_path, best_path)
            
    def load(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
        """Loads and returns all components for resuming."""
        state_path = os.path.join(checkpoint_path, "state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No checkpoint found at {state_path}")
            
        checkpoint = torch.load(state_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint.get('epoch', 0), checkpoint.get('global_step', 0)
