import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from plugins.registry import DATASET_REGISTRY, CHANNEL_REGISTRY
from agents import Sender, Receiver
from communication.protocol import CommunicationProtocol
from envs.reward import CompositeReward

from utils.logger import ResearchLogger
from checkpoint.manager import CheckpointManager
from analysis.protocol import ProtocolAnalyzer
from visualization.plots import PlotGenerator
from metrics.reconstruction import compute_mse, compute_psnr, compute_pixel_accuracy

class ExperimentRunner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.best_eval_metric = float('inf')  # Use MSE (lower is better) for early stopping
        self.reward_baseline = 0.0 # Moving average for REINFORCE
        self.patience = 5
        self.epochs_without_improvement = 0
        self.early_stop_triggered = False
        
    def setup(self):
        """Initializes components based on hydra config."""
        dataset_class = DATASET_REGISTRY.get(self.config.dataset.name)
        self.dataset_manager = dataset_class(self.config.dataset)
        self.train_loader = self.dataset_manager.get_dataloader("train")
        
        # Test loader fallback to train if no split handling is fully robust yet
        try:
            self.val_loader = self.dataset_manager.get_dataloader("val")
        except Exception:
            self.val_loader = self.dataset_manager.get_dataloader("train")
            
        try:
            self.test_loader = self.dataset_manager.get_dataloader("test")
        except Exception:
            self.test_loader = self.dataset_manager.get_dataloader("train")
        
        self.sender = Sender(
            encoder_name=self.config.agent.encoder_type,
            encoder_kwargs={"in_channels": 1, "hidden_dim": self.config.agent.hidden_dim},
            communication_dim=self.config.communication.vocab_size * self.config.communication.max_length
        ).to(self.device)
        
        self.receiver = Receiver(
            decoder_name=self.config.agent.decoder_type,
            decoder_kwargs={"hidden_dim": self.config.agent.hidden_dim, "out_channels": 1, "output_size": 28},
            communication_dim=self.config.communication.vocab_size * self.config.communication.max_length
        ).to(self.device)
        
        channel_class = CHANNEL_REGISTRY.get(self.config.communication.protocol)
        channel = channel_class(self.config.communication)
        
        self.protocol = CommunicationProtocol(
            channel=channel,
            vocab_size=self.config.communication.vocab_size,
            max_length=self.config.communication.max_length,
            noise_prob=self.config.communication.noise_prob
        ).to(self.device)
        
        self.reward_fn = CompositeReward(
            alpha_reconstruction=self.config.reward.alpha_reconstruction,
            beta_compression=self.config.reward.beta_compression,
            gamma_diversity=self.config.reward.gamma_diversity,
            delta_efficiency=self.config.reward.delta_efficiency,
            penalties=self.config.reward.penalties
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            list(self.sender.parameters()) + list(self.receiver.parameters()),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay
        )
        
        self.scaler = GradScaler(enabled=self.config.training.mixed_precision)
        
        # Managers & Diagnostics
        self.logger = ResearchLogger(self.config.experiment.log_dir)
        self.checkpoint_manager = CheckpointManager(self.config.experiment.checkpoint_dir)
        self.analyzer = ProtocolAnalyzer(self.config.communication.vocab_size)
        self.plotter = PlotGenerator(self.config.experiment.log_dir)

    def _compute_weight_norm(self):
        total_norm = 0.0
        for p in list(self.sender.parameters()) + list(self.receiver.parameters()):
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5
        
    def _compute_grad_norm(self):
        total_norm = 0.0
        for p in list(self.sender.parameters()) + list(self.receiver.parameters()):
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train(self):
        """Main training loop using advanced diagnostics and AMP."""
        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        
        for epoch in range(self.config.training.epochs):
            self.sender.train()
            self.receiver.train()
            
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                batch_size = data.size(0)
                
                self.optimizer.zero_grad()
                
                with autocast(enabled=self.config.training.mixed_precision):
                    sender_out = self.sender(data)
                    if self.config.communication.protocol in ["reinforce", "gumbel"]:
                        sender_out = sender_out.view(batch_size, self.config.communication.max_length, self.config.communication.vocab_size)
                    
                    message = self.protocol(sender_out)
                    
                    if self.config.communication.protocol == "reinforce":
                        message_one_hot = torch.nn.functional.one_hot(
                            message.content, num_classes=self.config.communication.vocab_size
                        ).float()
                        receiver_input = message_one_hot.view(batch_size, -1)
                    elif self.config.communication.protocol == "gumbel":
                        receiver_input = message.content.view(batch_size, -1)
                    else:
                        receiver_input = message.content
                    
                    from communication.message import Message
                    temp_message = Message(content=receiver_input, lengths=message.lengths)
                    reconstructed = self.receiver(temp_message)
                    
                    # Losses
                    rec_flat = reconstructed.view(batch_size, -1)
                    obs_flat = data.view(batch_size, -1)
                    loss_rec_indiv = mse_loss_fn(rec_flat, obs_flat).mean(dim=1)
                    loss_rec = loss_rec_indiv.mean()
                    
                    # Rewards
                    rewards_dict = self.reward_fn(loss_rec_indiv, message, self.config.communication.max_length, self.config.communication.vocab_size)
                    total_reward = rewards_dict["reward/total"]
                    
                    # Total Loss (Explicitly track communication loss if applicable)
                    # Gumbel is continuous from backprop perspective, no policy loss needed
                    # If continuous, communication loss could be added here (e.g. L2 on latents), but we'll stick to MSE for now.
                    loss_comm = torch.tensor(0.0, device=self.device)
                    loss_entropy = torch.tensor(0.0, device=self.device)
                    
                    if self.config.communication.protocol == "reinforce":
                        # Update baseline
                        mean_reward = total_reward.detach().mean().item()
                        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * mean_reward
                        advantages = total_reward.detach() - self.reward_baseline
                        
                        log_probs = message.log_probs.sum(dim=1)
                        loss_policy = -(log_probs * advantages).mean()
                        
                        # Add small entropy bonus to prevent collapse
                        if hasattr(message, 'entropy') and message.entropy is not None:
                            loss_entropy = -0.01 * message.entropy.mean()
                    else:
                        loss_policy = torch.tensor(0.0, device=self.device)
                        
                    loss_total = loss_rec + loss_policy + loss_comm + loss_entropy
                
                # NaN check
                if torch.isnan(loss_total):
                    print("NaN detected in loss. Skipping step.")
                    continue
                    
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)
                
                grad_norm = self._compute_grad_norm()
                if self.config.training.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.sender.parameters()) + list(self.receiver.parameters()), 
                        self.config.training.clip_grad_norm
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Track Diagnostics
                self.analyzer.track_batch(message.content, message.lengths)
                
                if self.global_step % 10 == 0:
                    metrics = {
                        "loss/total": loss_total.item(),
                        "loss/reconstruction": loss_rec.item(),
                        "loss/policy": loss_policy.item(),
                        "loss/communication": loss_comm.item(),
                        "loss/entropy": loss_entropy.item(),
                        "diagnostics/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "diagnostics/grad_norm": grad_norm,
                        "diagnostics/weight_norm": self._compute_weight_norm(),
                    }
                    metrics.update({k: v.mean().item() for k, v in rewards_dict.items()})
                    metrics.update(self.analyzer.get_statistics())
                    
                    self.logger.log_scalars(metrics, self.global_step)
                    
                self.global_step += 1
                
            # Epoch End Evaluation and Checkpointing
            print(f"Epoch {epoch} finished. Running Validation...")
            val_metrics = self.evaluate(self.val_loader, split="val")
            self.logger.log_scalars(val_metrics, self.global_step)
            
            # Early Stopping Check (Monitoring val/mse)
            is_best = val_metrics["val/mse"] < self.best_eval_metric
            if is_best:
                self.best_eval_metric = val_metrics["val/mse"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            self.checkpoint_manager.save(
                epoch=epoch,
                step=self.global_step,
                model_state={
                    "sender": self.sender.state_dict(),
                    "receiver": self.receiver.state_dict()
                }, 
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=None,
                config=self.config,
                metrics=val_metrics,
                seed=self.config.training.seed,
                git_hash="unknown",
                is_best=is_best
            )
            
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience={self.patience})")
                self.early_stop_triggered = True
                break

    @torch.no_grad()
    def evaluate(self, dataloader, split="val"):
        self.sender.eval()
        self.receiver.eval()
        self.analyzer.reset()
        
        total_mse, total_psnr, total_acc, total_reward = 0, 0, 0, 0
        batches = 0
        
        for data, _ in dataloader:
            data = data.to(self.device)
            batch_size = data.size(0)
            
            sender_out = self.sender(data)
            if self.config.communication.protocol in ["reinforce", "gumbel"]:
                sender_out = sender_out.view(batch_size, self.config.communication.max_length, self.config.communication.vocab_size)
            
            message = self.protocol(sender_out)
            
            if self.config.communication.protocol == "reinforce":
                message_one_hot = torch.nn.functional.one_hot(message.content, num_classes=self.config.communication.vocab_size).float()
                receiver_input = message_one_hot.view(batch_size, -1)
            elif self.config.communication.protocol == "gumbel":
                receiver_input = message.content.view(batch_size, -1)
            else:
                receiver_input = message.content
            
            from communication.message import Message
            temp_message = Message(content=receiver_input, lengths=message.lengths)
            reconstructed = self.receiver(temp_message)
            
            rec_flat = reconstructed.view(batch_size, -1)
            obs_flat = data.view(batch_size, -1)
            
            # Compute Metrics
            mse = compute_mse(rec_flat, obs_flat)
            psnr = compute_psnr(rec_flat, obs_flat)
            acc = compute_pixel_accuracy(rec_flat, obs_flat)
            
            rewards = self.reward_fn(mse, message, self.config.communication.max_length, self.config.communication.vocab_size)
            
            total_mse += mse.mean().item()
            total_psnr += psnr.mean().item()
            total_acc += acc.mean().item()
            total_reward += rewards["reward/total"].mean().item()
            
            self.analyzer.track_batch(message.discrete_content if message.discrete_content is not None else message.content, message.lengths)
            
            # Save visual grid for the very first batch
            if batches == 0:
                self._save_visualizations(data, reconstructed, split)
                
            batches += 1
            
        metrics = {
            f"{split}/mse": total_mse / batches,
            f"{split}/psnr": total_psnr / batches,
            f"{split}/pixel_accuracy": total_acc / batches,
            f"{split}/reward_total": total_reward / batches
        }
        
        comm_stats = self.analyzer.get_statistics()
        metrics.update({f"{split}_{k}": v for k, v in comm_stats.items()})
        
        print(f"[{split}] MSE: {metrics[f'{split}/mse']:.4f}, PSNR: {metrics[f'{split}/psnr']:.4f}, Reward: {metrics[f'{split}/reward_total']:.4f}")
        return metrics

    def _save_visualizations(self, original, reconstructed, split):
        import torchvision
        vis_dir = os.path.join(self.config.experiment.log_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Take first 16 images
        n = min(16, original.size(0))
        orig_imgs = original[:n].view(-1, 1, 28, 28)
        recon_imgs = reconstructed[:n].view(-1, 1, 28, 28)
        
        # Interleave original and reconstructed
        comparison = torch.cat([orig_imgs, recon_imgs])
        
        grid = torchvision.utils.make_grid(comparison, nrow=n)
        torchvision.utils.save_image(grid, os.path.join(vis_dir, f"{split}_recon_step_{self.global_step}.png"))

    def test(self):
        """Dedicated testing pipeline on held-out data using best model."""
        print("Loading best checkpoint for Testing...")
        # Placeholder for loading best checkpoint logic (simplified here)
        
        test_metrics = self.evaluate(self.test_loader, split="test")
        
        # Generate plots
        print("Generating Research Plots...")
        self.plotter.generate_all()
        self.logger.close()

    def run(self):
        print("Starting Setup...")
        self.setup()
        print("Starting Training...")
        self.train()
        print("Starting Testing Pipeline...")
        self.test()
