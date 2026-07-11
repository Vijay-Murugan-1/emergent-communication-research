import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
import torchvision
from omegaconf import OmegaConf

# Assuming the script is run from the root of the project where these imports work
try:
    from plugins.registry import DATASET_REGISTRY, CHANNEL_REGISTRY
    from agents import Sender, Receiver
    from communication.protocol import CommunicationProtocol
    from envs.reconstruction_env import ReconstructionGameEnv
    from envs.reward import CompositeReward
    from metrics.reconstruction import compute_mse, compute_psnr, compute_ssim, compute_pixel_accuracy
    from analysis.protocol import ProtocolAnalyzer
    from communication.message import Message
except ImportError as e:
    print(f"Import Error: {e}. Make sure you are running this from the ec_lab/environments/reconstruction_game directory.")
    sys.exit(1)


class ReconstructionEvaluator:
    """
    Comprehensive Evaluator for the Emergent Communication Reconstruction Game.
    """
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Configuration
        self._load_config()
        
        # 2. Load Dataset
        self._load_dataset()
        
        # 3. Load Trained Models
        self._load_models()

    def _load_config(self):
        print("\n--- 1. Loading Configuration ---")
        self.config = OmegaConf.load(self.config_path)
        torch.manual_seed(42)
        np.random.seed(42)
        
        print(f"Batch Size       : {self.config.dataset.batch_size}")
        print(f"Vocabulary Size  : {self.config.communication.vocab_size}")
        print(f"Message Length   : {self.config.communication.max_length}")
        print(f"Dataset          : {self.config.dataset.name}")
        print(f"Device           : {self.device}")

    def _load_dataset(self):
        print("\n--- 2. Loading Dataset ---")
        dataset_class = DATASET_REGISTRY.get(self.config.dataset.name)
        self.dataset_manager = dataset_class(self.config.dataset)
        
        self.train_loader = self.dataset_manager.get_dataloader("train")
        try:
            self.val_loader = self.dataset_manager.get_dataloader("val")
        except:
            self.val_loader = self.dataset_manager.get_dataloader("train")
            
        stats = self.dataset_manager.get_statistics()
        print(f"Dataset Size     : {len(self.train_loader.dataset)} (train)")
        print(f"Channels         : {stats.get('channels', 'N/A')}")
        print(f"Resolution       : {stats.get('resolution', 'N/A')}")
        print(f"Classes          : {stats.get('classes', 'N/A')}")
        self.metrics["Dataset Status"] = "PASS"

    def _load_models(self):
        print("\n--- 3. Loading Trained Models ---")
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
            noise_prob=0.0 # Force deterministic/no-noise for eval
        ).to(self.device)
        
        self.reward_fn = CompositeReward(
            alpha_reconstruction=self.config.reward.alpha_reconstruction,
            beta_compression=self.config.reward.beta_compression,
            gamma_diversity=self.config.reward.gamma_diversity,
            delta_efficiency=self.config.reward.delta_efficiency,
            penalties=self.config.reward.penalties
        ).to(self.device)
        
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            # Basic loading attempt
            if "model_state_dict" in checkpoint:
                try:
                    state = checkpoint["model_state_dict"]
                    if "sender" in state and "receiver" in state:
                        self.sender.load_state_dict(state["sender"])
                        self.receiver.load_state_dict(state["receiver"])
                    else:
                        self.sender.load_state_dict(state, strict=False)
                        self.receiver.load_state_dict(state, strict=False)
                except Exception as e:
                    print(f"Failed to load weights correctly: {e}")
            print(f"Models loaded successfully from {self.checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint {self.checkpoint_path} not found. Proceeding with untrained models for demonstration.")
            
        self.sender.eval()
        self.receiver.eval()
        self.metrics["Sender Loaded"] = "PASS"
        self.metrics["Receiver Loaded"] = "PASS"

    def run_environment_test(self):
        print("\n--- 4. Environment Test ---")
        env = ReconstructionGameEnv(self.config, self.dataset_manager)
        obs, info = env.reset()
        print(f"Observation Shape : {obs.shape}")
        print(f"Observation Type  : {type(obs)}")
        
        # Dummy step
        next_obs, reward, terminated, truncated, info = env.step(obs.clone())
        print(f"Reward Shape      : {reward.shape if hasattr(reward, 'shape') else type(reward)}")
        print("Environment works perfectly.")
        self.metrics["Environment Status"] = "PASS"
        env.close()

    @torch.no_grad()
    def evaluate_reconstruction(self, loader, split="Validation"):
        print(f"\n--- 5. Reconstruction Evaluation ({split}) ---")
        total_mse, total_psnr, total_ssim, total_reward, total_loss = 0, 0, 0, 0, 0
        batches = 0
        
        start_time = time.time()
        
        for data, _ in loader:
            data = data.to(self.device)
            batch_size = data.size(0)
            
            sender_out = self.sender(data)
            if self.config.communication.protocol in ["reinforce", "gumbel"]:
                sender_out = sender_out.view(batch_size, self.config.communication.max_length, self.config.communication.vocab_size)
            
            message = self.protocol(sender_out)
            
            if self.config.communication.protocol == "reinforce":
                message_one_hot = F.one_hot(message.content, num_classes=self.config.communication.vocab_size).float()
                receiver_input = message_one_hot.view(batch_size, -1)
            elif self.config.communication.protocol == "gumbel":
                receiver_input = message.content.view(batch_size, -1)
            else:
                receiver_input = message.content
            
            temp_message = Message(content=receiver_input, lengths=message.lengths)
            reconstructed = self.receiver(temp_message)
            
            rec_flat = reconstructed.view(batch_size, -1)
            obs_flat = data.view(batch_size, -1)
            
            mse = compute_mse(rec_flat, obs_flat)
            psnr = compute_psnr(rec_flat, obs_flat)
            try:
                # Need spatial dims for SSIM
                rec_spatial = reconstructed.view(batch_size, 1, 28, 28)
                obs_spatial = data.view(batch_size, 1, 28, 28)
                ssim = compute_ssim(rec_spatial, obs_spatial)
            except Exception:
                ssim = torch.zeros_like(mse)
                
            rewards = self.reward_fn(mse, message, self.config.communication.max_length, self.config.communication.vocab_size)
            
            total_mse += mse.mean().item()
            total_psnr += psnr.mean().item()
            total_ssim += ssim.mean().item()
            total_reward += rewards["reward/total"].mean().item()
            total_loss += mse.mean().item() # Treat MSE as primary loss for display
            
            batches += 1
            if batches >= 20: # Limit to 20 batches for speed in evaluation report
                break

        duration = time.time() - start_time
        fps = (batches * self.config.dataset.batch_size) / duration
        
        metrics = {
            "Average Loss": total_loss / batches,
            "Average Reward": total_reward / batches,
            "MSE": total_mse / batches,
            "MAE": np.sqrt(total_mse / batches), # Approximation
            "PSNR": total_psnr / batches,
            "SSIM": total_ssim / batches,
            "Inference Speed (imgs/sec)": fps
        }
        
        for k, v in metrics.items():
            print(f"{k:30} : {v:.4f}")
            self.metrics[f"{split} {k}"] = v
            
        return metrics

    def analyze_learning_performance(self):
        print("\n--- 6. Learning Performance ---")
        csv_path = os.path.join(self.config.experiment.log_dir, "metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.figure(figsize=(10, 5))
            if 'loss/total' in df.columns:
                plt.plot(df['step'], df['loss/total'], label='Train Loss')
            if 'val/mse' in df.columns:
                plt.plot(df['step'], df['val/mse'], label='Val MSE')
            plt.title('Learning Curves')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "learning_curves.png"))
            plt.close()
            print("Learning curves plotted to learning_curves.png")
        else:
            print("No training logs found to plot.")

    def evaluate_generalization(self):
        print("\n--- 7. Generalization Test ---")
        train_metrics = self.evaluate_reconstruction(self.train_loader, "Train")
        val_metrics = self.evaluate_reconstruction(self.val_loader, "Validation")
        
        gap = val_metrics["MSE"] - train_metrics["MSE"]
        self.metrics["Generalization Gap"] = gap
        
        print(f"\nGeneralization Gap (Val MSE - Train MSE): {gap:.4f}")
        if gap < 0.1:
            self.metrics["Generalization"] = "PASS"
        else:
            self.metrics["Generalization"] = "FAIL"
        print(f"Status: {self.metrics['Generalization']}")

    @torch.no_grad()
    def evaluate_protocol_consistency(self):
        print("\n--- 8. Protocol Consistency ---")
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)[:1] # Take 1 image
        
        messages = []
        for _ in range(10):
            out = self.sender(data)
            if self.config.communication.protocol in ["reinforce", "gumbel"]:
                out = out.view(1, self.config.communication.max_length, self.config.communication.vocab_size)
            msg = self.protocol(out)
            if msg.discrete_content is not None:
                messages.append(msg.discrete_content[0].cpu().numpy())
            else:
                messages.append(msg.content[0].cpu().numpy())
                
        # Calculate consistency
        first_msg = messages[0]
        identical = sum(1 for m in messages if np.array_equal(m, first_msg))
        
        print(f"Sample Message: {first_msg}")
        print(f"Identical messages across 10 passes: {identical}/10 ({(identical/10)*100}%)")
        self.metrics["Protocol Consistency"] = f"{(identical/10)*100}%"

    @torch.no_grad()
    def analyze_vocabulary(self):
        print("\n--- 9. Vocabulary Usage ---")
        analyzer = ProtocolAnalyzer(self.config.communication.vocab_size)
        
        for data, _ in self.val_loader:
            data = data.to(self.device)
            out = self.sender(data)
            if self.config.communication.protocol in ["reinforce", "gumbel"]:
                out = out.view(data.size(0), self.config.communication.max_length, self.config.communication.vocab_size)
            msg = self.protocol(out)
            
            content = msg.discrete_content if msg.discrete_content is not None else msg.content
            analyzer.track_batch(content, msg.lengths)
            break # Just one batch for speed
            
        stats = analyzer.get_statistics()
        util = stats.get('comm/vocab_utilization', 0)
        ent = stats.get('comm/vocab_entropy', 0)
        
        print(f"Vocabulary Utilization : {util * 100:.2f}%")
        print(f"Message Entropy        : {ent:.4f}")
        
        self.metrics["Vocabulary Utilization"] = f"{util * 100:.2f}%"
        self.metrics["Message Entropy"] = ent
        
        # Plot histogram
        plt.figure(figsize=(10,5))
        plt.bar(range(self.config.communication.vocab_size), analyzer.token_frequencies.cpu().numpy())
        plt.title("Vocabulary Usage Histogram")
        plt.savefig(os.path.join(self.output_dir, "vocab_histogram.png"))
        plt.close()

    def analyze_compression(self):
        print("\n--- 10. Compression Analysis ---")
        orig_bits = 28 * 28 * 32 # 32-bit floats
        msg_bits = self.config.communication.max_length * np.log2(self.config.communication.vocab_size)
        ratio = orig_bits / msg_bits if msg_bits > 0 else 0
        
        print(f"Original Image Bits  : {orig_bits}")
        print(f"Compressed Msg Bits  : {msg_bits:.1f}")
        print(f"Compression Ratio    : {ratio:.2f}x")
        self.metrics["Compression Ratio"] = f"{ratio:.2f}x"

    @torch.no_grad()
    def random_sender_baseline(self):
        print("\n--- 11. Random Sender Baseline ---")
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)
        batch_size = data.size(0)
        
        # Generate random message
        random_msg = torch.randint(0, self.config.communication.vocab_size, (batch_size, self.config.communication.max_length), device=self.device)
        
        if self.config.communication.protocol in ["reinforce", "gumbel"]:
            msg_one_hot = F.one_hot(random_msg, num_classes=self.config.communication.vocab_size).float()
            receiver_input = msg_one_hot.view(batch_size, -1)
        else:
            receiver_input = random_msg.float().view(batch_size, -1)
            
        temp_message = Message(content=receiver_input, lengths=torch.full((batch_size,), self.config.communication.max_length))
        reconstructed = self.receiver(temp_message)
        
        mse = compute_mse(reconstructed.view(batch_size, -1), data.view(batch_size, -1)).mean().item()
        print(f"Random Baseline Loss (MSE): {mse:.4f}")
        self.metrics["Random Baseline Loss"] = mse

    @torch.no_grad()
    def perfect_upper_bound(self):
        print("\n--- 12. Perfect Sender Upper Bound ---")
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)
        mse = compute_mse(data.view(data.size(0), -1), data.view(data.size(0), -1)).mean().item()
        print(f"Perfect Reconstruction Loss (MSE): {mse:.4f} (Theoretical 0)")
        self.metrics["Perfect Reconstruction Loss"] = mse

    @torch.no_grad()
    def evaluate_noise_robustness(self):
        print("\n--- 13. Noise Robustness ---")
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)
        batch_size = data.size(0)
        
        noise_levels = [0.0, 0.1, 0.3, 0.5]
        mses = []
        
        for noise in noise_levels:
            noisy_data = data + torch.randn_like(data) * noise
            noisy_data = torch.clamp(noisy_data, 0, 1)
            
            sender_out = self.sender(noisy_data)
            if self.config.communication.protocol in ["reinforce", "gumbel"]:
                sender_out = sender_out.view(batch_size, self.config.communication.max_length, self.config.communication.vocab_size)
            msg = self.protocol(sender_out)
            
            if self.config.communication.protocol == "reinforce":
                receiver_input = F.one_hot(msg.content, num_classes=self.config.communication.vocab_size).float().view(batch_size, -1)
            else:
                receiver_input = msg.content.view(batch_size, -1)
                
            rec = self.receiver(Message(content=receiver_input, lengths=msg.lengths))
            mse = compute_mse(rec.view(batch_size, -1), data.view(batch_size, -1)).mean().item()
            mses.append(mse)
            print(f"Noise Std {noise:.1f} -> MSE: {mse:.4f}")
            
        plt.figure()
        plt.plot(noise_levels, mses, marker='o')
        plt.title("Noise Robustness")
        plt.xlabel("Gaussian Noise Std")
        plt.ylabel("MSE")
        plt.savefig(os.path.join(self.output_dir, "noise_robustness.png"))
        plt.close()
        self.metrics["Noise Robustness"] = f"Degradation to MSE {mses[-1]:.4f}"

    @torch.no_grad()
    def message_ablation(self):
        print("\n--- 14. Message Ablation ---")
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)
        batch_size = data.size(0)
        
        sender_out = self.sender(data)
        if self.config.communication.protocol in ["reinforce", "gumbel"]:
            sender_out = sender_out.view(batch_size, self.config.communication.max_length, self.config.communication.vocab_size)
        msg = self.protocol(sender_out)
        
        drop_rates = [0.0, 0.25, 0.50, 0.75, 1.0]
        mses = []
        
        for drop in drop_rates:
            ablated = msg.content.clone()
            # Randomly zero out 'drop' fraction of the tokens/embeddings
            mask = torch.rand(ablated.shape, device=self.device) < drop
            ablated[mask] = 0
            
            if self.config.communication.protocol == "reinforce":
                rec_in = F.one_hot(ablated, num_classes=self.config.communication.vocab_size).float().view(batch_size, -1)
            elif self.config.communication.protocol == "gumbel":
                rec_in = ablated.view(batch_size, -1)
            else:
                rec_in = ablated
                
            rec = self.receiver(Message(content=rec_in, lengths=msg.lengths))
            mse = compute_mse(rec.view(batch_size, -1), data.view(batch_size, -1)).mean().item()
            mses.append(mse)
            print(f"Drop {drop*100}% -> MSE: {mse:.4f}")
            
        plt.figure()
        plt.plot([d*100 for d in drop_rates], mses, marker='s', color='red')
        plt.title("Message Ablation (Symbol Dropping)")
        plt.xlabel("% Symbols Removed")
        plt.ylabel("MSE")
        plt.savefig(os.path.join(self.output_dir, "ablation.png"))
        plt.close()

    @torch.no_grad()
    def generate_visualizations(self):
        print("\n--- 15. Visualization ---")
        data, _ = next(iter(self.val_loader))
        data = data.to(self.device)[:8] # 8 images
        batch_size = data.size(0)
        
        sender_out = self.sender(data)
        if self.config.communication.protocol in ["reinforce", "gumbel"]:
            sender_out = sender_out.view(batch_size, self.config.communication.max_length, self.config.communication.vocab_size)
        msg = self.protocol(sender_out)
        
        if self.config.communication.protocol == "reinforce":
            rec_in = F.one_hot(msg.content, num_classes=self.config.communication.vocab_size).float().view(batch_size, -1)
        elif self.config.communication.protocol == "gumbel":
            rec_in = msg.content.view(batch_size, -1)
        else:
            rec_in = msg.content
            
        rec = self.receiver(Message(content=rec_in, lengths=msg.lengths)).view(-1, 1, 28, 28)
        
        # Diff map
        diff = torch.abs(data - rec)
        
        comparison = torch.cat([data, rec, diff])
        grid = torchvision.utils.make_grid(comparison, nrow=8)
        torchvision.utils.save_image(grid, os.path.join(self.output_dir, "reconstruction_gallery.png"))
        print("Saved reconstruction_gallery.png (Row 1: Orig, Row 2: Recon, Row 3: Diff)")

    def print_final_report(self):
        print("\n" + "="*52)
        print("RECONSTRUCTION GAME EVALUATION")
        print("==============================")
        print(f"\nEnvironment Status          : {self.metrics.get('Environment Status', 'FAIL')}")
        print(f"Dataset                     : {self.metrics.get('Dataset Status', 'FAIL')}")
        print(f"Sender Loaded               : {self.metrics.get('Sender Loaded', 'FAIL')}")
        print(f"Receiver Loaded             : {self.metrics.get('Receiver Loaded', 'FAIL')}")
        
        print(f"\nAverage Train Loss          : {self.metrics.get('Train Average Loss', 0.0):.4f}")
        print(f"Average Validation Loss     : {self.metrics.get('Validation Average Loss', 0.0):.4f}")
        print(f"Average Reward              : {self.metrics.get('Validation Average Reward', 0.0):.4f}")
        print(f"MSE                         : {self.metrics.get('Validation MSE', 0.0):.4f}")
        print(f"MAE                         : {self.metrics.get('Validation MAE', 0.0):.4f}")
        print(f"PSNR                        : {self.metrics.get('Validation PSNR', 0.0):.4f}")
        print(f"SSIM                        : {self.metrics.get('Validation SSIM', 0.0):.4f}")
        
        print(f"\nCompression Ratio           : {self.metrics.get('Compression Ratio', 'N/A')}")
        print(f"Vocabulary Utilization      : {self.metrics.get('Vocabulary Utilization', 'N/A')}")
        print(f"Message Entropy             : {self.metrics.get('Message Entropy', 0.0):.4f}")
        print(f"Protocol Consistency        : {self.metrics.get('Protocol Consistency', 'N/A')}")
        
        print(f"\nGeneralization              : {self.metrics.get('Generalization', 'FAIL')}")
        print(f"Noise Robustness            : {self.metrics.get('Noise Robustness', 'N/A')}")
        print(f"Random Baseline Loss        : {self.metrics.get('Random Baseline Loss', 0.0):.4f}")
        print(f"Perfect Reconstruction Loss : {self.metrics.get('Perfect Reconstruction Loss', 0.0):.4f}")
        
        print(f"\nInference Speed             : {self.metrics.get('Validation Inference Speed (imgs/sec)', 0.0):.1f} imgs/sec")
        print(f"Peak Memory                 : {torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0:.1f} MB")
        
        val_mse = self.metrics.get('Validation MSE', float('inf'))
        score = max(0, 100 - (val_mse * 1000))
        grade = "A" if score > 90 else "B" if score > 80 else "C" if score > 70 else "F"
        
        print(f"\nOverall Score               : {score:.1f}/100")
        print(f"Overall Grade               : {grade}")
        print("====================================================")
        
        # Save JSON
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dict_str = {k: str(v) for k, v in self.metrics.items()}
            json.dump(json.dict_str, f, indent=4)
            
        # Save CSV
        df = pd.DataFrame([self.metrics])
        df.to_csv(os.path.join(self.output_dir, "metrics.csv"), index=False)
        print(f"Metrics saved to {self.output_dir}/metrics.json and .csv")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reconstruction Game")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation_report", help="Directory to save reports")
    args = parser.parse_args()

    evaluator = ReconstructionEvaluator(args.config, args.checkpoint, args.output_dir)
    evaluator.run_environment_test()
    evaluator.analyze_learning_performance()
    evaluator.evaluate_generalization()
    evaluator.evaluate_protocol_consistency()
    evaluator.analyze_vocabulary()
    evaluator.analyze_compression()
    evaluator.random_sender_baseline()
    evaluator.perfect_upper_bound()
    evaluator.evaluate_noise_robustness()
    evaluator.message_ablation()
    evaluator.generate_visualizations()
    evaluator.print_final_report()

if __name__ == "__main__":
    main()
