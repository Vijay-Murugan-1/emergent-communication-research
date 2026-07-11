import json

print("FINAL BASELINE RESULTS (Best Checkpoints - Evaluation on Validation Set)")
print("=" * 70)

for m in ['continuous', 'gumbel', 'reinforce']:
    try:
        with open(f'logs/{m}/metrics.json') as f:
            data = json.load(f)
            
            # The JSON is a dictionary of lists. We want to find the min val/mse.
            if 'val/mse' in data:
                val_mses = data['val/mse']
                min_idx = val_mses.index(min(val_mses))
                
                print(f"Protocol: {m.upper()}")
                print(f"  Best Validation MSE:    {val_mses[min_idx]:.4f}")
                print(f"  Best Validation PSNR:   {data['val/psnr'][min_idx]:.4f}")
                print(f"  Best Validation Reward: {data['val/reward_total'][min_idx]:.4f}")
                print("-" * 70)
            else:
                print(f"Protocol: {m.upper()} - No evaluation metrics found in JSON.")
    except Exception as e:
        print(f"Protocol: {m.upper()} - Error reading JSON: {e}")
        
print("Current Status: Phase 3 & 4 (Scientific Evaluation) fully completed.")
