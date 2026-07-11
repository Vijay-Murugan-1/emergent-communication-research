import pandas as pd

print("FINAL BASELINE RESULTS (Best Checkpoints - Evaluation on Validation Set)")
print("=" * 70)

for m in ['continuous', 'gumbel', 'reinforce']:
    try:
        df = pd.read_csv(f'logs/{m}/metrics.csv')
        # Drop rows where val/mse is NaN to get the evaluation rows
        eval_df = df.dropna(subset=['val/mse'])
        
        if not eval_df.empty:
            # For Reinforce, it triggered early stopping. We want the best val/mse.
            best_row = eval_df.loc[eval_df['val/mse'].idxmin()]
            
            print(f"Protocol: {m.upper()}")
            print(f"  MSE:    {best_row['val/mse']:.4f}")
            print(f"  PSNR:   {best_row['val/psnr']:.4f}")
            print(f"  Reward: {best_row['val/reward_total']:.4f}")
            print("-" * 70)
        else:
            print(f"Protocol: {m.upper()} - No evaluation metrics found.")
    except Exception as e:
        print(f"Protocol: {m.upper()} - Error reading logs: {e}")
        
print("Current Status: Phase 3 & 4 (Scientific Evaluation) fully completed.")
