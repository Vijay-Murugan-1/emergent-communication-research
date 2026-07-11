import json

for m in ['continuous', 'gumbel', 'reinforce']:
    with open(f'logs/{m}/metrics.json') as f:
        d = json.load(f)
        print(f'{m.upper()}:')
        print(f'  Test MSE: {d.get("test/mse", ["N/A"])[-1]}')
        print(f'  Test PSNR: {d.get("test/psnr", ["N/A"])[-1]}')
        print(f'  Test Reward: {d.get("test/reward_total", ["N/A"])[-1]}')
        
print("\nCurrent Status: Baselines completed successfully.")
