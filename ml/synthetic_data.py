import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="syn.csv")
parser.add_argument("--hours", type=int, default=24*30)
args = parser.parse_args()

time = pd.date_range(start="2025-01-01", periods=args.hours, freq="H")
solar = np.clip(np.sin(np.arange(args.hours)/24*2*np.pi)*50 + 50 + np.random.randn(args.hours)*5, 0, 100)
wind = np.clip(np.sin(np.arange(args.hours)/24*4*np.pi)*30 + 30 + np.random.randn(args.hours)*3, 0, 100)
demand = solar*0.3 + wind*0.5 + 50 + np.random.randn(args.hours)*5

df = pd.DataFrame({"time": time, "solar": solar, "wind": wind, "demand": demand})
df.to_csv(args.out, index=False)
print(f"Synthetic data saved to {args.out}")
