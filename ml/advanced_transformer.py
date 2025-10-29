import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="ml/syn.csv")
parser.add_argument("--epochs", type=int, default=3)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.data)
features = df[["solar", "wind", "demand"]].values
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Prepare sequences
seq_len = 24
X, y = [], []
for i in range(len(features) - seq_len):
    X.append(features[i:i+seq_len])
    y.append(features[i+seq_len, 2])

# Convert to tensors (fix slow list warning)
X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define Transformer
class TransformerForecast(nn.Module):
    def __init__(self, input_size=3, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True  # batch_first for better performance
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = x[:, -1, :]  # last time step
        return self.fc(x)

model = TransformerForecast()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(args.epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save model
torch.save(model.state_dict(), "models/transformer_forecast.pt")
print("✅ Model saved to models/transformer_forecast.pt")

