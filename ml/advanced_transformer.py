import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------
#  MODEL DEFINITION
# ---------------------------------------------------
class TransformerForecast(nn.Module):
    """Transformer model for next-step energy demand forecasting."""
    def __init__(self, input_size=3, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = x[:, -1, :]  # use last time step representation
        return self.fc(x).squeeze(-1)

# ---------------------------------------------------
#  TRAINING FUNCTION
# ---------------------------------------------------
def train_model(data_path="ml/syn.csv", epochs=3):
    """
    Train the Transformer model on synthetic or real data and save weights.
    """

    print(f"🚀 Starting training using data: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    features = df[["solar", "wind", "demand"]].values

    # Normalize input features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Prepare sequences
    seq_len = 24
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(features[i + seq_len, 2])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = TransformerForecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"🧠 Training for {epochs} epochs on {len(X)} samples...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/transformer_forecast.pt")
    print("✅ Model saved to models/transformer_forecast.pt")


# ---------------------------------------------------
#  CLI ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Transformer model for EnergySphere.")
    parser.add_argument("--data", default="ml/syn.csv", help="Path to CSV training data.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    args = parser.parse_args()

    train_model(data_path=args.data, epochs=args.epochs)
