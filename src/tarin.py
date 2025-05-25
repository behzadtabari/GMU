import os

import torch, torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(__file__)

# Go up one level to reach the Folder root
base_dir = os.path.abspath(os.path.join(current_dir, '..'))


# ----------  GMU LAYER ----------
class GMU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        """
        in_dim      – dimensionality of each modality’s input vector
        hidden_dim  – size of the internal fused representation
        """
        super().__init__()
        # Linear transforms for each modality
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.fc2  = nn.Linear(in_dim, hidden_dim)
        # Gating network (takes concatenated modalities)
        self.gate = nn.Linear(2 * in_dim, hidden_dim)

    def forward(self, x1, x2):
        h1 = torch.tanh(self.fc1(x1))
        h2 = torch.tanh(self.fc2(x2))
        g  = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        return g * h1 + (1 - g) * h2  # fused representation
# ---------------------------------

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gmu = GMU(in_dim=20, hidden_dim=32)
        self.out = nn.Linear(32, 1)

    def forward(self, x1, x2):
        fused = self.gmu(x1, x2)
        return torch.sigmoid(self.out(fused)).squeeze(-1)

# ----------  DATA LOAD ----------
def load_split(path):
    df = pd.read_csv(path)
    x1 = torch.tensor(df.iloc[:, :20].values, dtype=torch.float32)
    x2 = torch.tensor(df.iloc[:, 20:40].values, dtype=torch.float32)
    y  = torch.tensor(df["label"].values, dtype=torch.float32)
    return TensorDataset(x1, x2, y)

# Construct the path to the data file
train_data_path = os.path.join(base_dir, 'data', 'gmu_train.csv')
test_data_path = os.path.join(base_dir, 'data', 'gmu_test.csv')

train_ds = load_split(train_data_path)
test_ds  = load_split(test_data_path)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64)

# ----------  TRAIN LOOP ----------
model = Classifier()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.BCELoss()

for epoch in range(20):
    model.train()
    for x1, x2, y in train_loader:
        opt.zero_grad()
        preds = model(x1, x2)
        loss  = lossf(preds, y)
        loss.backward()
        opt.step()

    # quick validation
    model.eval()
    with torch.no_grad():
        acc = []
        for x1, x2, y in test_loader:
            acc.append(((model(x1, x2) > .5) == y).float().mean())
    print(f"Epoch {epoch+1:>2}: test acc = {torch.stack(acc).mean():.3f}")
