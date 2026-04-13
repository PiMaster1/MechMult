from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import os


class ModMultDataset(Dataset):
    def __init__(self, p, pairs):
        self.p = p
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, b = self.pairs[index]
        return self.pairs[index], (a * b) % self.p


class ModMultTransformer(nn.Module):
    def __init__(self, p, d_model, n_heads=1, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(p, d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn':  nn.MultiheadAttention(d_model, n_heads),
                'fc1':   nn.Linear(d_model, 4 * d_model),
                'fc2':   nn.Linear(4 * d_model, d_model),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])
        self.unembed = nn.Linear(d_model, p)

    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(0, 1)
        for block in self.blocks:
            attn, _ = block['attn'](x, x, x)
            x = block['norm1'](x + attn)
            v = block['fc1'](x)
            v = F.gelu(v)
            v = block['fc2'](v)
            x = block['norm2'](x + v)
        x = self.unembed(x)
        return x[-1]


p = 97
N_TRIALS = 5
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


if __name__ == '__main__':
    os.makedirs('trials_p97', exist_ok=True)

    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        print(f"\n{'='*50}")
        print(f"Trial {trial}")
        print(f"{'='*50}")

        pairs = torch.tensor([[a, b] for a in range(p) for b in range(p)])
        perm  = torch.randperm(len(pairs))
        pairs = pairs[perm]
        train_split = pairs[:int(0.3 * len(pairs))]
        test_split  = pairs[int(0.3 * len(pairs)):]

        train_data   = ModMultDataset(p, train_split)
        test_data    = ModMultDataset(p, test_split)
        train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
        test_loader  = DataLoader(test_data,  batch_size=512, shuffle=False)

        model     = ModMultTransformer(p=p, d_model=128, n_heads=1, num_layers=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
        criterion = nn.CrossEntropyLoss()
        metrics   = []
        grokked_at = None

        for epoch in range(10000):
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                output = model(inputs)
                loss   = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            correct = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long()
                    output = model(inputs)
                    preds  = output.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
            test_acc = correct / len(test_data)

            metrics.append({'epoch': epoch, 'train_loss': train_loss, 'test_acc': test_acc})

            if epoch % 100 == 0:
                print(f"epoch {epoch} | train loss {train_loss:.4f} | test acc {test_acc:.4f}")

            if test_acc == 1.0 and grokked_at is None:
                grokked_at = epoch
            if grokked_at is not None and epoch >= grokked_at + 500:
                print(f"Grokked at epoch {grokked_at}, stopping early")
                break

        torch.save(model.state_dict(), f'trials_p97/checkpoint_t{trial}.pt')
        with open(f'trials_p97/metrics_t{trial}.json', 'w') as f:
            json.dump({'trial': trial, 'grokked_at': grokked_at, 'grokked': grokked_at is not None, 'metrics': metrics}, f)
        print(f"Saved trials_p97/checkpoint_t{trial}.pt | grokked_at={grokked_at}")