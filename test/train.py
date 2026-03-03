#!/usr/bin/env python3
"""
Simple 2-layer MLP training script.
Modified to FORCE visible CPU↔GPU memory transfers for profiling.

Key changes:
- Create inputs/targets on CPU each step, then .to(device) -> forces HtoD copies on CUDA.
- Keep loss.item() -> forces a small DtoH (GPU -> CPU) copy/sync on CUDA.
- (Optional) pin_memory() best-effort to make HtoD copies more “memcpy-like” in traces.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device:", device)

    model = TwoLayerMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    input_dim = 512
    num_classes = 10

    model.train()

    for step in range(20):  # small loop for profiling
        # ---- Force HtoD transfers (when CUDA) ----
        cpu_inputs = torch.randn(batch_size, input_dim)  # CPU
        cpu_targets = torch.randint(0, num_classes, (batch_size,))  # CPU

        if use_cuda:
            # Best-effort pinning (helps produce clearer memcpy HtoD events)
            non_blocking_in = False
            non_blocking_tg = False
            try:
                cpu_inputs = cpu_inputs.pin_memory()
                non_blocking_in = True
            except Exception:
                pass
            try:
                cpu_targets = cpu_targets.pin_memory()
                non_blocking_tg = True
            except Exception:
                pass

            inputs = cpu_inputs.to(device, non_blocking=non_blocking_in)   # HtoD
            targets = cpu_targets.to(device, non_blocking=non_blocking_tg) # HtoD
        else:
            inputs = cpu_inputs
            targets = cpu_targets

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            # Forces a small DtoH transfer/sync on CUDA
            loss_value = loss.item()
            print(f"Step {step:02d} | Loss: {loss_value:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()