"""
Complex PyTorch training pipeline.
Includes: DataLoader, text transformer, Stable Diffusion, ViT classifier.
"""
with __import__("torch").profiler.record_function("TB:train.Block2 torch=torch.device|torch.cuda.is_available|torch.stack|torch.utils.data.DataLoader|torch.optim.AdamW|torch.nn.CrossEntropyLoss|..."):
    import random
    from typing import Dict, Optional
    
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    
    from toy_transformer_pkg.data import ImageDataset, TextDataset
    from toy_transformer_pkg.models import (
        StableDiffusionPipeline,
        TextTransformer,
        ViTClassifier,
    )
    
    
    def get_device() -> torch.device:
        with __import__("torch").profiler.record_function("TB:train.get_device.Block4 torch=torch.device|torch.cuda.is_available"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def collate_text(batch: list) -> Dict[str, torch.Tensor]:
        with __import__("torch").profiler.record_function("TB:train.collate_text.Block8 torch=torch.stack"):
            """Collate text batch."""
            input_ids = torch.stack([b["input_ids"] for b in batch])
            attention_mask = torch.stack([b["attention_mask"] for b in batch])
            labels = torch.stack([b["labels"] for b in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    
    def train_text_transformer(
        device: torch.device,
        num_epochs: int = 2,
        batch_size: int = 16,
        lr: float = 1e-4,
    ) -> None:
        with __import__("torch").profiler.record_function("TB:train.train_text_transformer.Block12 torch=torch.utils.data.DataLoader|torch.optim.AdamW|torch.nn.CrossEntropyLoss"):
            """Train text transformer with DataLoader."""
            dataset = TextDataset(num_samples=256, max_length=64, vocab_size=1000)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_text,
                num_workers=0,
            )
            model = TextTransformer(
                vocab_size=1000, max_length=64, num_classes=3, d_model=128, num_layers=2
            )
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            model.train()
        with __import__("torch").profiler.record_function("TB:train.train_text_transformer.Block13 torch=torch.nn.utils.clip_grad_norm_"):
            for epoch in range(num_epochs):
                total_loss = 0.0
                with __import__("torch").profiler.record_function("TB:train.train_text_transformer.Block16 torch=torch.nn.utils.clip_grad_norm_"):
                    for batch in dataloader:
                        with __import__("torch").profiler.record_function("TB:train.train_text_transformer.Block17 torch=torch.nn.utils.clip_grad_norm_"):
                            input_ids = batch["input_ids"].to(device)
                            attention_mask = batch["attention_mask"].to(device)
                            labels = batch["labels"].to(device)
                            optimizer.zero_grad()
                            logits = model(input_ids=input_ids, attention_mask=attention_mask)
                            loss = criterion(logits, labels)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            total_loss += loss.item()
                print(f"[TextTransformer] Epoch {epoch+1} loss={total_loss/len(dataloader):.4f}")
    
    
    def train_stable_diffusion(
        device: torch.device,
        num_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
    ) -> None:
        with __import__("torch").profiler.record_function("TB:train.train_stable_diffusion.Block21 torch=torch.utils.data.DataLoader|torch.optim.AdamW"):
            """Train Stable Diffusion (VAE + UNet) on synthetic images."""
            dataset = ImageDataset(
                num_samples=64, image_size=64, num_channels=3, num_classes=1
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            pipeline = StableDiffusionPipeline()
            pipeline.to(device)
            optimizer = torch.optim.AdamW(pipeline.parameters(), lr=lr)
            pipeline.train()
        with __import__("torch").profiler.record_function("TB:train.train_stable_diffusion.Block22 torch=torch.randn_like|torch.nn.functional.mse_loss|torch.randint"):
            for epoch in range(num_epochs):
                total_loss = 0.0
                with __import__("torch").profiler.record_function("TB:train.train_stable_diffusion.Block25 torch=torch.randn_like|torch.nn.functional.mse_loss|torch.randint"):
                    for images, _ in dataloader:
                        with __import__("torch").profiler.record_function("TB:train.train_stable_diffusion.Block26 torch=torch.randn_like|torch.randint.long|torch.nn.functional.mse_loss|torch.randint"):
                            images = images.to(device)
                            latents = pipeline.encode_image(images)
                            noise = torch.randn_like(latents, device=device)
                            timesteps = torch.randint(0, 1000, (images.size(0),), device=device).long()
                            noisy = latents + 0.1 * noise
                            optimizer.zero_grad()
                            pred = pipeline.unet(noisy, timestep=timesteps)
                            loss = nn.functional.mse_loss(pred, noise)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                print(f"[StableDiffusion] Epoch {epoch+1} loss={total_loss/len(dataloader):.4f}")
    
    
    def train_vit_classifier(
        device: torch.device,
        num_epochs: int = 2,
        batch_size: int = 8,
        lr: float = 1e-4,
    ) -> None:
        with __import__("torch").profiler.record_function("TB:train.train_vit_classifier.Block30 torch=torch.utils.data.DataLoader|torch.optim.AdamW|torch.nn.CrossEntropyLoss"):
            """Train ViT classifier on synthetic images."""
            dataset = ImageDataset(
                num_samples=128, image_size=224, num_channels=3, num_classes=10
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )
            model = ViTClassifier(
                img_size=224,
                patch_size=16,
                embed_dim=192,
                num_heads=3,
                num_layers=2,
                num_classes=10,
            )
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            model.train()
        with __import__("torch").profiler.record_function("TB:train.train_vit_classifier.Block31 torch=torch.nn.utils.clip_grad_norm_"):
            for epoch in range(num_epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                with __import__("torch").profiler.record_function("TB:train.train_vit_classifier.Block34 torch=torch.nn.utils.clip_grad_norm_"):
                    for images, labels in dataloader:
                        with __import__("torch").profiler.record_function("TB:train.train_vit_classifier.Block35 torch=torch.nn.utils.clip_grad_norm_"):
                            images = images.to(device)
                            labels = labels.to(device)
                            optimizer.zero_grad()
                            logits = model(images)
                            loss = criterion(logits, labels)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            total_loss += loss.item()
                            preds = logits.argmax(dim=-1)
                            correct += (preds == labels).sum().item()
                            total += labels.size(0)
                acc = correct / max(1, total)
                print(f"[ViT] Epoch {epoch+1} loss={total_loss/len(dataloader):.4f} acc={acc:.4f}")
    
    
    def main() -> None:
        with __import__("torch").profiler.record_function("TB:train.main.Block39 torch=torch.manual_seed"):
            torch.manual_seed(42)
            random.seed(42)
            device = get_device()
            print(f"Device: {device}")
    
            print("\n--- Text Transformer ---")
            train_text_transformer(device)
    
            print("\n--- Stable Diffusion ---")
            train_stable_diffusion(device)
    
            print("\n--- ViT Classifier ---")
            train_vit_classifier(device)
    
            print("\nDone.")
    
    
    if __name__ == "__main__":
        main()
