import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = "models/classifier.pt"


class BotnetMLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(BotnetMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class BotnetClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📟 Используется устройство: {self.device}")

    def train(self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 64, lr: float = 0.001):
        """
        Обучает модель на новых признаках.
        """
        X = df.drop(columns=["label", "source_file"], errors="ignore").values
        y = self.label_encoder.fit_transform(df["label"].values)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        input_size = X.shape[1]
        self.model = BotnetMLP(input_size=input_size, num_classes=len(set(y))).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"📚 Эпоха {epoch + 1}/{epochs} | Потери: {total_loss:.4f}")

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'classes': self.label_encoder.classes_,
            'input_size': self.model.model[0].in_features  # сохраняем размерность входа
        }, MODEL_PATH)
        print("💾 Модель сохранена.")

    def load_model(self, input_size: int):
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        saved_input_size = checkpoint.get("input_size")
        if saved_input_size and saved_input_size != input_size:
            raise ValueError(f"❌ Размерность признаков не совпадает: модель ожидала {saved_input_size}, а получили {input_size}. Переобучите модель.")
        self.model = BotnetMLP(input_size=input_size, num_classes=len(checkpoint['classes']))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.label_encoder.classes_ = checkpoint['classes']
        print("📥 Модель загружена.")

    def predict(self, df: pd.DataFrame) -> List[str]:
        X = df.drop(columns=["source_file", "label"], errors="ignore").values
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return self.label_encoder.inverse_transform(preds)
