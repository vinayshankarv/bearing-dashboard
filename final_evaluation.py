# ==========================================
# FINAL EVALUATION PIPELINE (DASHBOARD READY)
# ==========================================

import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# PATHS
# ==========================================
train_dir = r"D:/SET2_EXPANDED_PROJECT/data_augmented/train"
test_dir  = r"D:/SET2_EXPANDED_PROJECT/data_augmented/test"

# Output folder
os.makedirs("metrics", exist_ok=True)

# ==========================================
# TRANSFORMS
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================
# DATASET
# ==========================================
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Classes:", test_dataset.classes)

# ==========================================
# MODELS
# ==========================================
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RCA_EfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,  
            num_classes=0
        )

        self.feat_dim = self.backbone.num_features

        self.attention = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim // 8, self.feat_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)

        attn = self.attention(feat)

        feat = feat + (feat * attn)

        feat = self.dropout(feat)

        return self.classifier(feat)

# ==========================================
# MODEL LOADING
# ==========================================
def load_timm_model(name, path):
    model = timm.create_model(name, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_custom_model(model_class, path):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


models = {
    "CNN": load_custom_model(CNNModel, "models/cnn_baseline.pth"),
    "ResNet50": load_timm_model("resnet50", "models/resnet_50.pth"),
    "EfficientNet": load_timm_model("efficientnet_b0", "models/efficientnet_b0.pth"),
    "DeiT": load_timm_model("deit_small_patch16_224", "models/deit_small.pth"),
    "RCA_EfficientNet": load_custom_model(RCA_EfficientNet, "models/RCA_EfficientNet.pth")
}

# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate(model):
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ==========================================
# RUN EVALUATION
# ==========================================
results = {}

for name, model in models.items():
    print(f"Evaluating {name}...")
    results[name] = evaluate(model)

# ==========================================
# SAVE METRICS
# ==========================================
summary_rows = []

for name, (y_true, y_pred, y_prob) in results.items():

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    summary_rows.append({
        "Model": name,
        "Accuracy": round(acc * 100, 4),
        "Precision": round(prec * 100, 4),
        "Recall": round(rec * 100, 4),
        "F1 Score": round(f1 * 100, 4),
        "Kappa": round(kappa, 4)
    })

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = test_dataset.classes  
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(f"metrics/cm_{name}.csv")

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(
        f"metrics/roc_{name}.csv", index=False
    )

    # Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({
        "precision": precision_vals,
        "recall": recall_vals
    }).to_csv(f"metrics/pr_{name}.csv", index=False)

# ==========================================
# SAVE COMPARISON TABLE
# ==========================================
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by="Accuracy", ascending=False)

summary_df.to_csv("metrics/model_comparison.csv", index=False)

print("\n✅ Evaluation complete. Metrics saved in 'metrics/' folder.")