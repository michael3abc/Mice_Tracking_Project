import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Behaviors_Models import Behavior1D_CNN, BehaviorBiLSTM
from utils import Sliding_windows, Encode_labels, impute_windows

#1. 取X,y
csv_path = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\dataset\keypoints_with_behavior.csv"
df = pd.read_csv(csv_path)
df = df[(df["behavior"] != "unknown") & (df["behavior"] != "drink")]
df = df.sort_values(by=['video_id', 'frame']).reset_index(drop=True)

X, y = Sliding_windows(df)
X = impute_windows(X)
X_min = X.min(axis=(0,1), keepdims=True)
X_max = X.max(axis=(0,1), keepdims=True)
X_mm = (X - X_min) / (X_max - X_min + 1e-6)
np.savez("minmax_values.npz", X_min=X_min, X_max=X_max)

y_encoded, le = Encode_labels(y)

X_tensor = torch.tensor(X_mm).permute(0, 2, 1).float()   # shape: (N, 16, 32)
y_true = y_encoded           
print("🔢 LabelEncoder 對應如下：")
for i, label in enumerate(le.classes_):
    print(f"{i}: {label}")
# model = Behavior1D_CNN(num_features= 16, num_classes=7)
model =BehaviorBiLSTM(input_dim=16, num_classes=7)
best_model = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\model\best_epoch200_BehaviorBiLSTM.pth"
model.load_state_dict(torch.load(best_model, map_location='cpu'))
model.eval()


with torch.no_grad():
    y_pred = torch.argmax(model(X_tensor), dim=1).numpy() #dim=1 :抽出最大值 => 預測結果

# === 6. 畫 confusion matrix ===
labels = le.classes_
cm = confusion_matrix(y_true, y_pred)
cm_prob = cm.astype('float')/cm.sum(axis = 1, keepdims=True)


disp = ConfusionMatrixDisplay(confusion_matrix=cm_prob, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix on Full CSV (Validation)")
plt.tight_layout()
plt.show()