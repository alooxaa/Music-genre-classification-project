import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. Load data: X = 89 features, y = genre
df = pd.read_csv("features.csv")
X = df.drop(columns=["file", "label"])
y = df["label"]
print("X:", X.shape, "| genres:", y.value_counts().to_dict())

# 2. Model: scaling + neural network (89 → 128 → 64 → 32 → 3)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42),
)

# 3. Evaluation: 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
print("Accuracy per fold:", scores.round(2))
print(f"Mean accuracy: {scores.mean():.3f}")

pred = cross_val_predict(model, X, y, cv=cv)
print(classification_report(y, pred))
ConfusionMatrixDisplay.from_predictions(y, pred, cmap="Blues")
plt.title("Confusion matrix — MLP")
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")

# 4. Final training on all 30 songs and saving the model
model.fit(X, y)
mlp = model.named_steps["mlpclassifier"]
print(f"Training finished after {mlp.n_iter_} epochs, final loss = {mlp.loss_:.4f}")

plt.figure()
plt.plot(mlp.loss_curve_)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("MLP training curve")
plt.savefig("loss_curve.png", dpi=120, bbox_inches="tight")

joblib.dump({"model": model, "columns": list(X.columns), "name": "MLP"}, "genre_model.joblib")
print("Model saved → genre_model.joblib")
