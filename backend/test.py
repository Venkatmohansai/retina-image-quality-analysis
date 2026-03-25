import os
import numpy as np
import joblib

from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "dataset")
)

ARTIFACTS_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "artifacts")
)

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "svm_selectkbest_best.pkl"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("🚀 Loading trained SVM model...")

model = joblib.load(MODEL_PATH)

print("✅ Model loaded successfully")
print("Model type:", type(model))


# --------------------------------------------------
# LOAD TEST DATASET
# --------------------------------------------------

X_test = []
y_test = []

test_dir = os.path.join(DATASET_DIR, "test")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"❌ Test folder not found: {test_dir}")

print(f"\n📂 Scanning test dataset: {test_dir}")


for cls in ["0", "1"]:   # 0 = bad quality, 1 = good quality

    folder = os.path.join(test_dir, cls)
    label = int(cls)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Missing class folder: {folder}")

    for img in os.listdir(folder):

        img_path = os.path.join(folder, img)

        try:

            features = extract_features(img_path)

            X_test.append(features.flatten())

            y_test.append(label)

        except Exception as e:

            print("⚠ Skipping corrupted image:", img_path)
            print("Reason:", e)


X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"\n✅ Loaded {len(X_test)} test samples")


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

print("\n🧠 Running SVM predictions...")

y_pred = model.predict(X_test)


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n📊 FINAL TEST PERFORMANCE")
print("===================================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Bad Quality", "Good Quality"]
    )
)

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# OPTIONAL: SAVE RESULTS FOR PAPER REPORTING
# --------------------------------------------------

results_path = os.path.join(
    ARTIFACTS_DIR,
    "test_results.txt"
)

with open(results_path, "w") as f:

    f.write("FINAL TEST PERFORMANCE\n")
    f.write("======================\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["Bad Quality", "Good Quality"]
        )
    )

    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))


print(f"\n📁 Results saved to: {results_path}")