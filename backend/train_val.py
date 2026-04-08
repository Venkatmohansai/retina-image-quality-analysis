import os
import numpy as np
import joblib

from feature_extraction import extract_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "dataset")
)

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# --------------------------------------------------
# DATA LOADER FUNCTION
# --------------------------------------------------

def load_data(split):

    X = []
    y = []

    split_dir = os.path.join(DATASET_DIR, split)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Missing folder: {split_dir}")

    print(f"\n📂 Loading {split} dataset from:", split_dir)

    for cls in ["0", "1"]:   # 0 = BAD, 1 = GOOD

        class_dir = os.path.join(split_dir, cls)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        label = int(cls)

        for img in os.listdir(class_dir):

            img_path = os.path.join(class_dir, img)

            try:

                features = extract_features(img_path)

                X.append(features.flatten())
                y.append(label)

            except Exception as e:

                print("⚠ Skipping:", img_path)
                print("Reason:", e)

    return np.array(X), np.array(y)


# --------------------------------------------------
# LOAD TRAIN + VALIDATION DATA
# --------------------------------------------------

X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# --------------------------------------------------
# RANDOM FOREST CLASSIFIER
# --------------------------------------------------

model = RandomForestClassifier(

    n_estimators=300,        # number of trees
    max_depth=25,            # prevents overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced", # handles BAD/GOOD imbalance
    random_state=42,
    n_jobs=-1                # uses all CPU cores

)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\n🧠 Training Random Forest classifier...")

model.fit(X_train, y_train)


# --------------------------------------------------
# VALIDATION RESULTS
# --------------------------------------------------

print("\n📊 VALIDATION RESULTS")

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(
    y_val,
    y_pred,
    target_names=["BAD", "GOOD"]
))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_val, y_pred))


# --------------------------------------------------
# SAVE MODEL FOR FLASK APP
# --------------------------------------------------

model_path = os.path.join(
    ARTIFACTS_DIR,
    "random_forest_selectkbest_best.pkl"
)

joblib.dump(model, model_path)

print("\n✅ Model saved successfully at:")
print(model_path)
