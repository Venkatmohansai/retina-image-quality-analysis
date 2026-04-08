import os
import cv2
import joblib

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from feature_extraction import extract_features


# ======================================================
# INITIALIZE APP
# ======================================================

app = Flask(__name__)


# ======================================================
# PATH SETTINGS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "random_forest_selectkbest_best.pkl"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ======================================================
# LOAD TRAINED MODEL
# ======================================================

model = joblib.load(MODEL_PATH)

print("✅ Random Forest model loaded successfully")


# ======================================================
# FEATURE DESCRIPTIONS
# ======================================================

FEATURE_DESCRIPTIONS = {

"C_Mean": "Average brightness of the retinal image.",
"C_Std": "Intensity variation indicating vessel contrast strength.",
"C_Skew": "Asymmetry of grayscale distribution.",
"C_Kurt": "Sharpness of intensity peaks.",
"C_Ent": "Entropy measuring texture complexity.",
"C_Q1": "Lower quartile intensity.",
"C_Med": "Median brightness level.",
"C_Q3": "Upper quartile intensity.",

"C_Contrast_0°": "Horizontal texture contrast (0°).",
"C_Corr_0°": "Horizontal pixel correlation (0°).",
"C_En_0°": "Horizontal texture energy (0°).",
"C_Hom_0°": "Horizontal homogeneity (0°).",

"C_Corr_45°": "Diagonal correlation (45°).",
"C_En_45°": "Diagonal energy (45°).",

"C_Contrast_90°": "Vertical contrast (90°).",
"C_Corr_90°": "Vertical correlation (90°).",
"C_En_90°": "Vertical energy (90°).",
"C_Hom_90°": "Vertical homogeneity (90°).",

"C_Corr_135°": "Diagonal correlation (135°).",
"C_En_135°": "Diagonal energy (135°).",


"D_Mean": "Mean intensity after downsampling.",
"D_Std": "Contrast variation after downsampling.",
"D_Ent": "Entropy after resolution reduction.",
"D_Q1": "Lower quartile after downsampling.",
"D_Med": "Median after downsampling.",
"D_Q3": "Upper quartile after downsampling.",

"D_Corr_0°": "Horizontal correlation (0°).",
"D_En_0°": "Horizontal energy (0°).",

"D_Corr_45°": "Diagonal correlation (45°).",
"D_En_45°": "Diagonal energy (45°).",

"D_Corr_90°": "Vertical correlation (90°).",
"D_En_90°": "Vertical energy (90°).",

"D_Corr_135°": "Diagonal correlation (135°).",
"D_En_135°": "Diagonal energy (135°).",


"Sym1": "Left–right symmetry difference.",
"Sym2": "Top–bottom symmetry difference.",
"Sym3": "Main diagonal symmetry difference.",
"Sym4": "Secondary diagonal symmetry difference.",


"F_Skew": "Filtered skewness after smoothing.",
"F_Kurt": "Filtered kurtosis after smoothing.",

"F_Corr_0°": "Filtered correlation (0°).",
"F_En_0°": "Filtered energy (0°).",

"F_Corr_45°": "Filtered correlation (45°).",
"F_En_45°": "Filtered energy (45°).",

"F_Corr_90°": "Filtered correlation (90°).",
"F_En_90°": "Filtered energy (90°).",

"F_Corr_135°": "Filtered correlation (135°).",
"F_En_135°": "Filtered energy (135°).",


"W_RLHStd": "Std of LH wavelet band.",
"W_RLHKurt": "Kurtosis of LH band.",
"W_GLHStd": "Std of HL band.",
"W_GLHSkew": "Skewness of HL band.",
"W_GLHKurt": "Kurtosis of HL band.",
"W_GHLSkew": "Skewness of HH band.",
"W_GHLKurt": "Kurtosis of HH band.",
"W_BLHSkew": "Skewness low-frequency LH band.",
"W_BLHKurt": "Kurtosis low-frequency LH band.",
"W_BHLMean": "Mean of HL band.",
"W_BHLSkew": "Skewness HL band.",
"W_BHHSkew": "Skewness HH band"
}


# ======================================================
# MAIN ROUTE
# ======================================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename:

            filename = secure_filename(file.filename)

            image_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            file.save(image_path)

            img = cv2.imread(image_path)

            if img is None:
                error = "Invalid image file"
                return render_template(
                    "index.html",
                    error=error
                )

            features = extract_features(image_path)

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            class_index = list(model.classes_).index(prediction)

            confidence = round(
                probabilities[class_index] * 100,
                2
            )

            result = (
                "GOOD QUALITY IMAGE"
                if prediction == 1
                else "BAD QUALITY IMAGE"
            )

            image_url = f"/uploads/{filename}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        error=error,
        model_name="Random Forest"
    )


# ======================================================
# FEATURE PAGE
# ======================================================

@app.route("/features")
def features():

    return render_template(
        "features.html",
        features=FEATURE_DESCRIPTIONS
    )


# ======================================================
# SERVE UPLOADS
# ======================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":

    app.run(debug=True)
