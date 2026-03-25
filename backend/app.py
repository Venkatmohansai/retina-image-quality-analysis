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

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "svm_selectkbest_best.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ======================================================
# LOAD TRAINED MODEL
# ======================================================

model = joblib.load(MODEL_PATH)

print("✅ SVM model loaded successfully")


# ======================================================
# FEATURE DESCRIPTIONS (WITH DEGREE ORIENTATION)
# ======================================================

FEATURE_DESCRIPTIONS = {

# ---------- C FEATURES (Original Image) ----------

"C_Mean": "Average brightness of the retinal image.",
"C_Std": "Intensity variation indicating vessel contrast strength.",
"C_Skew": "Asymmetry of grayscale distribution revealing illumination imbalance.",
"C_Kurt": "Sharpness of intensity peaks indicating vessel edge clarity.",
"C_Ent": "Entropy measuring structural complexity of retinal texture.",
"C_Q1": "Lower quartile intensity detecting darker regions.",
"C_Med": "Median brightness representing stable illumination level.",
"C_Q3": "Upper quartile intensity representing highlight regions.",

"C_Contrast_0°": "Texture contrast along horizontal direction (0°).",
"C_Corr_0°": "Pixel correlation along horizontal direction (0°).",
"C_En_0°": "Texture uniformity along horizontal direction (0°).",
"C_Hom_0°": "Intensity smoothness along horizontal direction (0°).",

"C_Corr_45°": "Pixel correlation along diagonal direction (45°).",
"C_En_45°": "Texture uniformity along diagonal direction (45°).",

"C_Contrast_90°": "Texture contrast along vertical direction (90°).",
"C_Corr_90°": "Pixel correlation along vertical direction (90°).",
"C_En_90°": "Texture uniformity along vertical direction (90°).",
"C_Hom_90°": "Intensity smoothness along vertical direction (90°).",

"C_Corr_135°": "Pixel correlation along diagonal direction (135°).",
"C_En_135°": "Texture uniformity along diagonal direction (135°).",


# ---------- D FEATURES (Downsampled Image) ----------

"D_Mean": "Mean intensity after resolution reduction.",
"D_Std": "Contrast variation after downsampling.",
"D_Ent": "Texture randomness after resolution reduction.",
"D_Q1": "Lower quartile intensity after downsampling.",
"D_Med": "Median intensity after downsampling.",
"D_Q3": "Upper quartile intensity after downsampling.",

"D_Corr_0°": "Horizontal correlation at reduced resolution (0°).",
"D_En_0°": "Horizontal texture energy at reduced resolution (0°).",

"D_Corr_45°": "Diagonal correlation at reduced resolution (45°).",
"D_En_45°": "Diagonal texture energy at reduced resolution (45°).",

"D_Corr_90°": "Vertical correlation at reduced resolution (90°).",
"D_En_90°": "Vertical texture energy at reduced resolution (90°).",

"D_Corr_135°": "Diagonal correlation at reduced resolution (135°).",
"D_En_135°": "Diagonal texture energy at reduced resolution (135°).",


# ---------- SYMMETRY FEATURES ----------

"Sym1": "Difference between left and right halves of retina.",
"Sym2": "Difference between upper and lower halves of retina.",
"Sym3": "Difference across main diagonal symmetry.",
"Sym4": "Difference across secondary diagonal symmetry.",


# ---------- FILTERED FEATURES ----------

"F_Skew": "Intensity asymmetry after Gaussian smoothing.",
"F_Kurt": "Sharpness of filtered intensity distribution.",

"F_Corr_0°": "Filtered horizontal correlation (0°).",
"F_En_0°": "Filtered horizontal texture energy (0°).",

"F_Corr_45°": "Filtered diagonal correlation (45°).",
"F_En_45°": "Filtered diagonal texture energy (45°).",

"F_Corr_90°": "Filtered vertical correlation (90°).",
"F_En_90°": "Filtered vertical texture energy (90°).",

"F_Corr_135°": "Filtered diagonal correlation (135°).",
"F_En_135°": "Filtered diagonal texture energy (135°).",


# ---------- WAVELET FEATURES ----------

"W_RLHStd": "Standard deviation of LH wavelet band (horizontal edges).",
"W_RLHKurt": "Kurtosis of LH wavelet band.",
"W_GLHStd": "Standard deviation of HL wavelet band (vertical edges).",
"W_GLHSkew": "Skewness of HL wavelet band.",
"W_GLHKurt": "Kurtosis of HL wavelet band.",
"W_GHLSkew": "Skewness of HH wavelet band (diagonal edges).",
"W_GHLKurt": "Kurtosis of HH wavelet band.",
"W_BLHSkew": "Skewness of low-frequency LH components.",
"W_BLHKurt": "Kurtosis of low-frequency LH components.",
"W_BHLMean": "Mean value of HL wavelet band.",
"W_BHLSkew": "Skewness of HL wavelet components.",
"W_BHHSkew": "Skewness of HH wavelet components."

}


# ======================================================
# MAIN PAGE ROUTE
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

            image_path = os.path.join(UPLOAD_FOLDER, filename)

            file.save(image_path)

            img = cv2.imread(image_path)

            if img is None:
                error = "Invalid image file"
                return render_template("index.html", error=error)

            features = extract_features(image_path)

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]

            class_index = list(model.classes_).index(prediction)

            confidence = round(probabilities[class_index] * 100, 2)

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
        model_name="Support Vector Machine (SVM)"
    )


# ======================================================
# FEATURES PAGE ROUTE
# ======================================================

@app.route("/features")
def features():

    return render_template(
        "features.html",
        features=FEATURE_DESCRIPTIONS
    )


# ======================================================
# SERVE UPLOADED IMAGE
# ======================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(UPLOAD_FOLDER, filename)


# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":

    app.run(debug=True)