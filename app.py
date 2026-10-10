import os
import re
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

# ---------- Load artifacts ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# ---------- Minimal preprocessing (match training pipeline as closely as possible) ----------
# Your notebook uses NLTK stopwords + lemmatization. To avoid runtime downloads in Flask,
# we implement the same steps except lemmatization/stopword exactness.
# If you want exact matching, we can also package NLTK artifacts.
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _ = nltk.data.find("corpora/stopwords")
    _ = nltk.data.find("corpora/wordnet")

    _STOP_WORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()

    def clean_text(text: str) -> str:
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        tokens = [_LEMMATIZER.lemmatize(w) for w in tokens if w not in _STOP_WORDS and len(w) > 2]
        return " ".join(tokens)

except Exception:

    # Fallback: lowercasing + strip HTML/URLs/punct, no stopword removal/lemmatization.
    def clean_text(text: str) -> str:
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = text.split()
        tokens = [w for w in tokens if len(w) > 2]
        return " ".join(tokens)


def predict_sentiment(text: str):
    cleaned = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])

    pred_int = model.predict(tfidf_vec)[0]
    # probability for the predicted class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(tfidf_vec)[0][pred_int]
    else:
        proba = float("nan")

    label = label_encoder.inverse_transform([pred_int])[0]
    confidence = None if np.isnan(proba) else round(float(proba) * 100.0, 2)
    return label, confidence


# ---------- Flask app ----------
app = Flask(__name__, template_folder="templates")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()

    if not review:
        return jsonify({"error": "Review text is required"}), 400

    label, confidence = predict_sentiment(review)
    return jsonify({"label": str(label), "confidence": confidence})


if __name__ == "__main__":
    # debug=True for development only
    app.run(host="127.0.0.1", port=5000, debug=True)

