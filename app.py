from flask import Flask, request, jsonify, render_template
import pickle
import re
import string

app = Flask(__name__)


# Nettoyage du texte
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

# Charger modèle et vectoriseur
model = pickle.load(open("model_svm.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({"error": "Veuillez fournir un champ 'message'"}), 400

    message = data['message']
    cleaned = clean_text(message)
    vector = tfidf.transform([cleaned])

    # Prédiction
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    result = {
        "message": message,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": {
            "spam": round(proba[1] * 100, 2),
            "ham": round(proba[0] * 100, 2)
        }
    }

    return jsonify(result)

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
