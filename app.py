from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the summarizer model (downloads on first run)
print("Loading summarizer model...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model loaded!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")

    if not text or len(text.strip()) < 50:
        return jsonify({"error": "Please enter at least 50 characters."}), 400

    result = summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)
    summary = result[0]["summary_text"]

    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)