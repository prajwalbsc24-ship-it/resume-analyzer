from flask import Flask, render_template, request, jsonify
import PyPDF2
from analyzer import analyze_resume, JOB_ROLES

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

def extract_text(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
        return text.strip()
    except Exception:
        return ""

@app.route('/')
def home():
    roles = list(JOB_ROLES.keys())
    return render_template('index.html', roles=roles)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    role = request.form.get('role', 'Data Analyst')

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    text = extract_text(file)
    if not text:
        return jsonify({"error": "Could not read text from PDF"}), 400

    result = analyze_resume(text, role)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)