import re
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
STOP_WORDS = set(stopwords.words('english'))

JOB_ROLES = {
    "Data Analyst": {
        "python": 3, "sql": 3, "data analysis": 3,
        "excel": 2, "tableau": 2, "power bi": 2,
        "statistics": 2, "communication": 1, "problem solving": 1
    },
    "ML Engineer": {
        "python": 3, "machine learning": 3, "deep learning": 3,
        "tensorflow": 2, "pytorch": 2, "scikit-learn": 2,
        "data preprocessing": 2, "sql": 1, "docker": 1
    },
    "Web Developer": {
        "html": 3, "css": 3, "javascript": 3, "react": 2,
        "nodejs": 2, "rest api": 2, "git": 2,
        "sql": 1, "responsive design": 1
    }
}

SUGGESTIONS = {
    "tableau":          "Add Tableau or Power BI — critical for data roles.",
    "power bi":         "Mention Power BI or any BI tool you've used.",
    "machine learning": "Include ML projects — even college-level ones count.",
    "deep learning":    "Add deep learning coursework or projects.",
    "tensorflow":       "List TensorFlow or Keras in your skills section.",
    "pytorch":          "Include PyTorch if used in any project.",
    "docker":           "Mention Docker or containerization experience.",
    "react":            "Add React.js to your skills or project tech stack.",
    "nodejs":           "Include Node.js or any backend framework.",
    "git":              "List Git/GitHub under version control tools.",
    "sql":              "Add SQL and any database you've worked with.",
    "statistics":       "Mention statistical methods or tools like NumPy/SciPy.",
    "problem solving":  "Use 'problem solving' explicitly in your resume summary.",
    "rest api":         "Mention REST API design or consumption in projects.",
    "responsive design":"Add responsive design or Bootstrap to your skills.",
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

def extract_spacy_entities(text):
    doc = nlp(text[:100000])
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "GPE", "SKILL", "WORK_OF_ART"):
            entities.add(ent.text.lower().strip())
    noun_chunks = set(chunk.text.lower().strip() for chunk in doc.noun_chunks)
    return entities.union(noun_chunks)

def compute_tfidf_keywords(text, top_n=10):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=300,
            sublinear_tf=True
        )
        matrix = vectorizer.fit_transform([cleaned])
        scores = zip(
            vectorizer.get_feature_names_out(),
            matrix.toarray()[0]
        )
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return [
            {"word": word, "score": round(float(score), 2)}
            for word, score in sorted_scores[:top_n]
            if score > 0
        ]
    except Exception:
        return []

def match_skills(resume_text, role):
    resume_lower = resume_text.lower()
    skills = JOB_ROLES.get(role, JOB_ROLES["Data Analyst"])

    found, missing = [], []
    total_weight = found_weight = 0

    for skill, weight in skills.items():
        total_weight += weight
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, resume_lower):
            found.append(skill)
            found_weight += weight
        else:
            missing.append(skill)

    weighted_score = round((found_weight / total_weight) * 100, 1) if total_weight else 0
    skill_pct   = round(len(found) / (len(found) + len(missing)) * 100, 1) if (found or missing) else 0
    missing_pct = round(100 - skill_pct, 1)

    return found, missing, weighted_score, skill_pct, missing_pct

def generate_suggestions(missing, tfidf_keywords):
    suggestions = []
    top_words = {kw["word"] for kw in tfidf_keywords[:5]}

    for skill in missing:
        tip = SUGGESTIONS.get(skill)
        if tip:
            priority = " (high priority)" if skill in top_words else ""
            suggestions.append(f"<strong>{skill.title()}{priority}:</strong> {tip}")

    suggestions.append(
        "Use exact skill names from the job description — ATS systems match keywords precisely."
    )
    return suggestions[:6]

def analyze_resume(resume_text, role="Data Analyst"):
    found, missing, nlp_score, found_pct, missing_pct = match_skills(resume_text, role)
    tfidf_keywords = compute_tfidf_keywords(resume_text, top_n=10)
    spacy_entities = list(extract_spacy_entities(resume_text))[:8]
    suggestions    = generate_suggestions(missing, tfidf_keywords)

    return {
        "role":           role,
        "nlp_score":      nlp_score,
        "found":          found,
        "missing":        missing,
        "found_pct":      found_pct,
        "missing_pct":    missing_pct,
        "total_skills":   len(found) + len(missing),
        "tfidf_keywords": tfidf_keywords,
        "spacy_entities": spacy_entities,
        "suggestions":    suggestions
    }
