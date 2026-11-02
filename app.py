# Save this code inside app.py
import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page configurations
st.set_page_config(page_title="Subjective Answer Evaluator", layout="centered", page_icon="📝")

# Ensure local NLTK assets are downloaded securely
@st.cache_resource
def load_nltk_resources():
    for res in ['punkt', 'punkt_tab', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

load_nltk_resources()

# Cache the heavy SentenceTransformer model separately to avoid unpickling version mismatches
@st.cache_resource
def load_sentence_transformer(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# --- ROBUST EVALUATOR CLASS COMPATIBLE WITH STREAMLIT ---
class SubjectiveAnswerEvaluator:
    def __init__(self, encoder):
        self.encoder = encoder
        self.spell = SpellChecker()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()

    def preprocess_text(self, text):
        text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = word_tokenize(text_clean)
        return [w for w in tokens if w not in self.stop_words]

    def compute_semantic_similarity(self, ref_text, student_text):
        emb1 = self.encoder.encode(ref_text, convert_to_tensor=True)
        emb2 = self.encoder.encode(student_text, convert_to_tensor=True)
        similarity = util.cos_sim(emb1, emb2).item()
        similarity_score = round(max(0, similarity) * 5.0, 2)
        return similarity_score, round(similarity * 100, 2)

    def compute_keyword_accuracy(self, ref_text, student_text):
        ref_tokens = set(self.preprocess_text(ref_text))
        student_tokens = set(self.preprocess_text(student_text))
        if not ref_tokens:
            return 0.0, 0.0
        overlap = ref_tokens.intersection(student_tokens)
        keyword_ratio = len(overlap) / len(ref_tokens)
        keyword_score = round(min(keyword_ratio, 1.0) * 3.0, 2)
        return keyword_score, round(keyword_ratio * 100, 2)

    def compute_grammar_and_spelling(self, student_text):
        tokens = word_tokenize(re.sub(r'[^a-zA-Z\s]', '', student_text.lower()))
        if not tokens:
            return 0.0, 0.0
        misspelled = self.spell.unknown(tokens)
        correct_count = len(tokens) - len(misspelled)
        accuracy_ratio = correct_count / len(tokens)
        spell_score = round(accuracy_ratio * 2.0, 2)
        return spell_score, round(accuracy_ratio * 100, 2)

    def evaluate_answer(self, ref_text, student_text):
        sem_score, sem_pct = self.compute_semantic_similarity(ref_text, student_text)
        kw_score, kw_pct = self.compute_keyword_accuracy(ref_text, student_text)
        sp_score, sp_pct = self.compute_grammar_and_spelling(student_text)
        total_score = round(sem_score + kw_score + sp_score, 2)
        return {
            "Total Score (Out of 10)": total_score,
            "Semantic Similarity Score (Max 5.0)": sem_score,
            "Keyword Match Score (Max 3.0)": kw_score,
            "Spelling & Grammar Score (Max 2.0)": sp_score,
            "Metrics Breakdown": {
                "Semantic Agreement": f"{sem_pct}%",
                "Keyword Coverage": f"{kw_pct}%",
                "Spelling Accuracy": f"{sp_pct}%"
            }
        }

# Initialize model and evaluator
try:
    encoder_model = load_sentence_transformer('all-MiniLM-L6-v2')
    evaluator = SubjectiveAnswerEvaluator(encoder=encoder_model)
except Exception as e:
    st.error(f"Error initializing ML models: {e}")
    evaluator = None

# App title and description
st.title("📝 Subjective Answer Evaluator Dashboard")
st.markdown("Evaluate student answers against syllabus reference guidelines dynamically using NLP and semantic embeddings.")

if evaluator is None:
    st.error("Could not load the evaluator pipeline safely. Please check dependencies.")
else:
    # Define Subjects & Reference Data
    subject_files = {
        "ADA (Algorithms)": "ADA.txt",
        "MC (Microcontrollers)": "MC.txt",
        "CN (Computer Networks)": "CN.txt"
    }

    selected_subject = st.selectbox("Select Subject Module:", list(subject_files.keys()))

    # Extended 10-Question sets for each subject from the provided syllabus/exam PDFs
    questions = {
        "ADA (Algorithms)": [
            "Q1: What is Dynamic Programming?",
            "Q2: Explain the Divide and Conquer strategy.",
            "Q3: Explain asymptotic notations Big Oh, Big Omega and Big Theta.",
            "Q4: Explain the general plan for analyzing the efficiency of a recursive algorithm.",
            "Q5: What is the difference between Greedy and Dynamic Programming approaches?",
            "Q6: Explain the Breadth First Search (BFS) algorithm with an example.",
            "Q7: Discuss Kruskal's algorithm for Minimum Spanning Tree (MST).",
            "Q8: Describe the Floyd-Warshall algorithm for all-pairs shortest paths.",
            "Q9: Explain the concept of NP-Completeness.",
            "Q10: Discuss the backtracking approach with the N-Queens problem."
        ],
        "MC (Microcontrollers)": [
            "Q1: Explain the architecture of 8051 Microcontroller.",
            "Q2: Describe the pin diagram and pin configurations of the 8051.",
            "Q3: Explain the addressing modes of 8051 with examples.",
            "Q4: Discuss the differences between a Microprocessor and a Microcontroller.",
            "Q5: Explain the interrupt structure of the 8051 microcontroller.",
            "Q6: What are the TMOD and TCON registers in 8051?",
            "Q7: Describe how serial communication is done in 8051.",
            "Q8: Explain the memory organization (RAM and ROM layout) of 8051.",
            "Q9: Discuss the interfacing of a 7-segment display with 8051.",
            "Q10: What is the purpose of the Program Status Word (PSW) register?"
        ],
        "CN (Computer Networks)": [
            "Q1: What is the OSI Model and explain its layers?",
            "Q2: Differentiate between IPv4 and IPv6 protocols.",
            "Q3: Explain the working mechanism of Transmission Control Protocol (TCP).",
            "Q4: What is the difference between TCP and UDP?",
            "Q5: Describe the CSMA/CD and CSMA/CA protocols.",
            "Q6: Explain how Domain Name System (DNS) works.",
            "Q7: Discuss the functions and responsibilities of the Data Link Layer.",
            "Q8: What is routing? Explain Distance Vector Routing algorithm.",
            "Q9: Describe the RSA public-key cryptography algorithm.",
            "Q10: Explain the working of HTTP and HTTPS protocols."
        ]
    }

    selected_question = st.selectbox("Choose Question:", questions[selected_subject])

    student_ans = st.text_area("Enter Student's Answer:", height=150,
                               placeholder="Type or paste the student's answer here...")

    if st.button("Evaluate Answer", type="primary"):
        if not student_ans.strip():
            st.warning("Please enter a student answer to begin evaluation.")
        else:
            # Helper function to load reference answers
            ref_file = os.path.join("data", subject_files[selected_subject])
            ref_content = ""
            if os.path.exists(ref_file):
                with open(ref_file, "r", encoding="utf-8") as f:
                    ref_content = f.read()
            else:
                # Standard default references if files aren't created locally
                defaults = {
                    "ADA (Algorithms)": "Dynamic programming is an algorithmic technique used to solve optimization problems by breaking them down into smaller subproblems.",
                    "MC (Microcontrollers)": "The 8051 microcontroller is an 8-bit architecture consisting of an 8-bit CPU, 4KB on-chip ROM, 128 bytes on-chip RAM.",
                    "CN (Computer Networks)": "The Open Systems Interconnection (OSI) model is a conceptual framework that standardizes network communications into seven distinct layers."
                }
                ref_content = defaults[selected_subject]

            # Clean reference to pull target line if unstructured
            ref_search = re.search(r"Reference:\s*(.*)", ref_content)
            reference_text = ref_search.group(1) if ref_search else ref_content

            # Execute evaluation model
            results = evaluator.evaluate_answer(reference_text, student_ans)

            # Display Evaluation Report
            st.success("Evaluation Completed successfully!")

            # Display overall score metrics with clean components
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Score", f"{results['Total Score (Out of 10)']} / 10.0")
            col2.metric("Semantic Similarity", f"{results['Semantic Similarity Score (Max 5.0)']} / 5.0")
            col3.metric("Spelling/Grammar Check", f"{results['Spelling & Grammar Score (Max 2.0)']} / 2.0")

            st.markdown("### Detailed breakdown Analysis")
            st.json(results["Metrics Breakdown"])
