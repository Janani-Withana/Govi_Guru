from flask import Blueprint, current_app, request, jsonify
import logging
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from firebase_admin import firestore

from app.scripts import retrieve_answer, generate_refined_answer

chat_bp = Blueprint("chat", __name__)

MODEL_PATH = "Janani-Withana/sinhala-farming-qa-model"
FAISS_INDEX_PATH = "app/data/faiss_index.bin"
QA_CSV_PATH = "app/data/sinhala_farming_data.csv"

# Load Model, Data, FAISS
try:
    embedding_model = SentenceTransformer(MODEL_PATH)
    df = pd.read_csv(QA_CSV_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("Model, data, and FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading model or data: {e}")
    embedding_model = None


@chat_bp.route("/chat", methods=["POST"])
def chat():
    if embedding_model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        data = request.json
        user_query = data.get("message", "")
        user_id = data.get("user_id", "anonymous")

        if not user_query:
            return jsonify({'error': 'No question provided'}), 400

        best_question, best_answer = retrieve_answer(
            user_query=user_query,
            embedding_model=embedding_model,
            df=df,
            faiss_index_path=FAISS_INDEX_PATH,
            threshold=0.8
        )

        print(best_question,best_answer)

        if best_answer:
            final_answer = generate_refined_answer(user_query, best_answer)
        else:
            final_answer = "සුදුසු පිළිතුරක් නොමැත ⚠️"

        #Store chat history in Firestore
        chat_data = {
            "question": user_query,
            "answer": final_answer,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        current_app.db.collection("chat_history").document(user_id).collection("messages").add(chat_data)

        print(f"final answer:{final_answer}")

        return jsonify({"reply": final_answer})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error."}), 500
