import os
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    models,
    util
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# 1) Fine-Tuning Function
# -----------------------------
def fine_tune_embeddings(
    csv_path: str,
    base_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    output_dir: str = "fine_tuned_multilingual_model",
    epochs: int = 1,
    batch_size: int = 32,
    warmup_steps: int = 100,
):
    print(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    print(f"Reading Q&A CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    train_examples = []
    for _, row in df.iterrows():
        question = str(row["Question"])
        answer = str(row["Answer"])
        
        # Positive example
        train_examples.append(InputExample(texts=[question, answer], label=1.0))


    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    model.save(output_dir)
    print(f"Fine-tuning complete. Model saved to: {output_dir}")

# -----------------------------
# 2) Build FAISS Index
# -----------------------------
def build_faiss_index(csv_path: str, model_path: str, faiss_index_path: str = "faiss_index.bin"):
    print(f"Loading fine-tuned model from: {model_path}")
    embedding_model = SentenceTransformer(model_path)

    df = pd.read_csv(csv_path)
    questions = df["Question"].astype(str).tolist()

    print("Computing embeddings for all questions...")
    question_embeddings = embedding_model.encode(questions, normalize_embeddings=True)

    dimension = question_embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatIP(dimension)  # Cosine similarity (use with normalized vectors)
    index.add(np.array(question_embeddings, dtype=np.float32))

    # Write index to disk
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index built and saved to: {faiss_index_path}")

    return df

# -----------------------------
# 3) Retrieve Best Match
# -----------------------------
def retrieve_answer(
    user_query: str,
    embedding_model: SentenceTransformer,
    df: pd.DataFrame,
    faiss_index_path: str = "faiss_index.bin",
    threshold: float = 0.8
):
    """
    Retrieve the most relevant answer using FAISS similarity search.
    Returns (best_question, best_answer) or (None, None) if no good match.
    """
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)

    # Encode query
    query_embedding = embedding_model.encode([user_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Search top K=1
    distances, best_match_idx = index.search(query_embedding, k=1)
    distance_score = distances[0][0]

    print(f"Distance score: {distance_score:.4f}")

    # If the best match score is below threshold, treat it as no relevant answer
    if distance_score >= threshold:
        best_idx = best_match_idx[0][0]
        best_question = df.iloc[best_idx]["Question"]
        best_answer = df.iloc[best_idx]["Answer"]
        return best_question, best_answer
    else:
        return None, None

# -----------------------------
# 4) Refine with FlanT5-base
# -----------------------------
def generate_refined_answer(
    user_query: str,
    retrieved_answer: str,
    model_name: str = "google/flan-t5-base",
    max_length: int = 512
) -> str:
    if not retrieved_answer:
        return "No suitable context found to generate an answer."
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        prompt = (
            f"Answer the following question based on the given context.\n\n"
            f"Context: {retrieved_answer}\n"
            f"Question: {user_query}\n"
            f"Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_length=max_length, num_beams=5, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
        final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if not final_answer:
            return retrieved_answer

        return final_answer
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return retrieved_answer if retrieved_answer else "No suitable answer found."


# -----------------------------
# 5) Optional: Run once to fine-tune & build index
# -----------------------------
if __name__ == "__main__":
    QA_CSV_PATH = "data/sinhala_farming_data.csv"
    BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    FINETUNED_MODEL_DIR = "models/fine_tuned_multilingual_model"
    FAISS_INDEX_PATH = "data/faiss_index.bin"

    # A) Fine-tune
    fine_tune_embeddings(
        csv_path=QA_CSV_PATH,
        base_model_name=BASE_MODEL,
        output_dir=FINETUNED_MODEL_DIR,
        epochs=1,
        batch_size=32,
        warmup_steps=100
    )

    # B) Build index
    df = build_faiss_index(
        csv_path=QA_CSV_PATH,
        model_path=FINETUNED_MODEL_DIR,
        faiss_index_path=FAISS_INDEX_PATH
    )

    print("Done.")
