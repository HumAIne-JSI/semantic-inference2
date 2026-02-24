import json
import requests
import threading
import csv
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from bleurt import score
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Global Mutex for API access
api_mutex = threading.Lock()

# Load evaluation data from JSON file
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
    

# Query the API with streaming support and graphFormat="aas"
def query_api(query):
    try:
        with api_mutex:
            resp = requests.post(
                "http://127.0.0.1:5001/semantic-search",
                json={
                    "query": {"output": query},
                    "n": 1,
                    "graphFormat": "aas"
                },
                stream=True,
                headers={'Accept': 'text/event-stream'}
            )
            resp.raise_for_status()

            full = ""
            for chunk in resp.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                if chunk.startswith("data: "):
                    payload = chunk[6:]
                    try:
                        obj = json.loads(payload)
                        if "chunk" in obj:
                            full += obj["chunk"]
                        elif "error" in obj:
                            logging.error(f"API error: {obj['error']}")
                            return ""
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error in stream: {e}")
            time.sleep(4)  # same pacing as eval_with_streaming
            return full

    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        return ""

# Normalize text
def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

# Initialize scorers/models once
bleurt_scorer = score.BleurtScorer()
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_sbert_similarity(expected, generated):
    # embs = bert_model.encode([ref, hyp])
    # return np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
    embeddings = bert_model.encode([expected, generated])
    cosine_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    return cosine_sim

def evaluate_metrics(query, expected, generated, vectorizer, rouge_scorer_obj, bert_scorer, smoothing_function):
    results = {}
    try:
        # Normalize text
        normalized_expected = normalize_text(expected)
        normalized_generated = normalize_text(generated)

        # BLEU Score
        results["BLEU"] = sentence_bleu(
            [normalized_expected.split()], 
            normalized_generated.split(), 
            smoothing_function=smoothing_function
        )

        # ROUGE Scores
        rouge_scores = rouge_scorer_obj.score(normalized_expected, normalized_generated)
        results["ROUGE-1"] = rouge_scores["rouge1"].fmeasure
        results["ROUGE-2"] = rouge_scores["rouge2"].fmeasure
        results["ROUGE-L"] = rouge_scores["rougeL"].fmeasure

        # Cosine Similarity
        tfidf_matrix = vectorizer.fit_transform([normalized_expected, normalized_generated])
        results["Cosine Similarity"] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # BERT Scores
        # BERT Scores (positional args: candidates, references)
        bert_p, bert_r, bert_f1 = bert_scorer.score(
            [normalized_generated],
            [normalized_expected]
        )
        results["BERT Precision"] = bert_p.item()
        results["BERT Recall"] = bert_r.item()
        results["BERT F1"] = bert_f1.item()


        # BLEURT Score
        bleurt_score = bleurt_scorer.score(
            references=[normalized_expected],
            candidates=[normalized_generated]
        )[0]
        results["BLEURT Score"] = bleurt_score

        # Sentence-BERT similarity
        sbert_similarity = calculate_sbert_similarity(normalized_expected, normalized_generated)
        results["Sentence-BERT Cosine Similarity"] = sbert_similarity

        # Token-based metrics
        expected_tokens = normalized_expected.split()
        generated_tokens = normalized_generated.split()
        common_tokens = set(expected_tokens).intersection(set(generated_tokens))
        precision = len(common_tokens) / len(generated_tokens) if generated_tokens else 0
        recall = len(common_tokens) / len(expected_tokens) if expected_tokens else 0
        results["Precision"] = precision
        results["Recall"] = recall
        results["F1-Score"] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Exact match
        results["Exact Match"] = normalized_expected == normalized_generated
        
        
        #result.update({"Query": query, "Expected": expected, "Generated": generated})
        #all_results.append(result)
        # Log completion of evaluation for monitoring
        logging.info(f"Completed evaluation for query: {query[:50]}...")
        
        
    except Exception as e:
        logging.error(f"Error in evaluation for query: {query}. Error: {e}")
    return results


def evaluate(data, output_csv_path):
    vectorizer = TfidfVectorizer()
    rouge_obj = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)
    smooth_fn = SmoothingFunction().method1

    totals = {m: 0.0 for m in [
        "BLEU","ROUGE-1","ROUGE-2","ROUGE-L","Cosine Similarity",
        "BERT Precision","BERT Recall","BERT F1","BLEURT","Sentence-BERT Cosine Similarity",
        "Token Precision","Token Recall","Token F1","Exact Match"
    ]}

    all_results = []
    for entry in data:
        q = entry["query"]
        gold = entry["expected_results"][0]  # adjust if your JSON structure differs

        logging.info(f"Querying AAS-graph API for: {q[:40]}...")
        gen = query_api(q)
        if not gen:
            logging.warning(f"No output for: {q[:40]}...")
            continue

        metrics = evaluate_metrics(q, gold, gen, vectorizer, rouge_obj, bert_scorer, smooth_fn)
        metrics.update({"Query": q, "Expected": gold, "Generated": gen})
        all_results.append(metrics)
        logging.info(f"Results for query '{q[:50]}...':")
        print("Query: ", q)
        print("Generated: ", gen)
        print("Expected: ", gold)
        print("Metrics: ", metrics)

        for k in totals:
            totals[k] += metrics.get(k, 0.0)

    n = len(all_results)
    averages = {k: (v / n if n else 0.0) for k, v in totals.items()}
    averages["Accuracy"] = averages.pop("Exact Match")

    # Write per-query and averages to CSV
        # -----------------------
    # Write per-query and averages to CSV
    # -----------------------
    # 1) gather all unique fieldnames
    all_fieldnames = set()
    for row in all_results:
        all_fieldnames.update(row.keys())
    all_fieldnames.update(averages.keys())
    # (ensure Query/Expected/Generated are first, then the rest sorted)
    front = [f for f in ["Query", "Expected", "Generated"] if f in all_fieldnames]
    rest = sorted(all_fieldnames - set(front))
    fieldnames = front + rest

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
        # write the averages row (with "Query":"AVERAGE")
        avg_row = {"Query": "AVERAGE", **averages}
        writer.writerow(avg_row)

    logging.info("Evaluation complete.")
    return all_results, averages

if __name__ == "__main__":
    eval_file = "evaluation_aas.json"
    out_csv = "aas_query_results.csv"
    data = load_data(eval_file)
    results, avg = evaluate(data, out_csv)
    print("Averages:", avg)
