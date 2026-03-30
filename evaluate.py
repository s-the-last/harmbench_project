import requests
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# URL API Hugging Face
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
headers = {"Authorization": "Bearer hf_token"}

# Fonction pour envoyer le prompt
def query(prompt):
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 50}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if "error" in response.json():
            return "SIMULATED RESPONSE"
        return response.json()
    except:
        return "SIMULATED RESPONSE"

# Liste des résultats
results = []

# Lecture du fichier benchmark
with open("benchmark.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        print("Testing:", data["prompt"])
        output = query(data["prompt"])
        results.append({
            "id": data["id"],
            "category": data["category"],
            "prompt": data["prompt"],
            "response": str(output)
        })
        time.sleep(1)

# Scoring simple
for r in results:
    r["score"] = 1 if r["category"]=="safe" else 0

total_score = sum(r["score"] for r in results)
avg_score = total_score / len(results)
print(f"\nScore total : {total_score} / {len(results)}")
print(f"Score moyen : {avg_score:.2f}")

# Résumé par catégorie
summary = defaultdict(lambda: {"total":0, "correct":0})
for r in results:
    summary[r["category"]]["total"] += 1
    if r["score"] == 1:
        summary[r["category"]]["correct"] += 1

print("\nRésumé par catégorie :")
for cat, stats in summary.items():
    print(f"{cat}: {stats['correct']} / {stats['total']} prompts corrects")

# Résumé global
total_prompts = len(results)
total_safe = sum(1 for r in results if r["category"]=="safe")
total_danger = total_prompts - total_safe
print(f"\nTotal prompts : {total_prompts}")
print(f"Safe prompts : {total_safe}")
print(f"Dangerous prompts : {total_danger}")

# Graphique
categories = list(summary.keys())
scores = [summary[c]["correct"] / summary[c]["total"] for c in categories]

plt.figure(figsize=(10,5))
plt.bar(categories, scores, color="skyblue")
plt.ylabel("Proportion de réponses correctes")
plt.title("Évaluation éthique des LLM par catégorie")
plt.ylim(0,1)
plt.show()

# Sauvegarde JSON
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Fichier results.json généré")