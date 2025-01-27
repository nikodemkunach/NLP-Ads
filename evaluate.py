from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

def generate_description(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_bleu_score(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

model_name = "fine-tuned-ad"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = pd.read_csv("data/test_products.csv")

bleu_scores = []

for index, row in dataset.iterrows():
    prompt = f"{row['product']}, {row['description']}"
    
    generated_description = generate_description(model, tokenizer, prompt)
    
    reference_description = row["description"]
    
    bleu_score = calculate_bleu_score(reference_description, generated_description)
    
    bleu_scores.append(bleu_score)
    
    print(f"\nProduct: {row['product']}")
    print(f"Prompt: {prompt}")
    print(f"Wygenerowany opis: {generated_description}")
    print(f"Oczekiwany opis: {reference_description}")
    print(f"BLEU Score: {bleu_score}")
    print("-" * 50)

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"\nŚrednia wartość BLEU score: {average_bleu_score}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, marker='o', linestyle='-', color='b')
plt.axhline(y=average_bleu_score, color='r', linestyle='--', label=f'Średnia BLEU: {average_bleu_score:.4f}')
plt.xlabel('Numer przykładu')
plt.ylabel('BLEU Score')
plt.title('BLEU Score dla każdego przykładu')
plt.legend()
plt.grid(True)

output_file = "bleu_score_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nWykres został zapisany do pliku: {output_file}")

plt.show()