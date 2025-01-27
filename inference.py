import torch
import pandas as pd
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from model_utils import load_model_and_tokenizer
from data_preprocessing import load_and_preprocess_data

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[1].strip()
    description = response.split("Reklama:")[0].replace("Opis:", "").strip()
    ad = response.split("Reklama:")[1].strip()
    return description, ad

def main():
    model, tokenizer = load_model_and_tokenizer("fine-tuned-ad")
    model = FastLanguageModel.for_inference(model)

    file_path='data/amazon_products_with_detailed_categories.csv'
    df = pd.read_csv(file_path)
    categories = df["category"].unique().tolist()

    print("Dostępne kategorie:")
    for i, category in enumerate(categories, start=1):
        print(f"{i}. {category}")

    try:
        category_number = int(input("Wybierz numer kategorii: "))
        if 1 <= category_number <= len(categories):
            selected_category = categories[category_number - 1]
        else:
            print("Nieprawidłowy numer kategorii.")
            return
    except ValueError:
        print("Nieprawidłowy numer kategorii.")
        return

    product = input("Podaj nazwę produktu: ")

    prompt = f"### Instruction:\nPodaj opis i reklamę dla produktu {product} z kategorii {selected_category}\n\n### Input:\nKategoria: {selected_category}\nProdukt: {product}\n\n### Response:"

    description, ad = generate_response(model, tokenizer, prompt)

    print("\nNazwa produktu:", product)
    print("Opis:", description)
    print("Hasło reklamowe:", ad)

if __name__ == "__main__":
    main()