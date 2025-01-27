import pandas as pd
from datasets import Dataset

def load_and_preprocess_data(file_path='data/train_products.csv'):
    df = pd.read_csv(file_path)

    dataset = Dataset.from_pandas(df)

    def formatting_prompts_func(examples):
        instructions = []
        inputs = []
        descriptions = []
        ads = []

        for category, product, description, ad in zip(examples['category'], examples['product'], examples['description'], examples['ad']):
            instruction = f"Podaj opis i reklamę dla produktu {product} z kategorii {category}"
            input = f"Kategoria: {category}\nProdukt: {product}"
            descriptions.append(description)
            ads.append(ad)
            instructions.append(instruction)
            inputs.append(input)

        texts = []
        for instruction, input, description, ad in zip(instructions, inputs, descriptions, ads):
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\nOpis: {description}\nReklama: {ad}"
            texts.append(text)

        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    return formatted_dataset