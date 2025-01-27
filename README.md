# Generator opisów oraz haseł reklamowych produktów

## Opis projektu
Ten projekt dotyczy przetwarzania danych z pliku CSV zawierającego szczegółowe kategorie produktów. Celem jest stworzenie modelu językowego, który generuje opisy i hasła reklamowe dla produktów w różnych kategoriach. Proces obejmuje podział danych, trening modelu, generowanie wyników oraz ocenę jakości opisu na podstawie metryk takich jak BLEU.

## Struktura projektu

### Pliki

- **`amazon_products_with_detailed_categories.csv`** - Plik CSV zawierający dane produktów.
- **`split_data.py`** - Skrypt do podziału danych na zbiory treningowy i testowy.
- **`train.py`** - Skrypt do trenowania modelu językowego z wykorzystaniem frameworka `unsloth`.
- **`model_utils.py`** - Funkcje pomocnicze do ładowania modelu i konfiguracji fine-tuningu (Lora).
- **`inference.py`** - Skrypt do generowania opisów i reklam na podstawie podanego produktu i kategorii.
- **`evaluate.py`** - Skrypt do oceny jakości generowanych opisów na podstawie metryki BLEU.
- **`data_preprocessing.py`** - Skrypt do przygotowywania danych wejściowych w odpowiednim formacie do trenowania modelu.
- **`bleu_score_plot.png`** - Wykres przedstawiający BLEU score dla przykładów testowych.
- **`train_metrics.png`** - Wykres przedstawiający metryki treningowe.

## Instrukcja użytkowania

### 1. Przygotowanie danych
Podziel dane na zbiory treningowy, walidacyjny i testowy za pomocą `split_data.py`. Plik CSV z danymi powinien znajdować się w folderze `data`.

```bash
python split_data.py
```

Wyniki:
- `data/train_products.csv`
- `data/test_products.csv`

### 2. Trenowanie modelu
Uruchom `train.py`, aby przeprowadzić fine-tuning modelu bazowego na dostosowanym zbiorze danych.

```bash
python train.py
```
Wynikowy model zostanie zapisany w folderze `fine-tuned-ad`.

### 3. Generowanie opisów i reklam
Użyj `inference.py` do generowania opisów i reklam na podstawie podanej kategorii oraz produktu.

```bash
python inference.py
```
Wybierz kategorię z listy, a następnie wprowadź nazwę produktu. Model wygeneruje opis i hasło reklamowe.

### 4. Ocena wyników
Użyj `evaluate.py` do oceny jakości generowanych opisów. Skrypt oblicza BLEU score dla każdego przykładu testowego i generuje wykres wyników.

```bash
python evaluate.py
```

Wynikowy wykres zostanie zapisany jako `bleu_score_plot.png`.

---

## Wykres BLEU Score

![BLEU Score](bleu_score_plot.png)

### Analiza wyników
- Wykres przedstawia BLEU score dla każdego przykładu testowego.
- Średnia wartość BLEU : 0,3249, jej wynik wskazuje na **jakość dopasowania generowanych opisów do oczekiwanych**.
- Linia przerywana reprezentuje średnią wartość BLEU, co pozwala szybko zidentyfikować przykłady o niższej niż średnia jakości.

---

## Wykres metryk treningowych

![Train Metrics](train_metrics.png)

## Kluczowe funkcje

### Split danych (`split_data.py`)
- **Funkcja**: Podział danych wejściowych na zbiory `train` i `test`.
- **Wejście**: `amazon_products_with_detailed_categories.csv`
- **Wyjście**: Zapisane pliki w folderze `data`.

### Trenowanie modelu (`train.py`)
- **Model bazowy**: `unsloth/Llama-3.2-3B`
- **Metody fine-tuningu**: PEFT, gradient checkpointing.
- **Hyperparametry**:
  - Batch size: 2
  - Learning rate: 2e-4
  - Maksymalna długość sekwencji: 2048
  - lora_alpha:16
  - r:16

### Inference (`inference.py`)
- Generowanie opisu i reklamy dla dowolnego produktu na podstawie wprowadzonej kategorii.
- Wykorzystanie modelu `fine-tuned-ad` do inference.

### Ocena BLEU (`evaluate.py`)
- BLEU score obliczany na podstawie porównania generowanego i oczekiwanego opisu.
- Wyniki są wizualizowane na wykresie.

---

## Wymagania systemowe

- Python 3.11.11
- Framework `transformers` i `unsloth`
- Biblioteki: `torch`, `pandas`, `datasets`, `matplotlib`, `nltk`

---
