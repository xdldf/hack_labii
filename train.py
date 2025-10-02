import argparse
import ast
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import evaluate
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_annotation(annotation_str: str) -> List[Tuple[int, int, str]]:
    """Парсит строку аннотации вида "[(0, 4, 'B-TYPE'), (5, 10, 'I-TYPE')]".
    Убирает метки 'O'.
    Возвращает список кортежей (start_char, end_char_exclusive, label).
    """
    if annotation_str is None or annotation_str == "" or annotation_str == "nan":
        return []
    try:
        items = ast.literal_eval(annotation_str)
        cleaned: List[Tuple[int, int, str]] = []
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            start, end, label = item
            if label == "O":
                continue
            cleaned.append((int(start), int(end), str(label)))
        return cleaned
    except Exception:
        return []


def collect_label_list(df: pd.DataFrame) -> List[str]:
    """Собирает все уникальные метки из датасета"""
    labels = set(["O"])  # всегда включаем O
    for ann in df["annotation"].astype(str).tolist():
        try:
            items = ast.literal_eval(ann)
            for _, _, lab in items:
                labels.add(lab)
        except Exception:
            continue
    # Стабильный порядок, O первым
    labels = sorted([l for l in labels if l != "O"])  # исключаем O
    return ["O"] + labels


@dataclass
class NERSample:
    text: str
    entities: List[Tuple[int, int, str]]


def build_samples(df: pd.DataFrame) -> List[NERSample]:
    """Создает список образцов из датафрейма"""
    samples: List[NERSample] = []
    for _, row in df.iterrows():
        text = str(row["sample"]) if not pd.isna(row["sample"]) else ""
        entities = parse_annotation(str(row["annotation"]))
        samples.append(NERSample(text=text, entities=entities))
    return samples


def align_labels_with_tokens(
    text: str,
    entities: List[Tuple[int, int, str]],
    tokenizer: AutoTokenizer,
    label_to_id: Dict[str, int],
    max_length: int,
) -> Dict[str, Any]:
    """Выравнивает аннотации с токенами и исправляет BIO последовательности"""
    # Токенизация с оффсетами
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    offsets = tokenized["offset_mapping"]
    labels = []

    # Определяем метку для каждого токена
    def token_label_for_span(tok_start: int, tok_end: int) -> str:
        best_label = "O"
        best_overlap = 0
        best_is_begin = False
        for (ent_start, ent_end, ent_label) in entities:
            # длина пересечения
            overlap = max(0, min(tok_end, ent_end) - max(tok_start, ent_start))
            if overlap <= 0:
                continue
            is_begin = tok_start == ent_start
            # Предпочитаем большее пересечение, при равенстве - начало
            if overlap > best_overlap or (overlap == best_overlap and is_begin and best_label == ent_label):
                if ent_label.startswith("B-") or ent_label.startswith("I-"):
                    base_label = ent_label.split("-", 1)[1]
                else:
                    base_label = ent_label
                best_is_begin = is_begin
                best_overlap = overlap
                best_label = ("B-" if is_begin else "I-") + base_label
        return best_label

    # Исправляем I- последовательности
    prev_label = None
    for (start, end) in offsets:
        if end == 0 and start == 0:
            # Специальный токен
            labels.append(-100)
            continue
            
        lab = token_label_for_span(start, end)
        
        # Исправляем I- теги
        if lab.startswith("I-"):
            base = lab.split("-", 1)[1]
            if not prev_label or not prev_label.endswith(base):
                lab = "B-" + base
                
        labels.append(label_to_id.get(lab, label_to_id["O"]))
        prev_label = lab

    tokenized["labels"] = labels
    tokenized.pop("offset_mapping", None)
    return tokenized


def compute_metrics_builder(id_to_label: Dict[int, str]):
    """Создает функцию для вычисления метрик"""
    metric = evaluate.load("seqeval")

    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results.get("overall_precision", precision_score(true_labels, true_predictions)),
            "recall": results.get("overall_recall", recall_score(true_labels, true_predictions)),
            "f1": results.get("overall_f1", f1_score(true_labels, true_predictions)),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a token classification model (NER) on span-annotated CSV")
    parser.add_argument("--train_path", type=str, default="./datasets/train_clean.csv")
    parser.add_argument("--model_name", type=str, default="DeepPavlov/rubert-base-cased")# xlm-roberta-base
    parser.add_argument("--output_dir", type=str, default="./models/deeppavlov") #xlmr-ner
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Загружаем данные
    df = pd.read_csv(args.train_path, sep=";")
    if "sample" not in df.columns or "annotation" not in df.columns:
        raise ValueError("Ожидаются колонки 'sample' и 'annotation' в CSV с разделителем ';'")

    # Создаем метки
    label_list = collect_label_list(df)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for l, i in label_to_id.items()}

    # Создаем образцы
    samples = build_samples(df)
    train_samples, val_samples = train_test_split(samples, test_size=0.05, random_state=args.seed)

    # Токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # Простая обертка для датасета
    class TorchDataset:
        def __init__(self, items: List[NERSample]):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            s = self.items[idx]
            return align_labels_with_tokens(
                text=s.text,
                entities=s.entities,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_length=args.max_length,
            )

    train_dataset = TorchDataset(train_samples)
    eval_dataset = TorchDataset(val_samples)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],  # отключаем wandb
        logging_steps=50,
        seed=args.seed,
        fp16=True if os.environ.get("USE_FP16", "1") == "1" else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_builder(id_to_label),
    )

    trainer.train()

    # Сохраняем артефакты
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Сохраняем список меток
    with open(os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for l in label_list:
            f.write(l + "\n")


if __name__ == "__main__":
    main()
