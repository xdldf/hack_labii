#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import ast

# допустимые метки
VALID_LABELS = {
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT",
}

def parse_annotation_safe(s):
    try:
        return ast.literal_eval(s) if pd.notna(s) and s.strip() else []
    except Exception:
        return []

def clean_dataset(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv, sep=";", quotechar='"')
    problems = []

    clean_rows = []
    for idx, row in df.iterrows():
        text = str(row["sample"])
        anns = parse_annotation_safe(str(row["annotation"]))

        clean_anns = []
        for ann in anns:
            if not isinstance(ann, (tuple, list)) or len(ann) != 3:
                problems.append((idx, f"Некорректный формат аннотации: {ann}"))
                continue

            start, end, label = ann

            # проверки
            if not isinstance(start, int) or not isinstance(end, int):
                problems.append((idx, f"Неверные индексы: {ann}"))
                continue
            if start < 0 or end > len(text) or start >= end:
                problems.append((idx, f"Спан выходит за пределы текста: {ann}, text='{text}'"))
                continue
            if label not in VALID_LABELS:
                problems.append((idx, f"Некорректная метка: {label}"))
                continue

            clean_anns.append((start, end, label))

        clean_rows.append({"sample": text, "annotation": str(clean_anns)})

    clean_df = pd.DataFrame(clean_rows)
    clean_df.to_csv(output_csv, sep=";", index=False)

    print(f"✅ Чистый датасет сохранён: {output_csv}")
    print(f"Всего строк: {len(df)}, проблемных: {len(problems)}")
    if problems:
        print("\n⚠️ Примеры проблем:")
        for p in problems[:10]:
            print(f"Строка {p[0]} → {p[1]}")

if __name__ == "__main__":
    clean_dataset("datasets/train.csv", "datasets/train_clean.csv")
