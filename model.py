import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path

# ─────────────────────────────────────────────
# 1. Загрузка данных
# ─────────────────────────────────────────────
print("Загружаем датасет...")
train_df = pd.read_csv("data/train_hierarchical.csv")
val_df   = pd.read_csv("data/val_hierarchical.csv")

print(f"Train: {len(train_df):,}")
print(f"Val:   {len(val_df):,}")
print(train_df['pair_type'].value_counts())

# ─────────────────────────────────────────────
# 2. Подготовка данных
# ─────────────────────────────────────────────
def build_examples(df):
    examples = []
    weights  = []
    for _, row in df.iterrows():
        examples.append(InputExample(
            texts=[str(row["anchor_abstract"]),
                   str(row["positive_abstract"])],
            label=float(row["relevance_score"]),
        ))
        weights.append(float(row["weight"]))
    return examples, weights

train_examples, train_weights = build_examples(train_df)
print(f"\nПримеров для обучения: {len(train_examples):,}")

# Валидационный evaluator
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_df["anchor_abstract"].tolist(),
    sentences2=val_df["positive_abstract"].tolist(),
    scores=val_df["relevance_score"].tolist(),
    name="val",
)

# ─────────────────────────────────────────────
# 3. Список моделей для дообучения
# ─────────────────────────────────────────────
models_to_finetune = [
    {
        "model_name":  "intfloat/multilingual-e5-base",
        "output_path": "models/e5-base-hierarchical",
        "batch_size":  32,
        "lr":          2e-5,
        "label":       "multilingual-e5-base файнтюн",
    },
    {
        "model_name":  "ai-forever/ruBert-large",
        "output_path": "models/rubert-large-hierarchical",
        "batch_size":  16,   # large модель — батч меньше
        "lr":          1e-5,
        "label":       "ruBert-large файнтюн",
    },
    {
        "model_name":  "ai-forever/ruSciBERT",
        "output_path": "models/ruscibert-hierarchical",
        "batch_size":  32,
        "lr":          2e-5,
        "label":       "ruSciBERT файнтюн",
    },
]

# ─────────────────────────────────────────────
# 4. Функция дообучения одной модели
# ─────────────────────────────────────────────
def finetune_model(config):
    print(f"\n{'='*65}")
    print(f"Дообучаем: {config['label']}")
    print(f"Базовая модель: {config['model_name']}")
    print(f"Сохранение: {config['output_path']}")
    print('='*65)

    # Проверяем GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")
        print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()

    # Загружаем модель
    print(f"\nЗагружаем {config['model_name']}...")
    model = SentenceTransformer(config["model_name"])

    # DataLoader с WeightedSampler
    sampler = WeightedRandomSampler(
        weights=torch.tensor(train_weights, dtype=torch.float),
        num_samples=len(train_weights),
        replacement=True,
    )
    train_dataloader = DataLoader(
        train_examples,
        batch_size=config["batch_size"],
        sampler=sampler,
    )

    # Loss
    train_loss = losses.CosineSimilarityLoss(model=model)

    # Warmup
    total_steps  = len(train_dataloader) * 3
    warmup_steps = int(total_steps * 0.1)
    print(f"Шагов всего: {total_steps:,}, warmup: {warmup_steps:,}")

    # Папки
    output_path     = Path(config["output_path"])
    checkpoint_path = Path(config["output_path"] + "-checkpoints")
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Обучение
    print("\nНачинаем обучение...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=3,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config["lr"]},
        output_path=str(output_path),
        evaluation_steps=1000,
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_path=str(checkpoint_path),
        checkpoint_save_steps=2000,
    )

    print(f"\n✅ Готово! Модель сохранена → {output_path}")

    # Освобождаем память перед следующей моделью
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(output_path)

# ─────────────────────────────────────────────
# 5. Запускаем дообучение всех моделей
# ─────────────────────────────────────────────
saved_models = {}

for config in models_to_finetune:
    try:
        saved_path = finetune_model(config)
        saved_models[config["label"]] = saved_path
    except Exception as e:
        print(f"\n❌ Ошибка при обучении {config['label']}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 6. Итог
# ─────────────────────────────────────────────
print("\n" + "="*65)
print("ИТОГ — сохранённые модели:")
print("="*65)
for label, path in saved_models.items():
    print(f"  {label:<35} → {path}")
print("="*65)