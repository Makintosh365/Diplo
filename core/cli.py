import typer
import pandas as pd
from pathlib import Path
from core.preprocessing import load_csv_files, extract_features, save_processed_data
from core.labeling import add_labels
from core.model import BotnetClassifier
from core.evaluator import evaluate_predictions

app = typer.Typer()


@app.command()
def train(data_dir: str = "data/raw", save_path: str = "data/processed/train.csv"):
    """
    Обучает модель на основе CSV файлов в data/raw/
    """
    typer.echo("📥 Загрузка данных...")
    df = load_csv_files(data_dir)
    df = extract_features(df)
    df = add_labels(df)
    save_processed_data(df, save_path)

    typer.echo("🧠 Обучение модели...")
    classifier = BotnetClassifier()
    classifier.train(df)
    classifier.save_model()
    typer.echo("✅ Модель обучена и сохранена!")


@app.command()
def predict(file: str):
    """
    Делает предсказание для нового CSV файла.
    """
    typer.echo(f"📥 Анализ файла: {file}")
    df = load_csv_files(Path(file).parent.as_posix())
    df = df[df["source_file"] == Path(file).name]
    df = extract_features(df)

    classifier = BotnetClassifier()
    classifier.load_model(input_size=df.shape[1] - 1)  # минус source_file
    predictions = classifier.predict(df)

    result_df = df.copy()
    result_df["predicted_label"] = predictions
    typer.echo(result_df[["source_file", "predicted_label"]].value_counts())
    typer.echo("✅ Анализ завершен!")


@app.command()
def evaluate(file: str = "data/processed/train.csv"):
    """
    Выводит метрики модели на обучающем датасете.
    """
    typer.echo("📊 Оценка модели...")
    df = pd.read_csv(file)
    df = add_labels(df)

    classifier = BotnetClassifier()
    classifier.load_model(input_size=df.shape[1] - 2)  # без source_file и label

    y_true = df["label"].tolist()
    y_pred = classifier.predict(df)

    evaluate_predictions(y_true, y_pred)
    typer.echo("✅ Оценка завершена.")


if __name__ == "__main__":
    app()
