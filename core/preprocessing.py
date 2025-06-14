import os
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def load_csv_files(input_dir: str) -> pd.DataFrame:
    """
    Загружает все CSV-файлы из директории и добавляет колонку source_file.
    """
    dataframes = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(input_dir, filename))
            df["source_file"] = filename
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует DataFrame: извлекает числовые и бинарные признаки, векторизует Info.
    """
    # Убедимся, что нужные колонки есть
    required_cols = ["No.", "Time", "Protocol", "Length", "Info", "source_file"]
    df = df[[col for col in required_cols if col in df.columns]].copy()

    # Преобразуем числовые колонки
    df["No."] = pd.to_numeric(df["No."], errors="coerce")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df.dropna(inplace=True)

    # Добавим delta_time
    df["delta_time"] = df["Time"].diff().fillna(0)

    # Бинарные признаки на основе Info
    df["Info"] = df["Info"].astype(str)
    df["has_syn"] = df["Info"].str.contains("SYN", case=False).astype(int)
    df["has_ack"] = df["Info"].str.contains("ACK", case=False).astype(int)
    df["has_http"] = df["Info"].str.contains("HTTP", case=False).astype(int)
    df["has_icmp"] = df["Info"].str.contains("ICMP", case=False).astype(int)

    # Кодируем Protocol
    proto_encoder = LabelEncoder()
    df["Protocol"] = proto_encoder.fit_transform(df["Protocol"])

    # Векторизация Info через TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_matrix = tfidf.fit_transform(df["Info"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])

    # Масштабируем числовые признаки
    numeric_cols = ["No.", "Time", "Length", "delta_time"]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Собираем итоговый DataFrame
    df = df.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)

    final_df = pd.concat(
        [df[["Protocol", "has_syn", "has_ack", "has_http", "has_icmp"] + numeric_cols], tfidf_df],
        axis=1
    )
    final_df["source_file"] = df["source_file"].values


    return final_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Сохраняет обработанный DataFrame в указанный CSV.
    """
    df.to_csv(output_path, index=False)
