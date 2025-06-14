import pandas as pd

def assign_label(row: pd.Series) -> str:
    """
    Присваивает метку классу на основе имени исходного файла.

    :param row: Одна строка DataFrame.
    :return: Строка с меткой.
    """
    filename = row["source_file"].lower()

    if "tcp" in filename:
        return "botnet_tcp"
    elif "icmp" in filename:
        return "botnet_icmp"
    elif "slowloic" in filename:
        return "botnet_slowloic"
    elif "http" in filename:
        return "botnet_http"
    elif "normal" in filename or "benign" in filename:
        return "normal"
    else:
        return "unknown"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонку 'label' в DataFrame, определяя тип трафика по имени файла.

    :param df: DataFrame с колонкой 'source_file'.
    :return: DataFrame с новой колонкой 'label'.
    """
    df["label"] = df.apply(assign_label, axis=1)
    return df
