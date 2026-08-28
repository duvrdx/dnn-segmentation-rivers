import os
import numpy as np
from sklearn.model_selection import KFold


def create_kfold_splits(paths_dict, n_splits=5, shuffle=True, random_state=42):
    """
    Cria splits de K-Fold Cross-Validation a partir de um dicionário de paths.

    Args:
        paths_dict: dict com listas de caminhos. Ex:
            {'red': [...], 'nir': [...], 'blue': [...],
             'green': [...], 'masks': [...], 'hydro': [...]}
        n_splits (int): número de folds (default=5)
        shuffle (bool): embaralhar antes de dividir (default=True)
        random_state (int): semente aleatória (default=42)

    Returns:
        list of (dict_train, dict_val): cada elemento contém
            (train_paths_dict, val_paths_dict) para um fold.
    """
    first_key = list(paths_dict.keys())[0]
    n_samples = len(paths_dict[first_key])

    for key, paths in paths_dict.items():
        if len(paths) != n_samples:
            raise ValueError(
                f"Listas com tamanhos inconsistentes: '{first_key}'={n_samples}, "
                f"'{key}'={len(paths)}"
            )

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    splits = []
    for train_idx, val_idx in kf.split(range(n_samples)):
        train_paths = {}
        val_paths = {}
        for key in paths_dict:
            paths = paths_dict[key]
            train_paths[key] = [paths[i] for i in train_idx]
            val_paths[key] = [paths[i] for i in val_idx]
        splits.append((train_paths, val_paths))

    return splits


def compute_fold_statistics(fold_metrics_list):
    """
    Calcula média e desvio padrão para cada métrica através dos folds.

    Args:
        fold_metrics_list: lista de dicts, cada um com as métricas de um fold.
            Ex: [{'iou': 0.86, 'wiou': 0.75}, {'iou': 0.87, 'wiou': 0.76}]

    Returns:
        dict com 'mean_<metrica>' e 'std_<metrica>' para cada métrica encontrada.
    """
    if not fold_metrics_list:
        return {}

    keys = fold_metrics_list[0].keys()
    stats = {}
    for key in keys:
        values = [m[key] for m in fold_metrics_list if key in m]
        if values:
            stats[f"mean_{key}"] = float(np.mean(values))
            stats[f"std_{key}"] = float(np.std(values, ddof=1))
    return stats


def save_fold_results(
    results_dir,
    experiment_name,
    backbone,
    fold_metrics,
    aggregated,
):
    """
    Salva resultados de experimento K-Fold em arquivo de texto.

    Args:
        results_dir (str): diretório para salvar
        experiment_name (str): nome do experimento/configuração
        backbone (str): nome do backbone
        fold_metrics (list): lista de dicts com métricas por fold
        aggregated (dict): dict com médias e desvios
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, "kfold_summary.txt")

    lines = []
    lines.append(f"=== K-FOLD SUMMARY ===")
    lines.append(f"Experimento: {experiment_name}")
    lines.append(f"Backbone: {backbone}")
    lines.append(f"Número de folds: {len(fold_metrics)}")
    lines.append("")

    for fold_idx, metrics in enumerate(fold_metrics):
        metrics_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items()]
        )
        lines.append(f"  Fold {fold_idx}: {metrics_str}")

    lines.append("")
    lines.append("-" * 60)
    for key, value in aggregated.items():
        if key.startswith("mean_"):
            metric_name = key.replace("mean_", "")
            std_key = f"std_{metric_name}"
            std_val = aggregated.get(std_key, 0.0)
            lines.append(f"  {metric_name}: {value:.4f} ± {std_val:.4f}")
    lines.append("-" * 60)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Resultados K-Fold salvos em: {filepath}")


def save_holdout_results(
    results_dir,
    experiment_name,
    backbone,
    metrics,
):
    """
    Salva resultados de experimento hold-out (cross-dataset).
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, "holdout_summary.txt")

    metrics_str = " | ".join(
        [f"{k}: {v:.4f}" for k, v in metrics.items()]
    )

    content = (
        f"=== HOLDOUT SUMMARY ===\n"
        f"Experimento: {experiment_name}\n"
        f"Backbone: {backbone}\n"
        f"{metrics_str}\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Resultados Hold-Out salvos em: {filepath}")


def format_summary_line(
    experiment_name,
    backbone,
    metrics,
    is_kfold=True,
):
    """
    Formata uma linha de resumo para o arquivo final de resultados.

    Args:
        experiment_name: nome do experimento
        backbone: nome do backbone
        metrics: dict com métricas (pode conter mean/std para kfold, ou direto para holdout)
        is_kfold: se True, formata como média ± std; senão, valores diretos

    Returns:
        str formatada
    """
    if is_kfold:
        mean_iou = metrics.get("mean_iou", metrics.get("mean_best_iou", 0.0))
        std_iou = metrics.get("std_iou", metrics.get("std_best_iou", 0.0))
        mean_wiou = metrics.get("mean_wiou", metrics.get("mean_best_wiou", 0.0))
        std_wiou = metrics.get("std_wiou", metrics.get("std_best_wiou", 0.0))
        return (
            f"Experimento: {experiment_name:<30} | Backbone: {backbone:<22}\n"
            f"  -> K-Fold (K={len(metrics.get('n_folds', 5))}): "
            f"IoU = {mean_iou:.4f} ± {std_iou:.4f} | "
            f"WIoU = {mean_wiou:.4f} ± {std_wiou:.4f}\n"
        )
    else:
        iou = metrics.get("best_iou", 0.0)
        wiou = metrics.get("best_wiou", 0.0)
        return (
            f"Experimento: {experiment_name:<30} | Backbone: {backbone:<22}\n"
            f"  -> Hold-out: IoU = {iou:.4f} | WIoU = {wiou:.4f}\n"
        )
