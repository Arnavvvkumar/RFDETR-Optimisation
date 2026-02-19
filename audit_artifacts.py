from pathlib import Path

import pandas as pd


MODEL_FILES = {
    "original": "models/rf-detr-base.pth",
    "pruned_1": "models/rf_detr_pruned_1.pth",
    "pruned_30": "models/rf_detr_pruned_30.pth",
    "pruned_40": "models/rf_detr_pruned_40.pth",
    "quantized": "models/rf_detr_quantized.onnx",
}


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def audit_model_sizes() -> list[dict]:
    rows = []
    for model_name, relative_path in MODEL_FILES.items():
        path = Path(relative_path)
        if path.exists():
            rows.append(
                {
                    "model": model_name,
                    "path": str(path),
                    "exists": True,
                    "size_mb": round(size_mb(path), 3),
                }
            )
        else:
            rows.append(
                {
                    "model": model_name,
                    "path": str(path),
                    "exists": False,
                    "size_mb": None,
                }
            )
    return rows


def audit_accuracy_loss() -> pd.DataFrame | None:
    benchmark_path = Path("results/benchmark_results.csv")
    if not benchmark_path.exists():
        return None

    df = pd.read_csv(benchmark_path)
    grouped = (
        df.groupby("model_type", as_index=False)[["precision", "recall", "f1", "ap@0.5"]]
        .mean()
        .rename(columns={"ap@0.5": "map50"})
    )

    base = grouped[grouped["model_type"] == "original"]
    if base.empty:
        grouped["delta_f1_vs_original"] = None
        grouped["delta_map50_vs_original"] = None
        return grouped

    base_f1 = float(base["f1"].iloc[0])
    base_map50 = float(base["map50"].iloc[0])

    grouped["delta_f1_vs_original"] = grouped["f1"] - base_f1
    grouped["delta_map50_vs_original"] = grouped["map50"] - base_map50
    return grouped


def main() -> None:
    print("=== Model Artifact Audit ===")
    model_rows = audit_model_sizes()
    available = [row for row in model_rows if row["exists"]]

    for row in model_rows:
        if row["exists"]:
            print(f"[OK] {row['model']:10s} -> {row['path']} ({row['size_mb']:.3f} MB)")
        else:
            print(f"[MISSING] {row['model']:10s} -> {row['path']}")

    quantized = next((row for row in available if row["model"] == "quantized"), None)
    if quantized:
        target = 5.0
        diff = quantized["size_mb"] - target
        print(f"\nQuantized size check: {quantized['size_mb']:.3f} MB (target ~{target:.1f} MB, diff {diff:+.3f} MB)")
    else:
        print("\nQuantized size check: cannot verify (quantized model is missing).")

    print("\n=== Accuracy/Loss Audit ===")
    acc_df = audit_accuracy_loss()
    if acc_df is None:
        print("Cannot verify accuracy loss: results/benchmark_results.csv is missing.")
        return

    print(acc_df.to_string(index=False))


if __name__ == "__main__":
    main()
