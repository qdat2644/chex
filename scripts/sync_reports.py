from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").rstrip() + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalize_text(content), encoding="utf-8")


def check_file(path: Path, expected: str) -> bool:
    if not path.exists():
        print(f"ERROR: Missing generated file: {path}", file=sys.stderr)
        return False

    current = normalize_text(path.read_text(encoding="utf-8"))
    expected_norm = normalize_text(expected)

    if current != expected_norm:
        print(f"ERROR: {path} is not synchronized.", file=sys.stderr)
        return False

    return True


def generate_tbd_latex() -> str:
    return r"""
\begin{table}[t]
\centering
\caption{Protocol-compliant results are not yet available.}
\label{tab:protocol-results}
\begin{tabular}{lccc}
\toprule
Model & Macro AUROC & Macro AUPRC & Macro F1 \\
\midrule
DenseNet-121 & TBD & TBD & TBD \\
ConvNeXt-Small & TBD & TBD & TBD \\
\bottomrule
\end{tabular}
\end{table}
""".strip() + "\n"


def generate_tbd_readme() -> str:
    return r"""
# CheXpert AI Workstation

> **Protocol status:** No protocol-compliant final benchmark artifact is available yet.

## Pre-specified benchmark table

| Finding | DenseNet-121 AUROC | ConvNeXt-Small AUROC | $\Delta$ AUROC | Paired DeLong $p_{\text{adj}}$ |
|---|---:|---:|---:|---:|
| Atelectasis | TBD | TBD | TBD | TBD |
| Cardiomegaly | TBD | TBD | TBD | TBD |
| Consolidation | TBD | TBD | TBD | TBD |
| Edema | TBD | TBD | TBD | TBD |
| Pleural Effusion | TBD | TBD | TBD | TBD |
| **Macro** | **TBD** | **TBD** | **TBD** | — |

Results must not be entered manually. README and LaTeX tables are generated from the final protocol artifact.
""".strip() + "\n"


def generate_live_latex_table(data: dict) -> str:
    m_conv = data.get("architectures", {}).get("convnext_small", {}).get("metrics_per_label", {})
    m_dense = data.get("architectures", {}).get("densenet121", {}).get("metrics_per_label", {})
    comp = data.get("model_comparison", {}).get("per_label_comparison", {})

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Locked-Test Empirical Frontal Chest Radiograph Diagnostic Performance. Point estimates denote AUROC with 95\% patient-level cluster bootstrap confidence intervals (2,000 resamples). $p$-values evaluated via paired DeLong test with Holm-Bonferroni correction.}",
        r"\label{tab:chexpert_benchmark}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Pathology Finding} & \textbf{DenseNet-121} & \textbf{ConvNeXt-Small} & \textbf{$\Delta$ AUROC (95\% CI)} & \textbf{DeLong $p_{\text{adj}}$} \\",
        r"\midrule",
    ]

    for label in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]:
        d_ci = m_dense.get(label, {}).get("ci_95", "TBD")
        c_ci = m_conv.get(label, {}).get("ci_95", "TBD")
        delta_ci = comp.get(label, {}).get("delta_auc_95_ci", "TBD")
        p_adj = comp.get(label, {}).get("delong_p_holm_adj")
        p_str = f"{p_adj:.3f}" if p_adj is not None else "TBD"

        lines.append(f"{label} & {d_ci} & {c_ci} & {delta_ci} & {p_str} \\\\")

    mean_d = data.get("architectures", {}).get("densenet121", {}).get("macro_auroc_95_ci", "TBD")
    mean_c = data.get("architectures", {}).get("convnext_small", {}).get("macro_auroc_95_ci", "TBD")
    delta_m = data.get("model_comparison", {}).get("macro_comparison", {}).get("macro_delta_95_ci", "TBD")

    lines.extend([
        r"\midrule",
        f"\\textbf{{Mean AUROC (Macro)}} & {mean_d} & {mean_c} & {delta_m} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Synchronize README and LaTeX tables strictly from final benchmark artifact.")
    parser.add_argument("--artifact", type=Path, help="Path to final benchmark_artifact.json")
    parser.add_argument("--check", action="store_true", help="CI check mode: fails with exit code 1 if files are out of sync")
    args = parser.parse_args()

    readme_path = PROJECT_ROOT / "README.md"
    tex_eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    tex_results_path = tex_eval_dir / "table_results.tex"
    tex_per_label_path = tex_eval_dir / "table_per_label.tex"
    tex_comp_path = tex_eval_dir / "model_comparison.tex"

    # Also keep protocol_v0_1 output in sync if requested
    tex_final_dir = PROJECT_ROOT / "outputs" / "final" / "protocol_v0_1"
    tex_final_results = tex_final_dir / "table_results.tex"

    artifact_file = args.artifact or PROJECT_ROOT / "outputs" / "final" / "protocol_v0_1" / "benchmark_artifact.json"

    if artifact_file.exists():
        try:
            data = json.loads(artifact_file.read_text(encoding="utf-8"))
            if data.get("artifact_type") == "official_scientific_benchmark" and data.get("architectures"):
                tex_content = generate_live_latex_table(data)
                readme_content = generate_tbd_readme()
            else:
                tex_content = generate_tbd_latex()
                readme_content = generate_tbd_readme()
        except Exception:
            tex_content = generate_tbd_latex()
            readme_content = generate_tbd_readme()
    else:
        tex_content = generate_tbd_latex()
        readme_content = generate_tbd_readme()

    targets = [
        (readme_path, readme_content),
        (tex_results_path, tex_content),
        (tex_per_label_path, tex_content),
        (tex_comp_path, tex_content),
    ]

    if artifact_file.exists():
        targets.append((tex_final_results, tex_content))

    if args.check:
        all_ok = True
        for path, content in targets:
            if not check_file(path, content):
                all_ok = False
        if not all_ok:
            sys.exit(1)
        print("All reports and LaTeX tables are perfectly synchronized with artifact.")
        sys.exit(0)

    for path, content in targets:
        write_text(path, content)
        print(f"Synchronized -> {path}")


if __name__ == "__main__":
    main()
