from __future__ import annotations

import csv
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(r"d:\gaoyuan\Desktop\beijing2")
LOG_DIR = PROJECT_ROOT / "张森老师——方法对比" / "fig2_compare_2" / "Log"
OUT_DIR = PROJECT_ROOT / "98787_article" / "20260422_article" / "补充文件"
CSV_OUT = OUT_DIR / "Supplementary Table 4.csv"
JSON_OUT = OUT_DIR / "Supplementary Table 4.json"


def read_text(name: str) -> str:
    return (LOG_DIR / name).read_text(encoding="utf-8", errors="replace")


def seconds_to_human(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds:.2f} s ({seconds / 60:.2f} min)"
    if seconds < 86400:
        return f"{seconds:.2f} s ({seconds / 3600:.2f} h)"
    return f"{seconds:.2f} s ({seconds / 86400:.2f} d)"


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.I | re.M)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def first_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text, flags=re.I | re.M)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def build_rows() -> list[dict[str, object]]:
    si = read_text("SI_log.txt")
    fasttree = read_text("FastTree_5000random.txt")
    iqtree = read_text("IQ-TREE_5000random.txt")
    raxml = read_text("RAxMl_5000random.txt")

    si_wall = first_float(r"TOTAL_TIME_SECONDS\s*=\s*([0-9.]+)", si) or first_float(
        r"Total execution time:\s*([0-9.]+)\s*seconds", si
    ) or first_float(r"Total Time\):\s*([0-9.]+)\s*秒", si)
    si_step1 = first_float(r"Step 1 \(Feature extraction\):\s*([0-9.]+)\s*s", si)
    si_step2 = first_float(r"Step 2 \(Vector computation\):\s*([0-9.]+)\s*s", si)
    si_step3 = first_float(r"Step 3 \(Visualization\):\s*([0-9.]+)\s*s", si)
    si_step1 = si_step1 or first_float(r"Step 1 .*?:\s*([0-9.]+)\s*秒", si)
    si_step2 = si_step2 or first_float(r"Step 2 .*?:\s*([0-9.]+)\s*秒", si)
    si_step3 = si_step3 or first_float(r"Step 3 .*?:\s*([0-9.]+)\s*秒", si)
    si_matrix = re.search(r"Feature matrix shape:\s*\((\d+),\s*(\d+)\)", si)
    si_weights = re.search(
        r"(?:Optimal weights|PCA Weights calculated):\s*Cos=([0-9.]+),\s*Dot=([0-9.]+),\s*Euc=([0-9.]+)",
        si,
    )

    fast_wall = first_float(r"Total time:\s*([0-9.]+)\s*seconds", fasttree)
    fast_unique = re.search(r"Unique:\s*(\d+)/(\d+)", fasttree)
    fast_bad = re.search(r"Bad splits:\s*(\d+)/(\d+)", fasttree)

    iq_wall = first_float(r"Total wall-clock time used:\s*([0-9.]+)\s*sec", iqtree)
    iq_cpu = first_float(r"Total CPU time used:\s*([0-9.]+)\s*sec", iqtree)
    iq_threads = first_int(r"running [A-Z]+ instruction set on\s+(\d+)\s+cores", iqtree) or first_int(
        r"-T\s+(\d+)", iqtree
    )
    iq_alignment = re.search(
        r"Alignment has\s+(\d+)\s+sequences with\s+(\d+)\s+columns,\s+(\d+)\s+distinct patterns",
        iqtree,
    )
    iq_seq = int(iq_alignment.group(1)) if iq_alignment else None
    iq_cols = int(iq_alignment.group(2)) if iq_alignment else None
    iq_patterns = int(iq_alignment.group(3)) if iq_alignment else None

    rax_wall = first_float(r"Overall execution time(?: for full ML analysis)?:\s*([0-9.]+)\s*secs", raxml)
    rax_threads = first_int(r"-T\s+(\d+)", raxml)
    rax_patterns = first_int(r"Alignment has\s+(\d+)\s+distinct alignment patterns", raxml)
    rax_gaps = first_float(r"Proportion of gaps.*?:\s*([0-9.]+)%", raxml)
    rax_bootstrap = first_int(r"Executing\s+(\d+)\s+rapid bootstrap", raxml) or first_int(r"-#\s*(\d+)", raxml)

    if si_wall is None or fast_wall is None or iq_wall is None or rax_wall is None:
        raise RuntimeError("At least one required wall-time value could not be parsed from the logs.")

    rows = [
        {
            "Method": "SI multi-modal fusion workflow",
            "Primary output": "Similarity Index (SI) and vector-based pandemic-potential signal",
            "Input size": f"{si_matrix.group(1)} strains x {si_matrix.group(2)} features" if si_matrix else "5001 strains",
            "Feature/model settings": (
                f"Feature extraction, vector computation, visualization; "
                f"weights Cos={si_weights.group(1)}, Dot={si_weights.group(2)}, Euc={si_weights.group(3)}"
                if si_weights
                else "Feature extraction, vector computation, visualization"
            ),
            "Key parameters": "Step times: feature extraction "
            f"{si_step1:.2f}s; vector computation {si_step2:.2f}s; visualization {si_step3:.2f}s",
            "Threads": "Not reported",
            "Bootstrap/support": "Not applicable",
            "Wall-clock time (s)": round(si_wall, 3),
            "Wall-clock time": seconds_to_human(si_wall),
            "CPU time": "Not reported",
            "Relative wall time vs SI": "1.00x",
            "Notes": "End-to-end SI workflow time directly reported in SI_log.txt.",
            "Source log": "SI_log.txt",
        },
        {
            "Method": "FastTree",
            "Primary output": "Approximate maximum-likelihood phylogeny",
            "Input size": f"{fast_unique.group(2)} input taxa; {fast_unique.group(1)} unique taxa" if fast_unique else "5000 taxa",
            "Feature/model settings": "GTR-CAT approximation with 20 rate categories; NNI, SPR and ML-NNI optimization",
            "Key parameters": "FastTree 2.1.11; SH-like local support values; "
            + (f"bad splits {fast_bad.group(1)}/{fast_bad.group(2)}" if fast_bad else "bad splits not parsed"),
            "Threads": "Not reported",
            "Bootstrap/support": "SH-like support, 1000 resamples",
            "Wall-clock time (s)": round(fast_wall, 3),
            "Wall-clock time": seconds_to_human(fast_wall),
            "CPU time": "Not reported",
            "Relative wall time vs SI": f"{fast_wall / si_wall:.2f}x",
            "Notes": "Runtime is similar to SI but produces only a tree, without the SI/SDI fusion framework.",
            "Source log": "FastTree_5000random.txt",
        },
        {
            "Method": "IQ-TREE",
            "Primary output": "Maximum-likelihood phylogeny with ultrafast bootstrap support",
            "Input size": f"{iq_seq} sequences x {iq_cols} alignment columns; {iq_patterns} site patterns",
            "Feature/model settings": "GTR model; IQ-TREE 3.0.1; Intel vectorized binary",
            "Key parameters": "Command includes -T 24 -m GTR -B 1000",
            "Threads": str(iq_threads),
            "Bootstrap/support": "1000 ultrafast bootstrap replicates",
            "Wall-clock time (s)": round(iq_wall, 3),
            "Wall-clock time": seconds_to_human(iq_wall),
            "CPU time": seconds_to_human(iq_cpu) if iq_cpu is not None else "Not reported",
            "Relative wall time vs SI": f"{iq_wall / si_wall:.2f}x",
            "Notes": "Wall time and CPU time are directly reported by IQ-TREE.",
            "Source log": "IQ-TREE_5000random.txt",
        },
        {
            "Method": "RAxML",
            "Primary output": "Maximum-likelihood phylogeny with rapid bootstrap analysis",
            "Input size": f"5000 sequences; {rax_patterns} alignment patterns; {rax_gaps:.2f}% gaps",
            "Feature/model settings": "GTRGAMMA model; RAxML 8.2.12 PThreads",
            "Key parameters": "raxmlHPC-PTHREADS -T 24 -x 12345 -p 12345 -#100 -m GTRGAMMA -f a",
            "Threads": str(rax_threads),
            "Bootstrap/support": f"{rax_bootstrap} rapid bootstrap replicates plus best-scoring ML tree",
            "Wall-clock time (s)": round(rax_wall, 3),
            "Wall-clock time": seconds_to_human(rax_wall),
            "CPU time": "Not separately reported",
            "Relative wall time vs SI": f"{rax_wall / si_wall:.2f}x",
            "Notes": "Overall execution time is parsed from the final RAxML report.",
            "Source log": "RAxMl_5000random.txt",
        },
    ]
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    columns = [
        "Method",
        "Primary output",
        "Input size",
        "Feature/model settings",
        "Key parameters",
        "Threads",
        "Bootstrap/support",
        "Wall-clock time (s)",
        "Wall-clock time",
        "CPU time",
        "Relative wall time vs SI",
        "Notes",
        "Source log",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    JSON_OUT.write_text(json.dumps({"columns": columns, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {JSON_OUT}")
    for row in rows:
        print(row["Method"], row["Wall-clock time"], row["Relative wall time vs SI"])


if __name__ == "__main__":
    main()
