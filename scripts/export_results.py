"""Export evaluation reports to CSV and a formatted Excel workbook."""

import argparse
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

METRICS = ("AURC", "J80", "AUROC-S", "AUROC-M", "MAE", "J50")
DATASETS = {
    "controlled": "Controlled 200",
    "core": "CORE-AVS (all)",
    "core_a": "CORE-AVS A",
    "core_b": "CORE-AVS B",
    "core_c": "CORE-AVS C",
    "avsegformer": "AVSegFormer",
}
METHODS = {
    "final_dual": "Set dual",
    "ablation_16ep": "Set dual (16 epochs)",
    "small_pool": "Small pool",
    "pooled_dual": "Pooled dual",
    "set_risk": "Set risk",
    "without_gap_difference": "w/o gap difference",
    "confidnet": "ConfidNet",
    "log_rms": "Log-RMS",
    "denseav": "DenseAV",
    "soft_dice": "Soft Dice",
    "deepsets": "DeepSets",
    "dual": "Set dual",
}
SCHEMA = [
    (
        "01_table1_core",
        "Table 1",
        ["Dataset", "Method", "AUROC-S", "AUROC-M", "Source record"],
    ),
    (
        "02_table2_controlled",
        "Table 2",
        ["Method", "AURC", "J80", "AUROC-S", "AUROC-M", "Source record"],
    ),
    (
        "03_run_summary",
        "Run summary",
        ["Experiment", "Dataset", "Method", *METRICS, "Source report", "Source record"],
    ),
    (
        "04_individual_runs",
        "Individual runs",
        [
            "Experiment",
            "Dataset",
            "Method",
            "Seed",
            *METRICS,
            "Source report",
            "Source record",
        ],
    ),
    (
        "05_extended_results",
        "Extended results",
        ["Dataset", "Method", "Seed", *METRICS, "Source record"],
    ),
    (
        "06_statistics",
        "Statistics",
        [
            "Experiment",
            "Dataset",
            "Method",
            "Metric",
            "Mean",
            "Sample SD",
            "ddof",
            "Source report",
            "Source record",
        ],
    ),
    (
        "07_confidence_intervals",
        "Confidence intervals",
        [
            "Dataset",
            "Method",
            "Seed",
            "Metric",
            "Estimate",
            "Lower 95%",
            "Upper 95%",
            "Source record",
        ],
    ),
    (
        "08_reference_comparison",
        "Reference comparison",
        ["Report", "Record", "Expected", "Observed", "Difference", "Status"],
    ),
]
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", NS)


def column(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def parse_record(key, experiment):
    parts = [p.strip() for p in key.split("/")]
    dataset = DATASETS.get(parts[0], parts[0])
    summary = parts[-1] == "mean_std"
    raw = (
        parts[1]
        if len(parts) > 2 or not summary
        else ("ablation_16ep" if "16" in experiment else "final_dual")
    )
    raw = re.sub(r"\s+\(.*\)$", "", raw)
    raw = re.sub(r"_seed\d+$", "", raw)
    method = METHODS.get(raw, raw)
    seed = int(parts[-1]) if parts[-1].isdigit() else None
    return dataset, method, seed, summary


def collect_tables(directory):
    tables = {name: [] for name, _, _ in SCHEMA}
    for filename, experiment in [
        ("main_tables.json", "Main tables"),
        ("main_three_runs.json", "25 epochs (10+15)"),
        ("compact_three_runs.json", "16 epochs (4+12)"),
        ("extended/metrics.json", "Architecture and transfer"),
    ]:
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(path)
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        for key, values in results.items():
            dataset, method, seed, summary = parse_record(key, experiment)
            if summary:
                tables["03_run_summary"].append(
                    [experiment, dataset, method]
                    + [
                        f"{values[m]['mean']:.3f} +/- {values[m]['sample_std']:.3f}"
                        if m in values
                        else None
                        for m in METRICS
                    ]
                    + [filename, key]
                )
                for metric in METRICS:
                    if metric in values:
                        tables["06_statistics"].append(
                            [
                                experiment,
                                dataset,
                                method,
                                metric,
                                values[metric]["mean"],
                                values[metric]["sample_std"],
                                1,
                                filename,
                                key,
                            ]
                        )
            elif filename == "main_tables.json":
                if dataset == "Controlled 200":
                    tables["02_table2_controlled"].append(
                        [method] + [values.get(m) for m in METRICS[:4]] + [key]
                    )
                else:
                    tables["01_table1_core"].append(
                        [
                            dataset,
                            method,
                            values.get("AUROC-S"),
                            values.get("AUROC-M"),
                            key,
                        ]
                    )
            elif filename.startswith("extended/"):
                tables["05_extended_results"].append(
                    [dataset, method, seed] + [values.get(m) for m in METRICS] + [key]
                )
            else:
                tables["04_individual_runs"].append(
                    [experiment, dataset, method, seed]
                    + [values.get(m) for m in METRICS]
                    + [filename, key]
                )
            for metric, bounds in values.get("95% image-bootstrap CI", {}).items():
                tables["07_confidence_intervals"].append(
                    [dataset, method, seed, metric, values[metric], *bounds, key]
                )
    order = [
        "Soft Dice",
        "ConfidNet",
        "Log-RMS",
        "DenseAV",
        "Pooled dual",
        "Set risk",
        "Set dual",
        "w/o gap difference",
    ]
    tables["02_table2_controlled"].sort(key=lambda r: order.index(r[0]))
    report = directory / "comparison.json"
    if report.exists():
        for r in json.loads(report.read_text(encoding="utf-8"))["results"]:
            tables["08_reference_comparison"].append(
                [
                    r[k]
                    for k in (
                        "report",
                        "record",
                        "expected",
                        "observed",
                        "difference",
                        "status",
                    )
                ]
            )
    return tables


def write_workbook(path, tables):
    """Fill the bundled workbook template using standard OpenXML cells."""
    template = Path(__file__).with_name("results_template.xlsx")
    with (
        zipfile.ZipFile(template) as source,
        zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for name in source.namelist():
            data = source.read(name)
            match = re.fullmatch(r"xl/worksheets/sheet(\d+)\.xml", name)
            if match:
                table_name, _, headers = SCHEMA[int(match[1]) - 1]
                rows = tables[table_name]
                sheet = ET.fromstring(data)
                cells = sheet.find(f"{{{NS}}}sheetData")
                sample = cells.find(f"{{{NS}}}row[@r='5']")
                styles = {
                    re.sub(r"\d", "", c.get("r")): c.get("s", "0") for c in sample
                }
                for row in list(cells):
                    if int(row.get("r")) >= 5:
                        cells.remove(row)
                for r, values in enumerate(rows, 5):
                    row = ET.SubElement(
                        cells, f"{{{NS}}}row", r=str(r), ht="24", customHeight="1"
                    )
                    for c, value in enumerate(values, 1):
                        if value is None:
                            continue
                        cell = ET.SubElement(
                            row,
                            f"{{{NS}}}c",
                            r=f"{column(c)}{r}",
                            s=styles.get(column(c), "0"),
                        )
                        if isinstance(value, (int, float)):
                            ET.SubElement(cell, f"{{{NS}}}v").text = repr(value)
                        else:
                            cell.set("t", "inlineStr")
                            ET.SubElement(
                                ET.SubElement(cell, f"{{{NS}}}is"), f"{{{NS}}}t"
                            ).text = str(value)
                end = f"{column(len(headers))}{max(4, 4 + len(rows))}"
                dimension = sheet.find(f"{{{NS}}}dimension")
                if dimension is not None:
                    dimension.set("ref", f"A1:{end}")
                auto = sheet.find(f"{{{NS}}}autoFilter")
                if auto is None:
                    auto = ET.Element(f"{{{NS}}}autoFilter")
                    sheet.insert(list(sheet).index(cells) + 1, auto)
                auto.set("ref", f"A4:{end}")
                data = ET.tostring(sheet, encoding="utf-8", xml_declaration=True)
            target.writestr(name, data)


def export_results(directory):
    directory = Path(directory).resolve()
    tables = collect_tables(directory)
    output = directory / "tables"
    output.mkdir(exist_ok=True)
    for name, _, headers in SCHEMA:
        with (output / f"{name}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(tables[name])
    write_workbook(output / "reproduction_results.xlsx", tables)
    (output / "README.txt").write_text(
        "Open reproduction_results.xlsx in Excel or LibreOffice. Table 1 and Table 2 follow the manuscript.\n"
        "CSV files contain the same records and retain numeric precision.\n"
        "AURC and MAE: lower is better. J80, J50 and AUROC: higher is better.\n"
        "Blank cells mean the metric was not evaluated for that method, not zero.\n"
        "Run summaries use sample SD (ddof=1); numeric means and SDs are in Statistics.\n"
        "Confidence intervals use 10,000 paired-image bootstraps. This sheet is empty when intervals were skipped.\n"
        "Reference comparison reports equality at three decimals; expected and observed values remain separate.\n"
        "Each table includes source keys for the JSON reports in the parent directory.\n",
        encoding="utf-8",
    )
    print(f"Excel workbook: {output / 'reproduction_results.xlsx'}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export_results(parser.parse_args().directory)
