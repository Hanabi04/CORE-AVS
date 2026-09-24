import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.check = module("check_results")
        self.export = module("export_results")
        for target, source in self.check.REPORTS.items():
            path = self.output / target
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "reported_results" / source, path)

    def test_missing_or_changed_result_fails(self):
        result = self.check.compare_results(self.output)
        self.assertEqual(result["passed"], result["total"])
        path = self.output / "main_tables.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        first = next(iter(report["results"]))
        report["results"][first]["AURC"] += 0.02
        del report["results"][first]["J80"]
        path.write_text(json.dumps(report), encoding="utf-8")
        result = self.check.compare_results(self.output)
        self.assertEqual(result["total"] - result["passed"], 2)
        self.assertEqual(
            {r["status"] for r in result["results"] if r["status"] != "PASS"},
            {"MISSING", "DIFFERENT"},
        )

    def test_method_names_and_missing_metrics(self):
        tables = self.export.collect_tables(self.output)
        rows = {r[0]: r for r in tables["02_table2_controlled"]}
        self.assertEqual(len(rows), 8)
        self.assertIsNone(rows["Soft Dice"][3])
        self.assertIsNone(rows["Log-RMS"][1])
        self.assertIn("w/o gap difference", rows)

    def test_workbook_keeps_numeric_values_and_blanks(self):
        self.check.compare_results(self.output)
        folder = self.export.export_results(self.output)
        self.assertEqual(len(list(folder.glob("*.csv"))), 8)
        with zipfile.ZipFile(folder / "reproduction_results.xlsx") as archive:
            ns = {"s": self.export.NS}
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet2.xml"))
            cells = {c.get("r"): c for c in sheet.findall(".//s:c", ns)}
            self.assertNotEqual(cells["B5"].get("t"), "inlineStr")
            value = float(cells["B5"].find("s:v", ns).text)
            self.assertAlmostEqual(
                value,
                self.export.collect_tables(self.output)["02_table2_controlled"][0][1],
                places=14,
            )
            self.assertNotIn("D5", cells)
            self.assertEqual(sheet.find("s:autoFilter", ns).get("ref"), "A4:F12")


if __name__ == "__main__":
    unittest.main()
