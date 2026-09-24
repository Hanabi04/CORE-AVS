import hashlib
import json
import unittest
import torch
from datasets.cached import ROOT, load_cohort
from models.assignment_expert import RelativeSourceRegionMatcher
from runtime import predict
from metrics import evaluate


class AssetTests(unittest.TestCase):
    def test_manifest_alignment(self):
        for cohort, file, count in [
            ("train", "train_1316.json", 1316),
            ("controlled", "controlled_test_200.json", 200),
            ("core", "core_avs_manifest.json", 1073),
        ]:
            data = load_cohort(cohort)
            entries = json.loads((ROOT / "data/manifests" / file).read_text())[
                "entries"
            ]
            self.assertEqual(len(entries), count)
            for i, e in enumerate(entries):
                self.assertEqual(e["uid"], data["uids"][i])
                self.assertEqual(
                    e["feature_sha256"],
                    hashlib.sha256(
                        data["features"][3 * i : 3 * i + 3].tobytes()
                    ).hexdigest(),
                )
                self.assertTrue(e["fingerprints"])
                self.assertTrue(all(0 <= k < 21 for k in e["class_indices"]))

    def test_train_source_groups(self):
        def groups(filename):
            doc = json.loads((ROOT / "data/manifests" / filename).read_text())
            return {str(g) for e in doc["entries"] for g in e["source_group_ids"]}

        train = groups("train_1316.json")
        self.assertFalse(train & groups("controlled_test_200.json"))
        self.assertFalse(train & groups("core_avs_manifest.json"))

    def test_frozen_main_metrics(self):
        torch.set_num_threads(4)
        for cohort, expected_m in [
            ("controlled", 0.837225),
            ("core", 0.7819241936926804),
        ]:
            data = load_cohort(cohort)
            prediction, _ = predict(ROOT / "checkpoints/final_dual_seed2027.pth", data)
            result = evaluate(data, **prediction)
            self.assertLess(abs(result["AUROC-M"] - expected_m), 1e-4)
            if cohort == "controlled":
                self.assertAlmostEqual(result["AURC"], 0.622800924674619, places=5)
                self.assertAlmostEqual(result["J80"], 0.2720150053501129, places=5)

    def test_assignment_checkpoint(self):
        model = RelativeSourceRegionMatcher()
        ck = torch.load(
            ROOT / "checkpoints/assignment_expert.pth",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(ck["model"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 63490)
        result = model(
            torch.zeros(3, 256),
            torch.zeros(256),
            torch.tensor([1, 3]),
            torch.zeros(2, 8),
        )
        self.assertEqual(result.shape, (3, 21, 3))


if __name__ == "__main__":
    unittest.main()
