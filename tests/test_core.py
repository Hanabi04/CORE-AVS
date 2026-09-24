import unittest
import numpy as np
import torch
from models import SetAuditor
from models.curriculum_loss import curriculum_loss
from models.baselines import ClassAwareDeepSets
from metrics import auroc, evaluate, bootstrap_audio


class CoreTests(unittest.TestCase):
    def test_architecture_and_warmup(self):
        model = SetAuditor()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 18404)
        risk, logits = model(torch.zeros(3, 21, 13))
        loss = curriculum_loss(risk, logits, torch.ones(3), torch.arange(3), 0)
        loss.backward()
        self.assertIsNone(model.corruption.weight.grad)
        self.assertIsNotNone(model.risk.weight.grad)
        model.zero_grad(set_to_none=True)
        risk, logits = model(torch.zeros(3, 21, 13))
        loss = curriculum_loss(risk, logits, torch.ones(3), torch.arange(3), 0.1)
        loss.backward()
        self.assertIsNotNone(model.corruption.weight.grad)

    def test_auc_ties(self):
        self.assertEqual(auroc([0, 0], [0, 0]), 0.5)
        self.assertEqual(auroc([0, 1], [2, 3]), 1)
        self.assertEqual(auroc([2, 3], [0, 1]), 0)

    def test_deepsets_architecture(self):
        model = ClassAwareDeepSets()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 18412)
        risk, audio = model(torch.zeros(2, 21, 13))
        self.assertEqual(tuple(risk.shape), (2,))
        self.assertEqual(tuple(audio.shape), (2, 3))

    def test_selective_metrics(self):
        data = {"truth_risk": np.array([0, 1, 0.5, 0.25], np.float32)}
        result = evaluate(data, risk=np.array([0, 3, 2, 1]))
        expected = np.mean([0, 0.125, 0.25, 0.4375])
        self.assertAlmostEqual(result["AURC"], expected)
        self.assertAlmostEqual(result["J80"], 1 - 0.25)

    def test_bootstrap_pairing(self):
        probability = np.tile([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (4, 1)).astype(
            np.float32
        )
        self.assertEqual(
            bootstrap_audio(probability, 2027, 17),
            {"AUROC-S": [1.0, 1.0], "AUROC-M": [1.0, 1.0]},
        )


if __name__ == "__main__":
    unittest.main()
