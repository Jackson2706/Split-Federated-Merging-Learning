import os
import tempfile
import unittest

from collect_results import parse_log_metrics


class ParseLogMetricsTest(unittest.TestCase):
    def parse(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
            log.write(text)
            path = log.name
        try:
            return parse_log_metrics(path)
        finally:
            os.unlink(path)

    def test_classification_and_communication_report(self):
        metrics = self.parse("""epoch 1: loss=1.234 test_f1=99.9
=== Communication Report ===
client_to_edge_data_MB: 27326.25 MB
edge_to_cloud_data_MB: 57.80 MB
client_model_upload_MB: 144.45 MB
client_model_download_MB: 144.46 MB
edge_model_upload_MB: 1370.64 MB
edge_model_download_MB: 1370.65 MB
total_comm_MB: 30414.23 MB
|---- Best Validation Acc: 8.10%
|---- Test Acc: 8.20%
Total Run Time: 24040.2032s
""")
        self.assertEqual(metrics["best_val_top1"], 8.10)
        self.assertEqual(metrics["test_top1"], 8.20)
        self.assertEqual(metrics["runtime_s"], 24040.2032)
        self.assertEqual(metrics["total_comm_MB"], 30414.23)
        self.assertEqual(metrics["client_to_edge_data_MB"], 27326.25)
        self.assertEqual(metrics["edge_to_cloud_data_MB"], 57.80)
        self.assertEqual(metrics["client_model_upload_MB"], 144.45)
        self.assertEqual(metrics["client_model_download_MB"], 144.46)
        self.assertEqual(metrics["edge_model_upload_MB"], 1370.64)
        self.assertEqual(metrics["edge_model_download_MB"], 1370.65)
        self.assertEqual(metrics["status"], "complete")
        self.assertNotIn("test_f1", metrics)

    def test_segmentation_iou_dice_and_communication_report(self):
        metrics = self.parse("""=== Communication Report ===
client_to_edge_data_MB: 10.00 MB
edge_to_cloud_data_MB: 2.00 MB
client_model_upload_MB: 3.00 MB
client_model_download_MB: 4.00 MB
edge_model_upload_MB: 5.00 MB
edge_model_download_MB: 6.00 MB
total_comm_MB: 30.00 MB
|---- Best Validation IoU: 40.89%
|---- Test IoU: 40.89%  Dice: 55.59%
Total Run Time: 123.45s
""")
        self.assertEqual(metrics["best_iou"], 40.89)
        self.assertEqual(metrics["test_iou"], 40.89)
        self.assertEqual(metrics["test_dice"], 55.59)
        self.assertEqual(metrics["total_comm_MB"], 30.0)
        self.assertEqual(metrics["status"], "complete")

        nearby_dice = self.parse("""Best Validation IoU: 0.4089
Best-checkpoint Test IoU: 0.4089
Test Dice: 0.5559
Total Run Time: 12.00s
""")
        self.assertEqual(nearby_dice["test_iou"], 0.4089)
        self.assertEqual(nearby_dice["test_dice"], 0.5559)

    def test_crashed_truncated_log(self):
        metrics = self.parse("""Epoch 4 loss: 0.123
Traceback (most recent call last):
  File \"runner.py\", line 10, in main
RuntimeError: synthetic failure
""")
        self.assertEqual(metrics, {"status": "no_final_metric"})


if __name__ == "__main__":
    unittest.main()
