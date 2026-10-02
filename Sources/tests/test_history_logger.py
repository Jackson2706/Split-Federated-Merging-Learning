import csv
import tempfile
import unittest
from pathlib import Path

from ehsfp.communication import add_communication, new_communication_tracker
from ehsfp.history_logger import (
    HISTORY_COLUMNS,
    RoundHistoryLogger,
    communication_snapshot,
)


class RoundHistoryLoggerTest(unittest.TestCase):
    def test_exact_schema_rows_values_and_round_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            logger = RoundHistoryLogger(path)
            tracker = new_communication_tracker()

            for round_number in range(1, 4):
                start = communication_snapshot(tracker)
                add_communication(
                    tracker, "client_to_edge_MB", mb=round_number + 0.25
                )
                add_communication(
                    tracker, "edge_to_cloud_MB", mb=round_number * 2
                )
                logger.append_round(
                    round_number=round_number,
                    elapsed_time_s=round_number * 1.5,
                    train_loss=1.0 / round_number,
                    validation_metric_name="accuracy",
                    validation_metric_value=0.1 * round_number,
                    communication_start=start,
                    communication_current=tracker,
                    resources={
                        "client_cpu_percent": 20 + round_number,
                        "client_rss_bytes": 1000 * round_number,
                        "client_gpu_peak_allocated_bytes": 0,
                        "client_gpu_peak_reserved_bytes": 0,
                    },
                    client_memory_records=round_number * 3,
                    edge_memory_records=round_number,
                    event="client_timeout" if round_number == 2 else "",
                )

            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, list(HISTORY_COLUMNS))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["round"], "2")
            self.assertEqual(rows[1]["train_loss"], "0.5")
            self.assertEqual(rows[1]["validation_metric_name"], "accuracy")
            self.assertEqual(rows[1]["validation_metric_value"], "0.2")
            self.assertEqual(rows[1]["comm_client_to_edge_delta_MB"], "2.25")
            self.assertEqual(rows[1]["comm_edge_to_cloud_delta_MB"], "4.0")
            self.assertEqual(rows[1]["comm_total_delta_MB"], "6.25")
            self.assertEqual(rows[1]["client_cpu_percent"], "22")
            self.assertEqual(rows[1]["client_rss_bytes"], "2000")
            self.assertEqual(rows[1]["edge_rss_bytes"], "")
            self.assertEqual(rows[1]["client_memory_records"], "6")
            self.assertEqual(rows[1]["event"], "client_timeout")

    def test_interruption_leaves_parseable_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            tracker = new_communication_tracker()
            logger = RoundHistoryLogger(path)

            for round_number in range(1, 3):  # interrupted before rounds 3--5
                start = communication_snapshot(tracker)
                add_communication(tracker, "client_to_edge_MB", mb=round_number)
                logger.append_round(
                    round_number=round_number,
                    elapsed_time_s=0.5,
                    train_loss=0.25,
                    validation_metric_name="macro_f1",
                    validation_metric_value=0.75,
                    communication_start=start,
                    communication_current=tracker,
                )
            del logger

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["round"] for row in rows], ["1", "2"])
            self.assertTrue(all(set(row) == set(HISTORY_COLUMNS) for row in rows))

            # A resumed writer recognizes the frozen header and remains appendable.
            RoundHistoryLogger(path)


if __name__ == "__main__":
    unittest.main()
