import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from ehsfp.deadlock_diagnostics import start_deadlock_diagnostics


class DeadlockDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self._previous_anomaly_state = torch.is_anomaly_enabled()
        torch.autograd.set_detect_anomaly(False)

    def tearDown(self):
        torch.autograd.set_detect_anomaly(self._previous_anomaly_state)

    def _start(self, anomaly_enabled):
        env = {
            "EHSFP_DEADLOCK_DIAG": "1",
            "EHSFP_DEADLOCK_DIAG_SECONDS": "17",
        }
        if anomaly_enabled:
            env["EHSFP_DEADLOCK_DIAG_ANOMALY"] = "1"

        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        run_dir = Path(temporary_directory.name)

        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "ehsfp.deadlock_diagnostics.faulthandler.dump_traceback_later"
            ) as dump_traceback_later,
            mock.patch(
                "ehsfp.deadlock_diagnostics.faulthandler.cancel_dump_traceback_later"
            ),
            mock.patch("ehsfp.deadlock_diagnostics.threading.Thread") as thread,
            mock.patch("ehsfp.deadlock_diagnostics.atexit.register"),
        ):
            session = start_deadlock_diagnostics(str(run_dir))
            self.assertTrue(Path(session.log_path).is_file())
            dump_traceback_later.assert_called_once()
            call = dump_traceback_later.call_args
            self.assertEqual(call.args[0], 17)
            self.assertTrue(call.kwargs["repeat"])
            self.assertFalse(call.kwargs["exit"])
            self.assertIs(call.kwargs["file"], session._log)
            thread.return_value.start.assert_called_once_with()
            return session

    def test_base_flag_keeps_anomaly_detection_disabled(self):
        session = self._start(anomaly_enabled=False)
        self.assertFalse(torch.is_anomaly_enabled())
        session.stop()
        self.assertFalse(torch.is_anomaly_enabled())

    def test_anomaly_flag_enables_anomaly_detection_in_addition_to_timer(self):
        session = self._start(anomaly_enabled=True)
        self.assertTrue(torch.is_anomaly_enabled())
        session.stop()
        self.assertFalse(torch.is_anomaly_enabled())


if __name__ == "__main__":
    unittest.main()
