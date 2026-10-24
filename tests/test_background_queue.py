"""Exercise the real detached launcher with harmless stand-in training scripts."""
import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / 'training'


class BackgroundQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tumemo queue ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / 'run_all_nohup.sh'
        shutil.copyfile(HERE / self.script.name, self.script)
        shutil.copyfile(HERE / 'run_fusion_nohup.sh', self.root / 'run_fusion_nohup.sh')
        self.logs = self.root / 'logs with spaces'
        self.events = self.root / 'events'
        self.env = {**os.environ, 'QUEUE_TEST_DIR': str(self.root)}
        for index, name in ((1, 'run_v1_direct.sh'), (2, 'run_v2_finetune.sh'), (3, 'run_v3_frozen.sh'),
                            (4, 'run_v4_direct_5k.sh'), (5, 'run_v5_reset_fusion.sh'), (6, 'run_v6_tune_fusion.sh')):
            (self.root / name).write_text(f'''#!/usr/bin/env bash
set -euo pipefail
printf 'start {index}\\n' >> "$QUEUE_TEST_DIR/events"
printf '%s\\n' "$@"
trap 'printf "saved {index}\\n" >> "$QUEUE_TEST_DIR/events"; exit 130' TERM
if [[ -e "$QUEUE_TEST_DIR/block" ]]; then
  while [[ -e "$QUEUE_TEST_DIR/block" ]]; do sleep .05; done
fi
if [[ -e "$QUEUE_TEST_DIR/fail_{index}" ]]; then exit 7; fi
printf 'end {index}\\n' >> "$QUEUE_TEST_DIR/events"
''')
        self.addCleanup(self.stop_if_active)

    def invoke(self, action='start', *args):
        return subprocess.run(['bash', str(self.script), action, '--log-dir', str(self.logs), '--', *args],
                              env=self.env, capture_output=True, text=True, timeout=10)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        self.fail('Background queue did not reach expected state')

    def state(self):
        path = self.logs / 'status'
        return path.read_text().strip() if path.exists() else ''

    def stop_if_active(self):
        if (self.logs / 'queue.pid').exists():
            self.invoke('stop')
            deadline = time.monotonic() + 5
            while self.state().startswith(('running', 'stopping')) and time.monotonic() < deadline:
                time.sleep(.05)

    def test_sequential_commands_and_literal_arguments(self):
        payload = 'literal $(touch SHOULD_NOT_EXIST) ; spaces'
        result = self.invoke('start', '--output', payload)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.wait_for(lambda: self.state() == 'completed')
        self.assertEqual(self.events.read_text().splitlines(), ['start 1', 'end 1', 'start 2', 'end 2', 'start 3', 'end 3'])
        for name in ('run_v1_direct', 'run_v2_finetune', 'run_v3_frozen'):
            self.assertIn(payload, (self.logs / f'{name}.log').read_text())
        self.assertEqual((self.logs / 'queue.log').read_text().count('Command:'), 3)

    def test_failure_does_not_start_next_version(self):
        (self.root / 'fail_1').touch()
        self.invoke()
        self.wait_for(lambda: self.state().startswith('failed'))
        self.assertEqual(self.events.read_text().splitlines(), ['start 1'])

    def test_fusion_wrapper_runs_only_v5_then_v6(self):
        result = subprocess.run(['bash', str(self.root / 'run_fusion_nohup.sh'), '--log-dir', str(self.logs),
                                 '--', '--seed', '42'], env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.wait_for(lambda: self.state() == 'completed')
        self.assertEqual(self.events.read_text().splitlines(), ['start 5', 'end 5', 'start 6', 'end 6'])
        self.assertEqual((self.logs / 'queue.log').read_text().count('Command:'), 2)

    def test_all_suite_runs_six_versions_in_order(self):
        result = subprocess.run(['bash', str(self.script), '--suite', 'all', '--log-dir', str(self.logs),
                                 '--', '--seed', '42'], env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.wait_for(lambda: self.state() == 'completed')
        self.assertEqual(self.events.read_text().splitlines(),
                         [event for i in range(1, 7) for event in (f'start {i}', f'end {i}')])
        self.assertEqual((self.logs / 'queue.log').read_text().count('Command:'), 6)

    def test_hangup_lock_and_graceful_stop(self):
        (self.root / 'block').touch()
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.wait_for(lambda: self.events.exists())
        pid = int((self.logs / 'queue.pid').read_text().split()[0])
        os.kill(pid, signal.SIGHUP)
        time.sleep(.1)
        self.assertIn('Queue running', self.invoke('status').stdout)
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertEqual(self.invoke('stop').returncode, 0)
        self.wait_for(lambda: self.state().startswith('stopped'))
        self.assertEqual(self.events.read_text().splitlines(), ['start 1', 'saved 1'])
        self.assertNotEqual(self.invoke().returncode, 0)
        (self.root / 'block').unlink()
        self.assertEqual(self.invoke('start', '--resume').returncode, 0)
        self.wait_for(lambda: self.state() == 'completed')


if __name__ == '__main__':
    unittest.main()
