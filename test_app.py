"""Проверки ML, импорта, хранения и интерфейса. python -m unittest -v"""
import csv
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from main import (QApplication, Detector, Store, Window, demo_data, quality,
                  read_csv, write_csv)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = Detector()
        cls.detector.fit(demo_data(1000, 42, False))
        cls.rows = demo_data()
        cls.results = cls.detector.analyze(cls.rows)

    def test_independent_demo_quality(self):
        q = quality(self.rows, self.results)
        self.assertEqual(q['labeled'], 600)
        self.assertEqual(q['tp'] + q['fn'], 60)
        self.assertGreaterEqual(q['recall'], .8)
        self.assertGreaterEqual(q['precision'], .65)

    def test_labels_do_not_change_inference(self):
        from dataclasses import replace
        changed = [replace(r, label=1 - r.label) for r in self.rows]
        self.assertEqual(self.detector.analyze(changed), self.results)

    def test_training_guards(self):
        with self.assertRaises(ValueError):
            Detector().fit(self.rows)
        with self.assertRaises(ValueError):
            Detector().fit(demo_data(20, 42, False))
        with self.assertRaises(ValueError):
            Detector().analyze(self.rows)

    def test_csv_roundtrip_and_unlabeled(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'измерения.csv'
            rows = [replace(r, label=None) for r in self.rows[:10]]
            write_csv(path, rows)
            self.assertEqual(read_csv(path), rows)
            self.assertEqual(quality(rows, self.results[:10]), {'labeled': 0})
            write_csv(path, self.rows, self.results)
            self.assertEqual(len(read_csv(path)), 600)

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            write_csv(path, self.rows[:1])
            original = path.read_text(encoding='utf-8-sig')
            for replacement in ['nan', '-1', 'inf']:
                path.write_text(original.replace(str(self.rows[0].latency_ms), replacement), encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_csv(path)
            path.write_text(original + original.splitlines()[-1] + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_csv(path)
            path.write_text('wrong;header\n1;2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_csv(path)
            for source, replacement in [(str(self.rows[0].latency_ms), '1000001'),
                                        (str(self.rows[0].loss_pct), '101'),
                                        (self.rows[0].timestamp, 'не дата'),
                                        (self.rows[0].node, '=cmd')]:
                path.write_text(original.replace(source, replacement), encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_csv(path)

    def test_database_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.db'
            run = Store(path).save('тест', 'норма', self.rows, self.results)
            restored_rows, restored_results = Store(path).load(run)
            self.assertEqual(restored_rows, self.rows)
            self.assertEqual(restored_results, self.results)
            self.assertEqual(Store(path).history()[0][3], 600)

    def test_gui_demo_filter_history(self):
        import time
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            window = Window(Path(folder) / 'gui.db')
            window.show()
            window.demo()
            deadline = time.monotonic() + 30
            while window.worker is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            self.assertIsNone(window.worker)
            self.assertEqual(window.table.rowCount(), 600)
            window.only_bad.setChecked(True)
            self.assertEqual(window.table.rowCount(), sum(x['anomaly'] for x in window.results))
            window.node_filter.setCurrentText('BR-DEMO-01')
            self.assertTrue(all(window.rows[i].node == 'BR-DEMO-01' for i in window.visible_indices()))
            window.restore(0, 0)
            self.assertEqual(len(window.rows), 600)
            from dataclasses import replace
            window.only_bad.setChecked(False)
            window.rows = [replace(r, latency_ms=0) for r in window.rows[:1]]
            window.results = []
            window.reset_filter()
            window.refresh()
            window.tabs.setCurrentIndex(1)
            app.processEvents()
            self.assertFalse(window.plot.grab().isNull())
            window.close()
            app.processEvents()


if __name__ == '__main__':
    unittest.main()
