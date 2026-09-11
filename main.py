"""Анализ качества связи. Учебный проект Семёнова А.В., ИИ-26.

Запуск: python main.py. Измерения демонстрационные, к сети оператора
приложение не подключается. Модель обучается на отдельном эталоне нормы.
"""
from __future__ import annotations

from contextlib import closing
import csv
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget, QHeaderView,
)

ROOT = Path(__file__).resolve().parent
FEATURES = ('latency_ms', 'jitter_ms', 'loss_pct', 'utilization_pct')
FIELDS = ('timestamp', 'node', *FEATURES, 'label')
MODEL_NAME = 'Isolation Forest • 200 деревьев • random_state=42'
MAX_ROWS = 10000


@dataclass(frozen=True)
class Measurement:
    timestamp: str
    node: str
    latency_ms: float
    jitter_ms: float
    loss_pct: float
    utilization_pct: float
    label: int | None = None


def matrix(rows):
    return np.array([[getattr(r, f) for f in FEATURES] for r in rows], dtype=float)


def demo_data(count=600, seed=77, anomalies=True):
    """Эталон и контрольный набор генерируются с разными seed.

    label применяется только для оценки, в признаки модели не включается.
    Четыре типа отклонений показывают разные симптомы, а не причины аварии.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2026, 6, 15, 8)
    rows = []
    for i in range(count):
        values = [max(1, rng.normal(18, 3)), max(.1, rng.normal(3, .6)),
                  max(0, rng.normal(.15, .07)), np.clip(rng.normal(45, 9), 1, 80)]
        bad = anomalies and i % 10 == 0
        if bad:
            kind = (i // 10) % 4
            if kind == 0:
                values[0], values[1] = rng.uniform(90, 180), rng.uniform(20, 45)
            elif kind == 1:
                values[2] = rng.uniform(4, 12)
            elif kind == 2:
                values[3], values[0] = rng.uniform(92, 99), rng.uniform(50, 95)
            else:
                values[1], values[2] = rng.uniform(14, 30), rng.uniform(1.5, 4)
        rows.append(Measurement((base + timedelta(minutes=i)).isoformat(' '),
                                f'BR-DEMO-{i % 3 + 1:02}',
                                *[round(float(v), 3) for v in values], int(bad)))
    return rows


def read_csv(path):
    """Весь файл проверяется до замены текущего набора, без частичного импорта."""
    if Path(path).stat().st_size > 10 * 1024 * 1024:
        raise ValueError('Файл превышает 10 МБ.')
    result, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter=';')
        required = set(FIELDS) - {'label'}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError('Неверные столбцы. Используйте шаблон из папки data; разделитель — ;')
        for line, item in enumerate(reader, 2):
            try:
                stamp = datetime.fromisoformat(item['timestamp'].strip())
                if stamp.tzinfo is not None:
                    raise ValueError('время должно быть местным, без часового пояса')
                timestamp = stamp.isoformat(' ')
                node = item['node'].strip()
                if not re.fullmatch(r'[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_.-]{0,39}', node):
                    raise ValueError('недопустимый идентификатор узла')
                values = [float(item[f].replace(',', '.')) for f in FEATURES]
                if not all(np.isfinite(v) and v >= 0 for v in values):
                    raise ValueError('показатели должны быть конечными и неотрицательными')
                if values[2] > 100 or values[3] > 100:
                    raise ValueError('процент не может превышать 100')
                if values[0] > 1000000 or values[1] > 1000000:
                    raise ValueError('задержка и джиттер не должны превышать 1 000 000 мс')
                label_text = (item.get('label') or '').strip()
                if label_text not in ('', '0', '1'):
                    raise ValueError('label должен быть пустым, 0 или 1')
                if (timestamp, node) in seen:
                    raise ValueError('повтор времени и узла')
                seen.add((timestamp, node))
                result.append(Measurement(timestamp, node, *values,
                                          int(label_text) if label_text else None))
                if len(result) > MAX_ROWS:
                    raise ValueError(f'допускается не более {MAX_ROWS} строк')
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                raise ValueError(f'Строка {line}: {error}') from error
    if not result:
        raise ValueError('CSV не содержит измерений.')
    return result


def write_csv(path, rows, results=None):
    extra = ('anomaly', 'score', 'symptom') if results is not None else ()
    with open(path, 'w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FIELDS, *extra), delimiter=';')
        writer.writeheader()
        for i, row in enumerate(rows):
            record = asdict(row)
            if results is not None:
                record.update(results[i])
            writer.writerow(record)


def symptom(row):
    """Пороговые подсказки объясняют показатели, не подменяя решение ML."""
    messages = []
    if row.latency_ms > 50:
        messages.append('повышенная задержка')
    if row.jitter_ms > 10:
        messages.append('нестабильная задержка')
    if row.loss_pct > 1:
        messages.append('потери пакетов')
    if row.utilization_pct > 85:
        messages.append('высокая загрузка')
    return ', '.join(messages) or 'нет пороговых превышений; сравните с эталоном'


class Detector:
    def __init__(self):
        self.model = None
        self.baseline_size = 0

    def fit(self, baseline):
        if len(baseline) < 100:
            raise ValueError('Для обучения требуется не менее 100 измерений нормы.')
        if any(r.label == 1 for r in baseline):
            raise ValueError('Эталон нормы содержит label=1. Исключите аномальные записи.')
        if np.any(np.ptp(matrix(baseline), axis=0) == 0):
            raise ValueError('Каждый показатель эталона должен иметь ненулевой разброс.')
        model = IsolationForest(n_estimators=200, contamination=.02,
                                random_state=42, n_jobs=1)
        model.fit(matrix(baseline))
        self.model, self.baseline_size = model, len(baseline)

    def analyze(self, rows):
        if self.model is None:
            raise ValueError('Сначала обучите модель на эталоне нормы.')
        scores = self.model.decision_function(matrix(rows))
        return [dict(anomaly=int(s < 0), score=round(float(s), 6),
                     symptom=symptom(r) if s < 0 else 'в пределах модели нормы')
                for r, s in zip(rows, scores)]


def quality(rows, results):
    pairs = [(r.label, x['anomaly']) for r, x in zip(rows, results) if r.label is not None]
    if not pairs:
        return {'labeled': 0}
    truth, predicted = zip(*pairs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return dict(labeled=len(pairs), precision=float(precision), recall=float(recall),
                f1=float(f1), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


class Store:
    def __init__(self, path):
        self.path = Path(path)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript('''CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY, created TEXT, source TEXT,
                baseline TEXT, metrics TEXT);
                CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY, run_id INTEGER REFERENCES runs(id),
                timestamp TEXT, node TEXT, latency_ms REAL, jitter_ms REAL,
                loss_pct REAL, utilization_pct REAL, label INTEGER,
                anomaly INTEGER, score REAL, symptom TEXT);''')

    def save(self, source, baseline, rows, results):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('PRAGMA foreign_keys=ON')
            cursor = db.execute('INSERT INTO runs(created,source,baseline,metrics) VALUES(?,?,?,?)',
                                (datetime.now().isoformat(' ', timespec='seconds'), source,
                                 baseline, json.dumps(quality(rows, results))))
            run_id = cursor.lastrowid
            db.executemany('''INSERT INTO measurements(run_id,timestamp,node,latency_ms,
                jitter_ms,loss_pct,utilization_pct,label,anomaly,score,symptom)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                [(run_id, *asdict(r).values(), x['anomaly'], x['score'], x['symptom'])
                 for r, x in zip(rows, results)])
        return run_id

    def history(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute('''SELECT r.id,r.created,r.source,COUNT(m.id),SUM(m.anomaly)
                FROM runs r JOIN measurements m ON m.run_id=r.id
                GROUP BY r.id ORDER BY r.id DESC''').fetchall()

    def load(self, run_id):
        with closing(sqlite3.connect(self.path)) as db, db:
            records = db.execute('''SELECT timestamp,node,latency_ms,jitter_ms,loss_pct,
                utilization_pct,label,anomaly,score,symptom FROM measurements
                WHERE run_id=? ORDER BY id''', (run_id,)).fetchall()
        rows = [Measurement(*r[:7]) for r in records]
        results = [dict(anomaly=r[7], score=r[8], symptom=r[9]) for r in records]
        return rows, results


class Worker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, action):
        super().__init__()
        self.action = action

    def run(self):
        try:
            self.completed.emit(self.action())
        except Exception as error:
            self.failed.emit(str(error))


class Plot(QWidget):
    def __init__(self):
        super().__init__()
        self.rows, self.results = [], []
        self.setMinimumHeight(220)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('white'))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QColor('#223344'))
        painter.drawText(18, 24, 'Задержка по времени, мс • красные точки — аномалии ML')
        if not self.rows:
            return
        indices = sorted(range(len(self.rows)), key=lambda i: self.rows[i].timestamp)
        start = datetime.fromisoformat(self.rows[indices[0]].timestamp)
        times = [(datetime.fromisoformat(self.rows[i].timestamp) - start).total_seconds() for i in indices]
        span, maximum = max(times[-1], 1), max(1., max(r.latency_ms for r in self.rows) * 1.1)
        width, height = self.width() - 85, self.height() - 80
        painter.drawText(8, 48, f'{maximum:.0f}')
        painter.drawText(22, int(45 + height), '0')
        painter.drawLine(55, 40, 55, 45 + height)
        painter.drawLine(55, 45 + height, 55 + width, 45 + height)
        for i, time in zip(indices, times):
            x, y = 55 + int(time / span * width), 45 + height - int(self.rows[i].latency_ms / maximum * height)
            bad = self.results and self.results[i]['anomaly']
            painter.setPen(QPen(QColor('#c0392b' if bad else '#2875ae'), 4 if bad else 2))
            painter.drawPoint(x, y)
        painter.setPen(QColor('#223344'))
        painter.drawText(55, self.height() - 12, self.rows[indices[0]].timestamp)
        painter.drawText(self.width() - 180, self.height() - 12, self.rows[indices[-1]].timestamp)


class Window(QMainWindow):
    def __init__(self, database=None):
        super().__init__()
        self.setWindowTitle('Анализ качества связи — учебный проект | Семёнов А.В., ИИ-26')
        self.resize(1180, 760)
        self.store = Store(database or ROOT / 'telecom.db')
        self.detector = Detector()
        self.rows, self.results = [], []
        self.source, self.baseline_name = '', ''
        self.worker = None
        central, layout = QWidget(), QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)
        title = QLabel('АНАЛИЗ КАЧЕСТВА СВЯЗИ')
        title.setStyleSheet('font-size:22px;font-weight:600;color:#184f72;')
        layout.addWidget(title)
        layout.addWidget(QLabel('Брест • учебный стенд • без подключения к сети РУП «Белтелеком»'))
        bar = QHBoxLayout()
        self.buttons = []
        for label, callback in [('Демонстрация', self.demo), ('Эталон CSV', self.train_csv),
                                ('Импорт CSV', self.import_csv), ('Анализировать', self.analyze),
                                ('Экспорт CSV', self.export_csv)]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            bar.addWidget(button)
            self.buttons.append(button)
        layout.addLayout(bar)
        self.status = QLabel('Начните с «Демонстрация» или загрузите эталон нормы и измерения CSV.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.model_label = QLabel('Модель не обучена')
        layout.addWidget(self.model_label)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        data_tab, data_layout = QWidget(), QVBoxLayout()
        data_tab.setLayout(data_layout)
        filters = QHBoxLayout()
        self.node_filter, self.only_bad = QComboBox(), QCheckBox('Только аномалии')
        self.node_filter.addItem('Все узлы')
        self.node_filter.currentTextChanged.connect(self.refresh)
        self.only_bad.toggled.connect(self.refresh)
        filters.addWidget(self.node_filter)
        filters.addWidget(self.only_bad)
        filters.addStretch()
        data_layout.addLayout(filters)
        self.table = self.make_table(['Время', 'Узел', 'Задержка, мс', 'Джиттер, мс',
                                      'Потери, %', 'Загрузка, %', 'Решение ML', 'Оценка'])
        data_layout.addWidget(self.table)
        self.detail = QLabel('Выберите строку для пояснения результата.')
        self.detail.setWordWrap(True)
        data_layout.addWidget(self.detail)
        self.table.itemSelectionChanged.connect(self.show_detail)
        self.tabs.addTab(data_tab, 'Измерения')
        analytics, analytics_layout = QWidget(), QVBoxLayout()
        analytics.setLayout(analytics_layout)
        self.plot = Plot()
        analytics_layout.addWidget(self.plot)
        self.summary = self.make_table(['Узел', 'Измерений', 'Аномалий', 'Доля, %', 'Средняя задержка, мс'])
        analytics_layout.addWidget(self.summary)
        self.tabs.addTab(analytics, 'Аналитика')
        self.metrics = QTextEdit()
        self.metrics.setReadOnly(True)
        self.tabs.addTab(self.metrics, 'Оценка качества')
        self.history_table = self.make_table(['Запуск', 'Дата', 'Источник', 'Измерений', 'Аномалий'])
        self.history_table.cellDoubleClicked.connect(self.restore)
        self.tabs.addTab(self.history_table, 'История — двойной щелчок для открытия')
        self.reload_history()

    @staticmethod
    def make_table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def background(self, action, done):
        if self.worker is not None:
            return
        for button in self.buttons:
            button.setEnabled(False)
        self.history_table.setEnabled(False)
        self.status.setText('Выполняется обработка…')
        self.worker = Worker(action)
        self.worker.completed.connect(done)
        self.worker.failed.connect(self.error)
        self.worker.finished.connect(self.finish)
        self.worker.start()

    def finish(self):
        self.worker.deleteLater()
        self.worker = None
        for button in self.buttons:
            button.setEnabled(True)
        self.history_table.setEnabled(True)

    def error(self, text):
        self.status.setText('Операция не выполнена: ' + text)
        QMessageBox.warning(self, 'Проверка данных', text)

    def demo(self):
        def work():
            detector = Detector()
            detector.fit(demo_data(1000, 42, False))
            rows = demo_data()
            return detector, rows, detector.analyze(rows)
        def done(payload):
            self.detector, self.rows, self.results = payload
            self.source = 'Демонстрационные измерения seed=77'
            self.baseline_name = 'Демонстрационная норма: 1000 строк, seed=42'
            self.accept_result()
        self.background(work, done)

    def train_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Эталон нормальной работы', str(ROOT / 'data'), 'CSV (*.csv)')
        if not path:
            return
        def work():
            detector = Detector()
            detector.fit(read_csv(path))
            return detector
        def done(detector):
            self.detector = detector
            self.baseline_name = Path(path).name
            self.results = []
            self.model_label.setText(f'{MODEL_NAME} | Эталон: {detector.baseline_size} строк; {self.baseline_name}')
            self.status.setText('Модель обучена. Загрузите измерения и выполните анализ.')
            self.refresh()
        self.background(work, done)

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Измерения для анализа', str(ROOT / 'data'), 'CSV (*.csv)')
        if path:
            try:
                rows = read_csv(path)
                self.rows, self.results, self.source = rows, [], Path(path).name
                self.reset_filter()
                self.refresh()
                self.status.setText(f'Загружено {len(rows)} измерений. Нажмите «Анализировать».')
            except (ValueError, OSError, UnicodeError, csv.Error) as error:
                self.error(str(error))

    def analyze(self):
        if not self.rows or self.detector.model is None:
            self.error('Загрузите измерения и обучите модель либо запустите демонстрацию.')
            return
        def done(results):
            self.results = results
            self.accept_result()
        self.background(lambda: self.detector.analyze(self.rows), done)

    def accept_result(self):
        try:
            run_id = self.store.save(self.source, self.baseline_name, self.rows, self.results)
            saved = f'Сохранён запуск № {run_id}.'
        except sqlite3.Error as error:
            saved = f'Не удалось сохранить: {error}. Результат доступен для экспорта.'
        self.model_label.setText(f'{MODEL_NAME} | Эталон: {self.baseline_name}')
        self.reset_filter()
        self.refresh()
        self.reload_history()
        self.status.setText(f'Измерений: {len(self.rows)} | Аномалий: {sum(x["anomaly"] for x in self.results)} | {saved}')

    def reset_filter(self):
        self.node_filter.blockSignals(True)
        self.node_filter.clear()
        self.node_filter.addItems(['Все узлы', *sorted({r.node for r in self.rows})])
        self.node_filter.blockSignals(False)

    def visible_indices(self):
        return [i for i, row in enumerate(self.rows)
                if (self.node_filter.currentText() in ('Все узлы', row.node))
                and (not self.only_bad.isChecked() or self.results and self.results[i]['anomaly'])]

    def refresh(self):
        indices = self.visible_indices()
        self.table.setRowCount(len(indices))
        for pos, i in enumerate(indices):
            row = self.rows[i]
            result = self.results[i] if self.results else None
            values = [row.timestamp, row.node, *[f'{getattr(row, f):.3f}' for f in FEATURES],
                      ('Аномалия' if result['anomaly'] else 'Норма') if result else '—',
                      f'{result["score"]:.4f}' if result else '—']
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, i)
                if result and result['anomaly']:
                    item.setBackground(QColor('#ffeded'))
                self.table.setItem(pos, col, item)
        self.detail.setText(f'Показано строк: {len(indices)}. Экспорт сохраняет текущий фильтр.')
        self.plot.rows = [self.rows[i] for i in indices]
        self.plot.results = [self.results[i] for i in indices] if self.results else []
        self.plot.update()
        nodes = sorted({self.rows[i].node for i in indices})
        self.summary.setRowCount(len(nodes))
        for pos, node in enumerate(nodes):
            selected = [i for i in indices if self.rows[i].node == node]
            bad = sum(self.results[i]['anomaly'] for i in selected) if self.results else None
            values = [node, len(selected), bad if bad is not None else '—',
                      f'{bad / len(selected) * 100:.1f}' if bad is not None else '—',
                      f'{np.mean([self.rows[i].latency_ms for i in selected]):.2f}']
            for col, value in enumerate(values):
                self.summary.setItem(pos, col, QTableWidgetItem(str(value)))
        q = quality(self.rows, self.results) if self.results else {'labeled': 0}
        if q['labeled']:
            self.metrics.setPlainText(
                'ОЦЕНКА НА ВСЁМ РАЗМЕЧЕННОМ НАБОРЕ (фильтр не влияет)\n\n'
                f'Размечено измерений: {q["labeled"]}\n'
                f'Precision = {q["precision"]:.4f}\nRecall = {q["recall"]:.4f}\nF1 = {q["f1"]:.4f}\n\n'
                f'TP: {q["tp"]}   FP: {q["fp"]}   FN: {q["fn"]}   TN: {q["tn"]}\n\n'
                'TP — верно обнаруженные отклонения; FP — ложные тревоги;\n'
                'FN — пропущенные отклонения; TN — верно определённая норма.\n\n'
                'label не входит в признаки модели. Отрицательная оценка означает аномалию,\n'
                'но не вероятность аварии. Пороговые пояснения не определяют решение ML.\n\n'
                'Демонстрационные результаты относятся к синтетическому стенду.\n'
                'Для практического применения нужен репрезентативный эталон реальной сети\n'
                'и независимая проверка разметки специалистом.')
        else:
            self.metrics.setPlainText('Метрики недоступны: выполните анализ набора с заполненным label.\n'
                                      'Для неразмеченных измерений доступны решения модели и аналитика.')

    def show_detail(self):
        items = self.table.selectedItems()
        if items and self.results:
            i = items[0].data(Qt.ItemDataRole.UserRole)
            self.detail.setText('Пояснение: ' + self.results[i]['symptom'] +
                                '. Это симптом, а не установленная причина неисправности.')

    def export_csv(self):
        indices = self.visible_indices()
        if not indices or not self.results:
            self.error('Нет проанализированных строк для экспорта.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Экспорт текущего фильтра', str(ROOT / 'analysis.csv'), 'CSV (*.csv)')
        if path:
            try:
                write_csv(path, [self.rows[i] for i in indices], [self.results[i] for i in indices])
                self.status.setText(f'Экспортировано строк: {len(indices)}; {path}')
            except OSError as error:
                self.error(str(error))

    def reload_history(self):
        try:
            records = self.store.history()
            self.history_table.setRowCount(len(records))
            for row, record in enumerate(records):
                for col, value in enumerate(record):
                    self.history_table.setItem(row, col, QTableWidgetItem(str(value)))
        except sqlite3.Error as error:
            self.status.setText('Ошибка чтения истории: ' + str(error))

    def restore(self, row, column):
        try:
            run_id = int(self.history_table.item(row, 0).text())
            self.rows, self.results = self.store.load(run_id)
            self.source = f'История: запуск {run_id}'
            self.reset_filter()
            self.refresh()
            self.tabs.setCurrentIndex(0)
            self.status.setText(f'Открыт сохранённый запуск № {run_id}. Модель для повторного анализа обучается отдельно.')
        except (sqlite3.Error, ValueError) as error:
            self.error(str(error))

    def closeEvent(self, event):
        if self.worker is not None:
            self.status.setText('Дождитесь завершения обработки, затем закройте окно.')
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Arial', 10))
    try:
        window = Window()
    except (sqlite3.Error, OSError) as error:
        QMessageBox.critical(None, 'Не удалось открыть базу данных',
                             f'{error}\nРаспакуйте проект в папку с правом записи.')
        return 1
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())

