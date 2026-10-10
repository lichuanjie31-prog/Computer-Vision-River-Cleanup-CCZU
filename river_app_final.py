"""常州大学水面垃圾检测桌面程序。

依赖和模型权重的安装方式见 README.md。
"""
from pathlib import Path
import os
import sys
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resource_path(relative_path):
    """Locate packaged resources or files next to this script, independent of cwd."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / relative_path


def read_image(path):
    """Read images from paths containing Chinese or other Unicode characters."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def write_image(path, image):
    """Save an annotated image safely when the output path contains Unicode."""
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"无法编码图片: {path.name}")
    encoded.tofile(str(path))


class BatchWorker(QThread):
    """Keep disk I/O and model inference off the Qt GUI thread."""

    progress = pyqtSignal(str, object, int)
    problem = pyqtSignal(str)

    def __init__(self, model, folder):
        super().__init__()
        self.model = model
        self.folder = Path(folder)
        self.output = self.folder / "Results_Output_V2"
        self.stop_event = threading.Event()
        self.processed = 0
        self.total = 0

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            files = sorted(
                p for p in self.folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            self.total = len(files)
            self.output.mkdir(parents=True, exist_ok=True)
            for image_path in files:
                if self.stop_event.is_set():
                    break
                try:
                    image = read_image(image_path)
                    if image is None:
                        raise ValueError("图像无法解码")
                    results = self.model.predict(image, conf=0.5, verbose=False)
                    if self.stop_event.is_set():
                        break
                    annotated = results[0].plot()
                    write_image(self.output / image_path.name, annotated)
                    self.processed += 1
                    self.progress.emit(
                        image_path.name, annotated, len(results[0].boxes)
                    )
                except Exception as exc:
                    self.problem.emit(f"{image_path.name}: {exc}")
        except Exception as exc:
            self.problem.emit(f"批量任务失败: {exc}")


class RiverMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.data_points = [0] * 20
        self.model = None

        model_path = resource_path("weights/best.pt")
        if not model_path.is_file():
            model_path = resource_path("best.pt")
        self.model_error = None
        try:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"未找到模型权重文件。请将 Release 中的 best.pt 放入 "
                    f"{resource_path('weights/best.pt')}"
                )
            self.model = YOLO(str(model_path))
        except Exception as exc:
            self.model_error = str(exc)

        self.init_ui()
        if self.model_error:
            self.log(f"模型加载失败: {self.model_error}")
        else:
            self.log(f"模型加载成功: {model_path}")

    def init_ui(self):
        self.setWindowTitle("常州大学 - 水面垃圾检测系统 V2.0")
        self.resize(1150, 720)
        self.setStyleSheet("background-color: #1e272e; color: white;")

        main = QVBoxLayout(self)
        main.setContentsMargins(20, 15, 20, 15)
        header = QHBoxLayout()
        logo = QLabel()
        logo_path = resource_path("cczu_logo.png")
        if logo_path.is_file():
            logo.setPixmap(
                QPixmap(str(logo_path)).scaled(
                    90, 90, Qt.AspectRatioMode.KeepAspectRatio
                )
            )
        header.addWidget(logo)
        title = QLabel("智慧河道：数据处理分析中心 (V2.0)")
        title.setStyleSheet(
            "font-size: 26px; font-weight: bold; color: #3498db; margin-left: 15px;"
        )
        header.addWidget(title, stretch=1)
        main.addLayout(header)

        body = QHBoxLayout()
        self.preview_label = QLabel("等待任务启动...")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet(
            "border: 3px solid #3498db; border-radius: 15px; background: black;"
        )
        body.addWidget(self.preview_label, 2)

        chart_col = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.fig.patch.set_facecolor("#1e272e")
        self.ax.set_facecolor("#1e272e")
        self.ax.tick_params(colors="white", labelsize=8)
        self.canvas = FigureCanvas(self.fig)
        chart_col.addWidget(QLabel(
            "实时垃圾数量波动", alignment=Qt.AlignmentFlag.AlignCenter
        ))
        chart_col.addWidget(self.canvas)
        body.addLayout(chart_col, 1)
        main.addLayout(body)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(105)
        self.console.setStyleSheet(
            "background: #0c1117; color: #1dd1a1;"
            "font-family: Consolas; border: 1px solid #3498db; padding: 10px;"
        )
        main.addWidget(self.console)

        buttons = QHBoxLayout()
        self.btn_cam = QPushButton("模式A：开启实时监控")
        self.btn_batch = QPushButton("模式B：批量分析文件夹")
        self.btn_stop = QPushButton("停止任务")
        for button, color in (
            (self.btn_cam, "#3498db"),
            (self.btn_batch, "#9b59b6"),
            (self.btn_stop, "#e74c3c"),
        ):
            button.setStyleSheet(
                f"QPushButton{{background:{color}; color:white;"
                "border-radius:8px; padding:12px; font-weight:bold; font-size:14px;}"
            )
            buttons.addWidget(button)
        main.addLayout(buttons)

        self.btn_cam.clicked.connect(self.start_camera)
        self.btn_batch.clicked.connect(self.start_batch)
        self.btn_stop.clicked.connect(self.stop_all)

    def log(self, message):
        self.console.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def check_model(self):
        if self.model is not None:
            return True
        QMessageBox.critical(
            self, "无法启动", self.model_error or "模型加载失败"
        )
        return False

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def start_camera(self):
        if not self.check_model():
            return
        if self.worker is not None:
            self.log("请先停止或等待批量任务结束")
            return
        self.stop_camera()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.stop_camera()
            self.log("ERROR: 无法打开摄像头，请检查权限或其他软件占用")
            return
        self.timer.start(30)
        self.log("实时监控已启动")

    def start_batch(self):
        if not self.check_model() or self.worker is not None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "选择包含水面垃圾图片的文件夹"
        )
        if not folder:
            return
        self.stop_camera()
        self.worker = BatchWorker(self.model, folder)
        self.worker.progress.connect(self.on_batch_progress)
        self.worker.problem.connect(lambda message: self.log(f"ERROR: {message}"))
        self.worker.finished.connect(self.on_batch_finished)
        self.btn_cam.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.log(f"开始批量分析: {folder}")
        self.worker.start()

    def on_batch_progress(self, filename, annotated, count):
        self.show_image(annotated)
        self.log(f"{filename} | 识别数量: {count} | 已保存")

    def on_batch_finished(self):
        worker = self.worker
        if worker is None:
            return
        outcome = "已取消" if worker.stop_event.is_set() else "已完成"
        self.log(
            f"批量任务{outcome}: 成功处理 {worker.processed}/{worker.total} 张，"
            f"输出目录: {worker.output}"
        )
        self.btn_cam.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.worker = None
        worker.deleteLater()

    def show_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self.preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )

    def update_frame(self):
        if self.cap is None:
            self.timer.stop()
            return
        try:
            ok, frame = self.cap.read()
            if not ok:
                self.log("ERROR: 摄像头读取失败，已停止")
                self.stop_camera()
                return
            result = self.model.predict(frame, conf=0.3, verbose=False)[0]
            count = len(result.boxes)
            self.show_image(result.plot())
            self.data_points = self.data_points[1:] + [count]
            self.ax.clear()
            self.ax.plot(self.data_points, color="#3498db", linewidth=2)
            self.ax.set_ylim(0, max(5, max(self.data_points) + 2))
            self.ax.set_title("实时垃圾统计", color="white")
            self.ax.tick_params(colors="white", labelsize=8)
            self.canvas.draw_idle()
        except Exception as exc:
            self.log(f"ERROR: 检测异常，已停止: {exc}")
            self.stop_camera()

    def stop_all(self):
        self.stop_camera()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log("已发送批量任务停止请求，当前推理结束后退出")
        else:
            self.log("任务已停止")
        self.preview_label.clear()
        self.preview_label.setText("任务已停止\n常州大学软件工程项目组")

    def closeEvent(self, event):
        self.stop_camera()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        plt.close(self.fig)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RiverMonitor()
    window.show()
    sys.exit(app.exec())
