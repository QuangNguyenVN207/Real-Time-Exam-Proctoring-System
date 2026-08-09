import sys
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import traceback
from datetime import datetime
from PySide6.QtCore import Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve, QPointF, QRectF
from PySide6.QtGui import (QPainter, QColor, QPen, QLinearGradient, QRadialGradient,
                           QPainterPath, QTextCharFormat, QFont, QTextCursor)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QTextEdit, QPushButton, QComboBox, QLabel, QHBoxLayout, QFrame, QMessageBox)

# 🔥 1. IMPORT SIÊU AI CỦA BẠN (Thay cho transformers/Whisper cũ)
from whisper.audio_pipeline import AudioPipeline


# =========================================================================
# WIDGET ĐỒ HỌA SÓNG ÂM (Giữ nguyên vẹn 100% để giao diện đẹp)
# =========================================================================
class WaveformWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.waves = []
        self.target_waves = []
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_waves)
        self.animation_timer.start(30)  # Faster updates for smoother animation
        self.is_recording = False
        self.transition_speed = 0.15  # Controls how fast waves transition

    def start_animation(self):
        self.is_recording = True
        self.waves = [0.1] * 30  # Start with small waves

    def stop_animation(self):
        self.is_recording = False
        self.target_waves = [0] * 30

    def update_audio_data(self, data):
        if len(data) > 0:
            normalized = np.abs(data) / np.max(np.abs(data) + 1e-10)
            if self.is_recording:
                chunk_size = len(normalized) // 30
                self.target_waves = [
                    normalized[i:i + chunk_size].mean() * 1.2
                    for i in range(0, len(normalized), chunk_size)
                ][:30]
                self.target_waves = [
                    w * (1 + np.random.uniform(-0.3, 0.3)) for w in self.target_waves]
            else:
                self.target_waves = [0] * 30
            self.update()

    def update_waves(self):
        if not self.waves:
            self.waves = [0] * 30
            self.target_waves = [0] * 30

        for i in range(len(self.waves)):
            if self.is_recording:
                target = self.target_waves[i] * \
                    (1 + np.sin(time.time() * 6 + i) * 0.15)
                target *= 1 + np.cos(time.time() * 4) * 0.1 
            else:
                target = 0
            self.waves[i] += (target - self.waves[i]) * 0.2
        self.update()

    def paintEvent(self, event):
        if not self.waves:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        try:
            width = self.width()
            height = self.height()
            center_y = height / 2
            bar_width = width / (len(self.waves) * 1.5)
            max_height = height * 0.85 

            gradient = QLinearGradient(0, 0, 0, height)
            if self.is_recording:
                gradient.setColorAt(0, QColor(46, 204, 113))
                gradient.setColorAt(0.5, QColor(39, 174, 96))
                gradient.setColorAt(1, QColor(33, 150, 83))
            else:
                gradient.setColorAt(0, QColor(46, 204, 113, 200))
                gradient.setColorAt(1, QColor(33, 150, 83, 200))

            painter.setPen(Qt.NoPen)
            painter.setBrush(gradient)

            for i, amplitude in enumerate(self.waves):
                x = width * i / len(self.waves)
                wave_effect = np.sin(time.time() * 4 + i * 0.5) * 0.08
                bar_height = max_height * (amplitude + wave_effect)

                if self.is_recording:
                    pulse = 1 + np.sin(time.time() * 5) * 0.08
                    bar_height *= pulse

                rect = QRectF(
                    x + bar_width/2,
                    center_y - bar_height/2,
                    bar_width,
                    bar_height
                )

                if self.is_recording and amplitude > 0.1:
                    glow = QPainterPath()
                    glow.addRoundedRect(rect, bar_width/2, bar_width/2)
                    painter.fillPath(glow, QColor(46, 204, 113, 40))

                painter.drawRoundedRect(rect, bar_width/2, bar_width/2)

        finally:
            painter.end()


# =========================================================================
# GIAO DIỆN CHÍNH & LUỒNG XỬ LÝ AI
# =========================================================================
class WhisperGUI(QMainWindow):
    update_text = Signal(str)

    def __init__(self):
        super().__init__()
        self.history_text = []  # Lưu lịch sử tin nhắn
        
        self.statusBar().showMessage("Đang nạp AI...")
        
        self.init_ui()
        self.init_whisper()
        self.update_text.connect(self.update_display)

    def init_ui(self):
        main_layout = QVBoxLayout()
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        controls_layout = QHBoxLayout()

        model_label = QLabel("Hệ thống:")
        self.model_combo = QComboBox()
        # 🔥 Đổi tên giao diện cho oai
        self.model_combo.addItems(["AI Giám Thị (PhoWhisper + PhoBERT)"]) 

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)

        controls_layout.addWidget(model_label)
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(self.record_button)
        controls_frame.setLayout(controls_layout)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: white; font-size: 14px; }")

        self.waveform = WaveformWidget()

        main_layout.addWidget(controls_frame)
        main_layout.addWidget(self.waveform)
        main_layout.addWidget(self.text_display)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.setWindowTitle("Real-time AI Proctoring")
        self.setGeometry(100, 100, 800, 600)

    def init_whisper(self):
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        # Blocksize nhỏ (0.1s) để sóng âm chạy mượt, AI xử lý riêng
        self.blocksize = int(self.sample_rate * 0.1) 
        self.process_thread = None
        self.pipeline = None
        
        self.load_model()

    def load_model(self):
        self.statusBar().showMessage("⏳ Đang nạp mô hình AI Giám thị...")
        QApplication.processEvents() # Ép GUI cập nhật chữ
        try:
            # 🔥 2. KHỞI TẠO BỘ NÃO AI CỦA BẠN
            if self.pipeline is None:
                self.pipeline = AudioPipeline()
            self.statusBar().showMessage("✅ Đã kết nối Hệ thống AI thành công! Sẵn sàng thu âm.")
        except Exception as e:
            self.statusBar().showMessage(f"❌ Lỗi nạp AI: {str(e)}")
            print(f"Error loading AI Pipeline: {str(e)}")

    def process_audio(self):
        """Luồng ngầm gom âm thanh đủ 3.5s và đưa cho AI phân tích"""
        if self.pipeline is None:
            return

        audio_buffer = np.array([], dtype=np.float32)
        try:
            while self.recording:
                if self.audio_queue.empty():
                    time.sleep(0.05)
                    continue

                audio_data = self.audio_queue.get()
                audio_data = audio_data.flatten().astype(np.float32)
                audio_buffer = np.concatenate([audio_buffer, audio_data])

                # 🔥 3. KIỂM TRA: Gom đủ 5 GIÂY thì ném vào AI
                if len(audio_buffer) >= self.sample_rate * 4:
                    chunk_to_process = audio_buffer.copy()
                    audio_buffer = np.array([], dtype=np.float32) # Xóa buffer hứng cái mới

                    try:
                        # 🔥 4. AI PHÂN TÍCH
                        result = self.pipeline.process_audio(chunk_to_process)
                        
                        if result and result.get("status") != "idle":
                            status = result.get("status", "")
                            text = result.get("transcription", "")
                            reason = result.get("fusion_reason", "")
                            timestamp = datetime.now().strftime('%H:%M:%S')

                            # 🔥 5. TRANG TRÍ MÀU SẮC LÊN GIAO DIỆN
                            if status == "alert":
                                msg = f"🚨 [{timestamp}] GIAN LẬN: '{text}'\n   ↳ Lý do: {reason}"
                            else:
                                msg = f"✅ [{timestamp}] AN TOÀN: '{text}'"

                            self.update_text.emit(msg)

                    except Exception as e:
                        print(f"Lỗi khi AI dịch: {str(e)}")
                        traceback.print_exc()

        except Exception as e:
            print(f"Lỗi luồng Audio: {str(e)}")
            traceback.print_exc()

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.pipeline:
            self.load_model()

        self.recording = True
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: #c0392b; color: white;")
        self.waveform.start_animation()

        # Bật luồng phân tích AI
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.process_thread.start()

        # Bật Mic thu âm liên tục
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=self.blocksize
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("")
        self.waveform.stop_animation()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def audio_callback(self, indata, frames, time, status):
        """Hứng tín hiệu mic đẩy vào buffer (và vẽ sóng âm)"""
        if status:
            pass
        self.audio_queue.put(indata.copy())
        self.waveform.update_audio_data(indata.copy())

    def update_display(self, text):
        """Cập nhật text lên cửa sổ GUI mỗi khi AI trả kết quả"""
        self.history_text.append(text)
        
        # Nối các câu lịch sử lại, cách nhau 1 dòng trắng
        display_text = "\n\n".join(self.history_text)
        
        self.text_display.setPlainText(display_text)
        
        # Tự động cuộn xuống dòng cuối cùng
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_display.setTextCursor(cursor)

    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        event.accept()


def main():
    app = QApplication(sys.argv)
    # Style tối ưu giao diện app
    app.setStyle("Fusion")
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()