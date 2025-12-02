import sys
import os
import pygame
import miditoolkit
import tempfile
import copy
import bisect

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QFileDialog, QStyle, QMessageBox, QComboBox
)
from PyQt5.QtGui import QFont, QPainter, QColor, QBrush, QPen, QIcon, QFontMetrics
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, QObject, QThread

# --- Your Model Imports ---
# Mocking imports for standalone execution if models are not present
try:
    from src.model.pianoformer import PianoT5Gemma
    from src.model.generate import batch_performance_render, map_midi
    import torch
    MODELS_AVAILABLE = True
except ImportError:
    print("Warning: Model-related libraries not found. Running in UI-only mode.")
    MODELS_AVAILABLE = False
    # Create dummy classes and functions to avoid crashing
    class PianoT5Gemma:
        @staticmethod
        def from_pretrained(*args, **kwargs): return None
    def batch_performance_render(*args, **kwargs): return [None]
    def map_midi(*args, **kwargs): return None
    class torch:
        @staticmethod
        def cuda(): return type('cuda', (), {'is_available': lambda: False})()
        bfloat16 = None


# --- NEW: Language Management ---
class LanguageManager(QObject):
    """Manages language strings and signals changes."""
    language_changed = pyqtSignal()

    STRINGS = {
        'en': {
            # Window & Titles
            'window_title': "Pianist Transformer",
            'main_title': " ",
            # Player Bar
            'select_demo': "— Select a Demo —",
            # File Loader
            'file_path_label': "File Path",
            'load_midi': "Load MIDI",
            # Parameters
            'temperature': "Temperature",
            'top_p': "Top-p",
            # Main Controls
            'original_score': "Original",
            'version_button': "V{0}",
            'start_render': "Start Render",
            'render_again': "Render Again",
            'cancel_render': "Cancel Render",
            'save_render': "Save MIDI",
            'save_editable': "Save Editable MIDI",
            # Dialogs & Messages
            'select_midi_dialog_title': "Select MIDI File",
            'save_render_dialog_title': "Save MIDI",
            'save_editable_dialog_title': "Save Editable MIDI",
            'warning_title': "Warning",
            'error_title': "Error",
            'overwrite_title': "Confirm Overwrite",
            'render_notice_title': "Render Notification",
            'no_midi_loaded_warning': "Please load a MIDI file to render first.",
            'overwrite_warning': "All render slots are full. Do you want to overwrite version V{0}?",
            'model_load_fail_error': "Failed to load the model. Rendering is not possible.",
            # Progress Widget
            'progress_initializing': "Initializing...",
            'progress_loading_model': "Loading model...",
            'progress_rendering': "Rendering...",
            'progress_cancelling': "Cancelling...",
            # Render Worker Errors
            'render_cancelled_by_user': "Rendering was cancelled by the user.",
            'render_error': "Render error: {0}",
        },
        'zh': {
            # Window & Titles
            'window_title': "Pianist Transformer",
            'main_title': " ",
            # Player Bar
            'select_demo': "— 选择试听DEMO —",
            # File Loader
            'file_path_label': "文件路径",
            'load_midi': "载入 MIDI",
            # Parameters
            'temperature': "Temperature",
            'top_p': "Top-p",
            # Main Controls
            'original_score': "原乐谱",
            'version_button': "V{0}",
            'start_render': "开始渲染",
            'render_again': "再次渲染",
            'cancel_render': "取消渲染",
            'save_render': "保存渲染 MIDI",
            'save_editable': "保存可编辑 MIDI",
            # Dialogs & Messages
            'select_midi_dialog_title': "选择MIDI文件",
            'save_render_dialog_title': "保存渲染后的MIDI",
            'save_editable_dialog_title': "保存可编辑的渲染MIDI",
            'warning_title': "警告",
            'error_title': "错误",
            'overwrite_title': "确认覆盖",
            'render_notice_title': "渲染通知",
            'no_midi_loaded_warning': "请先载入一个用于渲染的MIDI文件。",
            'overwrite_warning': "所有渲染槽位已满。是否要覆盖版本 V{0}?",
            'model_load_fail_error': "模型加载失败，无法进行渲染。",
            # Progress Widget
            'progress_initializing': "初始化中...",
            'progress_loading_model': "正在加载模型...",
            'progress_rendering': "正在渲染...",
            'progress_cancelling': "正在取消...",
            # Render Worker Errors
            'render_cancelled_by_user': "渲染被用户取消。",
            'render_error': "渲染出错: {0}",
        }
    }

    def __init__(self, initial_lang='zh'):
        super().__init__()
        self._language = initial_lang

    def set_language(self, lang):
        if lang in self.STRINGS and self._language != lang:
            self._language = lang
            self.language_changed.emit()

    def tr(self, key):
        return self.STRINGS[self._language].get(key, f"<{key}>")

# --- PianoRollWidget (No changes needed) ---
class PianoRollWidget(QWidget):
    seek_requested = pyqtSignal(float)
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(60); self.notes = []; self.progress = 0.0
        self.min_pitch = 0; self.max_pitch = 127; self.setCursor(Qt.PointingHandCursor)
    def set_notes(self, midi_notes, total_duration_sec, min_pitch, max_pitch, tick_to_time_map):
        self.notes = []; self.min_pitch = min_pitch; self.max_pitch = max_pitch
        if not midi_notes or total_duration_sec == 0: self.update(); return
        for note in midi_notes:
            start_sec = tick_to_time_map[min(note.start, len(tick_to_time_map) - 1)]
            end_sec = tick_to_time_map[min(note.end, len(tick_to_time_map) - 1)]
            start_norm = start_sec / total_duration_sec
            duration_norm = (end_sec - start_sec) / total_duration_sec
            self.notes.append((start_norm, duration_norm, note.pitch))
        self.update()
    def set_progress(self, progress):
        self.progress = max(0.0, min(1.0, progress)); self.update()
    def mousePressEvent(self, event):
        if self.notes: self.seek_requested.emit(event.x() / self.width())
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#3498db"), 2)); painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        if not self.notes: return
        pitch_range = self.max_pitch - self.min_pitch + 1
        lane_height = self.height() / pitch_range if pitch_range > 0 else self.height()
        note_color = QColor("#1c2833"); painter.setPen(Qt.NoPen); painter.setBrush(note_color)
        for start_norm, duration_norm, pitch_value in self.notes:
            x = start_norm * self.width(); w = max(1, duration_norm * self.width())
            inverted_pitch_offset = self.max_pitch - pitch_value
            y = inverted_pitch_offset * lane_height; h = max(1, lane_height)
            painter.drawRect(QRectF(x, y, w, h))
        progress_x = self.progress * self.width()
        painter.setPen(QPen(QColor("#000000"), 2))
        painter.drawLine(int(progress_x), 0, int(progress_x), self.height())

# --- CircularProgressBar (No changes needed, but will be controlled by translated text) ---
class CircularProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.status_text = "Initializing..."
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(200, 200)

    def setValue(self, value):
        self.value = int(value)
        self.update()

    def setText(self, text):
        self.status_text = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setBrush(QColor(0, 0, 0, 120))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

        rect = QRectF(20, 20, 160, 160)
        
        pen = QPen(QColor("#566573"), 15, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        pen.setColor(QColor("#3498db"))
        painter.setPen(pen)
        span_angle = int(-self.value * 3.6 * 16)
        painter.drawArc(rect, 90 * 16, span_angle)

        painter.setPen(QColor("#FFFFFF"))
        
        font = QFont("Segoe UI", 30, QFont.Bold)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, f"{self.value}%")

        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        status_rect = rect.adjusted(0, 60, 0, 0)
        painter.drawText(status_rect, Qt.AlignCenter | Qt.AlignTop, self.status_text)

# --- RenderWorker (Modified to emit error keys instead of strings) ---
class RenderWorker(QObject):
    progress = pyqtSignal(float)
    finished = pyqtSignal(object)
    error = pyqtSignal(object) # Can emit a string (key) or a tuple (key, arg)

    def __init__(self, model, midi_obj, temp, top_p, device):
        super().__init__()
        self.model = model
        self.midi_obj = midi_obj
        self.temp = temp
        self.top_p = top_p
        self.device = device
        self._is_cancelled = False

    def run(self):
        try:
            def _report_progress(progress_float):
                if self._is_cancelled:
                    # MODIFIED: Use key for translation
                    raise InterruptedError("render_cancelled_by_user")
                self.progress.emit(progress_float)

            result = batch_performance_render(
                self.model,
                [self.midi_obj],
                temperature=self.temp,
                top_p=self.top_p,
                device=self.device,
                progress_callback=_report_progress
            )
            if not self._is_cancelled:
                self.finished.emit(result[0])
        except InterruptedError as e:
            self.error.emit(str(e)) # Emit the key
        except Exception as e:
            import traceback
            traceback.print_exc()
            # MODIFIED: Emit a tuple (key, argument) for translation
            self.error.emit(('render_error', str(e)))

    def cancel(self):
        self._is_cancelled = True

# --- 主窗口 ---
class AIPianistWindow(QWidget):
    
    NUM_RENDER_SLOTS = 5

    def __init__(self):
        super().__init__()
        # --- NEW: Initialize Language Manager ---
        self.lang_manager = LanguageManager()
        self.lang_manager.language_changed.connect(self._retranslate_ui)

        pygame.init(); pygame.mixer.init()
        self.original_midi_obj = None; self.current_midi_obj = None
        self.rendered_midis = [None] * self.NUM_RENDER_SLOTS
        self.next_render_slot_index = 0
        self.active_slot_index = -1
        self.is_demo_mode = False
        self.total_duration_sec = 0; self.current_seek_time_sec = 0
        self.temp_midi_path = None; self.is_paused = True
        self.model = None
        self.device = "cuda" if MODELS_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        # <<< FIX: Initialize these attributes *before* calling initUI() >>>
        self.render_thread = None
        self.render_worker = None
        
        self.demo_midi_paths = [
            "data/midis/testset/performance/0.mid",
            "data/midis/testset/performance/7.mid",
            "data/midis/testset/performance/14.mid",
            "data/midis/testset/performance/17.mid",
            "data/midis/testset/performance/21.mid",
        ]
        self.valid_demo_paths = [p for p in self.demo_midi_paths if os.path.exists(p)]
        if len(self.valid_demo_paths) != len(self.demo_midi_paths):
            print("Warning: Some demo MIDI paths are invalid and were ignored.")

        print(f"Using device: {self.device}")
        self.initUI() # Now it's safe to call this
        self.apply_stylesheet()
        self.playback_timer = QTimer(self); self.playback_timer.setInterval(50)
        self.playback_timer.timeout.connect(self.update_playback_progress)

        self.progress_widget = CircularProgressBar(self)
        self.progress_widget.hide()
        
        # --- NEW: Set initial text for progress widget ---
        self.progress_widget.setText(self.tr('progress_initializing'))

    # --- NEW: Shortcut for translation ---
    def tr(self, key):
        return self.lang_manager.tr(key)

    def initUI(self):
        # MODIFIED: Window title is now translatable
        self.setWindowTitle(self.tr("window_title"))
        self.setGeometry(300, 300, 850, 480)
        main_layout = QVBoxLayout(); main_layout.setContentsMargins(20, 20, 20, 20); main_layout.setSpacing(15)
        
        player_layout = self.create_player_bar()
        # MODIFIED: Store title label as instance variable to retranslate it
        self.title_label = QLabel()
        title_font = QFont("Script MT Bold", 48); title_font.setItalic(True)
        if "Script MT Bold" not in QFont().family(): title_font = QFont("Segoe Script", 48)
        self.title_label.setFont(title_font); self.title_label.setAlignment(Qt.AlignCenter); self.title_label.setObjectName("TitleLabel")
        
        loader_layout = self.create_file_loader()
        params_layout = self.create_param_sliders()
        
        bottom_controls_layout = QHBoxLayout()
        bottom_controls_layout.setSpacing(10)
        
        # MODIFIED: Use tr() for button text
        self.switch_to_original_btn = QPushButton()
        self.switch_to_original_btn.setCheckable(True); self.switch_to_original_btn.setObjectName("VersionButton")
        self.switch_to_original_btn.setEnabled(False); self.switch_to_original_btn.clicked.connect(self.activate_original_midi)
        bottom_controls_layout.addWidget(self.switch_to_original_btn)

        self.render_slot_btns = []
        for i in range(self.NUM_RENDER_SLOTS):
            # MODIFIED: Use tr() for button text
            btn = QPushButton()
            btn.setCheckable(True); btn.setObjectName("VersionButton")
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, index=i: self.activate_rendered_midi(index))
            self.render_slot_btns.append(btn)
            bottom_controls_layout.addWidget(btn)
        
        bottom_controls_layout.addStretch(1)

        # MODIFIED: Use tr() for button text
        self.render_button = QPushButton(); self.render_button.setObjectName("ActionButton"); self.render_button.setFixedHeight(40)
        self.render_button.clicked.connect(self.start_or_cancel_rendering)
        self.save_render_button = QPushButton(); self.save_render_button.setFixedHeight(40)
        self.save_render_button.clicked.connect(self.save_rendered_midi); self.save_render_button.setEnabled(False)
        self.save_editable_button = QPushButton(); self.save_editable_button.setFixedHeight(40)
        self.save_editable_button.clicked.connect(self.save_editable_midi); self.save_editable_button.setEnabled(False)
        bottom_controls_layout.addWidget(self.render_button)
        bottom_controls_layout.addWidget(self.save_render_button)
        bottom_controls_layout.addWidget(self.save_editable_button)
        
        main_layout.addLayout(player_layout)
        main_layout.addWidget(self.title_label)
        main_layout.addStretch(1)
        main_layout.addLayout(loader_layout)
        main_layout.addLayout(params_layout)
        main_layout.addLayout(bottom_controls_layout)
        self.setLayout(main_layout)

        # --- NEW: Call retranslate to set initial text for all widgets ---
        self._retranslate_ui()

    # --- NEW: Method to update all UI text ---
    def _retranslate_ui(self):
        self.setWindowTitle(self.tr("window_title"))
        self.title_label.setText(self.tr("main_title"))
        
        # Player bar
        self.demo_selector.setItemText(0, self.tr('select_demo'))
        
        # File loader
        self.filepath_label.setText(self.tr("file_path_label"))
        self.load_midi_btn.setText(self.tr("load_midi"))

        # Parameters
        self.temp_label.setText(self.tr("temperature"))
        self.topp_label.setText(self.tr("top_p"))

        # Bottom controls
        self.switch_to_original_btn.setText(self.tr("original_score"))
        for i, btn in enumerate(self.render_slot_btns):
            btn.setText(self.tr("version_button").format(i + 1))
        
        self._update_render_button_text() # Special handling for stateful button
        self.save_render_button.setText(self.tr("save_render"))
        self.save_editable_button.setText(self.tr("save_editable"))

    # --- NEW: Helper to manage the render button's stateful text ---
    def _update_render_button_text(self):
        if self.render_thread and self.render_thread.isRunning():
            self.render_button.setText(self.tr("cancel_render"))
        else:
            has_renders = any(s is not None for s in self.rendered_midis)
            if self.original_midi_obj and has_renders:
                self.render_button.setText(self.tr("render_again"))
            else:
                self.render_button.setText(self.tr("start_render"))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.progress_widget.isVisible():
            self.progress_widget.move(
                (self.width() - self.progress_widget.width()) // 2,
                (self.height() - self.progress_widget.height()) // 2
            )

    def start_or_cancel_rendering(self):
        if self.render_thread and self.render_thread.isRunning():
            self._cancel_rendering()
        else:
            self._start_rendering()

    def _start_rendering(self):
        if not self.original_midi_obj:
            # MODIFIED: Use tr() for message box
            QMessageBox.warning(self, self.tr("warning_title"), self.tr("no_midi_loaded_warning"))
            return
        
        target_slot = self.next_render_slot_index
        is_overwrite = target_slot >= self.NUM_RENDER_SLOTS
        actual_slot = target_slot % self.NUM_RENDER_SLOTS
        if is_overwrite:
            # MODIFIED: Use tr() for message box
            reply = QMessageBox.question(self, self.tr('overwrite_title'), 
                self.tr('overwrite_warning').format(actual_slot + 1),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        if self.model is None:
            # MODIFIED: Use tr() for progress text
            self.progress_widget.setText(self.tr("progress_loading_model"))
            self.progress_widget.setValue(0)
            self.progress_widget.show()
            self.progress_widget.raise_()
            self.resizeEvent(None)
            QApplication.processEvents()
            self._initialize_model()
            if self.model is None:
                # MODIFIED: Use tr() for message box
                QMessageBox.critical(self, self.tr("error_title"), self.tr("model_load_fail_error"))
                self.progress_widget.hide()
                return

        self._set_render_controls_enabled(False)
        # MODIFIED: Use tr() for button text
        self.render_button.setText(self.tr("cancel_render"))

        # MODIFIED: Use tr() for progress text
        self.progress_widget.setText(self.tr("progress_rendering"))
        self.progress_widget.setValue(0)
        self.progress_widget.show()
        self.progress_widget.raise_()
        self.resizeEvent(None)

        self.render_thread = QThread()
        self.render_worker = RenderWorker(
            self.model,
            self.original_midi_obj,
            float(self.temp_value_label.text()),
            float(self.topp_value_label.text()),
            self.device
        )
        self.render_worker.moveToThread(self.render_thread)

        self.render_thread.started.connect(self.render_worker.run)
        self.render_worker.finished.connect(self._on_rendering_finished)
        self.render_worker.error.connect(self._on_rendering_error)
        self.render_worker.progress.connect(self._update_render_progress)
        
        self.render_worker.finished.connect(self.render_thread.quit)
        self.render_worker.error.connect(self.render_thread.quit)
        self.render_thread.finished.connect(self.render_worker.deleteLater)
        self.render_thread.finished.connect(self.render_thread.deleteLater)
        self.render_thread.finished.connect(self._on_thread_finished)

        self.render_thread.start()

    def _update_render_progress(self, progress_float):
        self.progress_widget.setValue(progress_float * 100)

    def _on_rendering_finished(self, rendered_midi):
        print("渲染成功完成!")
        self.progress_widget.hide()
        
        # 步骤 1: 先更新数据状态
        actual_slot = (self.next_render_slot_index) % self.NUM_RENDER_SLOTS
        self.rendered_midis[actual_slot] = rendered_midi
        self.next_render_slot_index += 1
        
        # 步骤 2: 激活新的MIDI版本。这个函数会调用 _update_ui_states()，
        # 它现在可以读取到更新后的数据，从而正确设置所有按钮的状态和文本。
        self.activate_rendered_midi(actual_slot)
        
        # 步骤 3: 最后再重新启用所有相关控件。
        # 这里的_update_ui_states()调用会再次确认UI状态，确保万无一失。
        self._set_render_controls_enabled(True)
        
        # 注意：我们不再需要在末尾单独调用 self._update_render_button_text()，
        # 因为 activate_rendered_midi 和 _set_render_controls_enabled 内部
        # 已经通过调用 _update_ui_states() 把它包含了。

    def _on_rendering_error(self, error_data):
        # MODIFIED: Handle translated error messages from worker
        if isinstance(error_data, tuple):
            key, arg = error_data
            message = self.tr(key).format(arg)
        else:
            message = self.tr(error_data)

        print(f"Render error or cancellation: {message}")
        self.progress_widget.hide()
        self._set_render_controls_enabled(True)
        QMessageBox.warning(self, self.tr("render_notice_title"), message)
        
        # MODIFIED: Use helper to set correct translated text
        self._update_render_button_text()

    def _on_thread_finished(self):
        print("渲染线程已结束。")
        self.render_thread = None
        self.render_worker = None

    def _cancel_rendering(self):
        if self.render_worker:
            print("正在请求取消渲染...")
            self.render_worker.cancel()
            # MODIFIED: Use tr() for progress text
            self.progress_widget.setText(self.tr("progress_cancelling"))

    def _set_render_controls_enabled(self, enabled):
        self.load_midi_btn.setEnabled(enabled)
        self.temp_slider.setEnabled(enabled)
        self.topp_slider.setEnabled(enabled)
        self.save_render_button.setEnabled(enabled)
        self.save_editable_button.setEnabled(enabled)
        
        if enabled:
            self._update_ui_states()

    def create_player_bar(self):
        layout = QHBoxLayout()
        self.rewind_btn = QPushButton(); self.rewind_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.rewind_btn.clicked.connect(self.rewind_music)
        self.play_btn = QPushButton(); self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setCheckable(True); self.play_btn.clicked.connect(self.toggle_playback)
        self.piano_roll = PianoRollWidget(); self.piano_roll.seek_requested.connect(self.seek_music)
        self.time_label = QLabel("00:00/00:00"); self.time_label.setObjectName("InfoLabel")
        
        self.demo_selector = QComboBox()
        self.demo_selector.setObjectName("DemoSelector")
        # MODIFIED: Use tr() for placeholder text
        self.demo_selector.addItem(self.tr('select_demo'))
        for path in self.valid_demo_paths:
            self.demo_selector.addItem(os.path.basename(path))
        self.demo_selector.currentIndexChanged.connect(self.on_demo_selected)
        
        # --- NEW: Language Switcher ---
        self.lang_selector = QComboBox()
        self.lang_selector.addItem("中文")
        self.lang_selector.addItem("English")
        self.lang_selector.currentIndexChanged.connect(self.on_language_selected)
        
        layout.addWidget(self.rewind_btn); layout.addWidget(self.play_btn); layout.addWidget(self.piano_roll, 1);
        layout.addWidget(self.time_label); layout.addWidget(self.demo_selector);
        layout.addWidget(self.lang_selector) # Add to layout
        
        return layout

    # --- NEW: Slot for language selector ---
    def on_language_selected(self, index):
        if index == 0:
            self.lang_manager.set_language('zh')
        elif index == 1:
            self.lang_manager.set_language('en')

    def on_demo_selected(self, index):
        if index == 0: return

        filepath = self.valid_demo_paths[index - 1]
        print(f"Loading demo for listening: {filepath}")
        try:
            demo_midi = miditoolkit.MidiFile(filepath)
            self._load_midi_into_player(demo_midi)
            self.is_demo_mode = True
            self._update_ui_states()
        except Exception as e:
            print(f"无法加载或解析DEMO MIDI文件: {e}")
            self.demo_selector.blockSignals(True)
            self.demo_selector.setCurrentIndex(0)
            self.demo_selector.blockSignals(False)
            self._update_ui_states()

    def create_file_loader(self):
        layout = QHBoxLayout()
        # MODIFIED: Use tr() for label text
        self.filepath_label = QLabel()
        self.filepath_label.setObjectName("FilePathLabel")
        # MODIFIED: Store button as instance variable to retranslate it
        self.load_midi_btn = QPushButton()
        self.load_midi_btn.setObjectName("load_btn")
        self.load_midi_btn.clicked.connect(self.load_midi_file)
        layout.addWidget(self.filepath_label, 1)
        layout.addWidget(self.load_midi_btn)
        return layout

    def create_param_sliders(self):
        layout = QHBoxLayout(); layout.setSpacing(15)
        # MODIFIED: Store labels as instance variables to retranslate them
        self.temp_label = QLabel()
        self.temp_slider = QSlider(Qt.Horizontal); self.temp_slider.setRange(0, 200); self.temp_slider.setValue(100)
        self.temp_value_label = QLabel("1.00"); self.temp_slider.valueChanged.connect(lambda val: self.temp_value_label.setText(f"{val/100.0:.2f}"))
        self.topp_label = QLabel()
        self.topp_slider = QSlider(Qt.Horizontal); self.topp_slider.setRange(0, 100); self.topp_slider.setValue(95)
        self.topp_value_label = QLabel("0.95"); self.topp_slider.valueChanged.connect(lambda val: self.topp_value_label.setText(f"{val/100.0:.2f}"))
        for widget in [self.temp_label, self.temp_slider, self.temp_value_label, self.topp_label, self.topp_slider, self.topp_value_label]: layout.addWidget(widget)
        layout.setStretch(1, 1); layout.setStretch(4, 1); return layout

    def format_time(self, current_sec, total_sec):
        curr_min, curr_sec_rem = divmod(current_sec, 60); total_min, total_sec_rem = divmod(total_sec, 60)
        return f"{int(curr_min):02d}:{int(curr_sec_rem):02d}/{int(total_min):02d}:{int(total_sec_rem):02d}"

    def _initialize_model(self):
        if not MODELS_AVAILABLE:
            print("Cannot initialize model, libraries not found.")
            self.model = None
            return
            
        print("Initializing model for the first time...")
        try:
            self.model = PianoT5Gemma.from_pretrained(
                "models/sft/",
                torch_dtype=torch.bfloat16
            )
            self.model.to(self.device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def save_rendered_midi(self):
        if not self.current_midi_obj or self.active_slot_index == -1 or self.is_demo_mode:
            print("No rendered MIDI active to save.")
            return
        options = QFileDialog.Options()
        default_filename = f"rendered_V{self.active_slot_index + 1}.mid"
        # MODIFIED: Use tr() for dialog title
        filePath, _ = QFileDialog.getSaveFileName(self, self.tr("save_render_dialog_title"), default_filename, "MIDI Files (*.mid *.midi)", options=options)
        if filePath:
            try:
                self.current_midi_obj.dump(filePath)
                print(f"Rendered MIDI saved to {filePath}")
            except Exception as e:
                print(f"Error saving MIDI file: {e}")

    def save_editable_midi(self):
        if not self.original_midi_obj or self.active_slot_index == -1 or self.is_demo_mode:
            print("No active render to map back to the original MIDI.")
            return
        
        active_rendered_midi = self.rendered_midis[self.active_slot_index]
        if not active_rendered_midi:
            print("Error: Active slot has no data.")
            return

        options = QFileDialog.Options()
        default_filename = f"editable_V{self.active_slot_index + 1}.mid"
        # MODIFIED: Use tr() for dialog title
        filePath, _ = QFileDialog.getSaveFileName(self, self.tr("save_editable_dialog_title"), default_filename, "MIDI Files (*.mid *.midi)", options=options)
        if filePath:
            try:
                mapped_midi = map_midi(self.original_midi_obj, active_rendered_midi)
                mapped_midi.dump(filePath)
                print(f"Editable MIDI saved to {filePath}")
            except Exception as e:
                print(f"Error saving MIDI file: {e}")

    def _load_midi_into_player(self, midi_to_load):
        self.reset_playback()
        self.current_midi_obj = midi_to_load
        if not self.current_midi_obj:
            self.piano_roll.set_notes([], 0, 0, 0, [])
            return
        all_notes = [note for inst in self.current_midi_obj.instruments for note in inst.notes]
        if not all_notes:
            self.total_duration_sec = 0
            self.piano_roll.set_notes([], 0, 0, 0, [])
        else:
            all_notes.sort(key=lambda x: x.start)
            min_pitch = min(note.pitch for note in all_notes); max_pitch = max(note.pitch for note in all_notes)
            tick_to_time_map = self.current_midi_obj.get_tick_to_time_mapping()
            self.total_duration_sec = tick_to_time_map[-1]
            self.piano_roll.set_notes(all_notes, self.total_duration_sec, min_pitch, max_pitch, tick_to_time_map)
        self.time_label.setText(self.format_time(0, self.total_duration_sec))

    def load_midi_file(self):
        options = QFileDialog.Options()
        # MODIFIED: Use tr() for dialog title
        filepath, _ = QFileDialog.getOpenFileName(self, self.tr("select_midi_dialog_title"), "", "MIDI Files (*.mid *.midi)", options=options)
        if not filepath: return
        try:
            self.reset_playback()
            self.is_demo_mode = False
            
            self.original_midi_obj = miditoolkit.MidiFile(filepath)
            self.rendered_midis = [None] * self.NUM_RENDER_SLOTS
            self.next_render_slot_index = 0
            
            self.activate_original_midi()

            # MODIFIED: Use helper to set correct translated text
            self._update_render_button_text()
            self.filepath_label.setText(os.path.basename(filepath)) # Show filename instead of full path
            
        except Exception as e:
            print(f"无法加载或解析MIDI文件: {e}")
            self.original_midi_obj = None
            self._load_midi_into_player(None)
            self._update_ui_states()
    
    def activate_original_midi(self):
        if not self.original_midi_obj:
            return
        
        self.is_demo_mode = False
        self.active_slot_index = -1
        self._load_midi_into_player(copy.deepcopy(self.original_midi_obj))
        self._update_ui_states()

    def activate_rendered_midi(self, index):
        if index < 0 or index >= self.NUM_RENDER_SLOTS or self.rendered_midis[index] is None:
            self._update_ui_states()
            return

        self.is_demo_mode = False
        self.active_slot_index = index
        self._load_midi_into_player(copy.deepcopy(self.rendered_midis[index]))
        self._update_ui_states()

    def _update_ui_states(self):
        has_original = self.original_midi_obj is not None

        if self.is_demo_mode:
            self.switch_to_original_btn.setEnabled(has_original)
            self.switch_to_original_btn.setChecked(False)
            for i, btn in enumerate(self.render_slot_btns):
                btn.setEnabled(self.rendered_midis[i] is not None)
                btn.setChecked(False)
            self.save_render_button.setEnabled(False)
            self.save_editable_button.setEnabled(False)
        else:
            if self.demo_selector.currentIndex() != 0:
                self.demo_selector.blockSignals(True)
                self.demo_selector.setCurrentIndex(0)
                self.demo_selector.blockSignals(False)

            self.switch_to_original_btn.setEnabled(has_original)
            self.switch_to_original_btn.setChecked(self.active_slot_index == -1 and has_original)
            for i, btn in enumerate(self.render_slot_btns):
                btn.setEnabled(self.rendered_midis[i] is not None)
                btn.setChecked(self.active_slot_index == i)
            is_render_active = self.active_slot_index != -1
            self.save_render_button.setEnabled(is_render_active)
            self.save_editable_button.setEnabled(is_render_active)
        
        self.render_button.setEnabled(has_original)
        self._update_render_button_text()


    def toggle_playback(self, checked):
        if not self.current_midi_obj: self.play_btn.setChecked(False); return
        if checked:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            progress = self.current_seek_time_sec / self.total_duration_sec if self.total_duration_sec > 0 else 0
            self.seek_music(progress, is_resume=self.is_paused)
        else:
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            if pygame.mixer.music.get_busy():
                elapsed_ms = pygame.mixer.music.get_pos()
                self.current_seek_time_sec += elapsed_ms / 1000.0
            pygame.mixer.music.stop(); self.is_paused = True; self.playback_timer.stop()

    def seek_music(self, progress, is_resume=False):
        if not self.current_midi_obj: return
        pygame.mixer.music.stop(); self._cleanup_temp_file()
        if not is_resume: self.current_seek_time_sec = self.total_duration_sec * progress
        tick_to_time = self.current_midi_obj.get_tick_to_time_mapping()
        target_tick = bisect.bisect_left(tick_to_time, self.current_seek_time_sec)
        new_midi = copy.deepcopy(self.current_midi_obj)
        last_tempo = None
        for tempo_event in self.current_midi_obj.tempo_changes:
            if tempo_event.time <= target_tick: last_tempo = copy.deepcopy(tempo_event)
            else: break
        kept_tempos = [t for t in new_midi.tempo_changes if t.time >= target_tick]
        if last_tempo:
            last_tempo.time = 0; kept_tempos.insert(0, last_tempo)
        for t in kept_tempos:
            if t.time > 0: t.time -= target_tick
        new_midi.tempo_changes = kept_tempos
        for instrument in new_midi.instruments:
            instrument.notes = [n for n in instrument.notes if n.start >= target_tick]
            for note in instrument.notes: note.start -= target_tick; note.end -= target_tick
            instrument.control_changes = [c for c in instrument.control_changes if c.time >= target_tick]
            for cc in instrument.control_changes: cc.time -= target_tick
            instrument.pitch_bends = [p for p in instrument.pitch_bends if p.time >= target_tick]
            for pb in instrument.pitch_bends: pb.time -= target_tick
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
            self.temp_midi_path = tf.name; new_midi.dump(tf.name)
        pygame.mixer.music.load(self.temp_midi_path); pygame.mixer.music.play()
        self.is_paused = False
        if not self.play_btn.isChecked(): self.play_btn.setChecked(True)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        if not self.playback_timer.isActive(): self.playback_timer.start()
        self.update_playback_progress()

    def update_playback_progress(self):
        if pygame.mixer.music.get_busy():
            elapsed_ms = pygame.mixer.music.get_pos()
            current_total_sec = self.current_seek_time_sec + (elapsed_ms / 1000.0)
            if current_total_sec < self.total_duration_sec:
                progress = current_total_sec / self.total_duration_sec
                self.piano_roll.set_progress(progress)
                self.time_label.setText(self.format_time(current_total_sec, self.total_duration_sec))
            else: self.reset_playback()
        elif not self.is_paused and self.play_btn.isChecked(): self.reset_playback()

    def rewind_music(self):
        if not self.current_midi_obj: return
        was_playing = not self.is_paused
        pygame.mixer.music.stop(); self.playback_timer.stop()
        self.current_seek_time_sec = 0
        self.piano_roll.set_progress(0.0)
        self.time_label.setText(self.format_time(0, self.total_duration_sec))
        if was_playing: self.seek_music(0)
        else:
            self.is_paused = True; self.play_btn.setChecked(False)
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def reset_playback(self):
        pygame.mixer.music.stop(); self.playback_timer.stop(); self._cleanup_temp_file()
        self.is_paused = True; self.current_seek_time_sec = 0
        self.play_btn.setChecked(False); self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.piano_roll.set_progress(0.0)
        duration = self.total_duration_sec if self.current_midi_obj else 0
        self.time_label.setText(self.format_time(0, duration))

    def _cleanup_temp_file(self):
        if self.temp_midi_path and os.path.exists(self.temp_midi_path):
            try: os.remove(self.temp_midi_path)
            except OSError as e: print(f"Error removing temp file {self.temp_midi_path}: {e}")
            self.temp_midi_path = None
    
    def apply_stylesheet(self):
        style = """
            QWidget { 
                background-color: #f0f0f0; color: #333; 
                font-family: 'Segoe UI', 'Microsoft YaHei', 'Arial', sans-serif; 
                font-size: 14px; 
            }
            #TitleLabel { color: #1c2833; }
            #InfoLabel, #FileIcon { font-size: 13px; color: #555; }
            #FileIcon { color: #3498db; font-weight: bold; padding-bottom: 3px; }
            
            QPushButton { 
                background-color: #3498db; color: white; border: none; 
                padding: 10px 20px; border-radius: 5px; 
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f618d; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
            
            QPushButton:!checkable { padding: 5px; min-width: 40px; }
            
            QPushButton#ActionButton { 
                font-size: 16px; font-weight: bold; background-color: #27ae60; 
            }
            QPushButton#ActionButton:hover { background-color: #229954; }
            QPushButton#ActionButton:disabled { background-color: #95a5a6; }
            
            QPushButton#VersionButton {
                background-color: #8d99ae; padding: 8px 12px; font-size: 13px;
                min-width: 60px;
            }
            QPushButton#VersionButton:hover { background-color: #7b889d; }
            QPushButton#VersionButton:disabled {
                background-color: #e9ecef; color: #adb5bd;
                border: 1px solid #ced4da;
            }
            QPushButton#VersionButton:checked {
                background-color: #e67e22; border: 2px solid #c85e00;
                font-weight: bold;
            }
            QPushButton#VersionButton:checked:hover { background-color: #d35400; }
            
            #FilePathLabel { 
                background-color: white; border: 1px solid #ccc; border-radius: 5px; 
                padding: 8px; color: #888; 
            }
            
            QComboBox {
                border: 1px solid #ccc; border-radius: 4px; padding: 5px;
                background-color: white; min-width: 100px; /* Adjusted min-width */
            }
            QComboBox#DemoSelector { min-width: 150px; }
            QComboBox:hover { border-color: #3498db; }
            QComboBox::drop-down {
                subcontrol-origin: padding; subcontrol-position: top right;
                width: 20px; border-left-width: 1px;
                border-left-color: #ccc; border-left-style: solid;
                border-top-right-radius: 3px; border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow { image: url(some_path); }
            
            QSlider::groove:horizontal { 
                border: 1px solid #bbb; background: white; 
                height: 8px; border-radius: 4px; 
            }
            QSlider::sub-page:horizontal { 
                background: #3498db; border: 1px solid #3498db; 
                height: 10px; border-radius: 4px; 
            }
            QSlider::handle:horizontal { 
                background: #3498db; border: 2px solid white; 
                width: 18px; height: 18px; margin: -6px 0; border-radius: 9px; 
            }
        """
        self.setStyleSheet(style)

    def closeEvent(self, event):
        self._cleanup_temp_file(); pygame.quit(); event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AIPianistWindow()
    window.show()
    sys.exit(app.exec_())