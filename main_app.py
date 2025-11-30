import sys
import os
import multiprocessing
import psutil
import platform 
# You need to ensure wmi is installed in your Python environment for this to work
# pip install wmi
try:
    import wmi 
except ImportError:
    wmi = None # Set wmi to None if not found, to prevent crash later

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QLineEdit, QTabWidget, QTextEdit, 
    QMessageBox, QFileDialog, QScrollArea, QGridLayout, QFrame,
    QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Slot

from matplotlib_canvas import MplWidget
from benchmarker_worker import SystemMonitor, BenchmarkWorker
import csv
import json
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPU Benchmark Tool")
        self.setGeometry(100, 100, 1000, 750) 

        # --- 1. Data and Thread Setup ---
        self.benchmark_history = []
        self.benchmark_thread = QThread()
        self.monitor_thread = QThread()
        
        self.monitor = SystemMonitor()
        self.monitor.moveToThread(self.monitor_thread)
        self.monitor.data_ready.connect(self.update_live_graph)
        self.monitor_thread.start()
        self.monitor.start_monitoring()

        # --- 2. Central Layout (The Card View) ---
        main_container = QWidget()
        self.setCentralWidget(main_container)
        
        outer_layout = QHBoxLayout(main_container)
        outer_layout.addStretch(1) 
        
        self.central_card = QFrame()
        self.central_card.setObjectName("CentralCard")
        self.central_card.setMinimumWidth(800)
        self.central_card.setMaximumWidth(850)
        
        outer_layout.addWidget(self.central_card)
        outer_layout.addStretch(1) 
        
        self.card_layout = QVBoxLayout(self.central_card)
        
        # --- 3. TAB WIDGET SETUP ---
        
        # 🔑 FIX 1: Set up Header first (top of the card)
        self._setup_header() 
        
        # 🔑 FIX 2: Set up System Info second (just below header)
        self._setup_system_info()
        
        # Tabs will now occupy the remaining space below the system info
        self.tab_widget = QTabWidget()
        self.card_layout.addWidget(self.tab_widget)
        
        # Build Tabs
        self.tab_widget.addTab(self._create_monitoring_tab(), "Monitor & Controls")
        self.tab_widget.addTab(self._create_results_tab(), "History & Reports")
        
        
    def _create_monitoring_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # The tab contents are now only controls and graphs
        self._setup_controls(layout)
        self._setup_graph(layout)
        
        return tab

    def _setup_system_info(self):
        # Section Title
        section_title = QLabel("🧠 System Information")
        section_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px; border-bottom: 2px solid #3e3e3e; padding-bottom: 5px;")
        self.card_layout.addWidget(section_title)
        
        # Data Retrieval
        cpu_freq = psutil.cpu_freq()
        
        info_grid = QGridLayout()
        info_grid.setSpacing(10)
        
        def add_spec(row, col, label, value, color_value=True):
            label_widget = QLabel(label + ":")
            label_widget.setStyleSheet("color: #cccccc;")
            value_widget = QLabel(value)
            if color_value:
                value_widget.setStyleSheet("color: #00bcd4; font-weight: bold;")
            
            info_grid.addWidget(label_widget, row, col)
            info_grid.addWidget(value_widget, row, col + 1)

        # 🔑 CRITICAL FIX: Attempt to get actual CPU name using WMI
        processor_name = "Unknown Processor (Run as Admin)"
        if platform.system() == "Windows" and wmi is not None:
            try:
                c = wmi.WMI()
                # Query the dedicated CPU class and get the name property
                processor_name = c.Win32_Processor()[0].Name.strip()
            except Exception:
                processor_name = f"Error reading CPU (Arch: {platform.machine()})"

        # Specs are stored as attributes for potential future updates
        add_spec(0, 0, "Processor", processor_name, False) # Now shows real name
        add_spec(0, 2, "Cores (Logical)", str(psutil.cpu_count(logical=True)))
        add_spec(1, 0, "Clock Speed (Current)", f"{cpu_freq.current/1000:.2f} GHz")
        add_spec(1, 2, "Architecture", platform.machine(), False)
        add_spec(2, 0, "Max Clock Speed", f"{cpu_freq.max/1000:.2f} GHz")
        add_spec(2, 2, "Memory (Total)", f"{psutil.virtual_memory().total / (1024**3):.1f} GB", False)

        self.card_layout.addLayout(info_grid)

    def _setup_controls(self, parent_layout):
        # Configuration Controls
        config_group = QFrame()
        config_group.setObjectName("ConfigGroup")
        config_layout = QHBoxLayout(config_group)
        
        # Test Type
        config_layout.addWidget(QLabel("Test:"))
        self.test_type_combo = QComboBox()
        self.test_type_combo.addItems(['integer', 'floating', 'cache', 'memory', 'branch', 'full']) 
        config_layout.addWidget(self.test_type_combo)

        # Cores
        config_layout.addWidget(QLabel("Cores:"))
        self.cores_input = QLineEdit(str(psutil.cpu_count(logical=True)))
        self.cores_input.setMaximumWidth(50)
        config_layout.addWidget(self.cores_input)

        # Duration
        config_layout.addWidget(QLabel("Duration (s):"))
        self.duration_input = QLineEdit("5")
        self.duration_input.setMaximumWidth(50)
        config_layout.addWidget(self.duration_input)
        
        # Spacer
        config_layout.addStretch(1)
        
        # Button
        self.start_button = QPushButton("🚀 Start Benchmark")
        self.start_button.clicked.connect(self.start_benchmark)
        self.start_button.setObjectName("StartButton")
        config_layout.addWidget(self.start_button)
        
        parent_layout.addWidget(config_group)

    def _setup_graph(self, parent_layout):
        # Live Graph Section
        graph_title = QLabel("📊 Real-Time Metrics (CPU, Memory)")
        graph_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px; border-bottom: 2px solid #3e3e3e; padding-bottom: 5px;")
        parent_layout.addWidget(graph_title)
        
        self.graph_widget = MplWidget()
        self.graph_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        parent_layout.addWidget(self.graph_widget)
        
        parent_layout.addStretch(1) 

    def _create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results Log Area
        results_title = QLabel("📋 Benchmark History & Reports")
        results_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px; border-bottom: 2px solid #3e3e3e; padding-bottom: 5px;")
        layout.addWidget(results_title)
        
        self.results_display = QTextEdit() 
        self.results_display.setReadOnly(True)
        self.results_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.results_display)
        layout.addWidget(scroll)

        # Report Button (Separate layout for centering)
        report_button_layout = QHBoxLayout()
        report_button_layout.addStretch(1)
        self.report_button = QPushButton("📄 Export Report (JSON/CSV)")
        self.report_button.clicked.connect(self.generate_report)
        report_button_layout.addWidget(self.report_button)
        report_button_layout.addStretch(1)
        
        layout.addLayout(report_button_layout)
        
        return tab
    
    def _setup_header(self):
        # Title area mimicking the screenshot
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon (Placeholder using QLabel)
        cpu_icon = QLabel("⚙️") 
        cpu_icon.setObjectName("HeaderIcon")
        cpu_icon.setStyleSheet("font-size: 48px; color: #00bcd4;")
        cpu_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("CPU Benchmark Tool")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 5px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        
        subtitle = QLabel("Test your CPU's speed and efficiency through simple operations.")
        subtitle.setStyleSheet("color: #cccccc; margin-bottom: 20px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        
        header_layout.addWidget(cpu_icon)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        
        # Add header to the main card layout
        self.card_layout.addLayout(header_layout)

    # --- REPORT GENERATION METHODS (remain the same) ---

    def generate_report(self):
        """Saves the history of benchmark results to a CSV or JSON file."""
        if not self.benchmark_history:
            QMessageBox.warning(self, "No Data", "Run at least one benchmark before generating a report.")
            return

        default_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'benchmark_report')

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Benchmark Report", 
            default_path,
            "CSV Files (*.csv);;JSON Files (*.json)"
        )

        if not file_path:
            return 

        if file_path.lower().endswith('.csv') or "CSV Files" in selected_filter:
            self._save_to_csv(file_path)
        elif file_path.lower().endswith('.json') or "JSON Files" in selected_filter:
            self._save_to_json(file_path)
        else:
            QMessageBox.critical(self, "Error", "Unsupported file format selected.")

    def _save_to_csv(self, file_path):
        """Helper to save history to a CSV file."""
        try:
            fieldnames = list(self.benchmark_history[0].keys())
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.benchmark_history)
                
            QMessageBox.information(self, "Success", f"CSV Report saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save CSV report: {e}")

    def _save_to_json(self, file_path):
        """Helper to save history to a JSON file."""
        try:
            with open(file_path, 'w') as jsonfile:
                json.dump(self.benchmark_history, jsonfile, indent=4)
                
            QMessageBox.information(self, "Success", f"JSON Report saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON report: {e}")
            
    # --- SLOTS (Communication Handlers) ---
    
    @Slot(dict)
    def update_live_graph(self, system_data: dict):
        self.graph_widget.canvas.update_plot(system_data)
        
    @Slot(dict)
    def handle_benchmark_result(self, results):
        """Receives data from BenchmarkWorker and updates the results tab."""
        self.benchmark_thread.quit()
        self.benchmark_thread.wait()
        
        self.benchmark_history.append(results)
        
        output = f"\n========================================\n"
        output += f"TEST RUN: {results['test_type'].upper()} (Single)\n"
        output += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"========================================\n"
        output += f"Cores Used: {results['cores_used']}\n"
        output += f"Total Time: {results['total_time_s']} s\n"
        output += f"Total Operations: {results['total_ops']:,}\n"
        
        test_type = results['test_type']
        score = results['score_ops_sec']
        
        if test_type == 'memory':
            score_label = f"Bandwidth (MB/s): {score:,.2f}"
        elif test_type == 'cache':
            score_label = f"Latency (ns): {score:,.2f}"
        elif test_type == 'branch':
            score_label = f"Penalty Ratio (x1000): {score:,.2f}"
        else:
            score_label = f"Final Score (Ops/Sec): {score:,.2f}"
            
        output += f"{score_label}\n"
        output += f"CPU Frequency: {results['system_info']} MHz\n"
        
        self.results_display.append(output)
        
        self.start_button.setEnabled(True)
        QMessageBox.information(self, "Finished", f"'{results['test_type']}' completed successfully! ✅")

    def start_benchmark(self):
        """Initializes and starts the long-running benchmark thread."""
        try:
            test_config = {
                'test_type': self.test_type_combo.currentText(),
                'cores': int(self.cores_input.text()),
                'duration': float(self.duration_input.text()),
            }
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please ensure Cores and Duration are valid numbers.")
            return

        self.start_button.setEnabled(False)
        
        if test_config['test_type'] == 'full':
            self.run_full_benchmark_sequence()
            return

        # --- Single Test Logging ---\
        start_message = (
            f"\n--- Starting {test_config['test_type'].upper()} Test ---\n"
            f"Running on {test_config['cores']} cores for {test_config['duration']} seconds...\n"
        )
        
        self.results_display.append(start_message)
        
        self.benchmark_worker = BenchmarkWorker(test_config)
        self.benchmark_worker.moveToThread(self.benchmark_thread)

        self.benchmark_thread.started.connect(self.benchmark_worker.run_benchmark)
        self.benchmark_worker.result_ready.connect(self.handle_benchmark_result)
        
        self.benchmark_thread.start()

    def closeEvent(self, event):
        """Safely clean up threads when the application closes."""
        if self.monitor_thread.isRunning():
            self.monitor.stop_monitoring()
            self.monitor_thread.quit()
            self.monitor_thread.wait()
        
        if self.benchmark_thread.isRunning():
            self.benchmark_thread.quit()
            self.benchmark_thread.wait()

        super().closeEvent(event)
        
    def run_full_benchmark_sequence(self):
        """
        Runs the full sequence of microbenchmarks and collects a single report.
        This method will coordinate multiple worker runs sequentially.
        """
        self.full_test_list = ['integer', 'floating', 'cache', 'memory', 'branch']
        self.full_results_aggregate = []
        self.current_full_test_index = 0
        
        self.results_display.append("\n--- INITIATING FULL BENCHMARK SUITE ---")
        
        self._start_next_full_test()

    def _start_next_full_test(self):
        """Starts the next test in the sequence or finishes the full run."""
        if self.current_full_test_index >= len(self.full_test_list):
            self._finish_full_benchmark()
            return

        test_type = self.full_test_list[self.current_full_test_index]
        cores = int(self.cores_input.text())
        duration = float(self.duration_input.text())
        
        status_message = (
            f"Running Test {self.current_full_test_index + 1}/{len(self.full_test_list)}: "
            f"{test_type.upper()} on {cores} cores..."
        )
        self.results_display.append(status_message)

        test_config = {
            'test_type': test_type,
            'cores': cores,
            'duration': duration,
            'is_full_run': True
        }
        
        self.benchmark_worker = BenchmarkWorker(test_config)
        self.benchmark_worker.moveToThread(self.benchmark_thread)
        self.benchmark_thread.started.connect(self.benchmark_worker.run_benchmark)
        self.benchmark_worker.result_ready.connect(self._handle_full_test_step) 
        self.benchmark_thread.start()
        
    def _finish_full_benchmark(self):
        """Compiles and reports the final results of the full suite with conditional metrics."""
        self.benchmark_worker.result_ready.disconnect(self._handle_full_test_step)
        
        output = f"\n--- FULL BENCHMARK SUITE COMPLETE! ---\n\n"
        final_score = 0
        
        for result in self.full_results_aggregate:
            score = result['score_ops_sec']
            test_type = result['test_type']
            
            if test_type == 'memory':
                score_label = f"Bandwidth (MB/s): {score:,.2f}"
            elif test_type == 'cache':
                score_label = f"Latency (ns): {score:,.2f}"
            elif test_type == 'branch':
                score_label = f"Penalty Ratio (x1000): {score:,.2f}"
            else:
                score_label = f"Score (Ops/Sec): {score:,.2f}"
            
            output += f"Test: {test_type.upper()}\n"
            output += f"  {score_label}\n"
            output += f"  Time: {result['total_time_s']} s\n\n"
            
            if test_type in ('integer', 'floating'):
                final_score += score

        output += f"--- COMPOSITE SCORE (Arithmetic Only): {final_score:,.2f} ---\n"
        self.results_display.append(output)
        
        self.start_button.setEnabled(True)
        QMessageBox.information(self, "Finished", "Full Benchmark Suite completed! 🎉")


    @Slot(dict)
    def _handle_full_test_step(self, results):
        """Handles the result of a single test during the full sequence."""
        # 1. Store result
        self.full_results_aggregate.append(results)
        
        # 2. Cleanup worker thread
        self.benchmark_thread.quit()
        self.benchmark_thread.wait()
        
        # Log the completed step result
        score = results['score_ops_sec']
        test_type = results['test_type']
        
        # Log completed step status
        if test_type == 'memory':
            status_line = f"-> {test_type.upper()} DONE. Bandwidth: {score:,.0f} MB/s.\n"
        elif test_type == 'cache':
            status_line = f"-> {test_type.upper()} DONE. Latency: {score:,.1f} ns.\n"
        elif test_type == 'branch':
            status_line = f"-> {test_type.upper()} DONE. Penalty: {score:,.2f}x.\n"
        else:
            status_line = f"-> {test_type.upper()} DONE. Score: {score:,.0f} Ops/Sec.\n"

        self.results_display.append(status_line)
        
        # 3. Advance to next test
        self.current_full_test_index += 1
        self._start_next_full_test()


if __name__ == "__main__":
    multiprocessing.freeze_support() 
    app = QApplication(sys.argv)
    
    # --- Load External Style Sheet ---
    try:
        style_path = os.path.join(os.path.dirname(__file__), 'style.qss')
        with open(style_path, 'r') as f:
            qss = f.read()
        app.setStyleSheet(qss)
    except FileNotFoundError:
        print("ERROR: style.qss file not found. Running with default theme.")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())