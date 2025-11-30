import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QWidget, QVBoxLayout
import numpy as np

# Define Dark Theme Colors
BG_COLOR = '#1a1a1a'
AXES_BG_COLOR = '#2b2b2b'
TEXT_COLOR = '#f0f0f0'
CPU_COLOR = '#00bcd4'     # Vibrant Cyan
MEM_COLOR = '#a8e61e'     # Lime Green

class MultiMplCanvas(FigureCanvas):
    def __init__(self, x_len=100, parent=None):
        # 🔑 FIX: Changed subplots to (2, 1) to remove the third graph (Temperature)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), facecolor=BG_COLOR, edgecolor=BG_COLOR, tight_layout=True)
        
        # Manually set spacing to ensure no overlap
        fig.subplots_adjust(hspace=0.5) 
        
        super().__init__(fig)
        
        self.x_len = x_len
        self.x_data = list(range(x_len))
        # 🔑 FIX: Removed 'temperature' from data structures
        self.data = {'cpu': [0] * x_len, 'memory': [0] * x_len} 
        
        # 🔑 FIX: Axes now only holds two references
        self.ax_cpu, self.ax_mem = axes
        
        plot_configs = [
            (self.ax_cpu, self.data['cpu'], CPU_COLOR, "CPU Utilization (%)", 0, 100),
            (self.ax_mem, self.data['memory'], MEM_COLOR, "Memory Utilization (%)", 0, 100),
            # Temperature plot definition is removed
        ]

        self._plot_refs = []
        for ax, data_list, color, title, y_min, y_max in plot_configs:
            ax.set_facecolor(AXES_BG_COLOR)
            
            # Set titles, labels, and ticks to match dark theme
            ax.set_title(title, color=TEXT_COLOR, fontsize=10)
            ax.set_xlabel("Time (Samples)", color=TEXT_COLOR, fontsize=8) 
            ax.set_ylim(y_min, y_max)
            
            # Set tick and spine colors
            ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=8)
            ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=8)
            for spine in ax.spines.values():
                 spine.set_color(TEXT_COLOR)
            
            line_ref, = ax.plot(self.x_data, data_list, color=color, linewidth=2.5)
            self._plot_refs.append(line_ref)

        self.draw()

    def update_plot(self, new_data: dict):
        """Updates the CPU and Memory plots."""
        # 🔑 FIX: Only track CPU and Memory keys
        data_keys = ['cpu', 'memory'] 
        
        for i, key in enumerate(data_keys):
            if key in new_data:
                new_value = new_data[key]
                if isinstance(new_value, (int, float)):
                    self.data[key] = self.data[key][1:] + [new_value]
                    self._plot_refs[i].set_ydata(self.data[key])
        
        self.draw()

class MplWidget(QWidget):
    """Container widget for the MultiMatplotlib Canvas."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MultiMplCanvas() 
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)