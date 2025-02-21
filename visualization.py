import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush

class SPHViewer2D(QWidget):
    def __init__(self, sph_simulation, parent=None):
        super().__init__(parent)
        self.sph_simulation = sph_simulation
        self.setWindowTitle("2D SPH Visualization (Warp Accelerated)")
        self.resize(1600, 800)

        # 设置一个定时器来刷新模拟
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(16)  

    def update_simulation(self):
        self.sph_simulation.sph_simulation_step()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        pen = QPen(QColor(0, 0, 255))
        painter.setPen(pen)
        brush = QBrush(QColor(0, 0, 255))
        painter.setBrush(brush)

        width = self.width()
        height = self.height()

        box_min = self.sph_simulation.box_min
        box_max = self.sph_simulation.box_max
        size = box_max - box_min  

        # map the particle positions to the screen
        for pos in self.sph_simulation.positions_cpu:
            x_ratio = (pos[0] - box_min[0]) / size[0]
            y_ratio = (pos[1] - box_min[1]) / size[1]

            px = x_ratio * width
            py = height - y_ratio * height  # y向上

            radius = 5
            painter.drawEllipse(int(px - radius/2),
                                int(py - radius/2),
                                radius, radius)
