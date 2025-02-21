import sys
import warp as wp
from PyQt5.QtWidgets import QApplication
import numpy as np

from backend_sph import SPHSimulation2D_Warp
from visualization import SPHViewer2D

def main():
   
    wp.init() 
    
    app = QApplication(sys.argv)

    sph_sim = SPHSimulation2D_Warp(
        num_particles=1600,
        box_min=np.array([0.0, 0.0]),
        box_max=np.array([20.0, 10.0])
    )

    viewer = SPHViewer2D(sph_sim)
    viewer.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
