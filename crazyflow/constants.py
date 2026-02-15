import numpy as np
from numpy.typing import NDArray

# Physical constants
GRAVITY: float = 9.81
ARM_LEN: float = 0.0325 * np.sqrt(2)
MIX_MATRIX: NDArray = np.array([[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]])
SIGN_MIX_MATRIX: NDArray = np.sign(MIX_MATRIX)
MASS: float = 0.035
J: NDArray = np.array([[2.3951e-5, 0, 0], [0, 2.3951e-5, 0], [0, 0, 3.2347e-5]])
J_INV: NDArray = np.linalg.inv(J)

# Setup Crazyflie Brushless Constants, obtained from CrazyflieBrushJax
# ARM_LEN: float = 0.050 #Exactly 50mm from center to motor center
# MIX_MATRIX: NDArray = np.array([[-0.5, -0.5, -1], [-0.5, 0.5, 1], [0.5, 0.5, -1], [0.5, -0.5, 1]])
# SIGN_MIX_MATRIX: NDArray = np.sign(MIX_MATRIX)
# MASS: float = 0.041
# J: NDArray = np.array([[3.4e-05, 0, 0], [0, 3.4e-05, 0], [0, 0, 5.8e-05]])
# J_INV: NDArray = np.linalg.inv(J)
