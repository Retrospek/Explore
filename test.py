import numpy as np

# Given values
g = 9.81  # Acceleration due to gravity (m/s^2)
m1 = 0.1  # kg
m2 = 0.05  # kg
dm1 = 0.001  # kg
dm2 = 0.001  # kg

# Compute partial derivatives
da_dm1 = g * (2 * m2) / (m1 + m2) ** 2
da_dm2 = g * (-2 * m1) / (m1 + m2) ** 2

# Compute uncertainty in acceleration
da = np.sqrt((da_dm1 * dm1) ** 2 + (da_dm2 * dm2) ** 2)
print(da)