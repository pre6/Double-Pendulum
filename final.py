import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math


grid_size = 1000
G = 9.81   #gravitational acceleration
M = 1.0   #mass
L = 1.0   #length
dt = 0.08

def derivative(a1, a2, p1, p2):


    ml2 = M * L * L
    cos12 = np.cos(a1 - a2)
    sin12 = np.sin(a1 - a2)
    da1 = 6 / ml2 * (2 * p1 - 3 * cos12 * p2) / (16 - 9 * cos12 * cos12)
    da2 = 6 / ml2 * (8 * p2 - 3 * cos12 * p1) / (16 - 9 * cos12 * cos12)
    dp1 = ml2 / -2 * (+da1 * da2 * sin12 + 3 * G / L * np.sin(a1))
    dp2 = ml2 / -2 * (-da1 * da2 * sin12 + 3 * G / L * np.sin(a2))


    return [da1, da2, dp1, dp2]

def  rk4(k1a1, k1a2, k1p1, k1p2, dt):
    [k1da1, k1da2, k1dp1, k1dp2] = derivative(k1a1, k1a2, k1p1, k1p2)

    k2a1 = k1a1 + k1da1 * dt / 2
    k2a2 = k1a2 + k1da2 * dt / 2
    k2p1 = k1p1 + k1dp1 * dt / 2
    k2p2 = k1p2 + k1dp2 * dt / 2

    [k2da1, k2da2, k2dp1, k2dp2] = derivative(k2a1, k2a2, k2p1, k2p2)

    k3a1 = k1a1 + k2da1 * dt / 2
    k3a2 = k1a2 + k2da2 * dt / 2
    k3p1 = k1p1 + k2dp1 * dt / 2
    k3p2 = k1p2 + k2dp2 * dt / 2

    [k3da1, k3da2, k3dp1, k3dp2] = derivative(k3a1, k3a2, k3p1, k3p2)

    k4a1 = k1a1 + k3da1 * dt
    k4a2 = k1a2 + k3da2 * dt
    k4p1 = k1p1 + k3dp1 * dt
    k4p2 = k1p2 + k3dp2 * dt

    [k4da1, k4da2, k4dp1, k4dp2] = derivative(k4a1, k4a2, k4p1, k4p2)

    return [
        k1a1 + (k1da1 + 2*k2da1 + 2*k3da1 + k4da1) * dt / 6,
        k1a2 + (k1da2 + 2*k2da2 + 2*k3da2 + k4da2) * dt / 6,
        k1p1 + (k1dp1 + 2*k2dp1 + 2*k3dp1 + k4dp1) * dt / 6,
        k1p2 + (k1dp2 + 2*k2dp2 + 2*k3dp2 + k4dp2) * dt / 6
    ]

x_values, y_values = np.meshgrid(np.linspace(-3.5, 3.5, grid_size), np.linspace(-3.5, 3.5, grid_size))

p1 = np.zeros_like(x_values)
p2 = np.zeros_like(y_values)

def update(frame):
    global x_values, y_values, p1, p2   # Declare x_values and y_values as global variables
    # Implement your calculation here to update the x_values and y_values arrays
    # For demonstration purposes, let's just add a small amount to each value

    if frame % 50 == 0:
        print('hi')
    [x_values, y_values, p1, p2] = rk4(x_values, y_values, p1, p2, dt)

    return plt.imshow(np.sin(x_values) * np.cos(y_values), cmap='viridis', extent=[-3.5, 3.5, -3.5, 3.5])

# Create the initial plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.sin(x_values) * np.cos(y_values), cmap='viridis', extent=[-3.5, 3.5, -3.5, 3.5])
plt.colorbar(im, ax=ax, label='Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CHAOS OF A DOUBLE PENDULUM')



# Create the animation
ani = FuncAnimation(fig, update, frames=range(500), interval=1)
# plt.show()
# Save the animation using Pillow
ani.save('double_pendulum_animation.gif', writer='pillow')

print('DONE')