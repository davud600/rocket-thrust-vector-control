import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

# Set up duration for animation
t0 = 0
t_end = 22
dt = 0.02
t = np.arange(0, t_end + dt, dt)

# Constants
g = -9.81  # [m/s^2]
m = 1000  # [kg]
F_g = m * g

# Rocket
x_i = 40
y_i = 0
v_y_i = 0
F_y_r = 10000

x = np.ones(len(t)) * x_i
y = np.ones(len(t)) * y_i
v_y = np.zeros(len(t))
a_y = np.zeros(len(t))

for i in range(len(t)):
    prev_a_y = a_y[i - 1] if i > 1 else 0
    prev_v_y = v_y[i - 1] if i > 1 else v_y_i
    prev_y = y[i - 1] if i > 1 else y_i

    F_y_net = F_y_r + F_g
    a_y[i] = F_y_net / m
    v_y[i] = prev_v_y + ((prev_a_y + a_y[i]) / 2) * dt
    y[i] = prev_y + ((prev_v_y + v_y[i]) / 2) * dt

    if y[i] <= 0:
        y[i] = 0

print(y)


############### Animation ###############
frame_amount = len(t)


def update_plot(num):
    rocket.set_data([x[num], x[num]], [y[num] - 1, y[num] + 10 - 1])
    vert_vel.set_data(t[0:num], v_y[0:num])

    return rocket, vert_vel


fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(2, 2)
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                    top=0.95, wspace=0.1, hspace=0.2)


# Simulation subplot
ax0 = fig.add_subplot(gs[:, 0], facecolor=(0.9, 0.9, 0.9))
ax0_x_lim = 80
ax0_y_lim = 100
plt.xlim(0, ax0_x_lim)
plt.ylim(0, ax0_y_lim)
plt.xticks(np.arange(0, ax0_x_lim + 1, 10))
plt.yticks(np.arange(0, ax0_y_lim + 1, 10))
plt.grid(True)

# Generate wind data
# x = np.linspace(0, ax0_x_lim, 20)
# y = np.linspace(0, ax0_y_lim, 20)
# X, Y = np.meshgrid(x, y)
# U = np.ones_like(X) * 0.1
# V = np.zeros_like(Y)
# Q = ax0.quiver(X, Y, U, V, pivot='middle', color='blue')

# Rocket
rocket, = ax0.plot(
    [x_i, x_i], [y_i - 1, y_i + 10 - 1], 'black', linewidth=10)
rocket_thrust_trace, = ax0.plot([x_i, x_i], [y_i, y_i], 'orange', linewidth=2)


# subplot 1
ax1 = fig.add_subplot(gs[1, 1], facecolor=(0.9, 0.9, 0.9))
ax1_x_lim = 15
ax1_y_lim = 15
plt.xlim(0, ax1_x_lim)
plt.ylim(0, ax1_y_lim)
plt.grid(True)
vert_vel, = ax1.plot([], [], 'b', linewidth=1)


# subplot 2
ax2 = fig.add_subplot(gs[0, 1], facecolor=(0.9, 0.9, 0.9))


ani = animation.FuncAnimation(
    fig, update_plot, frames=frame_amount, interval=20, repeat=True, blit=True)
plt.show()
