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
v_x_i = 0
v_y_i = 0

x1 = np.ones(len(t)) * x_i
y1 = np.ones(len(t)) * y_i
x2 = np.zeros(len(t))
y2 = np.zeros(len(t))
x3 = np.zeros(len(t))
y3 = np.zeros(len(t))
v_x = np.zeros(len(t))
v_y = np.zeros(len(t))
a_x = np.zeros(len(t))
a_y = np.zeros(len(t))

for i in range(len(t)):
    prev_a_y = a_y[i - 1] if i > 1 else 0
    prev_v_y = v_y[i - 1] if i > 1 else v_y_i
    prev_y = y1[i - 1] if i > 1 else y_i
    
    prev_a_x = a_x[i - 1] if i > 1 else 0
    prev_v_x = v_x[i - 1] if i > 1 else v_x_i
    prev_x = x1[i - 1] if i > 1 else x_i

    F_y = 11500
    F_x = 0

    # Force Y component
    F_net_y = F_y + F_g
    a_y[i] = F_net_y / m
    v_y[i] = prev_v_y + ((prev_a_y + a_y[i]) / 2) * dt
    y1[i] = prev_y + ((prev_v_y + v_y[i]) / 2) * dt
    
    # Force X component
    F_net_x = F_x
    a_x[i] = F_net_x / m
    v_x[i] = prev_v_x + ((prev_a_x + a_x[i]) / 2) * dt
    x1[i] = prev_x + ((prev_v_x + v_x[i]) / 2) * dt

    if y1[i] <= 0:
        y1[i] = 0

    omega = np.arccos(F_net_x / np.sqrt(np.power(F_net_x, 2) + np.power(F_net_y, 2)))
    r = 10
    n = F_net_y * 0.005
    alpha = omega

    x2[i] = r * np.cos(alpha) + x1[i]
    y2[i] = r * np.sin(alpha) + y1[i]
    
    x3[i] = -n * np.cos(alpha) + x1[i]
    y3[i] = -n * np.sin(alpha) + y1[i]


############### Animation ###############
frame_amount = len(t)


def update_plot(num):
    # rocket.set_data([x[num], x[num]], [y[num] + 1, y[num] + 10 - 1])
    rocket.set_data([x1[num], x2[num]], [y1[num], y2[num]])
    rocket_thrust_trace.set_data([x1[num], x3[num]], [y1[num], y3[num]])
    vert_vel.set_data(t[0:num], v_y[0:num])

    return rocket, vert_vel, rocket_thrust_trace


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
    [x_i, x_i], [y_i + 1, y_i + 10 - 1], 'black', linewidth=10)
rocket_thrust_trace, = ax0.plot([0, 0], [0, 0], 'orange', linewidth=2)


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
