import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

# Set up duration for animation
t0 = 0
t_end = 10
dt = 0.02
t = np.arange(0, t_end + dt, dt)

# Constants
g = -9.81  # [m/s^2]
m = 1000  # [kg]
F_g = m * g

# PID constants
K_p_x = 10000
K_d_x = 6000
K_i_x = 1

K_p_y = 6000
K_d_y = 10000
K_i_y = 1

# Rocket
x_i = np.random.randint(-50, 51)
y_i = np.random.randint(175, 210)
v_x_i = np.random.randint(-20, 21)
v_y_i = np.random.randint(-30, -10)

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

e_x = np.zeros(len(t))
e_x_dot = np.zeros(len(t))
e_x_int = np.zeros(len(t))
e_y = np.zeros(len(t))
e_y_dot = np.zeros(len(t))
e_y_int = np.zeros(len(t))

x_ref = 0
y_ref = 0

F_x_arr = np.zeros(len(t))
F_y_arr = np.zeros(len(t))

MIN_Y_FORCE = 0  # [Newton]
MAX_Y_FORCE = 30000  # [Newton]
MIN_X_FORCE = -30000  # [Newton]
MAX_X_FORCE = 30000  # [Newton]

for i in range(len(t)):
    prev_a_y = a_y[i - 1] if i > 1 else 0
    prev_v_y = v_y[i - 1] if i > 1 else v_y_i
    prev_y = y1[i - 1] if i > 1 else y_i
    
    prev_a_x = a_x[i - 1] if i > 1 else 0
    prev_v_x = v_x[i - 1] if i > 1 else v_x_i
    prev_x = x1[i - 1] if i > 1 else x_i

    prev_e_x = e_x[i - 1] if i > 1 else 0
    prev_e_x_int = e_x_int[i - 1] if i > 1 else 0
    
    prev_e_y = e_y[i - 1] if i > 1 else 0
    prev_e_y_int = e_y_int[i - 1] if i > 1 else 0

    # -- PID Controller -- start
    
    # Compute x comp. error
    e_x[i] = x_ref - prev_x
    e_x_dot[i] = (e_x[i] - prev_e_x) / dt
    e_x_int[i] = prev_e_x_int + (prev_e_x + e_x[i]) / 2 * dt
    
    # Compute y comp. error
    e_y[i] = y_ref - prev_y
    e_y_dot[i] = (e_y[i] - prev_e_y) / dt
    e_y_int[i] = prev_e_y_int + (prev_e_y + e_y[i]) / 2 * dt

    F_x = max(MIN_X_FORCE, min(MAX_X_FORCE, K_p_x * e_x[i-1] + K_d_x * e_x_dot[i-1] + K_i_x * e_x_int[i-1]))
    F_y = max(MIN_Y_FORCE, min(MAX_Y_FORCE, K_p_y * e_y[i-1] + K_d_y * e_y_dot[i-1] + K_i_y * e_y_int[i-1]))

    # -- PID Controller -- end
    
    # Opposite force of gravity (when resting on surface)
    F_g_o = 0 
    if y1[i] <= 0:
        F_g_o = -F_g

    # Force Y component
    F_net_y = F_y + F_g + F_g_o
    a_y[i] = F_net_y / m
    v_y[i] = prev_v_y + ((prev_a_y + a_y[i]) / 2) * dt
    y1[i] = prev_y + ((prev_v_y + v_y[i]) / 2) * dt

    # Surface constraint
    if y1[i] <= 0:
        a_y[i] = 0
        v_y[i] = 0
        y1[i] = 0
    
    # Force X component
    F_net_x = F_x
    a_x[i] = F_net_x / m
    v_x[i] = prev_v_x + ((prev_a_x + a_x[i]) / 2) * dt
    x1[i] = prev_x + ((prev_v_x + v_x[i]) / 2) * dt

    omega = np.arccos(F_net_x / np.sqrt(np.power(F_net_x, 2) + np.power(F_net_y, 2)))
    r = 10
    n = np.sqrt(np.power(F_x, 2) + np.power(F_y, 2)) * 0.001
    alpha = omega

    x2[i] = r * np.cos(alpha) + x1[i]
    y2[i] = r * np.sin(alpha) + y1[i]
    
    x3[i] = -n * np.cos(alpha) + x1[i]
    y3[i] = -n * np.sin(alpha) + y1[i]

    # graphs
    F_x_arr[i] = F_x
    F_y_arr[i] = F_y


############### Animation ###############
frame_amount = len(t)


def update_plot(num):
    rocket.set_data([x1[num], x2[num]], [y1[num], y2[num]])
    rocket_thrust_trace.set_data([x1[num], x3[num]], [y1[num], y3[num]])
    reference_point.set_data([x_ref - 2, x_ref + 2], [y_ref, y_ref])
    
    hori_acc.set_data(t[0:num], a_x[0:num])
    vert_acc.set_data(t[0:num], a_y[0:num])
    hori_vel.set_data(t[0:num], v_x[0:num])
    vert_vel.set_data(t[0:num], v_y[0:num])
    e_x_graph.set_data(t[0:num], e_x[0:num])
    e_y_graph.set_data(t[0:num], e_y[0:num])

    return rocket, rocket_thrust_trace, reference_point, \
        hori_acc, vert_acc, hori_vel, vert_vel, e_x_graph, e_y_graph


fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(3, 2)
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                    top=0.95, wspace=0.1, hspace=0.2)


# Simulation subplot
ax0 = fig.add_subplot(gs[:, 0], facecolor=(0.9, 0.9, 0.9))
ax0_x_lim = 80
ax0_y_lim = 200
plt.xlim(-ax0_x_lim, ax0_x_lim)
plt.ylim(0, ax0_y_lim)
plt.xticks(np.arange(-ax0_x_lim, ax0_x_lim + 1, 20))
plt.yticks(np.arange(0, ax0_y_lim + 1, 20))
plt.grid(True)
plt.title('Rocket Simulation', fontsize=12)

# Generate wind data
# x = np.linspace(0, ax0_x_lim, 20)
# y = np.linspace(0, ax0_y_lim, 20)
# X, Y = np.meshgrid(x, y)
# U = np.ones_like(X) * 0.1
# V = np.zeros_like(Y)
# Q = ax0.quiver(X, Y, U, V, pivot='middle', color='blue')

# Rocket
rocket, = ax0.plot(
    [x_i, x_i], [y_i + 1, y_i + 10 - 1], 'black', linewidth=5)
rocket_thrust_trace, = ax0.plot([0, 0], [0, 0], 'orange', linewidth=3)
reference_point, = ax0.plot([0, 0], [0, 0], 'green', linewidth=3)


# subplot 1
ax1 = fig.add_subplot(gs[0, 1], facecolor=(0.9, 0.9, 0.9))
ax1_x_lim = t_end
ax1_y_lim = 100
plt.xlim(0, ax1_x_lim)
plt.ylim(-ax1_y_lim, ax1_y_lim)
plt.grid(True)
hori_acc, = ax1.plot([], [], 'purple', linewidth=2, label='X Acceleration [m/s^2]')
vert_acc, = ax1.plot([], [], 'green', linewidth=2, label='Y Acceleration [m/s^2]')
plt.legend(loc='lower right', fontsize='small')


# subplot 2
ax2 = fig.add_subplot(gs[1, 1], facecolor=(0.9, 0.9, 0.9))
ax2_x_lim = t_end
ax2_y_lim = 100
plt.xlim(0, ax2_x_lim)
plt.ylim(-ax2_y_lim, ax2_y_lim)
plt.grid(True)
hori_vel, = ax2.plot([], [], 'purple', linewidth=2, label='X Velocity [m/s]')
vert_vel, = ax2.plot([], [], 'green', linewidth=2, label='Y Velocity [m/s]')
plt.legend(loc='lower right', fontsize='small')


# subplot 3
ax3 = fig.add_subplot(gs[2, 1], facecolor=(0.9, 0.9, 0.9))
ax3_x_lim = t_end
ax3_y_lim = 200
plt.xlim(0, ax3_x_lim)
plt.ylim(-ax3_y_lim, ax3_y_lim)
plt.grid(True)
e_x_graph, = ax3.plot([], [], 'purple', linewidth=2, label='X Error [m]')
e_y_graph, = ax3.plot([], [], 'green', linewidth=2, label='Y Error [m]')
plt.legend(loc='lower right', fontsize='small')


ani = animation.FuncAnimation(
    fig, update_plot, frames=frame_amount, interval=20, repeat=True, blit=True)
plt.show()
