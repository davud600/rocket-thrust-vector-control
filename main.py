import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

# Set up duration for animation
t0 = 0
t_end = 22
dt = 0.02
t = np.arange(0, t_end + dt, dt)


############### Animation ###############
frame_amount = len(t)


def update_plot(num):
    return


fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=(0.8, 0.8, 0.8))
gs = gridspec.GridSpec(1, 2)
plt.subplots_adjust(left=0.15, bottom=0.25, right=0.85,
                    top=0.75, wspace=0.15, hspace=0.2)


ax0 = fig.add_subplot(gs[:, 0:2], facecolor=(0.9, 0.9, 0.9))

x_lim = 120
y_lim = 40

plt.xlim(0, x_lim)
plt.ylim(0, y_lim)
plt.xticks(np.arange(0, x_lim + 1, 10))
plt.yticks([])
plt.grid(True)

ani = animation.FuncAnimation(
    fig, update_plot, frames=frame_amount, interval=20, repeat=True, blit=True)
plt.show()
