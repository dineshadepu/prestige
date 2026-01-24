import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ball_height.csv", delimiter=",", skiprows=1)
t = data[:, 0]
y = data[:, 1]

plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("Ball height")
plt.title("Bouncing ball (GPU DEM)")
plt.grid()
# plt.show()
plt.savefig("bouncing_ball.png")
