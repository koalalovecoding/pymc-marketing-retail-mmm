import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

# Carryover
t = np.arange(7)
ax[0].bar(t, np.exp(-0.5 * t), color="#F96332")
ax[0].set(title="Carryover", xlabel="Time after spend", ylabel="Effect")

# Saturation
x = np.linspace(0, 5, 200)
ax[1].plot(x, 1 - np.exp(-x), color="#F96332", lw=3)
ax[1].set(title="Saturation", xlabel="Media spend", ylabel="Incremental revenue")

# Clean style
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.show()