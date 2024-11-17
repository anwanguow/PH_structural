import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(0, 4, 500)
tau_values = [0.2, 0.4, 0.57, 0.8]

plt.figure(figsize=(8, 6), dpi=300)
for tau in tau_values:
    plt.plot(T, np.exp(-tau * T), label=f'$\\tau$ = {tau}')

plt.axhline(y=0.1, color='gray', linestyle='--', linewidth=1)
plt.yticks([0, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('$r_q - r_1$', fontsize=30)
plt.ylabel('$e^{-\\tau \cdot (r_q - r_1)}$', fontsize=30)
plt.legend(fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.grid(False)
plt.show()
