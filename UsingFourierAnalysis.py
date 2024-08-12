import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# System parameters
m1, m2 = 1000, 800  # masses in kg
k1, k2 = 100000, 80000  # stiffnesses in N/m
c1, c2 = 2000, 1600  # damping coefficients in Ns/m

# Ground motion parameters
A = 0.5  # amplitude in m/s^2
omega_input = 5  # angular frequency in rad/s

# Time array
t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

# Define matrices
M = np.array([[m1, 0], [0, m2]])
C = np.array([[c1 + c2, -c2], [-c2, c2]])
K = np.array([[k1 + k2, -k2], [-k2, k2]])

# Calculate natural frequencies and mode shapes
eigenvalues, eigenvectors = linalg.eig(K, M)
natural_frequencies = np.sqrt(eigenvalues.real)
mode_shapes = eigenvectors

# Ground motion function
def ground_motion(t):
    timeSeries = np.loadtxt(r"c:\Users\me_su\Desktop\elCentro_NS.txt")
    return timeSeries[:,0],timeSeries[:,1]

# Calculate ground motion in time domain
t, ug_t = ground_motion(t)
dt = t[1] - t[0]

# Perform FFT on ground motion
ug_f = np.fft.fft(ug_t)
freq = np.fft.fftfreq(t.shape[-1], dt)
omega = 2 * np.pi * freq

# Calculate Frequency Response Function (FRF)
def calculate_frf(omega, M, C, K):
    H = np.zeros((len(omega), 2, 2), dtype=complex)
    for i, w in enumerate(omega):
        Z = -w**2 * M + 1j * w * C + K
        H[i] = linalg.inv(Z)
    return H

H = calculate_frf(omega, M, C, K)

# Calculate response in frequency domain
X_f = np.zeros((len(omega), 2), dtype=complex)
for i in range(len(omega)):
    X_f[i] = H[i] @ np.array([1, 1]) * ug_f[i]

# Inverse FFT to get response in time domain
x1_t = np.fft.ifft(X_f[:, 0]).real
x2_t = np.fft.ifft(X_f[:, 1]).real

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(t, x1_t, label='Floor 1')
plt.plot(t, x2_t, label='Floor 2')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.title('Time Domain Response')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(t, ug_t)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Ground Motion')

plt.subplot(3, 2, 3)
plt.plot(freq, np.abs(X_f[:, 0]), label='Floor 1')
plt.plot(freq, np.abs(X_f[:, 1]), label='Floor 2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Domain Response')
plt.legend()
plt.xlim(0, 10)

plt.subplot(3, 2, 4)
plt.plot(freq, np.abs(ug_f))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Ground Motion Spectrum')
plt.xlim(0, 10)

plt.subplot(3, 2, 5)
plt.plot(freq, np.abs(H[:, 0, 0]), label='H11')
plt.plot(freq, np.abs(H[:, 1, 1]), label='H22')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Response Function')
plt.legend()
plt.xlim(0, 10)

plt.subplot(3, 2, 6)
plt.plot(natural_frequencies / (2 * np.pi), [0, 0], 'ro', label='Natural Frequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Natural Frequencies')
plt.legend()
plt.xlim(0, 10)

plt.tight_layout()
plt.show()

print(f"Natural frequencies: {natural_frequencies / (2 * np.pi)} Hz")
print(f"Mode shapes:\n{mode_shapes}")