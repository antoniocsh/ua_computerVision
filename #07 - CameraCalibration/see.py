import numpy as np

data = np.load('camera.npz')

intrinsics = data['intrinsics']
distortion = data['distortion']

print("Intrinsics:\n", intrinsics)
print("Distortion:\n", distortion)