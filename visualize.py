
# Lets visualize some jets using matplotlib
# We want a circular jet area, so we use a polar plot
import matplotlib.pyplot as plt
import numpy as np
import json
etas = []
phis = []
pts = []

def plot_jet_phi_theta(jet):
    px, py, pz = jet["px"], jet["py"], jet["pz"]
    pt = np.sqrt(px**2 + py**2)
    theta = np.arccos(pz / np.sqrt(px**2 + py**2 + pz**2))
    eta = -np.log(np.tan(theta/2))
    phi = np.arctan2(py, px)
    
    etas.append(eta)
    phis.append(phi)
    pts.append(pt)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

def plot_jet_3d(jet):
    px, py, pz, E = jet["px"], jet["py"], jet["pz"], jet["E"]
    ax.quiver(0, 0, 0, px, py, pz, length=E/1000, normalize=True)



# Plot all signal jets
with open("output/signal_jets.jsonl", "r") as f:
    for i, line in enumerate(f):
        jet_data = json.loads(line)
        for jet in jet_data['jets']:
            # plot_jet_phi_theta(jet)
            plot_jet_3d(jet)
# Add background jets too
with open("output/background_jets.jsonl", "r") as f:
    for i, line in enumerate(f):
        jet_data = json.loads(line)
        for jet in jet_data['jets']:
            # plot_jet_phi_theta(jet)
            plot_jet_3d(jet)

"""
plt.figure(figsize=(20,20))
plt.scatter(phis, etas, s=np.array(pts)/10, c=pts, cmap="viridis", alpha=0.7)
plt.xlabel("phi")
plt.ylabel("eta")
plt.colorbar(label="pt (GeV)")
plt.title("Jets in eta-phi plane")
plt.show()
"""
plt.title(f'Jets in 3D space')
ax.set_xlabel('px')
ax.set_ylabel('py')
ax.set_zlabel('pz')
ax.set_xbound(-3, 3)
ax.set_ybound(-3, 3)
ax.set_zbound(-3, 3)
plt.show()