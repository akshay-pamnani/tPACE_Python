import numpy as np
import matplotlib.pyplot as plt
from Sparsify import sparsify_sample
from Wiener import wiener

# Define points and simulate Wiener process
pts = np.linspace(0, 1, num=20)
tmp = wiener(50, pts)
tmp1 = sparsify_sample(tmp, pts, sparsity=[2, 4, 6])

# Get the length of each sample in the sparsified Lt and Ly lists
lt_lengths = [len(l) for l in tmp1['Lt']]
ly_lengths = [len(l) for l in tmp1['Ly']]
print("Lt lengths:", lt_lengths)
print("Ly lengths:", ly_lengths)

# Repeat with a fixed random seed for reproducibility
np.random.seed(1)
tmp2 = wiener(10, pts=np.arange(0, 1.1, 0.1))  # pts from 0 to 1 in steps of 0.1
np.random.seed(1)
tmp3 = wiener(10, pts=np.arange(0, 1.1, 0.1), sparsify=2)

# Test with fragmentation on more points
pts = np.arange(0, 1.02, 0.02)  # pts from 0 to 1 in steps of 0.02
tmp = wiener(1000, pts)
tmp1 = sparsify_sample(tmp, pts, sparsity=list(range(1, 6)), fragment=0.2)


# Example plot function for design plot (replace with your preferred plotting library)
def create_design_plot(lt, pts, show_legend=True, show_axes=False):
    
    for i, time_points in enumerate(lt):
        plt.scatter(time_points, [i] * len(time_points), marker='|')
    plt.xlabel('Time')
    plt.ylabel('Sample Index')
    plt.title("Design Plot")
    if show_legend:
        plt.legend(["Observations"])
    if show_axes:
        plt.grid(True)
    plt.show()


# Plot the design of the sparsified data
create_design_plot(tmp1['Lt'], pts, show_legend=True, show_axes=False)
