import matplotlib.pyplot as plt
import numpy as np

#import plotly.plotly as py

gaussian_numbers = np.random.randn(10)
print(gaussian_numbers)
plt.hist(gaussian_numbers)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# fig = plt.gcf()

# plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')