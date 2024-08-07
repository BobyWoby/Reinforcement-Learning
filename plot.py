import matplotlib.pyplot as plt
import pickle

ax = pickle.load(open("plot.pickle", "rb"))
plt.show()
