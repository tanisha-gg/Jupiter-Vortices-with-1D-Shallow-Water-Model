import matplotlib.pyplot as plt

class Visualization:
    def __init__(self, params):
        self.params = params

    def plot_quiver(self, xx, yy, u, v, t, title):
        plt.figure(figsize=(12, 8))
        plt.quiver(xx, yy, u, v)
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.title(title + " at t = %7.5f days" % t)
        plt.draw()
        plt.pause(0.001)

    def plot_field(self, xx, yy, field, t, title):
        plt.clf()
        plt.pcolormesh(xx / 1e3, yy / 1e3, field)
        plt.colorbar()
        plt.title(title + "at t = %7.5f days" % t)
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        plt.xlim([-self.params.Lx / 2e3, self.params.Lx / 2e3])
        plt.ylim([-self.params.Ly / 2e3, self.params.Ly / 2e3])
        plt.draw()
        plt.pause(0.001)
