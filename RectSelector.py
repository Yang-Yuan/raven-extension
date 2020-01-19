from matplotlib import pyplot as plt
import numpy as np


class RectSelector:

    def __init__(self, img):
        self.rect_x = []
        self.rect_y = []
        self.rects = None
        fig, ax = plt.subplots()
        ax.imshow(img, cmap  = 'binary')
        self.cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_close = fig.canvas.mpl_connect('close_event', self.on_close)
        plt.show()

    def on_press(self, event):
        print("press:  " + str(event.xdata) + "    " + str(event.ydata))
        self.rect_x.append(event.xdata)
        self.rect_y.append(event.ydata)

    def on_release(self, event):
        print("release:  " + str(event.xdata) + "    " + str(event.ydata))
        self.rect_x.append(event.xdata)
        self.rect_y.append(event.ydata)

    def on_close(self, event):
        print("Window is closed.")
        x = np.sort(np.array(self.rect_x).reshape(-1, 2, order = "C"), axis = -1)
        y = np.sort(np.array(self.rect_y).reshape(-1, 2, order = "C"), axis = -1)
        self.rects = np.floor(np.concatenate((x, y), axis = -1)).astype(np.int)
