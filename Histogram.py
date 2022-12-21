import matplotlib.pyplot as plt
import numpy as np


class Histogram:
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu

    def __init__(self, values: np.ndarray = None, data: np.ndarray = None) -> None:
        if values is not None:
            self.values = []
            if values.shape[-1] == 3:
                for i in range(0, 3):
                    histogram = np.histogram(
                        values[:, :, i], bins=256, range=(0, 256))[0]
                    self.values.append(histogram)
            else:
                histogram = np.histogram(
                    values, bins=256, range=(0, 256))[0]
                self.values.append(histogram)
        elif data is not None:
            self.values = [data]
        else:
            self.values = []

    def plot(self) -> None:
        # metoda wyswietlajaca histogram na podstawie atrybutu values
        bins = np.array(range(256))

        if len(self.values) == 3:
            fig, ax = plt.subplots(1, 3)
            colors = ["red", "green", "blue"]
            for i in range(3):
                ax[i].plot(bins, self.values[i], color=colors[i])
        else:
            plt.plot(bins, self.values[0], color="gray")
        plt.show()

    def to_cumulated(self) -> 'Histogram':
        # metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        histogram = Histogram()
        arr = []
        sum = 0

        for x in self.values:
            hist = x

            for i in range(256):
                for j in range(i):
                    sum += hist[j]
                arr.append(sum)
                sum = 0

            histogram.values.append(arr)
            arr = []

        return histogram
