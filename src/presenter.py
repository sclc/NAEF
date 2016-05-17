""" Presenter def """

import matplotlib.pyplot as plt
import math
import numpy as np

class Presenter():
    """ """
    def __init__(self):
        """ """
    def instant_plot_y (self, y_data_list, figure_title, x_label, y_label,legend_list, color_list):
        """ plot with matplotlib"""
        max_value_comparator = lambda a,b: a if a>=b else b
        min_value_comparator = lambda a,b: b if a>=b else a

        max_x_axis = -np.inf
        min_x_axis = np.inf
        max_y_axis = -np.inf
        min_y_axis = np.inf
        for idx in range(0,len(y_data_list)):
            Y_data = np.asarray([ y for y in y_data_list[idx] ]);
            X_data = np.asarray( [x for x in range(0,len(y_data_list[idx]))] )
            plt.plot(X_data, Y_data, label=legend_list[idx] ,c=color_list[idx])

            max_x_axis = max_value_comparator(max_x_axis, max(X_data))
            max_y_axis = max_value_comparator(max_y_axis, max(Y_data))
            min_x_axis = min_value_comparator(min_x_axis, min(X_data))
            min_y_axis = min_value_comparator(min_y_axis, min(Y_data))

        plt.xticks(np.arange(min_x_axis, max_x_axis+1, 5.0))
        plt.yticks(np.arange(min_y_axis, max_y_axis+1, 1.))
        plt.ylim(min_y_axis-0.5, max_y_axis)
        plt.title(figure_title, fontsize=20)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=30)
        plt.legend(fancybox=True, prop={"size":16})
        plt.grid(True, lw=2, ls="--", c=".75")
        plt.show()

    def instant_plot_y_log10 (self, y_data_list, figure_title, x_label, y_label,legend_list, color_list):
        """ plot with matplotlib"""
        max_value_comparator = lambda a,b: a if a>=b else b
        min_value_comparator = lambda a,b: b if a>=b else a

        max_x_axis = -np.inf
        min_x_axis = np.inf
        max_y_axis = -np.inf
        min_y_axis = np.inf
        for idx in range(0,len(y_data_list)):
            assert min(y_data_list[idx]) >=.0, "you have negative value, you cannot you log10 scaling"
            Y_data = np.asarray([ math.log(y,10) for y in y_data_list[idx] ]);
            X_data = np.asarray( [x for x in range(0,len(y_data_list[idx]))] )
            plt.plot(X_data, Y_data, label=legend_list[idx] ,c=color_list[idx])

            max_x_axis = max_value_comparator(max_x_axis, max(X_data))
            max_y_axis = max_value_comparator(max_y_axis, max(Y_data))
            min_x_axis = min_value_comparator(min_x_axis, min(X_data))
            min_y_axis = min_value_comparator(min_y_axis, min(Y_data))

        plt.xticks(np.arange(min_x_axis, max_x_axis+1, 5.0))
        plt.yticks(np.arange(min_y_axis, max_y_axis+1, 1.))
        plt.ylim(min_y_axis-0.5, max_y_axis)
        plt.title(figure_title, fontsize=20)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=30)
        plt.legend(fancybox=True, prop={"size":16})
        plt.grid(True, lw=2, ls="--", c=".75")
        plt.show()

def main():
    """ """
if __name__ == "__main__":
    """ """
    main()
