import numpy as np

from openmmqmmm.functions.functions_general import BC, ashexit, print_line_with_mainheader


class ASH_plot:
    def __init__(
        self,
        figuretitle="Plottyplot",
        num_subplots=1,
        dpi=200,
        imageformat="png",
        figsize=(9, 5),
        x_axislabel="X-axis",
        y_axislabel="Energy (X)",
        x_axislabels=None,
        y_axislabels=None,
        title="Plot-title",
        subplot_titles=None,
        xlimit=None,
        ylimit=None,
        backend="Agg",
        legend_pos=None,
        horizontal=False,
        tight_layout=True,
        padding=None,
    ):
        print_line_with_mainheader("ASH_energy_plot")
        import matplotlib.pyplot as plt

        self.working = True
        self.num_subplots = num_subplots
        self.imageformat = imageformat
        self.dpi = dpi
        self.figuretitle = figuretitle
        self.subplot_titles = subplot_titles
        self.x_axislabel = x_axislabel
        self.y_axislabel = y_axislabel
        # For multi-subplots
        self.x_axislabels = x_axislabels
        self.y_axislabels = y_axislabels
        # Legend position
        self.legend_pos = legend_pos
        # Labels for each series
        self.labels = []

        print("Subplots:", self.num_subplots)
        print("Figure size:", figsize)

        if self.num_subplots > 1:
            print(
                BC.WARNING,
                "Note: For multiple subplots use:\n ASH_plot(x_axislabels=['X1','X2','X3], y_axislabels=['Y1','Y2','Y3'], subplot_titles='Title1,'Title2','Title3']",
                BC.END,
            )
        else:
            print("X-axis label:", x_axislabel)
            print("Y-axis label:", y_axislabel)
            print("Title:", title)

        if self.num_subplots == 1:
            self.fig, ax = plt.subplots(figsize=figsize)
            self.axs = [ax]
            self.x_axislabels = x_axislabels
            self.y_axislabels = y_axislabels

            # X-limit and y-limit
            if xlimit is not None:
                self.axs[0].set_xlim(xlimit[0], xlimit[1])
            if ylimit is not None:
                self.axs[0].set_ylim(ylimit[0], ylimit[1])

        elif self.num_subplots == 2:
            if horizontal is True:
                print("Horizontal plot is true")
                self.fig, self.axs = plt.subplots(1, 2, figsize=figsize)
                if tight_layout is True:
                    print("Tight layout True")
                    self.fig.tight_layout()
                # Subplot padding
                if padding is not None:
                    self.fig.subplots_adjust(wspace=padding)

            else:
                self.fig, self.axs = plt.subplots(2, 1, figsize=figsize)
                if tight_layout is True:
                    print("Tight layout True")
                    self.fig.tight_layout()
                # Subplot padding
                if padding is not None:
                    self.fig.subplots_adjust(hspace=padding)

            self.axiscount = 0

            # X-limit and y-limit
            # TODO: Allow different limits for each subplot
            if xlimit is not None:
                self.axs[0].set_xlim(xlimit[0], xlimit[1])
                self.axs[1].set_xlim(xlimit[0], xlimit[1])
            if ylimit is not None:
                self.axs[0].set_ylim(ylimit[0], ylimit[1])
                self.axs[1].set_ylim(ylimit[0], ylimit[1])

        elif self.num_subplots == 3:
            if horizontal is True:
                self.fig, self.axs = plt.subplots(1, 3, figsize=figsize)  # a figure with a 1x4 grid of Axes
                self.axiscount = 0
            else:
                self.plotlistnames = ["upleft", "upright", "low"]
                self.fig, axs_dict = plt.subplot_mosaic([["upleft", "upright"], ["low", "low"]])
                self.axs = [axs_dict["upleft"], axs_dict["upright"], axs_dict["low"]]
                self.axiscount = 0
            if ylimit is not None:
                self.axs[0].set_ylim(ylimit[0], ylimit[1])
                self.axs[1].set_ylim(ylimit[0], ylimit[1])
                self.axs[2].set_ylim(ylimit[0], ylimit[1])
        elif self.num_subplots == 4:
            self.fig, axs = plt.subplots(2, 2, figsize=figsize)  # a figure with a 2x2 grid of Axes
            self.axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
            self.axiscount = 0

        self.addplotcount = 0

    def invert_x_axis(self, subplot):
        self.axs[subplot].invert_xaxis()

    def invert_y_axis(self, subplot):
        self.axs[subplot].invert_yaxis()

    def addseries(
        self,
        subplot,
        surfacedictionary=None,
        x_list=None,
        y_list=None,
        x_labels=None,
        label="Series",
        color="blue",
        pointsize=40,
        scatter=True,
        line=True,
        bar=False,
        scatter_linewidth=2,
        line_linewidth=1,
        barwidth=None,
        barcolors=None,
        marker="o",
        legend=True,
        x_scaling=1.0,
        y_scaling=1.0,
        xticklabelrotation=80,
        x_scale_log=False,
        y_scale_log=False,
        colormap="viridis",
        linestyle="-",
    ):
        import matplotlib.pyplot as plt

        print("Adding new series to ASH_plot object")

        if bar is True and (scatter is True or line is True):
            print("Error: you can not add a bar together with scatter and line at the same time")
            ashexit()

        self.addplotcount += 1
        curraxes = self.axs[subplot]

        self.labels.append(label)

        # Using x_list and y_list unless not provided
        if surfacedictionary is None:
            # If Python lists
            if not isinstance(x_list, (list, np.ndarray)) or not isinstance(y_list, (list, np.ndarray)):
                print(
                    BC.FAIL,
                    "Please provide either a valid x_list and y_list (can be Python lists or Numpy arrays) or a surfacedictionary (Python dict)",
                    BC.END,
                )
                print(f"x_list: {x_list}")
                print(f"y_list: {y_list}")
                ashexit()
            else:
                x = list(x_list)
                y = list(y_list)

        # Alernative dictionary option
        if surfacedictionary is not None:
            print("Using provided surfacedictionary")
            x = []
            y = []
            # Sorting keys dictionary before grabbing so that line-plot is correct
            for key in sorted(surfacedictionary.keys()):
                x.append(float(key))
                y.append(surfacedictionary[key])
        # Optional scaling of x or y-values
        x = [i * x_scaling for i in x]
        y = [i * y_scaling for i in y]

        # Scatterplot
        if scatter is True:
            curraxes.scatter(x, y, color=color, marker=marker, s=pointsize, linewidth=scatter_linewidth, label=label)
        # Lineplot
        if line is True:
            # Avoid legend for line if scatter is enabled
            if scatter is True:
                label = "_nolegend_"
            else:
                pass
            curraxes.plot(x, y, linestyle=linestyle, color=color, linewidth=line_linewidth, label=label)
        if bar is True:
            if barcolors is None:
                print("No barcolors keyword supplied. Will color the bars according to colormap:", colormap)
                # Scale data
                denominator = max(y) - min(y)
                scaled_data = [(datum - min(y)) / denominator for datum in y]
                barcolors = []
                cmap = plt.get_cmap(colormap)
                for decimal in scaled_data:
                    barcolors.append(cmap(decimal))
            if barwidth is None:
                curraxes.bar(x, y, color=barcolors, label=label)
            else:
                curraxes.bar(x, y, width=barwidth, color=barcolors, label=label)

        # Add labels to x-axis if
        if x_labels is not None:
            print("Adding xticks labels using rotation parameter:", xticklabelrotation)
            curraxes.set_xticks(x, minor=False)
            curraxes.set_xticklabels(x_labels, fontdict=None, minor=False, rotation=xticklabelrotation)
        # Log scale
        if x_scale_log is True:
            curraxes.set_xscale("log")
        if y_scale_log is True:
            curraxes.set_yscale("log")

        # Title/axis options for 1 vs multiple subplots
        if self.num_subplots == 1:
            curraxes.set_xlabel(self.x_axislabel)  # Add an x-label to the axes.
            curraxes.set_ylabel(self.y_axislabel)  # Add a y-label to the axes.
            curraxes.set_title(self.figuretitle)  # Add a title to the axes if provided
        else:
            if self.x_axislabels is None:
                print(BC.FAIL, "For multiple subplots, x_axislabels and y_axislabels must be set.", BC.END)
                ashexit()
            curraxes.set_xlabel(self.x_axislabels[subplot])  # Add an x-label to the axes.
            curraxes.set_ylabel(self.y_axislabels[subplot])  # Add a y-label to the axes.
            if self.subplot_titles is not None:
                curraxes.set_title(self.subplot_titles[subplot])  # Add a title to the axes if provided
        if legend is True:
            curraxes.legend(shadow=True, fontsize="small")  # Add a legend.

    def showplot(self):
        # Requires GUI backend
        import matplotlib.pyplot as plt

        plt.show()

    def savefig(self, filename, imageformat=None, dpi=None):
        import matplotlib.pyplot as plt

        # Change legend position
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
        if self.num_subplots == 1 and self.legend_pos is not None:
            self.axs[0].legend(loc="center left", bbox_to_anchor=(self.legend_pos[0], self.legend_pos[1]))

        if imageformat is None:
            imageformat = self.imageformat
        if dpi is None:
            dpi = self.dpi
        file = filename + "." + imageformat
        print(f"\nSaving plot to file: {file} with resolution: {dpi} ")
        plt.savefig(file, format=imageformat, dpi=self.dpi, bbox_inches="tight")
