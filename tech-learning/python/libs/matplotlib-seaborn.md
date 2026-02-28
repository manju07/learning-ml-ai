# Matplotlib & Seaborn — Data Visualization

## Table of Contents
- [Matplotlib Basics](#matplotlib-basics)
- [Figure and Axes](#figure-and-axes)
- [Plot Types](#plot-types)
- [Customization](#customization)
- [Subplots](#subplots)
- [Seaborn](#seaborn)
- [Statistical Plots](#statistical-plots)
- [Heatmaps and Pairplots](#heatmaps-and-pairplots)
- [Saving Figures](#saving-figures)

---

## Matplotlib Basics

```python
import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
```

### Two Interfaces

**Pyplot interface** (stateful, MATLAB-style):
```python
plt.plot(x, y)
plt.title("Title")
plt.show()
```

**Object-oriented interface** (recommended for complex plots):
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Title")
plt.show()
```

---

## Figure and Axes

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))  # figsize in inches

# Or with dpi
fig = plt.figure(figsize=(10, 6), dpi=100)
ax  = fig.add_subplot(111)   # 1 row, 1 col, 1st subplot

# Multiple axes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# axes is a 2D array: axes[0, 0], axes[0, 1], etc.

# Adjust layout
fig.tight_layout()           # prevent overlap
plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.show()
```

---

## Plot Types

### Line Plot

```python
x = np.linspace(0, 4 * np.pi, 200)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, np.sin(x), label="sin(x)", color="blue",  linewidth=2, linestyle="-")
ax.plot(x, np.cos(x), label="cos(x)", color="red",   linewidth=2, linestyle="--")
ax.plot(x, np.tan(x), label="tan(x)", color="green", linewidth=1, linestyle=":")
ax.set_ylim(-3, 3)
ax.legend(loc="upper right")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Trigonometric Functions")
ax.grid(True, alpha=0.3)
plt.show()
```

### Scatter Plot

```python
np.random.seed(42)
n = 200

x = np.random.randn(n)
y = 2 * x + np.random.randn(n) * 0.5
colors = np.random.rand(n)
sizes  = (np.random.rand(n) * 100) + 20

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap="viridis")
fig.colorbar(scatter, ax=ax, label="Color value")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Scatter Plot with Color and Size")
plt.show()
```

### Bar Chart

```python
categories = ["Python", "JavaScript", "Java", "C++", "Rust"]
values     = [32, 28, 18, 14, 8]
errors     = [2, 1.5, 2, 1, 0.5]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vertical bars
bars = axes[0].bar(categories, values, color="steelblue", edgecolor="black", yerr=errors, capsize=5)
axes[0].bar_label(bars, fmt="%.0f%%")   # label on bars
axes[0].set_title("Programming Language Popularity")
axes[0].set_ylabel("Usage %")

# Horizontal bars
axes[1].barh(categories, values, color=["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6"])
axes[1].set_title("Horizontal Bar Chart")
axes[1].set_xlabel("Usage %")

plt.tight_layout()
plt.show()
```

### Histogram

```python
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(3, 1.5, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Basic histogram
axes[0].hist(data1, bins=30, edgecolor="black", color="steelblue", alpha=0.7)
axes[0].set_title("Histogram")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

# Overlapping histograms with density
axes[1].hist(data1, bins=40, density=True, alpha=0.5, label="Group 1", color="blue")
axes[1].hist(data2, bins=40, density=True, alpha=0.5, label="Group 2", color="red")
axes[1].legend()
axes[1].set_title("Overlapping Histograms (Density)")

plt.tight_layout()
plt.show()
```

### Box Plot and Violin Plot

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

data = [np.random.normal(loc, 1, 100) for loc in [0, 1, 2, 3]]
labels = ["Group A", "Group B", "Group C", "Group D"]

# Box plot
bp = axes[0].boxplot(data, labels=labels, patch_artist=True)
colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
axes[0].set_title("Box Plot")
axes[0].set_ylabel("Value")

# Violin plot
parts = axes[1].violinplot(data, showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor("skyblue")
    pc.set_alpha(0.7)
axes[1].set_xticks([1, 2, 3, 4])
axes[1].set_xticklabels(labels)
axes[1].set_title("Violin Plot")

plt.tight_layout()
plt.show()
```

### Pie Chart

```python
labels  = ["Engineering", "HR", "Marketing", "Sales", "Finance"]
sizes   = [35, 15, 20, 18, 12]
explode = [0.05, 0, 0, 0, 0]   # slightly pull out first slice

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    explode=explode,
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.Set3.colors,
    shadow=True,
)
for text in autotexts:
    text.set_fontsize(12)
    text.set_fontweight("bold")
ax.set_title("Department Distribution")
plt.show()
```

### Heatmap (Matplotlib)

```python
data = np.random.rand(8, 10)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
fig.colorbar(im, ax=ax, label="Value")

# Add text annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

ax.set_title("Heatmap")
plt.tight_layout()
plt.show()
```

### 3D Plot

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")

# Surface plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Surface Plot")
plt.show()
```

---

## Customization

### Colors, Lines, and Markers

```python
# Colors: name, hex, RGB tuple, RGBA tuple
colors = ["red", "#2ecc71", (0.1, 0.2, 0.9), (0.5, 0.5, 0.5, 0.5)]

# Line styles: '-', '--', '-.', ':', ''
# Marker styles: 'o', 's', '^', 'D', '*', '+', 'x', '.', ','

fig, ax = plt.subplots()
ax.plot(x, y, color="#3498db", linestyle="--", linewidth=2,
        marker="o", markersize=5, markerfacecolor="red",
        markeredgecolor="black", markeredgewidth=1, alpha=0.7,
        label="My Line")
```

### Axes Configuration

```python
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.set_xlabel("X Label", fontsize=14, fontweight="bold")
ax.set_ylabel("Y Label", fontsize=14)
ax.set_title("Title", fontsize=16, pad=20)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(["zero", "two", "four", "six", "eight", "ten"], rotation=45)
ax.tick_params(axis="both", labelsize=12, length=6)
ax.invert_xaxis()             # flip x-axis
ax.invert_yaxis()
ax.set_xscale("log")          # log scale
ax.set_yscale("symlog")       # symmetric log scale
ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5)  # horizontal line
ax.axvline(x=5, color="r", linestyle=":")                   # vertical line
ax.axhspan(0.5, 1, alpha=0.3, color="green")                # horizontal span
```

### Annotations

```python
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))

# Text annotation with arrow
ax.annotate(
    "Local max",
    xy=(np.pi/2, 1),         # point to annotate
    xytext=(np.pi/2 + 1, 0.7),  # text position
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=12,
    color="red",
)

# Simple text
ax.text(0.5, 0.5, "Center text", transform=ax.transAxes,   # axes coordinates
        ha="center", va="center", fontsize=14)
```

### Styles

```python
plt.style.available   # list all styles

# Use a style
plt.style.use("seaborn-v0_8-darkgrid")
plt.style.use("ggplot")
plt.style.use("fivethirtyeight")
plt.style.use("dark_background")
plt.style.use("bmh")

# Temporary style
with plt.style.context("seaborn-v0_8"):
    fig, ax = plt.subplots()
    ax.plot(x, y)

# rcParams — global defaults
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "axes.labelsize": 12,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})
```

---

## Subplots

```python
# Grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Access axes
axes[0, 0].plot(x, np.sin(x))
axes[0, 1].plot(x, np.cos(x))
axes[1, 2].scatter(x[:20], y[:20])

# Share axes
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, sharex=True)

# GridSpec — complex layouts
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :2])    # row 0, cols 0-1
ax2 = fig.add_subplot(gs[0, 2])     # row 0, col 2
ax3 = fig.add_subplot(gs[1, :])     # row 1, all cols

ax1.plot(x, np.sin(x), title="Wide Plot")
ax2.scatter(np.random.randn(50), np.random.randn(50))
ax3.bar(range(10), np.random.rand(10))

plt.tight_layout()
plt.show()
```

---

## Seaborn

Seaborn is built on Matplotlib and provides a higher-level API for statistical visualization.

```python
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="darkgrid", palette="muted")
# styles: "darkgrid", "whitegrid", "dark", "white", "ticks"
# palettes: "muted", "bright", "deep", "pastel", "colorblind"

# Load example datasets
tips     = sns.load_dataset("tips")
iris     = sns.load_dataset("iris")
titanic  = sns.load_dataset("titanic")
flights  = sns.load_dataset("flights")
```

---

## Statistical Plots

### Distribution Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram with KDE
sns.histplot(tips["total_bill"], kde=True, bins=20, ax=axes[0])
axes[0].set_title("Histogram + KDE")

# KDE plot only
sns.kdeplot(tips["total_bill"], fill=True, ax=axes[1])
axes[1].set_title("KDE Plot")

# ECDF
sns.ecdfplot(tips["total_bill"], ax=axes[2])
axes[2].set_title("ECDF")

plt.tight_layout()
plt.show()

# Combined: rugplot + kdeplot
fig, ax = plt.subplots()
sns.kdeplot(tips["total_bill"], fill=True, ax=ax)
sns.rugplot(tips["total_bill"], ax=ax)
plt.show()
```

### Categorical Plots

```python
tips = sns.load_dataset("tips")

# Box plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, ax=axes[0])
axes[0].set_title("Box Plot")

# Violin plot
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips,
               split=True, ax=axes[1])
axes[1].set_title("Violin Plot")

# Bar plot (with CI)
sns.barplot(x="day", y="total_bill", hue="sex", data=tips,
            capsize=0.1, ax=axes[2])
axes[2].set_title("Bar Plot with CI")

plt.tight_layout()
plt.show()

# Point plot — mean with CI, connected by lines
sns.pointplot(x="day", y="total_bill", hue="sex", data=tips)
plt.show()

# Count plot — count of occurrences
sns.countplot(x="day", hue="sex", data=tips)
plt.show()

# Swarm plot — all points
sns.swarmplot(x="day", y="total_bill", data=tips, size=3)
plt.show()

# Strip plot — jittered points
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.show()
```

### Scatter and Regression

```python
# Scatter plot
sns.scatterplot(x="total_bill", y="tip", hue="sex", size="size",
                style="smoker", data=tips)
plt.show()

# Regression plot
sns.regplot(x="total_bill", y="tip", data=tips, scatter_kws={"alpha": 0.4})
plt.show()

# lmplot — regplot with facets
sns.lmplot(x="total_bill", y="tip", hue="sex", col="time",
           data=tips, height=5, aspect=0.8)
plt.show()

# Residual plot
sns.residplot(x="total_bill", y="tip", data=tips)
plt.show()
```

---

## Heatmaps and Pairplots

### Heatmap

```python
import seaborn as sns
import numpy as np
import pandas as pd

# Correlation heatmap
iris = sns.load_dataset("iris")
corr = iris.drop("species", axis=1).corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,         # show values
    fmt=".2f",          # format
    cmap="coolwarm",    # colormap
    vmin=-1, vmax=1,    # value range
    center=0,           # center colormap at 0
    square=True,        # square cells
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Heatmap")
plt.show()

# Clustered heatmap
sns.clustermap(
    corr,
    annot=True,
    cmap="coolwarm",
    figsize=(8, 8),
    method="ward",      # linkage method
    metric="euclidean",
)
plt.show()
```

### Pair Plot

```python
iris = sns.load_dataset("iris")

# Pairplot — all pairwise relationships
g = sns.pairplot(
    iris,
    hue="species",
    diag_kind="hist",   # diagonal: "hist" or "kde"
    plot_kws={"alpha": 0.6},
    height=2.5,
)
g.fig.suptitle("Iris Pairplot", y=1.02)
plt.show()

# Pairplot with regression
sns.pairplot(iris, hue="species", kind="reg", plot_kws={"scatter_kws": {"alpha": 0.3}})
plt.show()
```

### FacetGrid

```python
tips = sns.load_dataset("tips")

# FacetGrid — create grid of plots for subsets
g = sns.FacetGrid(tips, col="time", row="sex", height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.6)
g.add_legend()
g.set_axis_labels("Total Bill ($)", "Tip ($)")
g.fig.suptitle("Tips by Time and Sex", y=1.02)
plt.show()

# With histogram
g = sns.FacetGrid(tips, col="day", col_wrap=2, height=4)
g.map(sns.histplot, "total_bill", bins=15)
g.set_titles("{col_name}")
plt.show()
```

### Joint Plot

```python
# Joint distribution of two variables
g = sns.jointplot(
    x="total_bill", y="tip",
    data=tips,
    kind="hex",        # "scatter", "kde", "hist", "hex", "reg", "resid"
    color="steelblue",
    height=7,
)
g.set_axis_labels("Total Bill ($)", "Tip ($)")
plt.show()

# KDE joint plot
sns.jointplot(x="total_bill", y="tip", data=tips, kind="kde", fill=True)
plt.show()
```

---

## Saving Figures

```python
fig, ax = plt.subplots()
ax.plot(x, y)

# Save to file
fig.savefig("plot.png", dpi=300, bbox_inches="tight")
fig.savefig("plot.pdf", bbox_inches="tight")
fig.savefig("plot.svg")

# Save with transparent background
fig.savefig("plot.png", transparent=True, dpi=150)

# Save to buffer (in-memory)
import io
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
image_bytes = buf.read()

# Seaborn figures
g = sns.pairplot(iris)
g.savefig("pairplot.png", dpi=300, bbox_inches="tight")

plt.close("all")   # close all figures to free memory
```

---

## Quick Reference

### Color Palettes

```python
# Sequential
sns.color_palette("Blues")
sns.color_palette("viridis")

# Diverging
sns.color_palette("coolwarm")
sns.color_palette("RdBu")

# Qualitative
sns.color_palette("Set1")
sns.color_palette("tab10")

# Custom
sns.color_palette(["#e74c3c", "#3498db", "#2ecc71"])

# View palette
sns.color_palette("viridis", n_colors=8)
```

### Common Figure Sizes

```python
# Single plot
fig, ax = plt.subplots(figsize=(10, 6))

# Wide plot (e.g., time series)
fig, ax = plt.subplots(figsize=(14, 4))

# Square (e.g., scatter, heatmap)
fig, ax = plt.subplots(figsize=(8, 8))

# Tall (e.g., horizontal bar chart)
fig, ax = plt.subplots(figsize=(8, 12))

# Presentation (16:9)
fig, ax = plt.subplots(figsize=(16, 9))
```
