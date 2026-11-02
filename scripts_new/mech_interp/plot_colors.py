"""Shared colors for mechanistic-interpretability figures."""
from matplotlib.colors import LinearSegmentedColormap


# Sequential magnitudes: low values are blue and high values are red.
BLUE_TO_RED = LinearSegmentedColormap.from_list(
    "activation_blue_to_red", ("#2166ac", "#b2182b")
)

# Signed differences: negative values are blue and positive values are red.
SIGNED_BLUE_RED = "coolwarm"
