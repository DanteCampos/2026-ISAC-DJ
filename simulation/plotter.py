import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

def plot_isac_satisfaction_heatmap(
    data,
    realization=0,
    grid_size=300,
    cmap="viridis",
    fig=None,
    ax=None,
    no_legend=False,
    icon_size=200,
    no_sensor_text=False,
):

    # ============================================================
    # Extract metric
    # ============================================================

    comm_sla_satisfied = data["comm_sla_satisfied"][realization].reshape(grid_size, grid_size)
    sensing_sla_satisfied = data["sensing_sla_satisfied"][realization].reshape(grid_size, grid_size)

    # ============================================================
    # Apply masks
    # ============================================================

    mask_ue = data["ue_mask"][realization].reshape(
        grid_size,
        grid_size
    )
    mask_target = data["target_mask"][realization].reshape(
        grid_size,
        grid_size
    )


    z = np.full_like(comm_sla_satisfied, np.nan, dtype=float)
    valid_mask = mask_ue | mask_target
    z[valid_mask] = 0
    z[valid_mask & ~sensing_sla_satisfied & comm_sla_satisfied] = 1
    z[valid_mask & sensing_sla_satisfied & ~comm_sla_satisfied] = 2
    z[valid_mask & sensing_sla_satisfied & comm_sla_satisfied] = 3


    # ============================================================
    # Plot
    # ============================================================

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    r = data["CELL_RANGE"]

    c = plt.get_cmap(cmap)

    cmap = ListedColormap([
        c(0.05),   # dark purple
        c(0.35),   # blue-green
        c(0.65),   # green
        c(0.95),   # yellow
    ])

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5],
        cmap.N
    )


    im = ax.imshow(
        z,
        extent=[-r, r, -r, r],
        origin="lower",
        cmap=cmap,
        norm=norm
    )

    # ============================================================
    # Plot sensors
    # ============================================================

    sensor_mask = data["sensor_mask"][realization]

    sensor_x = data["sensor_x"][realization][sensor_mask]
    sensor_y = data["sensor_y"][realization][sensor_mask]

    boresights = data["sensor_boresights"][realization][sensor_mask]

    ax.scatter(
        sensor_x,
        sensor_y,
        c="red",
        s=icon_size,
        marker="s",
        label="Sensor",
        zorder=5
    )

    arrow_length = 0.12 * r

    for i in range(len(sensor_x)):

        dx = arrow_length * np.cos(boresights[i])
        dy = arrow_length * np.sin(boresights[i])

        ax.quiver(
            sensor_x[i],
            sensor_y[i],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.006,
            zorder=4
        )

        ax.quiver(
            sensor_x[i],
            sensor_y[i],
            -dx,
            -dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.006,
            zorder=4
        )
        if not no_sensor_text:
            ax.text(
                sensor_x[i],
                sensor_y[i],
                str(i),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                zorder=7
            )

    # ============================================================
    # Cell boundary
    # ============================================================

    circle = plt.Circle(
        (0, 0),
        r,
        fill=False,
        color="black",
        linestyle="--",
        linewidth=2
    )

    ax.add_patch(circle)

    # ============================================================
    # Base station
    # ============================================================

    ax.scatter(
        0,
        0,
        c="black",
        s=icon_size,
        marker="^",
        label="ISAC Base Station",
        zorder=5
    )

    for i in range(data["N_SECTORS"]):

        boresight = data["BS_BORESIGHTS"][i]

        dx = arrow_length * np.cos(boresight)
        dy = arrow_length * np.sin(boresight)

        ax.quiver(
            0,
            0,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            width=0.006,
            zorder=4
        )

    # ============================================================
    # Labels
    # ============================================================
    
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        pad=0.03,
        aspect=35,
        location="right",
        shrink=0.8
    )

    cbar.set_ticks([3, 2, 1, 0],)

    cbar.set_ticklabels([
        "Both",
        "SaaS",
        "CaaS",
        "None",
    ])

    cbar.set_label(
        "ISAC Slice SLAs Satisfied",
        labelpad=3,
        fontsize=8
    )
    
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_aspect("equal")

    ax.set_yticks([-400, -200, 0, 200, 400])

    return fig, ax