import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def hist_skew_matrix(
    df: pd.DataFrame,
    columns=None,
    nbins: int = 30,
    grid_cols: int = 2,
    density: bool = False,
    round_digits: int = 4,
    height_per_row: int = 300,
    width_per_col: int = 380,
    title: str = "Histogramme mit Schiefe (skew)",
    show_legend: bool = False,
):
    """
    Erzeugt eine Subplot-Matrix von Histogrammen für numerische Spalten und schreibt
    pro Subplot die Schiefe (skew) und n in den Titel.

    Rückgabe
    --------
    fig : plotly.graph_objects.Figure
    skews : pd.Series
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        columns = list(columns)

    grid_rows = max(1, math.ceil(len(columns) / grid_cols))
    n_cells = grid_rows * grid_cols

    # Wichtig: Anzahl subplot_titles = rows * cols
    fig = make_subplots(
        rows=grid_rows,
        cols=grid_cols,
        subplot_titles=[""] * n_cells
    )

    skews = {}
    for i, col in enumerate(columns):
        r = i // grid_cols + 1
        c = i % grid_cols + 1

        s = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
        skew_val = float(s.skew())
        skews[col] = skew_val

        fig.add_trace(
            go.Histogram(
                x=s,
                nbinsx=nbins,
                histnorm="probability density" if density else None,
                name=col,
                showlegend=show_legend,
            ),
            row=r, col=c
        )

        fig.update_xaxes(title_text=col, row=r, col=c)
        fig.update_yaxes(title_text="Dichte" if density else "Häufigkeit", row=r, col=c)

        # Titeltext zusammenbauen
        ann_text = f"{col} | Skew={skew_val:.{round_digits}f} | n={len(s)}"

        # Sicheres Setzen des Subplot-Titels:
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].text = ann_text
        else:
            # Fallback: Annotation direkt über dem Subplot platzieren
            fig.add_annotation(
                text=ann_text,
                x=0.5, y=1.08,
                xref="x domain", yref="y domain",
                showarrow=False,
                row=r, col=c
            )

    fig.update_layout(
        height=height_per_row * grid_rows,
        width=width_per_col * grid_cols,
        title=title,
        bargap=0.05,
        margin=dict(t=60, l=40, r=20, b=40),
        showlegend=show_legend
    )
    return fig, pd.Series(skews, name="skew")
