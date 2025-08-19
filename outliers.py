import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

def compute_iqr_report_df(
    df: pd.DataFrame,
    columns=None,
    whisker: float = 1.5,
    digits: int = 3
) -> pd.DataFrame:
    """
    Liefert je Spalte einen IQR-Report als DataFrame:
    Index = Feature-Namen, Spalten = ['Q1','Q3','IQR','lower','upper','n_outliers'].

    Hinweise:
    - Grenzen werden nach IQR-Methode berechnet (Q1 ± whisker*IQR).
    - Bei IQR==0 werden lower/upper auf Q1/Q3 gesetzt.
    - Nicht-numerische Spalten werden ignoriert (oder per 'columns' explizit wählen).
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns

    # numerische Werte (robust gegen Strings/NaN)
    num = df[columns].apply(pd.to_numeric, errors="coerce")

    # Quartile & IQR (spaltenweise)
    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr

    # IQR==0 sauber behandeln
    lower = lower.where(iqr != 0, q1)
    upper = upper.where(iqr != 0, q3)

    # Ausreißer zählen (Broadcasting spaltenweise)
    out_mask = (num.lt(lower, axis=1)) | (num.gt(upper, axis=1))
    n_out = out_mask.sum(axis=0).astype(int)

    report = pd.DataFrame({
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower": lower, "upper": upper,
        "n_outliers": n_out
    })

    # runden (nur die Float-Spalten)
    float_cols = ["Q1","Q3","IQR","lower","upper"]
    report[float_cols] = report[float_cols].round(digits)

    return report


def plot_box_outliers_matrix(
    df: pd.DataFrame,
    columns=None,
    grid_cols: int = 2,
    whisker: float = 1.5,
    bounds: pd.DataFrame | None = None,   # akzeptiert Report mit Spalten ['lower','upper', ...]
    title: str = "Boxplots mit Ausreißern",
    point_size_normal: int = 5,
    point_size_outlier: int = 6,
    color_normal: str = "blue",
    color_outlier: str = "red",
    opacity_normal: float = 0.6,
    opacity_outlier: float = 0.85,
    height_per_row: int = 300,
    width_per_col: int = 320,
    showlegend_once: bool = True,
):
    """
    Zeichnet pro Feature einen Boxplot und markiert Punkte außerhalb der IQR-Grenzen farbig.
    - Wenn 'bounds' None ist: Grenzen/Report via compute_iqr_report_df(df, columns, whisker) schätzen.
    - Wenn 'bounds' gegeben: erwartet Spalten 'lower' und 'upper' (weitere Spalten sind ok).
    Rückgabe: (fig, bounds_report_df)
    """
    # Spalten bestimmen
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    else:
        columns = list(columns)

    # Bounds/Report vorbereiten
    if bounds is None:
        bounds_report = compute_iqr_report_df(df, columns=columns, whisker=whisker, digits=6)
    else:
        # Sicherstellen, dass 'lower'/'upper' vorhanden sind
        if not {"lower", "upper"}.issubset(bounds.columns):
            raise ValueError("Übergebene 'bounds' benötigen Spalten ['lower','upper'].")
        # Nur relevante Spalten & Reihenfolge angleichen
        bounds_report = bounds.copy()
        bounds_report = bounds_report.loc[bounds_report.index.intersection(columns)]
        # auf gewünschte Reihenfolge reindizieren (Spalten, nicht 'columns'!)
        bounds_report = bounds_report.reindex(index=columns)

    # Falls einzelne gewünschte Spalten fehlen, filtern
    columns = [c for c in columns if c in bounds_report.index]

    grid_rows = math.ceil(len(columns) / grid_cols) if len(columns) else 1
    fig = make_subplots(rows=grid_rows, cols=grid_cols, subplot_titles=columns)

    for i, col in enumerate(columns):
        r = i // grid_cols + 1
        c = i % grid_cols + 1

        s = pd.to_numeric(df[col], errors="coerce")
        lo = bounds_report.loc[col, "lower"]
        up = bounds_report.loc[col, "upper"]

        is_out = ((s < lo) | (s > up)).fillna(False)

        # Box (ohne Punkte)
        fig.add_trace(
            go.Box(y=s, name=col, boxpoints=False, showlegend=False),
            row=r, col=c
        )
        # Punkte normal
        fig.add_trace(
            go.Scatter(
                y=s[~is_out],
                x=[col] * (~is_out).sum(),
                mode="markers",
                marker=dict(size=point_size_normal, color=color_normal, opacity=opacity_normal),
                name="Normal",
                showlegend=(showlegend_once and i == 0),
            ),
            row=r, col=c
        )
        # Punkte Ausreißer
        fig.add_trace(
            go.Scatter(
                y=s[is_out],
                x=[col] * (is_out).sum(),
                mode="markers",
                marker=dict(size=point_size_outlier, color=color_outlier, opacity=opacity_outlier),
                name="Ausreißer",
                showlegend=(showlegend_once and i == 0),
            ),
            row=r, col=c
        )

    fig.update_layout(
        height=max(1, height_per_row * grid_rows),
        width=width_per_col * grid_cols,
        title=title,
        margin=dict(t=60, l=40, r=20, b=40),
        showlegend=True
    )
    return fig, bounds_report

import pandas as pd

def iqr_clip_cols(
    df: pd.DataFrame,
    columns,
    whisker: float = 1.5,
    bounds: pd.DataFrame | None = None,
    return_bounds: bool = False
):
    """
    Clipt (winsorisiert) Werte in numerischen Spalten auf IQR-basierte Grenzen.

    Zweck
    -----
    Diese Funktion begrenzt Ausreißer nach der IQR-Methode.
    Für jede angegebene Spalte werden Unter- und Obergrenzen gemäß
    Q1 - whisker*IQR bzw. Q3 + whisker*IQR verwendet. Liegen Werte
    außerhalb dieser Grenzen, werden sie auf die jeweilige Grenze
    zurückgeschnitten (Clipping/Winsorizing). So bleiben alle Zeilen
    erhalten, extreme Werte verzerren jedoch weniger.

    Arbeitsweise
    ------------
    - Wenn `bounds` None ist, werden die IQR-Grenzen **aus `df` geschätzt**
      (Q1=0.25-Quantil, Q3=0.75-Quantil, IQR=Q3-Q1) und anschließend geclippt.
    - Wenn `bounds` übergeben wird, werden **nur diese Grenzen angewendet**
      (kein neues Schätzen). `bounds` muss ein DataFrame mit
      Index = Spaltennamen und Spalten = ["lower", "upper"] sein.

    Wichtiger Praxis-Hinweis (Leakage)
    ----------------------------------
    Im ML-Workflow sollten die Grenzen immer nur auf Basis der **Trainingsdaten**
    gelernt werden (z. B. `X_train`) und anschließend **identisch** auf
    Trainings- und Testdaten angewendet werden. Dadurch wird Data Leakage
    vermieden.

    Parameter
    ---------
    df : pd.DataFrame
        Eingabedaten (im ML-Fall typischerweise X oder X_train).
    columns : list[str] | Index
        Zu clipende Spalten (numerisch).
    whisker : float, default 1.5
        Faktor für die IQR-Grenzen (klassisch 1.5; größer = toleranter, kleiner = strenger).
    bounds : pd.DataFrame | None, default None
        Vorgegebene Grenzen mit Struktur:
        index = Spaltennamen, columns = ["lower", "upper"].
        Wird None übergeben, werden Grenzen aus `df` geschätzt.
    return_bounds : bool, default False
        Wenn True, wird zusätzlich das verwendete `bounds`-DataFrame zurückgegeben.

    Rückgabe
    --------
    df_out : pd.DataFrame
        Kopie von `df` mit geclippten Werten in `columns`.
    (df_out, bounds) : tuple
        Wenn `return_bounds=True`, zusätzlich das verwendete/neu geschätzte
        Grenzen-DataFrame.

    Anmerkungen
    -----------
    - Bei IQR == 0 (konstante Spalte) sind die Grenzen Q1==Q3; Clipping hat dann
      keine Auswirkung über den konstanten Wert hinaus.
    - Die Funktion berechnet Quantile auf numerisch konvertierten Werten
      (`errors="coerce"`). Nicht-numerische Spalten sollten vorab aus `columns`
      ausgeschlossen werden.

    Beispiel
    --------
    >>> # Grenzen auf X_train schätzen und anwenden
    >>> X_train_clipped, train_bounds = iqr_clip_cols(X_train, columns=num_cols, return_bounds=True)
    >>> # Dieselben Grenzen auf X_test anwenden (kein neues Fitten)
    >>> X_test_clipped = iqr_clip_cols(X_test, columns=num_cols, bounds=train_bounds)
    """
    df_out = df.copy()

    if bounds is None:
        num = df_out[columns].apply(pd.to_numeric, errors="coerce")
        q = num.quantile([0.25, 0.75])
        iqr = q.loc[0.75] - q.loc[0.25]
        lower = q.loc[0.25] - whisker * iqr
        upper = q.loc[0.75] + whisker * iqr
        bounds = pd.DataFrame({"lower": lower, "upper": upper})

    df_out[columns] = df_out[columns].clip(lower=bounds["lower"], upper=bounds["upper"], axis=1)

    if return_bounds:
        return df_out, bounds
    return df_out
