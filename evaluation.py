import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

def preview_regression_errors(
    y_true,
    y_pred,
    n: int = 10,
    sort_by: str | None = None,   # z.B. "abs_error_desc" oder "sq_error_desc"
    digits: int = 3,
    add_summary: bool = False     # hängt eine Summenzeile mit MAE/MAPE/RMSE an (über alle Zeilen)
) -> pd.DataFrame:
    """
    Erzeugt eine kompakte Vorschau-Tabelle: Tatsächlich vs. Vorhersage + Fehlermaße.

    Parameter
    ---------
    y_true, y_pred : array-like oder pd.Series gleicher Länge
    n : int
        Anzahl der Zeilen zur Anzeige (nach optionaler Sortierung).
    sort_by : str | None
        - None: keine Sortierung, es werden die ersten n Elemente gezeigt.
        - "abs_error_desc": nach Absolutfehler absteigend.
        - "sq_error_desc" : nach quadratischem Fehler absteigend.
        - "perc_abs_error_desc": nach absolutem Prozentfehler absteigend.
    digits : int
        Rundung in der Ausgabe.
    add_summary : bool
        Fügt eine Summenzeile mit MAE, MAPE (ohne y=0), RMSE über den GESAMTEN Input an.

    Rückgabe
    --------
    pd.DataFrame
        Tabelle mit Spalten:
        - "Tatsächlicher Wert"
        - "Vorhergesagter Wert"
        - "Abweichung (Residuum)"           = y_true - y_pred
        - "Abweichung (%)"                  = (Residuum / y_true) * 100, bei y_true=0 -> NaN
        - "Absoluter Fehler"                = |Residuum|
        - "Absoluter Fehler (%)"            = (|Residuum| / |y_true|) * 100, bei y_true=0 -> NaN
        - "Quadratischer Fehler"            = (Residuum)^2
    """
    # In Arrays/Series umwandeln und Index übernehmen, falls Series
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)

    resid = y_true - y_pred
    abs_err = resid.abs()
    sq_err = resid.pow(2)

    # Prozentfehler robust (bei y=0 -> NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        perc_err = resid / y_true * 100.0
        perc_abs_err = abs_err / y_true.abs() * 100.0
        perc_err = perc_err.replace([np.inf, -np.inf], np.nan)
        perc_abs_err = perc_abs_err.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        "Tatsächlicher Wert": y_true,
        "Vorhergesagter Wert": y_pred,
        "Abweichung (Residuum)": resid,
        "Abweichung (%)": perc_err,
        "Absoluter Fehler": abs_err,
        "Absoluter Fehler (%)": perc_abs_err,
        "Quadratischer Fehler": sq_err
    })

    # optional sortieren, dann Top-n auswählen
    if sort_by is None:
        df_out = df.head(n).copy()
    else:
        key = {
            "abs_error_desc": "Absoluter Fehler",
            "sq_error_desc": "Quadratischer Fehler",
            "perc_abs_error_desc": "Absoluter Fehler (%)"
        }.get(sort_by, None)
        if key is None:
            raise ValueError("sort_by unbekannt. Erlaubt: None, 'abs_error_desc', 'sq_error_desc', 'perc_abs_error_desc'.")
        df_out = df.sort_values(key, ascending=False).head(n).copy()

    # runden
    float_cols = df_out.select_dtypes(include="number").columns
    df_out[float_cols] = df_out[float_cols].round(digits)

    # optional Summary-Zeile (über alle Daten, nicht nur Top-n)
    if add_summary:
        mae = abs_err.mean()
        rmse = np.sqrt(sq_err.mean())
        mape = perc_abs_err.dropna().mean()  # MAPE: nur dort, wo y!=0
        summary = pd.Series({
            "Tatsächlicher Wert": np.nan,
            "Vorhergesagter Wert": np.nan,
            "Abweichung (Residuum)": np.nan,
            "Abweichung (%)": np.nan,
            "Absoluter Fehler": mae,
            "Absoluter Fehler (%)": mape,
            "Quadratischer Fehler": rmse**2
        }, name="⟨Summary (MAE/MAPE/RMSE²)⟩").round(digits)
        df_out = pd.concat([df_out, summary.to_frame().T], axis=0)

    return df_out

def evaluate_regression(
    model,
    X_train, y_train,
    X_test, y_test,
    *,                              # ab hier nur Keyword-Argumente
    cv_splits: int = 5,
    cv_shuffle: bool = True,
    cv_random_state: int = 42,
    scoring: str = "r2",            # Metrik für die Cross-Validation (z. B. "r2", "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error")
    return_print: bool = True,      # Ergebnisse ausgeben
    with_train_metrics: bool = False,   # zusätzlich Training-Metriken berechnen (Overfitting-Check)
    extra_test_metrics: bool = True,    # zusätzlich MAE, MedAE, MAPE zurückgeben
    return_model: bool = False,         # das gefittete Modell mit zurückgeben
    return_predictions: bool = False    # y_pred (Test) mit zurückgeben
) -> Dict[str, Any]:
    """
    Bewertet ein Regressionsmodell mit Cross-Validation (auf dem Trainingsset) und Test-Set-Metriken.

    Ablauf
    ------
    1) Cross-Validation auf X_train/y_train (KFold).
    2) Frische Modellkopie fitten (Train) und auf X_test vorhersagen.
    3) Test-Metriken berechnen (R², RMSE, optional MAE/MedAE/MAPE).
    4) Adjusted R² auf dem Test-Set berechnen (mit n_test und p Features).

    Hinweise
    --------
    - 'scoring' nutzt die Scikit-Learn-Scorer-Strings (z. B. "r2", "neg_root_mean_squared_error").
      Achtung: "neg_*" sind negative Fehler (Scikit-Learn-Konvention); die Funktion selbst berechnet RMSE_test positiv.
    - Adjusted R² nutzt n = Anzahl Test-Beobachtungen und p = Anzahl Features im Test.
    - Für robuste Ergebnisse ist es ideal, alle Preprocessing-Schritte (Clipping, Skalierung, Encoding)
      mit Pipelines zu kapseln und das Pipeline-Objekt hier zu übergeben.

    Rückgabe
    --------
    dict mit u. a.:
        - R2_test, Adj_R2_test, RMSE_test
        - (optional) MAE_test, MedAE_test, MAPE_test
        - CV_mean, CV_std, CV_metric, cv_splits
        - n_test, p_features
        - (optional) R2_train, RMSE_train
        - (optional) y_pred, model
    """
    # --- 1) Cross-Validation auf dem Trainingsset ---
    cv = KFold(n_splits=cv_splits, shuffle=cv_shuffle, random_state=cv_random_state)
    cv_scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores, ddof=1)) if len(cv_scores) > 1 else 0.0

    # --- 2) Modell fitten und Test vorhersagen ---
    fitted = clone(model).fit(X_train, y_train)
    y_pred = fitted.predict(X_test)

    # --- 3) Testmetriken ---
    r2_test = float(r2_score(y_test, y_pred))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Adjusted R²
    n_test = int(getattr(X_test, "shape", (len(y_test),))[0])
    p = int(getattr(X_test, "shape", (None, None))[1] or 0)
    denom = n_test - p - 1
    if denom <= 0:
        adj_r2_test = np.nan
    else:
        adj_r2_test = float(1 - (1 - r2_test) * (n_test - 1) / denom)

    results: Dict[str, Any] = {
        "R2_test": r2_test,
        "Adj_R2_test": adj_r2_test,
        "RMSE_test": rmse_test,
        "CV_mean": float(cv_mean),
        "CV_std": float(cv_std),
        "CV_metric": scoring,
        "n_test": n_test,
        "p_features": p,
        "cv_splits": int(cv_splits),
    }

    # Optional: zusätzliche Testmetriken
    if extra_test_metrics:
        mae = float(mean_absolute_error(y_test, y_pred))
        medae = float(median_absolute_error(y_test, y_pred))
        # MAPE robust: y_true == 0 → NaN ignorieren
        y_true = pd.Series(y_test)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_series = (np.abs(y_true - y_pred) / y_true.abs()) * 100.0
        mape = float(np.nanmean(np.where(np.isfinite(mape_series), mape_series, np.nan)))
        results.update({
            "MAE_test": mae,
            "MedAE_test": medae,
            "MAPE_test": mape,
        })

    # Optional: Train-Metriken (Overfitting-Schnellcheck)
    if with_train_metrics:
        y_pred_train = fitted.predict(X_train)
        results.update({
            "R2_train": float(r2_score(y_train, y_pred_train)),
            "RMSE_train": float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
        })

    # Optional: Rückgaben erweitern
    if return_model:
        results["model"] = fitted
    if return_predictions:
        results["y_pred"] = np.asarray(y_pred)

    # Ausgabe
    if return_print:
        print(f"RMSE (Test): {results['RMSE_test']:.4f}")
        print(f"R² (Test): {results['R2_test']:.4f}")
        print(f"Adjusted R² (Test): {results['Adj_R2_test']:.4f}" if np.isfinite(results['Adj_R2_test']) else "Adjusted R² (Test): n/a")
        if extra_test_metrics:
            print(f"MAE (Test): {results['MAE_test']:.4f} | MedAE: {results['MedAE_test']:.4f} | MAPE: {results['MAPE_test']:.2f}%")
        print(f"CV-{scoring.upper()} (Train)  Mittel: {results['CV_mean']:.4f} | Std: {results['CV_std']:.4f} | Folds: {cv_splits}")

        if with_train_metrics:
            print(f"RMSE (Train):       {results['RMSE_train']:.4f}")
            print(f"R² (Train):         {results['R2_train']:.4f}")

    return results

