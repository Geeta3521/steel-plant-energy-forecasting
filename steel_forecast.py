#!/usr/bin/env python3
"""
Forecasting steel-plant electricity demand: SARIMA vs SARIMAX (calendar / Fourier)
UCI "Steel Industry Energy Consumption" (id 851), DAEWOO Steel, Gwangyang, 2018.

USAGE
    python steel_forecast.py --csv Steel_industry_data.csv            # real run
    python steel_forecast.py --synthetic --quick                      # pipeline smoke test ONLY
Outputs go to ./outputs (figures/, tables/).

Sections map to the rubric:  R2 = eda(), R3 = baselines + sarima diagnostics,
R4 = sarimax variants + rolling_origin().
"""
import argparse, os, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

warnings.filterwarnings("ignore")
TARGET = "Usage_kWh"

# --------------------------------------------------------------------------- #
# 0. Data loading and hourly aggregation
# --------------------------------------------------------------------------- #
def load_real(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    return df

def make_synthetic(seed=0):
    """Synthetic 15-min data with the SAME columns as the UCI file. Smoke-test only."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01 00:15", "2019-01-01 00:00", freq="15min")
    h = (idx - pd.Timedelta(minutes=15)).hour.values + 0.25
    dow = (idx - pd.Timedelta(minutes=15)).dayofweek.values
    base = 12 + 8 * np.sin(2 * np.pi * (h - 6) / 24) ** 2 + 3 * (dow < 5)
    base = base * (1 - 0.4 * (dow == 6))
    usage = np.clip(base + rng.normal(0, 2.0, len(idx)), 0.5, None)
    df = pd.DataFrame({
        "date": idx, TARGET: usage,
        "Lagging_Current_Reactive.Power_kVarh": usage * 0.4 + rng.normal(0, 1, len(idx)),
        "Leading_Current_Reactive_Power_kVarh": rng.gamma(1, 1.5, len(idx)),
        "CO2(tCO2)": usage * 4e-4 + rng.normal(0, 1e-3, len(idx)),
        "Lagging_Current_Power_Factor": 80 + rng.normal(0, 8, len(idx)),
        "Leading_Current_Power_Factor": 95 + rng.normal(0, 8, len(idx)),
    })
    return df 

def to_hourly(df):
    d = df.copy()

    # Parse dates correctly: day/month/year
    d["date"] = pd.to_datetime(
        d["date"],
        format="%d/%m/%Y %H:%M",
        errors="raise"
    )

    # Assign each interval-ending reading to its hour
    d["hour"] = (
        d["date"] - pd.Timedelta(seconds=1)
    ).dt.floor("h")

    # Sum consumption and average other numeric columns
    aggregation = {TARGET: "sum"}

    for col in d.columns:
        if col not in ("date", "hour", TARGET):
            if pd.api.types.is_numeric_dtype(d[col]):
                aggregation[col] = "mean"

    hourly = d.groupby("hour").agg(aggregation)
    hourly.index.name = "ts"

    counts = d.groupby("hour").size()

    print("Original records:", len(d))
    print("Hourly bins:", len(hourly))
    print("First timestamp:", hourly.index.min())
    print("Last timestamp:", hourly.index.max())
    print("Readings per hour:")
    print(counts.value_counts().sort_index())

    # Keep the complete 2018 calendar year
    hourly = hourly.loc["2018-01-01 00:00":"2018-12-31 23:00"]
    hourly = hourly.asfreq("h")

    missing = int(hourly[TARGET].isna().sum())

    if missing:
        print("Missing hourly target values:", missing)
        hourly[TARGET] = hourly[TARGET].interpolate(
            method="time",
            limit=2,
            limit_area="inside"
        )

    if hourly[TARGET].isna().any():
        raise ValueError(
            "Missing hourly target values remain. "
            "Inspect the timestamps before forecasting."
        )

    return hourly, missing
# --------------------------------------------------------------------------- #
# 1. Future-known regressors (calendar + Fourier)
# --------------------------------------------------------------------------- #
def korean_holidays(index):
    import holidays
    kr = holidays.KR(years=sorted(set(index.year)))
    days = pd.to_datetime(list(kr.keys()))
    return pd.Series(index.normalize().isin(days).astype(float), index=index, name="holiday")

def calendar_features(index):
    """Day-of-week dummies (Mon = baseline) + holiday flag.
    'Weekend' is NOT added separately: it is an exact linear combination of the
    day-of-week dummies, so it would make the design matrix rank-deficient."""
    X = pd.get_dummies(pd.Series(index.dayofweek, index=index), prefix="dow", drop_first=True).astype(float)
    X["holiday"] = korean_holidays(index)
    return X

def fourier(index, Kd, Kw):
    """Daily harmonics j/24, j=1..Kd, plus WEEKLY harmonics j/168, j=1..Kw with j%7!=0.
    Harmonics j=7,14,.. of the 168-h period ARE the daily harmonics, so they are dropped
    to avoid duplicates/collinearity. Remaining j (6,8,13,15,...) are the sidebands that let the
    daily shape differ between weekdays and weekends."""
    t = np.arange(len(index)) + index[0].hour + 24 * index[0].dayofweek  # phase anchored to Mon 00:00
    cols = {}
    for j in range(1, Kd + 1):
        cols[f"d_sin{j}"] = np.sin(2 * np.pi * j * t / 24)
        cols[f"d_cos{j}"] = np.cos(2 * np.pi * j * t / 24)
    for j in range(1, Kw + 1):
        if j % 7 == 0:
            continue
        cols[f"w_sin{j}"] = np.sin(2 * np.pi * j * t / 168)
        cols[f"w_cos{j}"] = np.cos(2 * np.pi * j * t / 168)
    return pd.DataFrame(cols, index=index)

def select_K(y_train, Kd_grid=(2, 4, 6, 8, 12), Kw_grid=(0, 8, 15, 22, 29)):
    """Cheap K selection: OLS of y on [calendar + Fourier], pick lowest AIC on TRAINING data only."""
    best = None
    cal = calendar_features(y_train.index)
    for Kd in Kd_grid:
        for Kw in Kw_grid:
            X = pd.concat([cal, fourier(y_train.index, Kd, Kw)], axis=1)
            m = sm.OLS(y_train.values, sm.add_constant(X.values)).fit()
            if best is None or m.aic < best[0]:
                best = (m.aic, Kd, Kw)
    return best[1], best[2]

# --------------------------------------------------------------------------- #
# 2. EDA  (R2)
# --------------------------------------------------------------------------- #
def eda(df15, hr, fig, tab):
    y = hr[TARGET]
    # 2.1 overview
    fig_, ax = plt.subplots(2, 1, figsize=(13, 6))
    y.plot(ax=ax[0], lw=.4); ax[0].set_title("Hourly Usage_kWh, 2018"); ax[0].set_ylabel("kWh")
    y.rolling(24 * 7).mean().plot(ax=ax[1], color="C1"); ax[1].set_title("7-day rolling mean (trend)")
    plt.tight_layout(); fig_.savefig(f"{fig}/01_series_trend.png", dpi=130); plt.close(fig_)

    # 2.2 STL decomposition (daily)
    stl = STL(y, period=24, robust=True).fit()
    f = stl.plot(); f.set_size_inches(13, 8); f.savefig(f"{fig}/02_stl_period24.png", dpi=110); plt.close(f)

    # 2.3 hour x weekday profile
    prof = y.groupby([y.index.dayofweek, y.index.hour]).mean().unstack(0)
    prof.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_, ax = plt.subplots(figsize=(10, 4)); prof.plot(ax=ax)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean kWh"); ax.set_title("Mean daily profile by weekday")
    plt.tight_layout(); fig_.savefig(f"{fig}/03_hour_by_weekday.png", dpi=130); plt.close(fig_)

    # 2.4 stationarity tests
    rows = []
    for name, s in [("hourly level", y), ("seasonal diff (24)", y.diff(24).dropna()),
                    ("diff(1)", y.diff().dropna()), ("diff(1) then (24)", y.diff().diff(24).dropna())]:
        adf_p = adfuller(s, autolag="AIC")[1]
        kp_p = kpss(s, regression="c", nlags="auto")[1]
        rows.append({"series": name, "ADF p-value": adf_p, "KPSS p-value": kp_p,
                     "reading": ("stationary" if adf_p < .05 and kp_p > .05 else
                                 "unit root / non-stationary" if adf_p >= .05 and kp_p <= .05 else "conflicting")})
    pd.DataFrame(rows).round(4).to_csv(f"{tab}/stationarity_tests.csv", index=False)

    # 2.5 ACF / PACF (levels and seasonally differenced)
    fig_, ax = plt.subplots(2, 2, figsize=(14, 7))
    plot_acf(y, lags=24 * 8, ax=ax[0, 0], title="ACF hourly (8 days)")
    plot_pacf(y, lags=72, ax=ax[0, 1], title="PACF hourly (72 lags)", method="ywm")
    d24 = y.diff(24).dropna()
    plot_acf(d24, lags=24 * 8, ax=ax[1, 0], title="ACF after seasonal diff (24)")
    plot_pacf(d24, lags=72, ax=ax[1, 1], title="PACF after seasonal diff (24)", method="ywm")
    plt.tight_layout(); fig_.savefig(f"{fig}/04_acf_pacf.png", dpi=120); plt.close(fig_)

    # 2.6 leakage check: correlation with target, hourly and 15-min
    cols = [c for c in df15.columns if c not in ("date", TARGET) and pd.api.types.is_numeric_dtype(df15[c])]
    cal = calendar_features(hr.index); fo = fourier(hr.index, 2, 0)
    rows = []
    for c in cols:
        rows.append({"variable": c, "group": "EXCLUDED (measured same time as load)",
                     "corr_hourly": hr[c].corr(y), "corr_15min": df15[c].corr(df15[TARGET])})
    for c in ["dow_5", "dow_6", "holiday"]:
        if c in cal: rows.append({"variable": c, "group": "ALLOWED (known in advance)", "corr_hourly": cal[c].corr(y), "corr_15min": np.nan})
    for c in fo.columns[:4]:
        rows.append({"variable": c, "group": "ALLOWED (known in advance)", "corr_hourly": fo[c].corr(y), "corr_15min": np.nan})
    pd.DataFrame(rows).round(3).to_csv(f"{tab}/leakage_correlations.csv", index=False)
    return stl

# --------------------------------------------------------------------------- #
# 3. Metrics
# --------------------------------------------------------------------------- #
def metrics(actual, pred, scale):
    e = actual - pred
    nz = np.abs(actual) > 1e-6
    return {
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "MAPE": 100 * np.mean(np.abs(e[nz] / actual[nz])),
        "sMAPE": 100 * np.mean(2 * np.abs(e) / (np.abs(actual) + np.abs(pred) + 1e-9)),
        "WAPE": 100 * np.sum(np.abs(e)) / np.sum(np.abs(actual)),
        "MASE": np.mean(np.abs(e)) / scale,
    }

# --------------------------------------------------------------------------- #
# 4. Models
# --------------------------------------------------------------------------- #
# Model specs.  Seasonal differencing (D=1) and DAILY Fourier terms are substitutes: a period-24 sinusoid is
# annihilated by the lag-24 difference, so the Fourier variants use D=0 and let the harmonics carry the
# daily shape.  The calendar-only variant keeps D=1 (hour-of-day is then handled by differencing).
SPECS = {
    "SARIMA":           dict(order=(1, 0, 1), sorder=(1, 1, 0, 24), exog=None,       diff24=False),
    "SARIMAX_calendar": dict(order=(1, 0, 1), sorder=(1, 1, 0, 24), exog="calendar", diff24=True),
    "SARIMAX_fourier":  dict(order=(2, 0, 1), sorder=(1, 0, 0, 24), exog="fourier",  diff24=False),
    "SARIMAX_both":     dict(order=(2, 0, 1), sorder=(1, 0, 0, 24), exog="both",     diff24=False),
}

def build_exog(index, variant, Kd, Kw):
    kind = SPECS[variant]["exog"]
    if kind is None:
        return None
    cal = calendar_features(index)
    if kind == "calendar":
        return cal
    fo = fourier(index, Kd, Kw)
    return fo if kind == "fourier" else pd.concat([cal, fo], axis=1)

class RegSarima:
    """Regression with SARIMA errors, estimated in two stages (much faster than joint MLE in statsmodels):
       (1) beta by OLS (on lag-24 differences when the error model differences at 24),
       (2) SARIMA fitted by MLE to u = y - X beta.   Forecast = X_future beta + SARIMA forecast of u.
       Coefficients are consistent; standard errors from stage 2 ignore stage-1 uncertainty."""
    def __init__(self, spec, maxiter=40, low_memory=False):
        self.spec, self.maxiter, self.low_memory = spec, maxiter, low_memory
        self.beta, self.intercept = None, 0.0
    def _xb(self, X):
        return pd.Series(X.values @ self.beta + self.intercept, index=X.index)
    def _sarima(self, u):
        return SARIMAX(u, order=self.spec["order"], seasonal_order=self.spec["sorder"],
                       enforce_stationarity=False, enforce_invertibility=False)
    def fit(self, y, X):
        u = y
        if X is not None:
            Xv, yv = X.values, y.values
            if self.spec["diff24"]:
                self.beta = np.linalg.lstsq(Xv[24:] - Xv[:-24], yv[24:] - yv[:-24], rcond=None)[0]
                self.intercept = 0.0
            else:
                b = np.linalg.lstsq(np.column_stack([np.ones(len(Xv)), Xv]), yv, rcond=None)[0]
                self.intercept, self.beta = b[0], b[1:]
            u = y - self._xb(X)
        self.res = self._sarima(u).fit(disp=False, maxiter=self.maxiter, method="lbfgs", low_memory=self.low_memory)
        return self
    def update(self, y, X):
        u = y if X is None else y - self._xb(X)
        self.res = self._sarima(u).filter(self.res.params, low_memory=self.low_memory)   # keep parameters, refilter on new data
        return self
    def forecast(self, H, Xf):
        f = np.asarray(self.res.forecast(H))
        return f if Xf is None else f + self._xb(Xf).values

def rolling_origin(hr, variants, Kd, Kw, init_frac=0.6, step=168, H=168, refit_every=4,
                   max_origins=None, log=print):
    y = hr[TARGET]; n = len(y)
    origins = list(range(int(n * init_frac), n - H + 1, step))
    if max_origins:
        origins = origins[:max_origins]
    scale = np.mean(np.abs(y.values[24:origins[0]] - y.values[:origins[0] - 24]))   # in-sample seasonal-naive(24) MAE
    Xs = {v: build_exog(y.index, v, Kd, Kw) for v in variants}
    models, store = {}, []
    log(f"{len(origins)} origins, first train size {origins[0]}, MASE scale {scale:.3f}")
    for i, o in enumerate(origins):
        act = y.values[o:o + H]
        preds = {"snaive_24": np.tile(y.values[o - 24:o], H // 24 + 1)[:H],
                 "snaive_168": y.values[o - 168:o][:H]}
        for v in variants:
            t0 = time.time(); X = Xs[v]
            Xtr = None if X is None else X.iloc[:o]
            Xfc = None if X is None else X.iloc[o:o + H]
            if i % refit_every == 0 or v not in models:
                models[v] = RegSarima(SPECS[v], low_memory=True).fit(y.iloc[:o], Xtr)
            else:
                models[v].update(y.iloc[:o], Xtr)
            preds[v] = models[v].forecast(H, Xfc)
            log(f"  origin {i+1}/{len(origins)} {v:18s} {time.time()-t0:5.1f}s")
        for m, p in preds.items():
            store.append((o, m, act, p))
    rows = []
    order = list(preds)
    for h in (24, 168):
        for m in order:
            a = np.concatenate([s[2][:h] for s in store if s[1] == m])
            p = np.concatenate([s[3][:h] for s in store if s[1] == m])
            rows.append({"horizon": h, "model": m, **metrics(a, p, scale), "n_origins": len(origins)})
    return pd.DataFrame(rows), store, origins

# --------------------------------------------------------------------------- #
# 5. Residual diagnostics (R3) + significance
# --------------------------------------------------------------------------- #
def diagnostics(res, name, fig, tab, burn=24 * 8):
    r = pd.Series(res.resid).iloc[burn:]            # drop diffuse-initialisation burn-in
    lb = acorr_ljungbox(r, lags=[24, 48, 168], return_df=True)
    lb.to_csv(f"{tab}/ljungbox_{name}.csv")
    fig_, ax = plt.subplots(2, 2, figsize=(13, 7))
    r.plot(ax=ax[0, 0], lw=.4, title=f"{name}: residuals")
    r.hist(bins=60, ax=ax[0, 1]); ax[0, 1].set_title("Residual histogram")
    plot_acf(r, lags=24 * 8, ax=ax[1, 0], title="Residual ACF")
    sm.qqplot(r, line="s", ax=ax[1, 1]); ax[1, 1].set_title("Q-Q plot")
    plt.tight_layout(); fig_.savefig(f"{fig}/resid_{name}.png", dpi=120); plt.close(fig_)
    return lb

def diebold_mariano(e1, e2, h=1):
    """DM test on absolute errors, HAC variance with h-1 lags. Negative stat => model1 better."""
    d = np.abs(e1) - np.abs(e2); n = len(d); dm = d - d.mean()
    var = np.sum(dm ** 2) / n
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.sum(dm[k:] * dm[:-k]) / n
    from scipy import stats
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="outputs"); ap.add_argument("--quick", action="store_true",
                    help="few origins; for smoke tests")
    ap.add_argument("--refit-every", type=int, default=4)
    a = ap.parse_args()
    fig, tab = f"{a.out}/figures", f"{a.out}/tables"
    os.makedirs(fig, exist_ok=True); os.makedirs(tab, exist_ok=True)

    df15 = make_synthetic() if a.synthetic else load_real(a.csv)
    if a.synthetic: print("*** SYNTHETIC DATA: numbers are meaningless, pipeline test only ***")
    hr, nmiss = to_hourly(df15)
    print(f"hourly points: {len(hr)}, filled gaps: {nmiss}, min hourly kWh: {hr[TARGET].min():.2f}")

    eda(df15, hr, fig, tab)

    y = hr[TARGET]; init = int(len(y) * .6)
    Kd, Kw = select_K(y.iloc[:init]); print(f"K selected on training data by AIC: Kd={Kd}, Kw={Kw}")
    pd.DataFrame([{"Kd": Kd, "Kw": Kw}]).to_csv(f"{tab}/selected_K.csv", index=False)

    variants = ["SARIMA", "SARIMAX_calendar", "SARIMAX_fourier", "SARIMAX_both"]
    # initial-train fits for residual diagnostics + coefficient tables
    for v in ("SARIMA", "SARIMAX_both"):
        X = build_exog(y.index, v, Kd, Kw)
        mdl = RegSarima(SPECS[v]).fit(y.iloc[:init], None if X is None else X.iloc[:init])
        diagnostics(mdl.res, v, fig, tab)
        if mdl.beta is not None:
            pd.DataFrame({"coef": mdl.beta}, index=list(X.columns)).round(4).to_csv(f"{tab}/beta_{v}.csv")
        pd.DataFrame({"coef": mdl.res.params, "se": mdl.res.bse}).round(4).to_csv(f"{tab}/arma_params_{v}.csv")

    results, store, origins = rolling_origin(hr, variants, Kd, Kw, refit_every=a.refit_every,
                                             max_origins=3 if a.quick else None)
    results.round(3).to_csv(f"{tab}/rolling_origin_results.csv", index=False)
    print(results.round(3).to_string(index=False))

    # DM tests vs plain SARIMA, per horizon
    rows = []
    for h in (24, 168):
        base = np.concatenate([s[2][:h] - s[3][:h] for s in store if s[1] == "SARIMA"])
        for v in variants[1:]:
            e = np.concatenate([s[2][:h] - s[3][:h] for s in store if s[1] == v])
            st, p = diebold_mariano(e, base, h=h)
            rows.append({"horizon": h, "model": v, "vs": "SARIMA", "DM_stat": st, "p_value": p})
    pd.DataFrame(rows).round(4).to_csv(f"{tab}/dm_tests.csv", index=False)

    # example forecast plot (last origin)
    o, H = origins[-1], 168
    fig_, ax = plt.subplots(figsize=(13, 4))
    ax.plot(y.index[o - 96:o + H], y.values[o - 96:o + H], "k", lw=1, label="actual")
    for m in ("snaive_168", "SARIMA", "SARIMAX_both"):
        p = [s[3] for s in store if s[0] == o and s[1] == m][0]
        ax.plot(y.index[o:o + H], p, lw=1, label=m)
    ax.axvline(y.index[o], color="grey", ls=":"); ax.legend(); ax.set_title("168-h forecast, last origin")
    plt.tight_layout(); fig_.savefig(f"{fig}/05_forecast_example.png", dpi=130); plt.close(fig_)
    print("done ->", a.out)

if __name__ == "__main__":
    main()