"""Core routines for 'Persistence of sourcing responses to supplier disruptions in staple-crop trade'.

Inputs (DATA_DIR, default ./data), produced by download_data.py, extract_faostat.py, build_panel.py and policy.py:
  panel_dyad.parquet             reconciled bilateral flows (importer, supplier, crop, year)
  fao_prod.parquet               FAOSTAT production, area and yield
  cru_monthly.parquet            CRU-CY v4.10 country-average monthly tmx, pre, pet
  cru_crosswalk.csv              FAO area code -> CRU country file name
  policy_years_<set>.csv         audited export-restriction days in force per supplier, crop and year
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import least_squares

D = os.environ.get('DATA_DIR', 'data') + '/'
COMMODS = ['wheat', 'maize', 'rice', 'soybeans', 'barley', 'sorghum']
PRODITEM = {15: 'wheat', 56: 'maize', 27: 'rice', 236: 'soybeans', 44: 'barley', 83: 'sorghum'}
KMIN, KMAX = -3, 8
THR_PROD = float(np.log(0.85))   # production at least 15% below trend
THR_YLD = float(np.log(0.90))    # yield at least 10% below trend

# ---------------------------------------------------------------- loading

def load_panel(src_filter=None):
    p = pd.read_parquet(D + 'panel_dyad.parquet')
    if src_filter == 'importer':
        p = p[p.src == 'importer']
    p['tot'] = p.groupby(['imp', 'commod', 'year']).q.transform('sum')
    p['s'] = p.q / p.tot
    return p


def names():
    nm = pd.read_csv(D + 'country_names.csv')
    cw = pd.read_csv(D + 'cru_crosswalk.csv')[['code', 'cru']]
    return nm.drop(columns=[c for c in ['cru'] if c in nm.columns]).merge(cw, on='code', how='left')

# ---------------------------------------------------------------- production anomalies

def trend_anomaly(years, x, method='backward', window=10, minpts=7, half=7):
    """log(x_t) minus a linear trend.
    backward: trend fitted on the `window` preceding years only (information available before t);
    twosided: trend fitted on [t-half, t+half] excluding t (sensitivity analysis)."""
    lx = np.log(x)
    out = np.full(len(x), np.nan)
    for i, t in enumerate(years):
        if not np.isfinite(lx[i]):
            continue
        if method == 'backward':
            m = (years < t) & (years >= t - window) & np.isfinite(lx)
        else:
            m = (np.abs(years - t) <= half) & (years != t) & np.isfinite(lx)
        if m.sum() < minpts:
            continue
        b = np.polyfit(years[m] - t, lx[m], 1)
        out[i] = lx[i] - b[1]
    return out


def production_anomalies(method='backward'):
    pr = pd.read_parquet(D + 'fao_prod.parquet')
    pr['commod'] = pr['item'].map(PRODITEM)
    pr['area'] = pr['area'].replace({255: 15, 256: 15})
    pr = pr[(pr.area < 5000) & pr.commod.notna()]
    w = pr.pivot_table(index=['area', 'commod', 'year'], columns='elem', values='value', aggfunc='sum').reset_index()
    w = w.rename(columns={5510: 'prod', 5312: 'harea', 5412: 'yld'})
    w = w[(w['prod'] > 0) & (w['harea'] > 0)].copy()
    w['yld'] = w['prod'] / w['harea']
    res = []
    for (a, c), g in w.groupby(['area', 'commod']):
        g = g.sort_values('year')
        yrs = g.year.values.astype(float)
        g = g.assign(a_prod=trend_anomaly(yrs, g['prod'].values, method),
                     a_yld=trend_anomaly(yrs, g['yld'].values, method))
        res.append(g)
    return pd.concat(res)[['area', 'commod', 'year', 'prod', 'harea', 'yld', 'a_prod', 'a_yld']]

# ---------------------------------------------------------------- weather-predicted yield anomalies

def weather_features():
    cru = pd.read_parquet(D + 'cru_monthly.parquet')
    m = [f'm{i}' for i in range(1, 13)]
    cru[m] = cru[m].where(cru[m] > -998)          # -999 = missing
    tmx = cru[cru['var'] == 'tmx'].set_index(['cru', 'year'])[m]
    pre = cru[cru['var'] == 'pre'].set_index(['cru', 'year'])[m]
    pet = cru[cru['var'] == 'pet'].set_index(['cru', 'year'])[m]
    wb = pre - (pet * 30.4).reindex(pre.index)    # pet mm/day -> mm/month
    feats = []
    for name, X in [('tmx', tmx), ('wb', wb)]:
        X = X.sort_index()
        prev = X.groupby(level=0).shift(1)
        f = pd.DataFrame(index=X.index)
        agg = np.mean if name == 'tmx' else np.sum
        f[f'{name}_pSON'] = agg(prev[['m9', 'm10', 'm11']].values, axis=1)
        f[f'{name}_DJF'] = agg(np.c_[prev['m12'].values, X[['m1', 'm2']].values], axis=1)
        f[f'{name}_MAM'] = agg(X[['m3', 'm4', 'm5']].values, axis=1)
        f[f'{name}_JJA'] = agg(X[['m6', 'm7', 'm8']].values, axis=1)
        f[f'{name}_SON'] = agg(X[['m9', 'm10', 'm11']].values, axis=1)
        feats.append(f)
    return pd.concat(feats, axis=1).reset_index()


def weather_predicted_anomaly(anom, nm, countries, alpha=20.0, y0=1975):
    """Leave-one-year-out ridge prediction of the yield anomaly from seasonal Tmax and climatic
    water balance, per country and crop. Feature detrending and standardisation are estimated on
    the training years of each fold only and then applied to the held-out year."""
    wf = weather_features()
    cols = [c for c in wf.columns if c not in ('cru', 'year')]
    code2cru = dict(zip(nm.code, nm.cru))
    out = []
    for (a, c), g in anom[anom.area.isin(countries)].groupby(['area', 'commod']):
        cr = code2cru.get(a)
        if not isinstance(cr, str):
            continue
        X = wf[wf.cru == cr].set_index('year')[cols]
        g = g[(g.year >= y0)].set_index('year')
        idx = g.index.intersection(X.index)
        g = g.loc[idx]; Xg = X.loc[idx]
        ok = np.isfinite(g.a_yld.values) & np.isfinite(Xg.values).all(1)
        if ok.sum() < 25:
            continue
        yr = idx.values[ok].astype(float)
        Z0 = Xg.values[ok]; y = g.a_yld.values[ok]
        n, k = Z0.shape
        pred = np.empty(n)
        for i in range(n):
            mk = np.ones(n, bool); mk[i] = False
            yc = yr[mk].mean()
            B = np.polyfit(yr[mk] - yc, Z0[mk], 1)
            Zt = Z0[mk] - (np.outer(yr[mk] - yc, B[0]) + B[1])
            zi = Z0[i] - ((yr[i] - yc) * B[0] + B[1])
            mu_z, sd_z = Zt.mean(0), Zt.std(0) + 1e-9
            Zt = (Zt - mu_z) / sd_z; zi = (zi - mu_z) / sd_z
            yt = y[mk]; mu = yt.mean()
            beta = np.linalg.solve(Zt.T @ Zt + alpha * np.eye(k), Zt.T @ (yt - mu))
            pred[i] = mu + zi @ beta
        out.append(pd.DataFrame({'area': a, 'commod': c, 'year': yr.astype(int), 'a_yld': y, 'w_pred': pred}))
    return pd.concat(out)

# ---------------------------------------------------------------- exporter shares

def exporter_shares(p):
    ex = p.groupby(['exp', 'commod', 'year'], as_index=False).agg(xq=('q', 'sum'), xv=('v', 'sum'))
    ex['world'] = ex.groupby(['commod', 'year']).xq.transform('sum')
    ex['wsh'] = ex.xq / ex.world
    ex['uv'] = ex.xv / ex.xq
    wv = ex.groupby(['commod', 'year'])[['xv', 'xq']].sum()
    wuv = (wv.xv / wv.xq).rename('wuv').reset_index()
    ex = ex.merge(wuv, on=['commod', 'year'])
    ex['ruv'] = np.log(ex.uv / ex.wuv)
    return ex


def significant_exporters(ex, min_share=0.02, min_q=2e5):
    full = ex.set_index(['exp', 'commod', 'year'])[['wsh', 'xq']]
    idx = pd.MultiIndex.from_product([ex.exp.unique(), ex.commod.unique(), range(ex.year.min(), ex.year.max() + 1)],
                                     names=['exp', 'commod', 'year'])
    full = full.reindex(idx).fillna(0.0).reset_index().sort_values(['exp', 'commod', 'year'])
    for col in ['wsh', 'xq']:
        full[col + '_pre'] = full.groupby(['exp', 'commod'])[col].transform(lambda s: s.shift(1).rolling(3).mean())
    full['sig'] = (full.wsh_pre >= min_share) & (full.xq_pre >= min_q)
    return full

# ---------------------------------------------------------------- restrictions

def policy_years(kind='prohibition'):
    y = pd.read_csv(D + f'policy_years_{kind}.csv')
    return y


def build_events(anom, py, sig, wpred=None, thr_prod=THR_PROD, thr_yld=THR_YLD, yield_rule=True,
                 y_lo=1996, y_hi=2023, pol_lo=2007, pol_hi=2023, clean=3):
    sigs = sig[sig.sig][['exp', 'commod', 'year', 'wsh_pre', 'xq_pre']]
    allshock = anom.assign(sh=(anom.a_prod <= thr_prod) & ((anom.a_yld <= thr_yld) | (not yield_rule)))
    shock_set = set(map(tuple, allshock[allshock.sh][['area', 'commod', 'year']].values))
    pb = py[py.binding]
    bind = set(map(tuple, pb[['exp', 'commod', 'year']].values))
    flag = {(r.exp, r.commod, r.year): (r.spell_id, bool(r.cap12), bool(r.open)) for r in pb.itertuples()}
    disrupt = shock_set | bind
    # production shortfalls
    a = anom.rename(columns={'area': 'exp'}).merge(sigs, on=['exp', 'commod', 'year'])
    a['shock'] = (a.a_prod <= thr_prod) & ((a.a_yld <= thr_yld) | (not yield_rule))
    ev = a[a.shock & a.year.between(y_lo, y_hi)].copy()
    ev = ev[[not any((r.exp, r.commod, r.year - k) in disrupt for k in range(1, clean + 1)) for r in ev.itertuples()]]
    ev = ev.rename(columns={'year': 't0'})
    ev['type'] = 'production'
    # restriction onsets (first binding year after `clean` unrestricted, shortfall-free years)
    pe = []
    for (e, c), g in pb.groupby(['exp', 'commod']):
        for y in sorted(g.year):
            if pol_lo <= y <= pol_hi and not any((e, c, y - k) in disrupt for k in range(1, clean + 1)):
                pe.append((e, c, y))
    pe = pd.DataFrame(pe, columns=['exp', 'commod', 't0']).merge(sigs.rename(columns={'year': 't0'}), on=['exp', 'commod', 't0'])
    pe['type'] = 'restriction'
    an = anom.rename(columns={'area': 'exp', 'year': 't0'})[['exp', 'commod', 't0', 'a_prod', 'a_yld']]
    pe = pe.merge(an, on=['exp', 'commod', 't0'], how='left')
    events = pd.concat([ev[['exp', 'commod', 't0', 'type', 'a_prod', 'a_yld', 'wsh_pre', 'xq_pre']], pe], ignore_index=True)
    # number of earlier disruption years of the same supplier and crop in t0-15 .. t0-clean-1
    events['n_prior'] = [sum((r.exp, r.commod, r.t0 - k) in disrupt for k in range(clean + 1, 16)) for r in events.itertuples()]
    if wpred is not None:
        events = events.merge(wpred[['area', 'commod', 'year', 'w_pred']].rename(columns={'area': 'exp', 'year': 't0'}),
                              on=['exp', 'commod', 't0'], how='left')
    return events, shock_set, bind, flag


def dedupe_events(ev, win=2):
    """Merge onsets of the same supplier and crop within `win` years (earliest kept; type 'both'
    when a shortfall and a restriction coincide)."""
    ev = ev.sort_values(['exp', 'commod', 't0', 'type']).reset_index(drop=True)
    ev['t_prod_last'] = np.where(ev.type == 'production', ev.t0, np.nan)
    keep, last = [], {}
    for i, r in ev.iterrows():
        key = (r.exp, r.commod)
        if key in last and r.t0 - ev.loc[last[key], 't0'] <= win:
            j = last[key]
            if ev.loc[j, 'type'] != r.type:
                ev.loc[j, 'type'] = 'both'
            if r.type == 'production':
                ev.loc[j, 't_prod_last'] = np.nanmax([ev.loc[j, 't_prod_last'], r.t0])
            continue
        last[key] = i; keep.append(i)
    out = ev.loc[keep].copy().reset_index(drop=True)
    return out


def add_disruption_end(ev, bind, flag, shock_set, last_year, horizon=KMAX):
    """t1: last disruption year (shortfall year; last consecutive binding year of a restriction;
    a merged later shortfall extends it). censor_year: first later year in which the same
    supplier and crop is disrupted again (follow-up is truncated before it). open_end: the
    restriction was still in force after the last year of trade data."""
    t1s, cens, spell, cap12, opn = [], [], [], [], []
    for r in ev.itertuples():
        t = int(r.t0)
        sp, c12, op = np.nan, False, False
        if r.type in ('restriction', 'both'):
            # restriction may start in t0 or within the merge window
            ys = [y for y in range(t, t + 3) if (r.exp, r.commod, y) in bind]
            if ys:
                t = ys[0]
                sp, c12, op = flag[(r.exp, r.commod, t)]
                while (r.exp, r.commod, t + 1) in bind:
                    t += 1
                    f2 = flag[(r.exp, r.commod, t)]; c12 |= f2[1]; op |= f2[2]
        if not pd.isna(r.t_prod_last):
            t = max(t, int(r.t_prod_last))
        t = max(t, int(r.t0))
        t1s.append(t); spell.append(sp); cap12.append(c12); opn.append(op or (t >= last_year and r.type != 'production'))
        cy = np.nan
        for y in range(t + 1, t + horizon + 1):
            if (r.exp, r.commod, y) in shock_set or (r.exp, r.commod, y) in bind:
                cy = y; break
        cens.append(cy)
    ev = ev.copy()
    ev['t1'] = t1s; ev['censor_year'] = cens; ev['spell_id'] = spell; ev['cap12'] = cap12; ev['open_end'] = opn
    return ev


def build_pseudo(anom, sig, ex, shock_set, bind, y_lo=1996, y_hi=2023, calm=0.05, before=3, after=0, horizon=KMAX):
    """Calm (comparison) years: production within +-`calm` log points of trend, yield no more than
    `calm` below trend, and no shortfall or binding restriction of the same supplier and crop in the
    `before` preceding years (the same clean-history rule applied to disruption onsets). Follow-up is
    censored at the next disruption exactly as for events; `after` > 0 additionally requires
    `after` disruption-free years after t0 (the original symmetric-window definition)."""
    sigs = sig[sig.sig][['exp', 'commod', 'year', 'wsh_pre', 'xq_pre']]
    a = anom.rename(columns={'area': 'exp'}).merge(sigs, on=['exp', 'commod', 'year'])
    a = a[a.year.between(y_lo, y_hi) & (a.a_prod.abs() < calm) & (a.a_yld > -calm)]
    dis = shock_set | bind
    a = a[[not any((r.exp, r.commod, r.year + k) in dis for k in range(-before, after + 1)) for r in a.itertuples()]]
    ps = a.rename(columns={'year': 't0'}).copy()
    exr = ex.set_index(['exp', 'commod', 'year']).ruv
    ps['druv'] = [exr.get((r.exp, r.commod, r.t0), np.nan) -
                  np.nanmean([exr.get((r.exp, r.commod, r.t0 - l), np.nan) for l in (1, 2, 3)]) for r in ps.itertuples()]
    ps['type'] = 'calm'; ps['n_prior'] = 0; ps['t1'] = ps.t0
    cens = []
    for r in ps.itertuples():
        cy = np.nan
        for y in range(int(r.t0) + 1, int(r.t0) + horizon + 1):
            if (r.exp, r.commod, y) in dis:
                cy = y; break
        cens.append(cy)
    ps['censor_year'] = cens; ps['open_end'] = False; ps['cap12'] = False; ps['spell_id'] = np.nan
    return ps[['exp', 'commod', 't0', 't1', 'type', 'a_prod', 'a_yld', 'wsh_pre', 'xq_pre', 'n_prior', 'druv',
               'censor_year', 'open_end', 'cap12', 'spell_id']]

# ---------------------------------------------------------------- stacked sample

def stack(events, p, min_imp=2e4, min_dep=0.02, kmin=KMIN, kmax=KMAX, last_year=None, censor=True):
    """All importers of the crop with mean pre-period (t0-3..t0-1) imports >= min_imp t/yr, positive
    imports in each pre year, and mean pre-period share from the supplier >= min_dep.
    k = t - t0 before onset, k = t - t1 from the last disruption year on; years strictly inside a
    multi-year disruption are dropped; with censor=True years >= censor_year are dropped."""
    if last_year is None:
        last_year = int(p.year.max())
    tot = p.groupby(['imp', 'commod', 'year'], as_index=False).tot.first()
    totw = tot.pivot_table(index=['imp', 'commod'], columns='year', values='tot').fillna(0.0)
    rows = []
    pc = {c: g for c, g in p.groupby('commod')}
    for e in events.itertuples():
        g = pc[e.commod]
        t0 = int(e.t0); t1 = int(e.t1)
        hi = min(t1 + kmax, last_year)
        if censor and not pd.isna(e.censor_year):
            hi = min(hi, int(e.censor_year) - 1)
        yrs = [y for y in range(t0 + kmin, hi + 1) if (y < t0) or (y >= t1)]
        T = totw.xs(e.commod, level=1)
        pre_years = [t0 - 3, t0 - 2, t0 - 1]
        if not all(y in T.columns for y in pre_years):
            continue
        tpre = T[pre_years].mean(1)
        imps = tpre[(tpre >= min_imp) & (T[pre_years].min(1) > 0)].index
        imps = imps.drop(e.exp, errors='ignore')
        win = g[g.year.isin(yrs) & g.imp.isin(imps)]
        gj = win[win.exp == e.exp].set_index(['imp', 'year'])
        mi = pd.MultiIndex.from_product([imps, yrs], names=['imp', 'year'])
        qj = gj.q.reindex(mi).fillna(0.0); vj = gj.v.reindex(mi).fillna(0.0)
        Tk = T.loc[imps, [y for y in yrs if y in T.columns]].stack()
        Tk.index.names = ['imp', 'year']
        df = pd.DataFrame({'q': qj, 'v': vj, 'tot': Tk.reindex(mi)}).reset_index()
        df = df[df.tot > 0]
        df['s'] = df.q / df.tot
        d = df[df.year.isin(pre_years)].groupby('imp').s.mean()
        d = d[d >= min_dep]
        df = df[df.imp.isin(d.index)].copy()
        if df.empty:
            continue
        df['d'] = df.imp.map(d)
        df['k'] = np.where(df.year < t0, df.year - t0, df.year - t1)
        for col in ['eid', 'exp', 'commod', 'type']:
            df[col] = getattr(e, col)
        df['t0'] = t0; df['t1'] = t1
        oth = win[(win.exp != e.exp) & win.imp.isin(d.index)].groupby(['imp', 'year']).agg(oq=('q', 'sum'), ov=('v', 'sum'))
        df = df.merge(oth.reset_index(), on=['imp', 'year'], how='left')
        df['ruv'] = np.where((df.q > 0) & (df.oq > 0), np.log((df.v / df.q) / (df.ov / df.oq)), np.nan)
        rows.append(df)
    S = pd.concat(rows, ignore_index=True)
    S = S[S.k.between(kmin, kmax)]
    return S

# ---------------------------------------------------------------- estimation

def demean_multi(M, codes, w=None, tol=1e-9, maxit=2000):
    """Alternating projections for several fixed-effect dimensions (Gaure 2013); weighted
    group means when w is given (weighted within transformation)."""
    M = np.array(M, float, copy=True)
    if w is None:
        w = np.ones(M.shape[0])
    sw = [np.bincount(c, weights=w) for c in codes]
    for it in range(maxit):
        delta = 0.0
        for c, n in zip(codes, sw):
            sums = np.zeros((n.size, M.shape[1]))
            np.add.at(sums, c, M * w[:, None])
            adj = (sums / np.maximum(n, 1e-300)[:, None])[c]
            M -= adj
            delta = max(delta, float(np.abs(adj).max()))
        if len(codes) == 1 or delta < tol:
            break
    return M


def fe_solve(y, X, codes, w=None):
    A = np.column_stack([y, X])
    At = demean_multi(A, codes, w) if codes else A - np.average(A, axis=0, weights=w)
    yt, Xt = At[:, 0], At[:, 1:]
    if w is not None:
        s = np.sqrt(w); yt = yt * s; Xt = Xt * s[:, None]
    XtX = Xt.T @ Xt
    inv = np.linalg.pinv(XtX)
    b = inv @ Xt.T @ yt
    return b, yt - Xt @ b, Xt, inv


def fe_ols(df, ycol, xcols, fes, cl, w=None):
    """OLS with absorbed fixed effects and CR1 cluster-robust standard errors."""
    codes = [pd.factorize(df[f].values)[0] for f in fes]
    wv = df[w].values.astype(float) if w is not None else None
    b, u, X, inv = fe_solve(df[ycol].values.astype(float), np.column_stack([df[c].values for c in xcols]).astype(float), codes, wv)
    g = pd.factorize(df[cl].values)[0]
    G = g.max() + 1
    sc = np.zeros((G, X.shape[1])); np.add.at(sc, g, X * u[:, None])
    n, k = X.shape
    V = inv @ (sc.T @ sc) @ inv * (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), int(G), int(n)

# ---------------------------------------------------------------- relaxation fit (model-dependent summary)

def fit_exp(k, b):
    """b(k) = A exp(-k/tau) for k >= 0; returns tau and half-life tau*ln2."""
    k = np.asarray(k, float); b = np.asarray(b, float)
    m = np.isfinite(b)
    r = least_squares(lambda th: th[0] * np.exp(-k[m] / np.exp(th[1])) - b[m], [b[m][0], np.log(2.0)],
                      bounds=([-5, np.log(0.05)], [5, np.log(200.0)]))
    tau = float(np.exp(r.x[1]))
    return {'A': float(r.x[0]), 'tau': tau, 'half_life': tau * np.log(2)}


def km_median(durations, events):
    """Kaplan-Meier survival of time-to-next-disruption; returns (median, S(t) dict)."""
    d = np.asarray(durations, float); e = np.asarray(events, bool)
    ts = np.unique(d[e])
    S, out, med = 1.0, {}, np.nan
    for t in ts:
        n = (d >= t).sum(); x = ((d == t) & e).sum()
        S *= 1 - x / n
        out[int(t)] = S
        if np.isnan(med) and S <= 0.5:
            med = t
    return med, out
