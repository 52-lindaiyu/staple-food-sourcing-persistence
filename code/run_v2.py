"""Step 4: full analysis. Writes every number used in the manuscript and SI to OUT_DIR (default ./results).
Environment variables: DATA_DIR, OUT_DIR, B_MAIN, B_SUB, B_ROB (bootstrap replications), NPROC, SKIP_ROBUST."""
import os, sys, json, time, warnings
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[v] = '1'
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rm2 as rc

B_MAIN = int(os.environ.get('B_MAIN', 1999)); B_SUB = int(os.environ.get('B_SUB', 999)); B_ROB = int(os.environ.get('B_ROB', 499))
NPROC = int(os.environ.get('NPROC', 1))
OUT = os.environ.get('OUT_DIR', 'results') + '/'
os.makedirs(OUT, exist_ok=True)
KS = list(range(rc.KMIN, rc.KMAX + 1)); REF = -1
EU = [54, 68, 97, 15, 199, 98, 63, 198, 79, 167, 150, 174, 119, 27, 67, 134, 203, 210, 84, 104, 126, 183, 106, 50, 173, 11]
WIN = {'pre': [-3, -2], 'early': [0, 1], 'late': [2, 3, 4], 'tail': [5, 6, 7, 8]}
MARGIN = 0.05   # pre-specified equivalence margin (share ratio units, 5% of pre-disruption share)
T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)

# ------------------------------------------------------------------ base data
P_ALL = rc.load_panel(); P_IMP = rc.load_panel('importer'); NM = rc.names()
LAST = int(P_ALL.year.max())
ANOM = {'backward': rc.production_anomalies('backward'), 'twosided': rc.production_anomalies('twosided')}
EX = rc.exporter_shares(P_ALL); EX_IMP = rc.exporter_shares(P_IMP)
PY = {'prohibition': rc.policy_years('prohibition'), 'prohib_quota': rc.policy_years('prohib_quota')}
log('base data', P_ALL.shape)

SIG0 = rc.significant_exporters(EX, min_share=0.01, min_q=1e5)
WPRED = rc.weather_predicted_anomaly(ANOM['backward'], NM, SIG0[SIG0.sig].exp.unique(), alpha=20.0)
def skill(g):
    return pd.Series({'n_series': g.groupby(['area', 'commod']).ngroups, 'n_obs': len(g),
                      'r_oos': np.corrcoef(g.a_yld, g.w_pred)[0, 1],
                      'r2_oos': 1 - ((g.a_yld - g.w_pred) ** 2).sum() / ((g.a_yld - g.a_yld.mean()) ** 2).sum()})
wsk = WPRED.groupby('commod')[list(WPRED.columns)].apply(skill).reset_index()
wsk = pd.concat([wsk, skill(WPRED).to_frame().T.assign(commod='all'),
                 skill(WPRED[WPRED.commod != 'rice']).to_frame().T.assign(commod='all_excl_rice')], ignore_index=True)
wsk.to_csv(OUT + 'weather_skill.csv', index=False)
WPRED.to_csv(OUT + 'weather_prediction.csv', index=False)
log('weather skill', wsk.round(3).to_dict('records'))

imp_tot = P_ALL.groupby(['imp', 'commod', 'year']).q.sum().rename('M').reset_index()
exp_tot = P_ALL.groupby(['exp', 'commod', 'year']).q.sum().rename('Xq').reset_index().rename(columns={'exp': 'imp'})
prod_tot = ANOM['backward'][['area', 'commod', 'year', 'prod']].rename(columns={'area': 'imp'})
SSR = imp_tot.merge(exp_tot, on=['imp', 'commod', 'year'], how='left').merge(prod_tot, on=['imp', 'commod', 'year'], how='left').fillna(0.0)
SSR['ssr'] = SSR['prod'] / (SSR['prod'] + SSR.M - SSR.Xq).clip(lower=1.0)
XTOT = P_ALL.groupby(['exp', 'commod', 'year']).q.sum().rename('X').reset_index()
WTOT = P_ALL.groupby(['commod', 'year']).q.sum().rename('W').reset_index()
SRCSH = P_ALL.assign(mir=(P_ALL.src == 'mirror').astype(float)).set_index(['imp', 'exp', 'commod', 'year']).mir

MAIN = dict(anom='backward', thr_prod=rc.THR_PROD, thr_yld=rc.THR_YLD, yield_rule=True, min_share=0.02, min_q=2e5,
            policy='prohibition', min_imp=2e4, min_dep=0.02, src='all', clean=3, censor=True, excl_years=None,
            drop_intra_eu=False, weather_only=False, drop_cap12=False, ycap=3.0, design='within', y_hi=2023,
            calm_band=0.05, calm_after=0)

# ------------------------------------------------------------------ builders

def build(cfg):
    p = P_IMP if cfg['src'] == 'importer' else P_ALL
    ex = EX_IMP if cfg['src'] == 'importer' else EX
    anom = ANOM[cfg['anom']]
    sig = rc.significant_exporters(ex, min_share=cfg['min_share'], min_q=cfg['min_q'])
    ev, shock_set, bind, flag = rc.build_events(anom, PY[cfg['policy']], sig, wpred=WPRED, thr_prod=cfg['thr_prod'],
                                                thr_yld=cfg['thr_yld'], yield_rule=cfg['yield_rule'], clean=cfg['clean'],
                                                y_hi=cfg['y_hi'])
    ev = rc.dedupe_events(ev)
    ev = rc.add_disruption_end(ev, bind, flag, shock_set, LAST)
    ev_all = ev.copy()
    ev = ev[~ev.open_end]
    if cfg['excl_years']:
        ev = ev[~ev.t0.isin(cfg['excl_years'])]
    if cfg['weather_only']:
        ev = ev[(ev.type != 'production') | ((ev.w_pred <= -0.05) & (ev.commod != 'rice'))]
    if cfg['drop_cap12']:
        ev = ev[~ev.cap12]
    ev = ev.reset_index(drop=True); ev['eid'] = np.arange(len(ev))
    ps = rc.build_pseudo(anom, sig, ex, shock_set, bind, calm=cfg['calm_band'], before=cfg['clean'], after=cfg['calm_after'], y_hi=cfg['y_hi'])
    ps = ps.reset_index(drop=True); ps['eid'] = np.arange(len(ps)) + 100000
    S = rc.stack(ev, p, min_imp=cfg['min_imp'], min_dep=cfg['min_dep'], last_year=LAST, censor=cfg['censor'])
    SP = rc.stack(ps, p, min_imp=cfg['min_imp'], min_dep=cfg['min_dep'], last_year=LAST, censor=cfg['censor'])
    ST = pd.concat([S.assign(ev=1.0), SP.assign(ev=0.0)], ignore_index=True)
    ST = finish(ST, ycap=cfg['ycap'])
    if cfg['drop_intra_eu']:
        ST = ST[~(ST.imp.isin(EU) & ST.exp.isin(EU))]
    ST_broad = ST.copy()
    if cfg['design'] == 'within':
        both = ST.groupby('jc').ev.agg(['min', 'max'])
        ok = both[(both['min'] == 0) & (both['max'] == 1)].index
        ST = ST[ST.jc.isin(ok)]
    ST = ST.reset_index(drop=True)
    meta = {'ev_all': ev_all, 'ev': ev, 'ps': ps, 'broad': ST_broad.reset_index(drop=True)}
    return ST, meta


def finish(ST, ycap=3.0):
    ST = ST.copy()
    ST['R'] = ST.s / ST.d
    ST['y'] = np.minimum(ST.R, ycap) - 1.0
    ST['y5'] = np.minimum(ST.R, 5.0) - 1.0
    ST['ylog'] = np.log((ST.s + 0.01) / (ST.d + 0.01))
    ST = ST.merge(XTOT, on=['exp', 'commod', 'year'], how='left').merge(WTOT, on=['commod', 'year'], how='left')
    ST['M'] = (ST.X.fillna(0) - ST.q) / (ST.W - ST.tot)
    pre = ST[ST.k < 0].groupby(['eid', 'imp']).agg(M_pre=('M', 'mean'), q_pre=('q', 'mean'), tpre=('tot', 'mean'),
                                                     trade_pre=('q', lambda s: (s > 0).mean())).reset_index()
    ST = ST.merge(pre, on=['eid', 'imp'], how='left')
    ST['yrho'] = np.minimum((ST.s / ST.M) / (ST.d / ST.M_pre), ycap) - 1.0
    ST['yq'] = np.minimum(ST.q / ST.q_pre, ycap) - 1.0
    ST['dlogQ'] = np.log(ST.tot / ST.tpre)
    ST['trade'] = (ST.q > 0).astype(float)
    ST['yext'] = ST.trade - ST.trade_pre
    ST = ST.sort_values(['eid', 'imp', 'k'])
    post = ST.k >= 0
    ST['ever_back'] = np.nan
    ST.loc[post, 'ever_back'] = ST[post].groupby(['eid', 'imp']).trade.cummax()
    ST['absent_through'] = 1.0 - ST.ever_back
    ST = ST.merge(EX[['exp', 'commod', 'year', 'ruv']].rename(columns={'ruv': 'ruvj'}), on=['exp', 'commod', 'year'], how='left')
    rj = ST[ST.k < 0].groupby('eid').ruvj.mean().rename('ruvj_pre').reset_index()
    ST = ST.merge(rj, on='eid', how='left')
    ST['druvj'] = (ST.ruvj - ST.ruvj_pre)
    ST['ruvd'] = ST.ruv.clip(-2, 2)
    ST['dbin'] = pd.cut(ST.d, [0.0, 0.1, 0.25, 0.5, 1.01], labels=False, right=False)
    ST['bink'] = ST.dbin.astype(str) + '_' + ST.k.astype(str)
    ST['jc'] = ST.exp.astype(str) + '_' + ST.commod
    ST['jck'] = ST.jc + '_' + ST.k.astype(str)
    ST['cy'] = ST.commod + '_' + ST.year.astype(str)
    ST['icy'] = ST.imp.astype(str) + '_' + ST.cy
    ST['wgt'] = ST.d * ST.tpre
    ST['pre_year'] = ST.t0 - 1
    ST = ST.merge(SSR[['imp', 'commod', 'year', 'ssr']].rename(columns={'year': 'pre_year'}), on=['imp', 'commod', 'pre_year'], how='left')
    ST['capped'] = (ST.R >= ycap).astype(float)
    return ST.reset_index(drop=True)

# ------------------------------------------------------------------ estimation

def xcols(ST, ks=KS, ref=REF):
    names_ = []
    for k in ks:
        if k == ref:
            continue
        ST[f'e{k}'] = (ST.ev * (ST.k == k)).astype(float); names_.append(f'e{k}')
    return names_


def estimate(ST, outcome='y', fes=('bink', 'jck'), cl='jc', w=None, ks=KS, ref=REF, controls=()):
    d = ST[ST.k.isin(ks)].dropna(subset=[outcome] + list(controls)).reset_index(drop=True)
    xc = xcols(d, ks, ref)
    b, se, G, n = rc.fe_ols(d, outcome, xc + list(controls), list(fes), cl, w=w)
    bk = dict(zip(xc, b)); sk = dict(zip(xc, se))
    curve = np.array([0.0 if k == ref else bk[f'e{k}'] for k in ks])
    secur = np.array([0.0 if k == ref else sk[f'e{k}'] for k in ks])
    ctrl = {c: (float(b[len(xc) + i]), float(se[len(xc) + i])) for i, c in enumerate(controls)}
    return curve, secur, G, n, ctrl, d


def windows(curve, ks=KS):
    k = np.array(ks); out = {}
    for nm_, kk in WIN.items():
        m = np.isin(k, kk)
        out[nm_] = float(curve[m].mean()) if m.any() else np.nan
    out['R'] = out['late'] / out['early'] if out['early'] != 0 else np.nan
    return out

_G = {}
def _init(payload):
    _G.update(payload)


def _draw(seed):
    rng = np.random.default_rng(seed)
    blocks = _G['blocks']
    draw = rng.integers(0, len(blocks), len(blocks))
    idx = np.concatenate([blocks[j] for j in draw])
    y = _G['y'][idx]; X = _G['X'][idx]
    if X.sum(0).min() == 0:
        return None
    codes = [c[idx] for c in _G['codes']]
    w = _G['w'][idx] if _G['w'] is not None else None
    try:
        b, _, _, _ = rc.fe_solve(y, X, codes, w)
    except Exception:
        return None
    ks, ref = _G['ks'], _G['ref']
    bk = dict(zip(_G['xc'], b[:len(_G['xc'])]))
    curve = np.array([0.0 if k == ref else bk[f'e{k}'] for k in ks])
    return curve


def bootstrap(d, outcome, fes, w, ks, ref, B, cluster='jc', controls=()):
    if B <= 0:
        return None
    xc = [f'e{k}' for k in ks if k != ref]
    payload = {'y': d[outcome].values.astype(float),
               'X': np.column_stack([d[c].values for c in xc + list(controls)]).astype(float),
               'codes': [pd.factorize(d[f].values)[0] for f in fes],
               'w': d[w].values.astype(float) if w else None,
               'blocks': list(d.groupby(cluster).indices.values()), 'ks': ks, 'ref': ref, 'xc': xc}
    seeds = range(1, B + 1)
    if NPROC > 1:
        from multiprocessing import Pool
        with Pool(NPROC, initializer=_init, initargs=(payload,)) as pool:
            out = pool.map(_draw, seeds, chunksize=max(1, B // (NPROC * 4)))
    else:
        _init(payload); out = [_draw(s) for s in seeds]
    return np.array([o for o in out if o is not None])


def run(ST, label, group, B=0, outcome='y', fes=('bink', 'jck'), cl='jc', w=None, ks=KS, ref=REF, controls=(),
        boot_cluster='jc', extra=None):
    curve, se, G, n, ctrl, d = estimate(ST, outcome, fes, cl, w, ks, ref, controls)
    rec = {'group': group, 'label': label, 'outcome': outcome, 'fes': '+'.join(fes), 'n_obs': n, 'n_clusters': G,
           'n_events': int(d[d.ev == 1].eid.nunique()), 'n_calm': int(d[d.ev == 0].eid.nunique()),
           'n_event_exposures': int(d[d.ev == 1].groupby(['eid', 'imp']).ngroups),
           'n_event_dyads': int(d[d.ev == 1].groupby(['imp', 'exp', 'commod']).ngroups),
           'n_calm_exposures': int(d[d.ev == 0].groupby(['eid', 'imp']).ngroups), 'boot_cluster': boot_cluster}
    wnd = windows(curve, ks); rec.update(wnd)
    for c, (bb, ss) in ctrl.items():
        rec[f'ctrl_{c}'] = bb; rec[f'ctrl_{c}_se'] = ss
    if all(k in ks for k in range(0, 9)):
        f = rc.fit_exp(np.arange(9), curve[[ks.index(k) for k in range(9)]]); rec['hl_exp'] = f['half_life']
    cur = pd.DataFrame({'group': group, 'label': label, 'outcome': outcome, 'k': ks, 'b': curve, 'se': se})
    cur['n_ev_exposures'] = [int(((d.ev == 1) & (d.k == k)).sum()) for k in ks]
    cur['n_events_k'] = [int(d[(d.ev == 1) & (d.k == k)].eid.nunique()) for k in ks]
    cur['n_calm_exposures'] = [int(((d.ev == 0) & (d.k == k)).sum()) for k in ks]
    bs = bootstrap(d, outcome, fes, w, ks, ref, B, cluster=boot_cluster, controls=controls)
    if bs is not None and len(bs) > 20:
        cur['lo'] = np.percentile(bs, 2.5, axis=0); cur['hi'] = np.percentile(bs, 97.5, axis=0)
        # simultaneous (sup-t) 95% band over all estimated event times
        sd = bs.std(0); est = sd > 0
        tstat = np.abs((bs[:, est] - curve[est]) / sd[est]).max(1)
        crit = float(np.percentile(tstat, 95))
        cur['slo'] = np.where(est, curve - crit * sd, 0.0); cur['shi'] = np.where(est, curve + crit * sd, 0.0)
        rec['supt_crit'] = crit
        if label in SAVE_DRAWS:
            np.save(OUT + f'draws_{label}.npy', bs)
        W = np.array([[windows(r, ks)[nm_] for nm_ in ['pre', 'early', 'late', 'tail', 'R']] for r in bs])
        for j, nm_ in enumerate(['pre', 'early', 'late', 'tail']):
            col = W[:, j]
            rec[nm_ + '_lo'], rec[nm_ + '_hi'] = np.percentile(col, [2.5, 97.5])
            rec[nm_ + '_lo90'], rec[nm_ + '_hi90'] = np.percentile(col, [5, 95])
            rec[nm_ + '_p_ge0'] = float((col >= 0).mean())
            rec[nm_ + '_tost_equiv'] = bool((rec[nm_ + '_lo90'] > -MARGIN) and (rec[nm_ + '_hi90'] < MARGIN))
        R = W[:, 4]; e_ = W[:, 1]
        rec['p_early_lt0'] = float((e_ < 0).mean())
        neg = e_ < 0
        rec['p_R_le0'] = float(((R <= 0) & neg).mean()); rec['p_R_0_1'] = float(((R > 0) & (R < 1) & neg).mean())
        rec['p_R_ge1'] = float(((R >= 1) & neg).mean()); rec['p_early_ge0'] = float((~neg).mean())
        rec['R_lo'], rec['R_hi'] = np.nanpercentile(R[neg], [2.5, 97.5]) if neg.sum() > 10 else (np.nan, np.nan)
        rec['late_minus_early_lo'], rec['late_minus_early_hi'] = np.percentile(W[:, 2] - W[:, 1], [2.5, 97.5])
        if 'hl_exp' in rec:
            hl = [rc.fit_exp(np.arange(9), r[[ks.index(k) for k in range(9)]])['half_life'] for r in bs]
            rec['hl_exp_lo'], rec['hl_exp_hi'] = np.percentile(hl, [2.5, 97.5])
            rec['p_hl_lt3'] = float((np.array(hl) < 3).mean())
        rec['n_boot'] = int(len(bs))
    if extra:
        rec.update(extra)
    RECS.append(rec); CURVES.append(cur)
    log(group, label, 'n_ev=%d early=%.3f late=%.3f tail=%.3f' % (rec['n_events'], rec['early'], rec['late'], rec['tail']),
        '[%.3f,%.3f]' % (rec.get('late_lo', np.nan), rec.get('late_hi', np.nan)))
    return rec, cur

RECS, CURVES = [], []
SAVE_DRAWS = {'main', 'main_onset_year_bootstrap', 'extensive_margin', 'aligned_on_supply_restoration', 'crop_year_fe',
              'importer_crop_year_fe', 'onset_aligned'}

# ------------------------------------------------------------------ 1. main build and descriptives
ST, META = build(MAIN)
EV, PS, EVALL = META['ev'], META['ps'], META['ev_all']
nmap = dict(zip(NM.code, NM.name))
EVALL.assign(name=EVALL.exp.map(nmap)).to_csv(OUT + 'events_all_candidates.csv', index=False)
EV.assign(name=EV.exp.map(nmap)).to_csv(OUT + 'events.csv', index=False)
PS.assign(name=PS.exp.map(nmap)).to_csv(OUT + 'calm_years.csv', index=False)
ST.to_parquet(OUT + 'stacked_main.parquet', index=False)
in_ev = set(ST[ST.ev == 1].eid); in_ps = set(ST[ST.ev == 0].eid)
B0 = META['broad']
desc = {'n_candidate_events': int(len(EVALL)), 'n_open_end_dropped': int(EVALL.open_end.sum()),
        'n_events_built': int(len(EV)), 'n_events_broad': int(B0[B0.ev == 1].eid.nunique()),
        'n_events_within': len(in_ev), 'n_calm_built': int(len(PS)), 'n_calm_broad': int(B0[B0.ev == 0].eid.nunique()),
        'n_calm_within': len(in_ps), 'n_pairs_within': int(ST.jc.nunique()),
        'n_pairs_with_events_broad': int(B0[B0.ev == 1].jc.nunique()),
        'by_type': EV[EV.eid.isin(in_ev)].type.value_counts().to_dict(),
        'by_commod': EV[EV.eid.isin(in_ev)].commod.value_counts().to_dict(),
        'n_suppliers': int(EV[EV.eid.isin(in_ev)].exp.nunique()),
        'n_event_exposures': int(ST[ST.ev == 1].groupby(['eid', 'imp']).ngroups),
        'n_event_dyads': int(ST[ST.ev == 1].groupby(['imp', 'exp', 'commod']).ngroups),
        'n_event_importers': int(ST[ST.ev == 1].imp.nunique()),
        'n_calm_exposures': int(ST[ST.ev == 0].groupby(['eid', 'imp']).ngroups),
        'n_censored_events': int(EV[EV.eid.isin(in_ev)].censor_year.notna().sum()),
        'n_cap12_events': int(EV[EV.eid.isin(in_ev)].cap12.sum()),
        'years': [int(P_ALL.year.min()), LAST], 'mirror_share': float((P_ALL.src == 'mirror').mean()),
        'cap_share_event_post': float(ST[(ST.ev == 1) & (ST.k >= 0)].capped.mean()),
        'cap_share_calm_post': float(ST[(ST.ev == 0) & (ST.k >= 0)].capped.mean()),
        'cap_share_all': float(ST.capped.mean()),
        'B_MAIN': B_MAIN, 'B_SUB': B_SUB, 'B_ROB': B_ROB}
ev_in = EV[EV.eid.isin(in_ev)]
desc['wpred_le_m05_nonrice'] = int(((ev_in.type != 'restriction') & (ev_in.w_pred <= -0.05) & (ev_in.commod != 'rice')).sum())
desc['wpred_neg_share_prod'] = float((ev_in[ev_in.type != 'restriction'].w_pred < 0).mean())
desc['wpred_missing_prod'] = int(ev_in[ev_in.type != 'restriction'].w_pred.isna().sum())
log('main build', desc)

# event audit table
aud = EVALL.assign(name=EVALL.exp.map(nmap)).copy()
expo = ST[ST.ev == 1].groupby('eid').agg(n_exposures=('imp', 'nunique'), kmax_obs=('k', 'max')).reset_index()
aud = aud.merge(EV[['exp', 'commod', 't0', 'eid']], on=['exp', 'commod', 't0'], how='left').merge(expo, on='eid', how='left')
aud['in_estimation'] = aud.eid.isin(in_ev)
aud.to_csv(OUT + 'event_audit.csv', index=False)

# support: calm years per pair and balance
sup = ST.groupby('jc').apply(lambda g: pd.Series({'n_events': g[g.ev == 1].eid.nunique(), 'n_calm': g[g.ev == 0].eid.nunique(),
                                                    'ev_exposures': g[g.ev == 1].groupby(['eid', 'imp']).ngroups,
                                                    'calm_exposures': g[g.ev == 0].groupby(['eid', 'imp']).ngroups})).reset_index()
sup.to_csv(OUT + 'support_pairs.csv', index=False)
X0 = ST.drop_duplicates(['eid', 'imp']).copy()
X0['log_tpre'] = np.log(X0.tpre)
pre_rows = ST[ST.k < 0]
mir = pre_rows.assign(mir=[SRCSH.get((i, j, c, y), np.nan) for i, j, c, y in zip(pre_rows.imp, pre_rows.exp, pre_rows.commod, pre_rows.year)])
X0 = X0.merge(mir.groupby(['eid', 'imp']).mir.mean().rename('mirror_pre').reset_index(), on=['eid', 'imp'], how='left')
tr = ST[ST.k.isin([-3, -1])].pivot_table(index=['eid', 'imp'], columns='k', values='y')
X0 = X0.merge((tr[-1] - tr[-3]).rename('pretrend').reset_index(), on=['eid', 'imp'], how='left')
X0 = X0.merge(ST[ST.k < 0].groupby('eid').year.min().rename('y_first').reset_index(), on='eid', how='left')
X0['wsh_pre'] = X0.eid.map(pd.concat([EV.set_index('eid').wsh_pre, PS.set_index('eid').wsh_pre]))
bal = []
for v in ['d', 'log_tpre', 'trade_pre', 'mirror_pre', 'ssr', 'wsh_pre', 'pretrend', 't0']:
    z = X0.dropna(subset=[v])
    m1, m0 = z[z.ev == 1][v].mean(), z[z.ev == 0][v].mean()
    sd = np.sqrt((z[z.ev == 1][v].var() + z[z.ev == 0][v].var()) / 2)
    zz = z.assign(one=1.0)
    b, se, G, n = rc.fe_ols(zz, v, ['ev'], ['jc'], 'jc')
    bal.append({'var': v, 'mean_event': m1, 'mean_calm': m0, 'std_diff': (m1 - m0) / sd if sd > 0 else np.nan,
                'within_pair_diff': b[0], 'within_pair_se': se[0], 'n': n})
pd.DataFrame(bal).to_csv(OUT + 'balance.csv', index=False)

if os.environ.get('ONLY_ROB'):
    # run only the named robustness specifications (comma-separated labels of ROB, defined below) and stop
    _only = os.environ['ONLY_ROB'].split(',')
    _ROBX = {'clean_window_1y': dict(clean=1), 'clean_window_2y': dict(clean=2)}
    for lab in _only:
        cfg = dict(MAIN); cfg.update(_ROBX[lab])
        st_r, meta_r = build(cfg)
        meta_r['ev'].assign(name=meta_r['ev'].exp.map(nmap)).to_csv(OUT + f'events_{lab}.csv', index=False)
        run(st_r, lab, 'robust', B=B_ROB)
    pd.DataFrame(RECS).to_csv(OUT + 'summary_extra.csv', index=False)
    pd.concat(CURVES, ignore_index=True).to_csv(OUT + 'curves_extra.csv', index=False)
    sys.exit(0)

# ------------------------------------------------------------------ 2. main estimates
run(ST, 'main', 'main', B=B_MAIN)
run(ST, 'main_onset_year_bootstrap', 'main', B=B_MAIN, boot_cluster='t0')
run(ST, 'main_cluster_event', 'main', B=0, cl='eid')
BR = META['broad']
run(BR, 'broad_all_calm_years', 'design', B=B_SUB, fes=('bink',))
run(BR, 'broad_pair_k_fe', 'design', B=B_SUB, fes=('bink', 'jck'))
run(ST, 'crop_year_fe', 'design', B=B_SUB, fes=('bink', 'jck', 'cy'))
run(ST, 'importer_crop_year_fe', 'design', B=B_SUB, fes=('bink', 'jck', 'icy'))
run(ST, 'weighted_by_volume', 'design', B=B_SUB, w='wgt')
ST['w_ev'] = 1.0 / ST.groupby('eid').imp.transform('nunique')
run(ST, 'event_equal_weights', 'design', B=B_SUB, w='w_ev')
ST['ypp'] = ST.s - ST.d
run(ST, 'share_points', 'design', B=B_SUB, outcome='ypp', extra={'mean_d_event': float(ST[ST.ev == 1].drop_duplicates(['eid', 'imp']).d.mean())})
# onset-aligned trajectory: event time measured from the first disruption year for every event
EVo = EV[EV.eid.isin(in_ev)].assign(t1=lambda x: x.t0)
SO = rc.stack(EVo, P_ALL, last_year=LAST)
STO = finish(pd.concat([SO.assign(ev=1.0), ST[ST.ev == 0][SO.columns.intersection(ST.columns)].drop_duplicates(['eid', 'imp', 'year']).assign(ev=0.0)], ignore_index=True))
STO = STO[STO.jc.isin(ST.jc.unique())].reset_index(drop=True)
run(STO, 'onset_aligned', 'design', B=B_SUB)
# balanced cohort: events whose follow-up is observed (uncensored) through k=4
obs4 = ST[(ST.ev == 1)].groupby('eid').k.max()
bal_ids = set(obs4[obs4 >= 4].index)
obs4c = ST[(ST.ev == 0)].groupby('eid').k.max(); bal_c = set(obs4c[obs4c >= 4].index)
STB = ST[(ST.eid.isin(bal_ids) | ST.eid.isin(bal_c)) & (ST.k <= 4)].reset_index(drop=True)
run(STB, 'balanced_cohort_k4', 'design', B=B_SUB, ks=list(range(-3, 5)))

# ------------------------------------------------------------------ 3. margins and decomposition
run(ST, 'extensive_margin', 'margins', B=B_SUB, outcome='yext')
run(ST, 'absent_through_k', 'margins', B=B_SUB, outcome='absent_through', ks=list(range(0, 9)), ref=None)
run(ST, 'quantity_from_supplier', 'margins', B=B_SUB, outcome='yq')
run(ST, 'total_imports_log', 'margins', B=B_SUB, outcome='dlogQ')
run(ST, 'market_relative_share', 'margins', B=B_SUB, outcome='yrho')
run(ST, 'log_share', 'margins', B=B_SUB, outcome='ylog')
STI = ST[(ST.k < 0) | (ST.q > 0)].reset_index(drop=True)
run(STI, 'intensive_positive_trade', 'margins', B=B_SUB)
# re-entry among exposures with no purchase in the last disruption year (k = 0)
z0 = ST[(ST.k == 0) & (ST.trade == 0)][['eid', 'imp']]
STR = ST.merge(z0, on=['eid', 'imp'])
run(STR, 'reentry_after_zero_k0', 'margins', B=B_SUB, outcome='trade', ks=list(range(1, 9)), ref=None)

# ------------------------------------------------------------------ 4. supplier side
def supplier_panel(df, flag):
    rows = []
    Xs = XTOT.set_index(['exp', 'commod', 'year']).X; Ws = WTOT.set_index(['commod', 'year']).W
    An = ANOM['backward'].set_index(['area', 'commod', 'year']).a_prod; Rv = EX.set_index(['exp', 'commod', 'year']).ruv
    for e in df.itertuples():
        t0, t1 = int(e.t0), int(e.t1)
        pre = [t0 - 3, t0 - 2, t0 - 1]
        xpre = np.mean([Xs.get((e.exp, e.commod, y), 0.0) for y in pre])
        wpre = np.mean([Xs.get((e.exp, e.commod, y), 0.0) / Ws.get((e.commod, y), np.nan) for y in pre])
        rpre = np.nanmean([Rv.get((e.exp, e.commod, y), np.nan) for y in pre])
        hi = min(t1 + 8, LAST)
        if not pd.isna(e.censor_year):
            hi = min(hi, int(e.censor_year) - 1)
        for y in list(range(t0 - 3, t0)) + list(range(t1, hi + 1)):
            k = y - t0 if y < t0 else y - t1
            x = Xs.get((e.exp, e.commod, y), 0.0)
            rows.append({'eid': e.eid, 'jc': f'{e.exp}_{e.commod}', 'k': k, 'ev': flag, 'type': e.type, 't0': t0, 't1': t1,
                         'exp': e.exp, 'commod': e.commod,
                         'logx_rel': np.log((x + 1e3) / (xpre + 1e3)), 'wsh_rel': (x / Ws.get((e.commod, y), np.nan)) / wpre - 1 if wpre > 0 else np.nan,
                         'a_prod': An.get((e.exp, e.commod, y), np.nan), 'druvj': Rv.get((e.exp, e.commod, y), np.nan) - rpre})
    return pd.DataFrame(rows)

SUP = pd.concat([supplier_panel(EV[EV.eid.isin(in_ev)], 1.0), supplier_panel(PS[PS.eid.isin(in_ps)], 0.0)], ignore_index=True)
SUP['jck'] = SUP.jc + '_' + SUP.k.astype(str)
sup_rows = []
for oc in ['a_prod', 'logx_rel', 'wsh_rel', 'druvj']:
    d = SUP.dropna(subset=[oc]).copy()
    if oc == 'wsh_rel':
        d[oc] = d[oc].clip(-1, 3)
    xc = xcols(d)
    b, se, G, n = rc.fe_ols(d, oc, xc, ['jck'], 'jc')
    bk = dict(zip(xc, b)); sk = dict(zip(xc, se))
    for k in KS:
        sup_rows.append({'outcome': oc, 'k': k, 'b': 0.0 if k == REF else bk[f'e{k}'], 'se': 0.0 if k == REF else sk[f'e{k}'], 'n': n, 'G': G})
pd.DataFrame(sup_rows).to_csv(OUT + 'supplier_recovery.csv', index=False)
SUP.to_csv(OUT + 'supplier_panel.csv', index=False)

# post-availability alignment: first year from t1 on in which the supplier's exports reach 90% of the pre-period mean
def t_available(df):
    Xs = XTOT.set_index(['exp', 'commod', 'year']).X
    out = []
    for e in df.itertuples():
        pre = [int(e.t0) - 3, int(e.t0) - 2, int(e.t0) - 1]
        xpre = np.mean([Xs.get((e.exp, e.commod, y), 0.0) for y in pre])
        ta = np.nan
        for y in range(int(e.t1), min(int(e.t1) + 6, LAST) + 1):
            if Xs.get((e.exp, e.commod, y), 0.0) >= 0.9 * xpre:
                ta = y; break
        out.append(ta)
    return np.array(out)

EVa = EV[EV.eid.isin(in_ev)].copy(); EVa['t_av'] = t_available(EVa)
PSa = PS[PS.eid.isin(in_ps)].copy(); PSa['t_av'] = t_available(PSa)
EVa[['eid', 'exp', 'commod', 't0', 't1', 't_av']].to_csv(OUT + 'availability_dates.csv', index=False)
desc['n_events_supply_restored_by_t1'] = int((EVa.t_av == EVa.t1).sum())
desc['n_events_supply_restored_later'] = int((EVa.t_av > EVa.t1).sum())
desc['n_events_supply_not_restored'] = int(EVa.t_av.isna().sum())
desc['median_lag_restore'] = float((EVa.t_av - EVa.t1).median())
EVa2 = EVa.dropna(subset=['t_av']).assign(t1=lambda x: x.t_av.astype(int))
PSa2 = PSa.dropna(subset=['t_av']).assign(t1=lambda x: x.t_av.astype(int))
SA = rc.stack(EVa2, P_ALL, last_year=LAST); SAP = rc.stack(PSa2, P_ALL, last_year=LAST)
STA = finish(pd.concat([SA.assign(ev=1.0), SAP.assign(ev=0.0)], ignore_index=True))
both = STA.groupby('jc').ev.agg(['min', 'max']); STA = STA[STA.jc.isin(both[(both['min'] == 0) & (both['max'] == 1)].index)].reset_index(drop=True)
run(STA, 'aligned_on_supply_restoration', 'availability', B=B_SUB)

# ------------------------------------------------------------------ 5. heterogeneity
ST['n_prior_ev'] = ST.eid.map(EV.set_index('eid').n_prior).fillna(0)
ST['w_pred'] = ST.eid.map(EV.set_index('eid').w_pred)
evt = ST[ST.ev == 1]; pst = ST[ST.ev == 0]
def sub_ev(mask):
    return pd.concat([evt[mask(evt)], pst], ignore_index=True)
for c in rc.COMMODS:
    d = ST[ST.commod == c].reset_index(drop=True)
    if d[d.ev == 1].eid.nunique() >= 2:
        run(d, c, 'crop', B=B_SUB)
for lab, m in [('production_only', lambda d: d.type == 'production'),
               ('restriction_or_both', lambda d: d.type.isin(['restriction', 'both'])),
               ('first_disruption', lambda d: d.n_prior_ev == 0), ('repeat_disruption', lambda d: d.n_prior_ev >= 1),
               ('weather_validated_nonrice', lambda d: (d.type == 'production') & (d.w_pred <= -0.05) & (d.commod != 'rice')),
               ('onset_before_2007', lambda d: d.t0 < 2007), ('onset_2007_on', lambda d: d.t0 >= 2007)]:
    d = sub_ev(m)
    if d[d.ev == 1].eid.nunique() >= 3:
        run(d, lab, 'subgroup', B=B_SUB)
for lab, m in [('d<0.10', lambda d: d.d < 0.10), ('0.10<=d<0.25', lambda d: (d.d >= 0.10) & (d.d < 0.25)),
               ('d>=0.25', lambda d: d.d >= 0.25), ('ssr<0.5', lambda d: d.ssr < 0.5), ('ssr>=0.5', lambda d: d.ssr >= 0.5)]:
    run(ST[m(ST)].reset_index(drop=True), lab, 'importer_subgroup', B=B_SUB)
    run(ST[m(ST)].reset_index(drop=True), lab + '_extensive', 'importer_subgroup', B=B_SUB, outcome='yext')

def interaction_test(D, flag, label, level):
    D = D.dropna(subset=[flag]).copy()
    D['f'] = D[flag].astype(float)
    xs = xcols(D)
    for wn, kk in [('early', [0, 1]), ('late', [2, 3, 4]), ('tail', [5, 6, 7, 8]), ('pre', [-3, -2])]:
        D[wn + '_f'] = (D.ev * D.k.isin(kk) * D.f).astype(float); xs.append(wn + '_f')
    fes = ['bink', 'jck'] + (['binkf'] if level == 'importer' else [])
    D['binkf'] = D.bink + '_' + D.f.astype(int).astype(str)
    b, se, G, n = rc.fe_ols(D, 'y', xs, fes, 'jc')
    r = dict(zip(xs, zip(b, se)))
    return [{'test': label, 'term': t, 'b': r[t][0], 'se': r[t][1], 'G': G, 'n': n} for t in ['pre_f', 'early_f', 'late_f', 'tail_f']]
ST['repeat'] = np.where(ST.ev == 1, (ST.n_prior_ev >= 1).astype(float), 0.0)
ST['restr'] = np.where(ST.ev == 1, ST.type.isin(['restriction', 'both']).astype(float), 0.0)
ST['highdep'] = (ST.d >= 0.25).astype(float)
ST['lowssr'] = np.where(ST.ssr.notna(), (ST.ssr < 0.5).astype(float), np.nan)
it = []
for f, lab, lev in [('repeat', 'repeat_vs_first', 'event'), ('restr', 'restriction_vs_production', 'event'),
                    ('highdep', 'dependence_ge_0.25', 'importer'), ('lowssr', 'self_sufficiency_lt_0.5', 'importer')]:
    it += interaction_test(ST, f, lab, lev)
pd.DataFrame(it).to_csv(OUT + 'interaction_tests.csv', index=False)

# ------------------------------------------------------------------ 6. price descriptives (not causal)
run(ST, 'descriptive_supplier_price_adjusted', 'price', B=B_SUB, controls=('druvj',))
ST['logx_rel'] = np.log((ST.X.fillna(0) + 1e3) / (ST.eid.map(ST[ST.k < 0].groupby('eid').X.mean()).fillna(0) + 1e3))
run(ST, 'descriptive_export_volume_adjusted', 'price', B=B_SUB, controls=('logx_rel',))
run(ST[ST.ruvd.notna() | (ST.k < 0)].reset_index(drop=True), 'descriptive_dyad_price_path', 'price', B=0, outcome='ruvd')
psb = META['broad'][META['broad'].ev == 0].copy()
q80 = PS.druv.quantile(0.8); spike = set(PS[PS.druv >= q80].eid)
psb['ev'] = psb.eid.isin(spike).astype(float)
run(psb, 'price_spike_calm_years', 'price', B=B_SUB, fes=('bink', 'jck'), extra={'q80_druv': float(q80), 'n_spike': len(spike)})

# relationship-state transitions for exposures observed through k = 4
tr_rows = []
for flag_, g in ST[ST.k.between(0, 4)].groupby('ev'):
    kk = g.groupby(['eid', 'imp']).k.nunique()
    full = kk[kk == 5].index
    gg = g.set_index(['eid', 'imp']).loc[full].reset_index()
    pat = gg.groupby(['eid', 'imp']).trade.agg(['sum', 'first'])
    tr_rows.append({'ev': flag_, 'n': len(pat), 'trade_all_5': float((pat['sum'] == 5).mean()),
                    'trade_none_5': float((pat['sum'] == 0).mean()), 'intermittent': float(((pat['sum'] > 0) & (pat['sum'] < 5)).mean()),
                    'zero_at_k0': float((pat['first'] == 0).mean())})
pd.DataFrame(tr_rows).to_csv(OUT + 'transitions_k0_4.csv', index=False)

# cross-crop spillover: the disrupted supplier's share in the same importer's other crops in which the
# supplier is a significant exporter and is not itself disrupted during the window
SIGX = rc.significant_exporters(EX, 0.02, 2e5).set_index(['exp', 'commod', 'year']).sig
Psh = P_ALL.set_index(['imp', 'exp', 'commod', 'year']).s
TOTc = P_ALL.groupby(['imp', 'commod', 'year']).tot.first()
A_b = ANOM['backward']; shk = set(map(tuple, A_b[(A_b.a_prod <= rc.THR_PROD) & (A_b.a_yld <= rc.THR_YLD)][['area', 'commod', 'year']].values))
bnd = set(map(tuple, PY['prohibition'][PY['prohibition'].binding][['exp', 'commod', 'year']].values))
rows = []
keys = ST.drop_duplicates(['eid', 'imp'])[['eid', 'imp', 'exp', 'commod', 't0', 't1', 'ev', 'jc']]
yr_by = ST.groupby(['eid', 'imp'])[['year', 'k', 'bink']].apply(lambda g: g.values.tolist()).to_dict()
for r in keys.itertuples():
    for c2 in rc.COMMODS:
        if c2 == r.commod or not SIGX.get((r.exp, c2, r.t0), False):
            continue
        if any((r.exp, c2, y) in shk or (r.exp, c2, y) in bnd for y in range(r.t0 - 3, r.t1 + 9)):
            continue
        pre = [Psh.get((r.imp, r.exp, c2, r.t0 - l), 0.0) if TOTc.get((r.imp, c2, r.t0 - l), 0) > 0 else np.nan for l in (1, 2, 3)]
        if np.any(np.isnan(pre)) or np.mean(pre) < 0.02:
            continue
        d2 = np.mean(pre)
        for y, k, bk in yr_by[(r.eid, r.imp)]:
            if TOTc.get((r.imp, c2, y), 0) > 0:
                rows.append({'eid': r.eid, 'imp': r.imp, 'exp': r.exp, 'commod': r.commod, 'c2': c2, 'k': k, 'year': y,
                             'ev': r.ev, 'jc': r.jc, 'jck': f'{r.jc}_{k}', 'bink': bk, 't0': r.t0,
                             'y': min(Psh.get((r.imp, r.exp, c2, y), 0.0) / d2, 3.0) - 1.0})
SPL = pd.DataFrame(rows)
SPL.to_csv(OUT + 'spillover_panel.csv', index=False)
if len(SPL) and SPL.ev.sum() > 0:
    run(SPL, 'same_supplier_other_crops', 'spillover', B=B_SUB)

# ------------------------------------------------------------------ 7. HHI falsification
HH = P_ALL.assign(s2=P_ALL.s ** 2).groupby(['imp', 'commod', 'year']).s2.sum()
pre_share = EX.set_index(['exp', 'commod', 'year']).wsh
base = ST[['eid', 'imp', 'exp', 'commod', 'k', 'year', 't0', 'ev', 'jc', 'jck', 'bink']].copy()
base['hhi'] = [HH.get((i, c, y), np.nan) for i, c, y in zip(base.imp, base.commod, base.year)]
base = base.merge(base[base.k < 0].groupby(['eid', 'imp']).hhi.mean().rename('hhi_pre').reset_index(), on=['eid', 'imp'])
base['yhhi'] = (base.hhi / base.hhi_pre - 1).clip(-1, 3)
rows = []
keys = base.drop_duplicates(['eid', 'imp'])
for r in keys.itertuples():
    for c2 in rc.COMMODS:
        if c2 == r.commod:
            continue
        if np.mean([pre_share.get((r.exp, c2, r.t0 - l), 0.0) for l in (1, 2, 3)]) >= 0.005:
            continue
        pv = [HH.get((r.imp, c2, r.t0 - l), np.nan) for l in (1, 2, 3)]
        if np.any(np.isnan(pv)):
            continue
        rows.append((r.eid, r.imp, c2, np.mean(pv)))
oth = pd.DataFrame(rows, columns=['eid', 'imp', 'c2', 'hp']).merge(base[['eid', 'imp', 'exp', 'commod', 'k', 'year', 'ev', 'jc', 'jck', 'bink']], on=['eid', 'imp'])
oth['yhhi'] = ([HH.get((i, c, y), np.nan) for i, c, y in zip(oth.imp, oth.c2, oth.year)] / oth.hp - 1).clip(-1, 3)
for lab, d in [('hhi_own_crop', base), ('hhi_unrelated_crop', oth)]:
    run(d.dropna(subset=['yhhi']).reset_index(drop=True), lab, 'hhi', B=0, outcome='yhhi')

# ------------------------------------------------------------------ 8. recurrence of disruptions
SIGM = rc.significant_exporters(EX, 0.02, 2e5)
pairs = SIGM[SIGM.sig][['exp', 'commod']].drop_duplicates()
A = ANOM['backward']; sh = A[(A.a_prod <= rc.THR_PROD) & (A.a_yld <= rc.THR_YLD)]
pb = PY['prohibition'][PY['prohibition'].binding]
durs, evs = [], []
for r in pairs.itertuples():
    yrs = sorted(set(sh[(sh.area == r.exp) & (sh.commod == r.commod) & (sh.year >= 1975)].year) |
                 set(pb[(pb.exp == r.exp) & (pb.commod == r.commod)].year))
    # episodes: disruption years separated by at least one undisrupted year
    ep = [y for i, y in enumerate(yrs) if i == 0 or y - yrs[i - 1] > 1]
    for i, y in enumerate(ep):
        if i + 1 < len(ep):
            durs.append(ep[i + 1] - y); evs.append(True)
        else:
            durs.append(LAST - y); evs.append(False)
med, Sk = rc.km_median(durs, evs)
desc['recurrence_km_median_years'] = float(med) if np.isfinite(med) else None
desc['recurrence_S'] = {str(k): float(v) for k, v in Sk.items() if k <= 15}
desc['recurrence_n_episodes'] = len(durs)

# ------------------------------------------------------------------ 9. robustness grid (window means with bootstrap intervals)
ROB = [
    ('anomaly_two_sided', dict(anom='twosided')),
    ('thr_prod20_yld15', dict(thr_prod=float(np.log(0.80)), thr_yld=float(np.log(0.85)))),
    ('thr_prod10_yld05', dict(thr_prod=float(np.log(0.90)), thr_yld=float(np.log(0.95)))),
    ('no_yield_condition', dict(yield_rule=False)),
    ('exporter_share_1pct', dict(min_share=0.01, min_q=1e5)),
    ('exporter_share_5pct', dict(min_share=0.05)),
    ('importer_min_5kt', dict(min_imp=5e3)),
    ('importer_min_100kt', dict(min_imp=1e5)),
    ('dependence_min_5pct', dict(min_dep=0.05)),
    ('importer_reported_only', dict(src='importer')),
    ('excl_2007_2008', dict(excl_years=[2007, 2008])),
    ('excl_intra_EU', dict(drop_intra_eu=True)),
    ('restrictions_incl_quotas', dict(policy='prohib_quota')),
    ('drop_capped_end_dates', dict(drop_cap12=True)),
    ('clean_window_5y', dict(clean=5)),
    ('clean_window_1y', dict(clean=1)),
    ('no_censoring', dict(censor=False)),
    ('weather_validated_nonrice', dict(weather_only=True)),
    ('outcome_cap5', dict(ycap=5.0)),
    ('onset_until_2021', dict(y_hi=2021)),
    ('calm_symmetric_window', dict(calm_after=3)),
    ('calm_band_10pct', dict(calm_band=0.10)),
]
if not os.environ.get('SKIP_ROBUST'):
    for lab, upd in ROB:
        cfg = dict(MAIN); cfg.update(upd)
        try:
            st_r, _ = build(cfg)
            run(st_r, lab, 'robust', B=B_ROB)
        except Exception as ex_:
            log('robust fail', lab, repr(ex_))
    # importer-reported flows on the main sample of exposures
    stI, _ = build(dict(MAIN, src='importer'))
    keyI = ST[['eid', 'imp', 'exp', 'commod', 't0', 'ev']].drop_duplicates(['eid', 'imp'])
    m_ = stI.merge(keyI[['imp', 'exp', 'commod', 't0', 'ev']].drop_duplicates(), on=['imp', 'exp', 'commod', 't0', 'ev'])
    run(m_, 'importer_reported_main_exposures', 'robust', B=B_ROB)
    # winsorised ratio at the 99th percentile
    q99 = ST.R.quantile(0.99); STW = ST.assign(yw=np.minimum(ST.R, q99) - 1.0)
    run(STW, 'outcome_winsor_p99', 'robust', B=B_ROB, outcome='yw', extra={'q99': float(q99)})
    run(ST, 'outcome_log_share', 'robust', B=B_ROB, outcome='ylog')
    # leave-one-crop-out and leave-one-supplier-out (point estimates)
    loo = []
    for c in rc.COMMODS:
        d = ST[ST.commod != c].reset_index(drop=True)
        cv, _, _, _, _, _ = estimate(d); w_ = windows(cv); loo.append({'drop': 'crop', 'unit': c, **w_})
    for j in sorted(ST[ST.ev == 1].exp.unique()):
        d = ST[ST.exp != j].reset_index(drop=True)
        cv, _, _, _, _, _ = estimate(d); w_ = windows(cv); loo.append({'drop': 'supplier', 'unit': nmap.get(j, j), **w_})
    for t in sorted(EV[EV.eid.isin(in_ev)].t0.unique()):
        drop_ids = set(EV[EV.t0 == t].eid)
        d = ST[~((ST.ev == 1) & ST.eid.isin(drop_ids))].reset_index(drop=True)
        cv, _, _, _, _, _ = estimate(d); w_ = windows(cv); loo.append({'drop': 'onset_year', 'unit': int(t), **w_})
    pd.DataFrame(loo).to_csv(OUT + 'leave_one_out.csv', index=False)

# ------------------------------------------------------------------ 10. Egypt wheat case
eg = P_ALL[(P_ALL.imp == 59) & (P_ALL.commod == 'wheat')].copy()
eg['name'] = eg.exp.map(nmap)
eg[['year', 'exp', 'name', 'q', 's', 'src']].to_csv(OUT + 'egypt_wheat.csv', index=False)

pd.DataFrame(RECS).to_csv(OUT + 'summary.csv', index=False)
pd.concat(CURVES, ignore_index=True).to_csv(OUT + 'curves.csv', index=False)
desc['runtime_s'] = time.time() - T0
json.dump(desc, open(OUT + 'descriptives.json', 'w'), indent=1, default=float)
log('DONE')
