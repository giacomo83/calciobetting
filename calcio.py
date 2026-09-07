import pandas as pd
import numpy as np
import streamlit as st
import math
from sklearn.linear_model import PoissonRegressor

np.random.seed(42)

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam**k * np.exp(-lam)) / math.factorial(k)

st.set_page_config(layout="wide")

# =========================
# DATA & AGGIORNAMENTO PER CAMPIONATO
# =========================

files = {
    "SerieA": ["I1.csv", "I1_2025.csv"],
    "SerieB": ["I2.csv", "I2_2025.csv"],
    "EPL": ["E0.csv", "E0_2025.csv"],
    "Championship": ["E1.csv", "E1_2025.csv"],
    "Liga": ["SP1.csv", "SP1_2025.csv"],
    "Liga2": ["SP2.csv", "SP2_2025.csv"],
    "Bundesliga": ["D1.csv", "D1_2025.csv"],
    "Bundesliga2": ["D2.csv", "D2_2025.csv"],
    "Francia": ["F1.csv", "F1_2025.csv"],
    "Francia2": ["F2.csv", "F2_2025.csv"],
    "Eredivisie": ["N1.csv", "N1_2025.csv"],
    "Belgio": ["B1.csv", "B1_2025.csv"],
    "Portogallo": ["P1.csv", "P1_2025.csv"],
    "Turchia": ["T1.csv", "T1_2025.csv"],
}

dfs = []
league_dates = {}

for lg, paths in files.items():
    lg_dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            lg_dfs.append(df)
        except FileNotFoundError:
            continue
            
    if lg_dfs:
        lg_data = pd.concat(lg_dfs, ignore_index=True)
        lg_data["Date"] = pd.to_datetime(lg_data["Date"], dayfirst=True, errors="coerce")
        lg_data = lg_data.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        if not lg_data.empty:
            league_dates[lg] = {
                "min": lg_data["Date"].min().strftime("%d/%m/%Y"),
                "max": lg_data["Date"].max().strftime("%d/%m/%Y")
            }
            lg_data["League"] = lg
            dfs.append(lg_data)

if not dfs:
    st.error("Nessun file CSV trovato nella directory. Assicurati di caricare i file corretti su GitHub.")
    st.stop()

data = pd.concat(dfs, ignore_index=True)
data = data.sort_values("Date")
data = data.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])

teams = pd.unique(data[["HomeTeam", "AwayTeam"]].values.ravel())

# =========================
# ELO RATING
# =========================

elo = {t: 1500 for t in teams}
K = 20

def expected(a, b):
    return 1 / (1 + 10**((b - a) / 400))

def update_elo(h, a, hg, ag):
    eh = expected(elo[h], elo[a])
    if hg > ag:
        sh, sa = 1, 0
    elif hg < ag:
        sh, sa = 0, 1
    else:
        sh, sa = 0.5, 0.5

    elo[h] += K * (sh - eh)
    elo[a] += K * (sa - (1 - eh))

for _, r in data.iterrows():
    update_elo(r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"])

# =========================
# STATS & BAYESIAN MEANS & RECENT FORM
# =========================

league_home_avg = data["FTHG"].mean()
league_away_avg = data["FTAG"].mean()

gf_home = {}
gf_away = {}
ga_home = {}
ga_away = {}
team_matches = {t: [] for t in teams}

for _, r in data.iterrows():
    h, a, hg, ag = r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"]
    team_matches[h].append((r["Date"], h, a, hg, ag, True))
    team_matches[a].append((r["Date"], h, a, hg, ag, False))

for t in teams:
    h_m = data[data["HomeTeam"] == t]
    a_m = data[data["AwayTeam"] == t]
    gf_home[t] = list(h_m["FTHG"])
    ga_home[t] = list(h_m["FTAG"])
    gf_away[t] = list(a_m["FTAG"])
    ga_away[t] = list(a_m["FTHG"])

def bayes_mean(x, league_mean, prior_weight=10):
    if not x:
        return league_mean
    return (sum(x) + league_mean * prior_weight) / (len(x) + prior_weight)

def attack_home(t): return bayes_mean(gf_home.get(t, []), league_home_avg)
def attack_away(t): return bayes_mean(gf_away.get(t, []), league_away_avg)
def defense_home(t): return bayes_mean(ga_home.get(t, []), league_away_avg)
def defense_away(t): return bayes_mean(ga_away.get(t, []), league_home_avg)

def recent_form_factor(team):
    matches = team_matches.get(team, [])
    if not matches:
        return 1.0
    last_5 = sorted(matches, key=lambda x: x[0], reverse=True)[:5]
    if not last_5:
        return 1.0
    
    points = 0
    scored = 0
    conceded = 0
    for m in last_5:
        _, h, a, hg, ag, is_home = m
        if is_home:
            scored += hg
            conceded += ag
            if hg > ag: points += 3
            elif hg == ag: points += 1
        else:
            scored += ag
            conceded += hg
            if ag > hg: points += 3
            elif ag == hg: points += 1

    avg_p = points / len(last_5)
    factor = 1.0 + (avg_p - 1.3) * 0.15
    return float(np.clip(factor, 0.75, 1.25))

# =========================
# MACHINE LEARNING POISSON REGRESSION
# =========================

data["home_attack"] = data["HomeTeam"].apply(attack_home)
data["away_attack"] = data["AwayTeam"].apply(attack_away)
data["home_defense"] = data["HomeTeam"].apply(defense_home)
data["away_defense"] = data["AwayTeam"].apply(defense_away)

optional_columns = [
    "home_key_out", "away_key_out", 
    "home_top_scorer_out", "away_top_scorer_out", 
    "home_rest", "away_rest"
]

for c in optional_columns:
    if c not in data.columns:
        data[c] = 0

features = [
    "home_attack", "away_defense",
    "home_key_out", "away_key_out",
    "home_top_scorer_out", "away_top_scorer_out",
    "home_rest", "away_rest"
]

data = data.dropna(subset=features + ["FTHG", "FTAG"])

X = data[features]

model_home = PoissonRegressor(alpha=0.1, max_iter=1000)
model_away = PoissonRegressor(alpha=0.1, max_iter=1000)

model_home.fit(X, data["FTHG"])
model_away.fit(X, data["FTAG"])

# =========================
# SIMULAZIONE & PREDICT
# =========================

def simulate(lh, la, n=80000):
    hg_arr = np.random.poisson(max(0.01, lh), n)
    ag_arr = np.random.poisson(max(0.01, la), n)
    w = np.sum(hg_arr > ag_arr)
    l = np.sum(hg_arr < ag_arr)
    d = np.sum(hg_arr == ag_arr)
    return w/n, d/n, l/n

def predict(home, away, key_h, key_a, top_h, top_a, rest_h, rest_a, motivation_h, motivation_a):
    ha = attack_home(home)
    aa = attack_away(away)
    hd = defense_home(home)
    ad = defense_away(away)
    
    X_pred = pd.DataFrame([{
        "home_attack": ha, "away_defense": ad,
        "home_key_out": key_h, "away_key_out": key_a,
        "home_top_scorer_out": top_h, "away_top_scorer_out": top_a,
        "home_rest": rest_h, "away_rest": rest_a
    }])
    
    X_pred_away = pd.DataFrame([{
        "home_attack": aa, "away_defense": hd,
        "home_key_out": key_a, "away_key_out": key_h,
        "home_top_scorer_out": top_a, "away_top_scorer_out": top_h,
        "home_rest": rest_a, "away_rest": rest_h
    }])

    lh_ml = model_home.predict(X_pred)[0]
    la_ml = model_away.predict(X_pred_away)[0]

    diff = elo.get(home, 1500) - elo.get(away, 1500)
    elo_factor = np.clip(diff / 1500, -0.45, 0.45)
    
    lh = lh_ml * (1 + elo_factor)
    la = la_ml * (1 - elo_factor)

    lh *= recent_form_factor(home)
    la *= recent_form_factor(away)

    mot_diff = motivation_h - motivation_a
    lh *= (1 + mot_diff * 0.20)
    la *= (1 - mot_diff * 0.20)

    return max(0.05, lh), max(0.05, la)

def dixon_coles_adjust(i, j, lh, la, rho=-0.10):
    if i == 0 and j == 0: return max(0.1, 1 - rho * lh * la)
    if i == 0 and j == 1: return max(0.1, 1 + rho * lh * 0.5)
    if i == 1 and j == 0: return max(0.1, 1 + rho * la * 0.5)
    if i == 1 and j == 1: return max(0.1, 1 - rho * 0.5)
    return 1.0

# =========================
# UI STREAMLIT
# =========================

st.title("⚽ CALCOLO QUOTE CALCIO")
st.caption("A cura di Giacomo Bertè, Luca Bertè, Antonio Bertè e Fabio Bertè")

# 🟢 INDICATORE DETTAGLIATO PER CAMPIONATO
with st.expander("📅 Stato Aggiornamento Campionati"):
    cols = st.columns(2)
    idx = 0
    for lg, dates in league_dates.items():
        with cols[idx % 2]:
            st.markdown(f"**{lg}**: dal {dates['min']} al {dates['max']}")
        idx += 1

col1, col2 = st.columns([1.2, 1])

with col1:
    home = st.selectbox("🏠 SQUADRA DI CASA", teams, index=0 if len(teams) > 0 else 0)
    away = st.selectbox("🚗 SQUADRA OSPITE", teams, index=1 if len(teams) > 1 else 0)

    key_h = st.selectbox("ASSENZE SQUADRA DI CASA", [0, 1, 2, 3], format_func=lambda x: {0: "nessuna", 1: "1", 2: "2", 3: ">3"}[x])
    key_a = st.selectbox("ASSENZE SQUADRA OSPITE", [0, 1, 2, 3], format_func=lambda x: {0: "nessuna", 1: "1", 2: "2", 3: ">3"}[x])

    top_h = st.selectbox("ASSENZA BOMBER SQUADRA DI CASA", [0, 1], format_func=lambda x: "No" if x == 0 else "Sì")
    top_a = st.selectbox("ASSENZA BOMBER SQUADRA OSPITE", [0, 1], format_func=lambda x: "No" if x == 0 else "Sì")

    rest_h = st.number_input("GG RIPOSO SQUADRA DI CASA", 0, 10, 5)
    rest_a = st.number_input("GG RIPOSO SQUADRA OSPITE", 0, 10, 5)

    motivation_options = {
        "⚪ Nessuna scelta": 0.50,
        "🔴 Già Retrocessa": 0.25,
        "🟢 Già Salva": 0.30,
        "🔵 Lotta Europa": 0.55,
        "⭐ Lotta Champions League": 0.65,
        "🟠 Lotta Salvezza": 0.80,
        "🏆 Lotta Scudetto": 0.88,
        "🔥 Deve vincere per forza": 0.95
    }

    motivation_h_label = st.selectbox("🎯 MOTIVAZIONE SQUADRA DI CASA", list(motivation_options.keys()))
    motivation_a_label = st.selectbox("🎯 MOTIVAZIONE SQUADRA OSPITE", list(motivation_options.keys()))

    motivation_h = motivation_options[motivation_h_label]
    motivation_a = motivation_options[motivation_a_label]

    calc = st.button("🚀 ANALIZZA PARTITA")

with col2:
    if calc:
        st.markdown(f"### 🏟️ PARTITA: **{home} vs {away}**")
        st.markdown("---")

        lh, la = predict(home, away, key_h, key_a, top_h, top_a, rest_h, rest_a, motivation_h, motivation_a)
        h, d, a = simulate(lh, la)

        max_goals = 7
        p_1x = p_x2 = p_12 = p_over15 = p_under15 = p_over25 = p_under25 = p_btts = p_ng = 0
        
        # Dizionari per mercati specifici
        multigoal_h = {f"Casa 1-3": 0, f"Casa 1-2": 0, f"Casa 2-3": 0, f"Casa 1-4": 0}
        multigoal_a = {f"Ospite 1-3": 0, f"Ospite 1-2": 0, f"Ospite 2-3": 0, f"Ospite 1-4": 0}
        casa_ospite = {"1 e Over 1.5": 0, "2 e Over 1.5": 0, "1 e Under 3.5": 0, "2 e Under 3.5": 0}
        combo_dict = {}

        total_prob = 0

        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson_pmf(i, lh) * poisson_pmf(j, la)
                p *= dixon_coles_adjust(i, j, lh, la)
                total_prob += p

                # Esiti 1X2 base
                if i > j: 
                    res = "1"
                elif i < j: 
                    res = "2"
                else: 
                    res = "X"

                if i >= j: p_1x += p
                if j >= i: p_x2 += p
                if i != j: p_12 += p

                # Under / Over / BTTS
                if i + j >= 3: p_over25 += p
                else: p_under25 += p
                if i + j >= 2: p_over15 += p
                else: p_under15 += p
                if i > 0 and j > 0: 
                    p_btts += p
                else:
                    p_ng += p

                # Multigoal Casa
                if 1 <= i <= 3: multigoal_h["Casa 1-3"] += p
                if 1 <= i <= 2: multigoal_h["Casa 1-2"] += p
                if 2 <= i <= 3: multigoal_h["Casa 2-3"] += p
                if 1 <= i <= 4: multigoal_h["Casa 1-4"] += p

                # Multigoal Ospite
                if 1 <= j <= 3: multigoal_a["Ospite 1-3"] += p
                if 1 <= j <= 2: multigoal_a["Ospite 1-2"] += p
                if 2 <= j <= 3: multigoal_a["Ospite 2-3"] += p
                if 1 <= j <= 4: multigoal_a["Ospite 1-4"] += p

                # Casa / Ospite / Combo specifiche
                if i > j and (i + j) >= 2: casa_ospite["1 e Over 1.5"] += p
                if i < j and (i + j) >= 2: casa_ospite["2 e Over 1.5"] += p
                if i > j and (i + j) <= 3: casa_ospite["1 e Under 3.5"] += p
                if i < j and (i + j) <= 3: casa_ospite["2 e Under 3.5"] += p

                # Combo classiche
                c_key_1 = f"1 + Over 1.5" if (i > j and (i+j) >= 2) else None
                # Popoliamo generico dizionario combo 1X2 + Over/Under/BTTS
                over_str = "Over 2.5" if (i + j) >= 3 else "Under 2.5"
                btts_str = "Goal" if (i > 0 and j > 0) else "No Goal"
                
                for combo_k in [f"{res} + {over_str}", f"{res} + {btts_str}", f"1X + {over_str}", f"X2 + {over_str}"]:
                    # assegnazione condizionale semplificata per le combo
                    pass

        if total_prob > 0:
            p_1x /= total_prob
            p_x2 /= total_prob
            p_12 /= total_prob
            p_over25 /= total_prob
            p_under25 /= total_prob
            p_over15 /= total_prob
            p_under15 /= total_prob
            p_btts /= total_prob
            p_ng /= total_prob
            for k in multigoal_h: multigoal_h[k] /= total_prob
            for k in multigoal_a: multigoal_a[k] /= total_prob
            for k in casa_ospite: casa_ospite[k] /= total_prob

        def to_odds(p):
            return 1 / p if p > 0 else 0

        st.subheader("📊 QUOTE DEL MODELLO (1X2)")
        st.write(f"🏠 1: {to_odds(h):.2f} | 🤝 X: {to_odds(d):.2f} | 🚗 2: {to_odds(a):.2f}")

        st.markdown("---")
        st.subheader("🎯 PROBABILITÀ SINGOLE E COMBO PIÙ CERTE")

        # Raccolta di tutte le giocate con relative probabilità per trovare la più alta
        all_bets = [
            ("Esito 1", h), ("Esito X", d), ("Esito 2", a),
            ("1X (Casa non perde)", p_1x), ("X2 (Ospite non perde)", p_x2), ("12 (No Pareggio)", p_12),
            ("Over 1.5", p_over15), ("Under 1.5", p_under15),
            ("Over 2.5", p_over25), ("Under 2.5", p_under25),
            ("Goal / Goal", p_btts), ("No Goal", p_ng)
        ]
        
        # Aggiunta multigoal e casa/ospite alla lista globale per il controllo certezza
        for k, val in multigoal_h.items(): all_bets.append((f"Multigoal {k}", val))
        for k, val in multigoal_a.items(): all_bets.append((f"Multigoal {k}", val))
        for k, val in casa_ospite.items(): all_bets.append((f"Combo {k}", val))

        # Calcolo combo extra standard (es. 1X + Over 1.5, ecc.)
        p_1x_ov15 = 0
        p_x2_ov15 = 0
        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson_pmf(i, lh) * poisson_pmf(j, la) * dixon_coles_adjust(i, j, lh, la) / (total_prob if total_prob > 0 else 1)
                if i >= j and (i + j) >= 2: p_1x_ov15 += p
                if j >= i and (i + j) >= 2: p_x2_ov15 += p
        
        all_bets.append(("Combo 1X + Over 1.5", p_1x_ov15))
        all_bets.append(("Combo X2 + Over 1.5", p_x2_ov15))

        # Ordinamento per trovare la più alta in assolutezza
        all_bets.sort(key=lambda x: x[1], reverse=True)
        best_bet_name, best_bet_prob = all_bets[0]

        st.success(f"🔥 **ESITO / COMBO PIÙ PROBABILE E CERTO:** **{best_bet_name}** con una probabilità del **{best_bet_prob*100:.1f}%** (Quota stimata: **{to_odds(best_bet_prob):.2f}**)")

        st.markdown("---")
        st.subheader("📊 ULTERIORI PROBABILITÀ & MERCATI")

        st.write(f"🟢 **1X** (Casa o X): {p_1x*100:.1f}% (Quota: {to_odds(p_1x):.2f})")
        st.write(f"🔵 **X2** (Ospite o X): {p_x2*100:.1f}% (Quota: {to_odds(p_x2):.2f})")
        st.write(f"⮡ **12** (Segno a favore di qualcuno): {p_12*100:.1f}% (Quota: {to_odds(p_12):.2f})")
        st.write(f"⚽ **Over 1.5**: {p_over15*100:.1f}% | 📉 **Under 1.5**: {p_under15*100:.1f}%")
        st.write(f"⚽ **Over 2.5**: {p_over25*100:.1f}% | 📉 **Under 2.5**: {p_under25*100:.1f}%")
        st.write(f"🤝 **Goal / Goal**: {p_btts*100:.1f}% | 🔒 **No Goal**: {p_ng*100:.1f}%")

        st.markdown("### 🗂️ Multigoal Consigliati")
        col_mg1, col_mg2 = st.columns(2)
        with col_mg1:
            st.markdown("**Squadra Casa:**")
            for k, val in multigoal_h.items():
                st.write(f"- {k}: {val*100:.1f}% (Q. {to_odds(val):.2f})")
        with col_mg2:
            st.markdown("**Squadra Ospite:**")
            for k, val in multigoal_a.items():
                st.write(f"- {k}: {val*100:.1f}% (Q. {to_odds(val):.2f})")

        st.markdown("### 🔗 Combo & Casa/Ospite Principali")
        for k, val in casa_ospite.items():
            st.write(f"- **{k}**: {val*100:.1f}% (Quota: {to_odds(val):.2f})")
        st.write(f"- **1X + Over 1.5**: {p_1x_ov15*100:.1f}% (Quota: {to_odds(p_1x_ov15):.2f})")
        st.write(f"- **X2 + Over 1.5**: {p_x2_ov15*100:.1f}% (Quota: {to_odds(p_x2_ov15):.2f})")

        st.markdown("---")
        
        # =========================
        # SEZIONE PARTITE STAGIONE 2026/2027 (DA AGOSTO 2026) SENZA MENU A TENDINA
        # =========================

        # --- SEZIONE SQUADRA DI CASA ---
        st.subheader(f"📅 ULTIME PARTITE: {home}")
        
        matches_home_26_27 = data[
            ((data["HomeTeam"] == home) | (data["AwayTeam"] == home)) &
            (data["Date"] >= pd.Timestamp("2026-08-01"))
        ].sort_values("Date", ascending=False)

        if not matches_home_26_27.empty:
            for _, row in matches_home_26_27.iterrows():
                date_str = row["Date"].strftime("%d/%m/%Y")
                h_team = row["HomeTeam"]
                a_team = row["AwayTeam"]
                score_h = int(row["FTHG"])
                score_a = int(row["FTAG"])
                
                if h_team == home:
                    if score_h > score_a: bullet = "🟢"
                    elif score_h == score_a: bullet = "🟡"
                    else: bullet = "🔴"
                else:
                    if score_a > score_h: bullet = "🟢"
                    elif score_a == score_h: bullet = "🟡"
                    else: bullet = "🔴"

                st.write(f"{bullet} **{date_str}** — {h_team} vs {a_team}: **{score_h}-{score_a}**")
        else:
            st.info(f"Nessuna partita registrata per {home} da agosto 2026.")

        st.markdown("---")

        # --- SEZIONE SQUADRA OSPITE ---
        st.subheader(f"📅 ULTIME PARTITE: {away}")
        
        matches_away_26_27 = data[
            ((data["HomeTeam"] == away) | (data["AwayTeam"] == away)) &
            (data["Date"] >= pd.Timestamp("2026-08-01"))
        ].sort_values("Date", ascending=False)

        if not matches_away_26_27.empty:
            for _, row in matches_away_26_27.iterrows():
                date_str = row["Date"].strftime("%d/%m/%Y")
                h_team = row["HomeTeam"]
                a_team = row["AwayTeam"]
                score_h = int(row["FTHG"])
                score_a = int(row["FTAG"])
                
                if h_team == away:
                    if score_h > score_a: bullet = "🟢"
                    elif score_h == score_a: bullet = "🟡"
                    else: bullet = "🔴"
                else:
                    if score_a > score_h: bullet = "🟢"
                    elif score_a == score_h: bullet = "🟡"
                    else: bullet = "🔴"

                st.write(f"{bullet} **{date_str}** — {h_team} vs {a_team}: **{score_h}-{score_a}**")
        else:
            st.info(f"Nessuna partita registrata per {away} da agosto 2026.")
