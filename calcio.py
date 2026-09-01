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

def imp_prob(odds):
    return 1 / odds if odds > 0 else 0

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

    st.subheader("📊 QUOTE BOOKMAKERS")
    odd_h = st.number_input("Casa", 0.0, 20.0, 0.0, step=0.01)
    odd_d = st.number_input("X", 0.0, 20.0, 0.0, step=0.01)
    odd_a = st.number_input("Ospite", 0.0, 20.0, 0.0, step=0.01)

    calc = st.button("🚀 ANALIZZA CONFRONTO")

with col2:
    if calc:
        st.markdown(f"### 🏟️ PARTITA: **{home} vs {away}**")
        st.markdown("---")

        lh, la = predict(home, away, key_h, key_a, top_h, top_a, rest_h, rest_a, motivation_h, motivation_a)
        h, d, a = simulate(lh, la)

        max_goals = 7
        p_1x = p_x2 = p_over15 = p_under15 = p_over25 = p_under25 = p_btts = 0
        total_prob = 0

        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson_pmf(i, lh) * poisson_pmf(j, la)
                p *= dixon_coles_adjust(i, j, lh, la)
                total_prob += p

                if i >= j: p_1x += p
                if j >= i: p_x2 += p
                if i + j >= 3: p_over25 += p
                else: p_under25 += p
                if i + j >= 2: p_over15 += p
                else: p_under15 += p
                if i > 0 and j > 0: p_btts += p

        if total_prob > 0:
            p_1x /= total_prob
            p_x2 /= total_prob
            p_over25 /= total_prob
            p_under25 /= total_prob
            p_over15 /= total_prob
            p_under15 /= total_prob
            p_btts /= total_prob

        def to_odds(p):
            return 1 / p if p > 0 else 0

        st.subheader("📊 QUOTE DEL MODELLO")
        st.write(f"🏠 {to_odds(h):.2f} | 🤝 {to_odds(d):.2f} | 🚗 {to_odds(a):.2f}")

        st.subheader("📊 QUOTE DEI BOOKMAKERS")
        if odd_h > 0 and odd_d > 0 and odd_a > 0:
            st.write(f"🏠 {odd_h:.2f} | 🤝 {odd_d:.2f} | 🚗 {odd_a:.2f}")
        else:
            st.info("ℹ️ Inserisci le quote dei bookmakers per visualizzarle e confrontarle.")

        st.subheader("💡 SUGGERIMENTI DEL MODELLO")
        if odd_h > 0 and odd_d > 0 and odd_a > 0:
            ev_h = (h * odd_h) - 1
            ev_d = (d * odd_d) - 1
            ev_a = (a * odd_a) - 1

            has_heavy_favorite_home = (odd_h <= 2.00)
            has_heavy_favorite_away = (odd_a <= 2.00)
            high_double_chance_active = (p_1x > 0.70) or (p_x2 > 0.70)

            def display_safe_player_advice(ev, label, book_odd, target_type):
                if high_double_chance_active:
                    if target_type == "h" and p_1x > 0.70:
                        st.info(f"🛡️ **PRUDENZA SU {label}** — Il modello rileva un'alta copertura della doppia chance 1X ({p_1x*100:.1f}%). Meglio valutare la copertura o un esito prudente anziché la vittoria secca.")
                    elif target_type == "a" and p_x2 > 0.70:
                        st.info(f"🛡️ **PRUDENZA SU {label}** — Il modello rileva un'alta copertura della doppia chance X2 ({p_x2*100:.1f}%). Il match è molto chiuso, valuta la copertura.")
                    else:
                        st.warning(f"⚠️ **ATTENZIONE A {label}** — Elevato rischio di partita bloccata o pareggio in base ai flussi di probabilità.")
                elif has_heavy_favorite_home or has_heavy_favorite_away:
                    if book_odd <= 2.00:
                        st.success(f"🔥 **FAVORITO DI MERCATO ({label})** — Questa squadra ha i favori netti dei bookmaker (quota <= 2.00). Il mercato la vede vincente, segui il trend principale.")
                    else:
                        st.warning(f"❌ **NON PUNTARE SU {label}** — C'è un chiaro favorito forte dall'altra parte. Sconsigliato andare contro il mercato in questa situazione.")
                elif ev > 0.05:
                    st.success(f"🎯 **PUNTA SU {label}** — Questa quota è un vero affare! L'Agenzia di Scommesse la paga di più rispetto al reale rischio.")
                elif ev > 0:
                    st.info(f"👍 **CI PUÒ STARE SU {label}** — C'è un piccolo vantaggio, puoi metterla nella tua schedina.")
                else:
                    st.warning(f"🚫 **LASCIA PERDERE {label}** — Questa quota è troppo bassa rispetto alle reali probabilità. Ci guadagna solo l'Agenzia di Scommesse.")

            display_safe_player_advice(ev_h, "Casa (1)", odd_h, "h")
            display_safe_player_advice(ev_d, "Pareggio (X)", odd_d, "d")
            display_safe_player_advice(ev_a, "Ospite (2)", odd_a, "a")

            book_h = imp_prob(odd_h)
            book_d = imp_prob(odd_d)
            book_a = imp_prob(odd_a)
            
            error = np.mean([abs(h - book_h), abs(d - book_d), abs(a - book_a)])
            confidence = max(0, 1 - error * 3)

            st.subheader("🎯 CONFRONTO MODELLO-BOOKMAKERS")
            st.markdown(f"<h1 style='text-align:center; color:#2ecc71;'>{confidence*100:.1f}%</h1>", unsafe_allow_html=True)

            if confidence > 0.7:
                st.success("🟢 QUOTE DEL MODELLO VICINE A QUELLE DEI BOOKMAKERS")
            elif confidence > 0.4:
                st.warning("🟡 QUOTE DEL MODELLO INCERTE")
            else:
                st.error("🔴 QUOTE DEL MODELLO LONTANE DA QUELLE DEI BOOKMAKERS")
                st.warning(
                    "🚨 **ATTENZIONE VALUTARE CAUTELA**\n\n"
                    "Calcolata probabilità **totalmente diversa** rispetto all'Agenzia di Scommesse. "
                    "Questo può significare che:\n"
                    "1. **Super Value Bet:** Il modello ha individuato una quota sottovalutata dall'Agenzia di Scommesse.\n"
                    "2. **Informazione Mancante:** C'è un fattore critico (infortunio last-minute, turnover pesante, ecc.) che il modello statistico non può intercettare.\n\n"
                    "💡 *Consiglio di tutela:* Se decidi di seguire l'intuizione del modello su quote così distanti, **punta cifre simboliche o valuta coperture (Doppia Chance / Handicap)**."
                )
        else:
            st.info("ℹ️ Inserisci le quote dei bookmakers per attivare i suggerimenti e il confronto del modello.")

        st.markdown("---")
        st.subheader("📊 ULTERIORI PROBABILITÀ")

        st.write(f"🟢 1X - La squadra di casa non perde: {p_1x*100:.1f}%")
        st.write(f"🔵 X2 - La squadra ospite non perde: {p_x2*100:.1f}%")
        st.write(f"⚽ Over 1.5: {p_over15*100:.1f}%")
        st.write(f"📉 Under 1.5: {p_under15*100:.1f}%")
        st.write(f"⚽ Over 2.5: {p_over25*100:.1f}%")
        st.write(f"📉 Under 2.5: {p_under25*100:.1f}%")
        st.write(f"🤝 Goal/Goal: {p_btts*100:.1f}%")

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
                
                # Assegnazione pallino dal punto di vista della squadra 'home'
                if h_team == home:
                    if score_h > score_a:
                        bullet = "🟢" # Vittoria
                    elif score_h == score_a:
                        bullet = "🟡" # Pareggio
                    else:
                        bullet = "🔴" # Sconfitta
                else:
                    if score_a > score_h:
                        bullet = "🟢" # Vittoria
                    elif score_a == score_h:
                        bullet = "🟡" # Pareggio
                    else:
                        bullet = "🔴" # Sconfitta

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
                
                # Assegnazione pallino dal punto di vista della squadra 'away'
                if h_team == away:
                    if score_h > score_a:
                        bullet = "🟢" # Vittoria
                    elif score_h == score_a:
                        bullet = "🟡" # Pareggio
                    else:
                        bullet = "🔴" # Sconfitta
                else:
                    if score_a > score_h:
                        bullet = "🟢" # Vittoria
                    elif score_a == score_h:
                        bullet = "🟡" # Pareggio
                    else:
                        bullet = "🔴" # Sconfitta

                st.write(f"{bullet} **{date_str}** — {h_team} vs {a_team}: **{score_h}-{score_a}**")
        else:
            st.info(f"Nessuna partita registrata per {away} da agosto 2026.")
