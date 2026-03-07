import io
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


st.set_page_config(page_title="Elite Soccer Analytics", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_embeddings(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def kpis(player_df: pd.DataFrame) -> dict:
    if player_df.empty:
        return {"Players": 0, "Avg npxG/90": 0, "Avg xA/90": 0}
    return {
        "Players": player_df["player_id"].nunique(),
        "Avg npxG/90": round(player_df.get("per90_npxg", pd.Series([0])).mean(), 2),
        "Avg xA/90": round(player_df.get("per90_xa", pd.Series([0])).mean(), 2),
    }


def radar_chart(player_row: pd.Series, role_avg: pd.Series):
    metrics = [
        "per90_npxg",
        "per90_xa",
        "per90_progressive_passes",
        "per90_progressive_carries",
        "per90_pressures",
        "per90_tackles",
        "per90_interceptions",
        "per90_passes_final_third",
    ]
    label_map = {
        "per90_npxg": "Non-penalty xG (per 90)",
        "per90_xa": "Expected Assists (per 90)",
        "per90_progressive_passes": "Progressive Passes (per 90)",
        "per90_progressive_carries": "Progressive Carries (per 90)",
        "per90_pressures": "Pressures (per 90)",
        "per90_tackles": "Tackles (per 90)",
        "per90_interceptions": "Interceptions (per 90)",
        "per90_passes_final_third": "Passes into Final Third (per 90)",
    }
    metrics = [m for m in metrics if m in player_row.index]
    player_vals = [player_row.get(m, 0) for m in metrics]
    role_vals = [role_avg.get(m, 0) for m in metrics]

    labels = [label_map.get(m, m) for m in metrics]
    df = pd.DataFrame({"metric": labels, "player": player_vals, "role_avg": role_vals})
    fig = px.line_polar(df, r="player", theta="metric", line_close=True, title="Player Radar (per 90 minutes)")
    fig.add_trace(px.line_polar(df, r="role_avg", theta="metric", line_close=True).data[0])
    fig.update_traces(fill="toself")
    return fig


def generate_pdf_report(player_id: int, player_row: pd.Series) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Player Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Player ID: {player_id}")
    y = 690
    for col, val in player_row.items():
        if col.startswith("per90_"):
            c.drawString(50, y, f"{col}: {round(float(val), 3)}")
            y -= 16
            if y < 100:
                c.showPage()
                y = 750
    c.save()
    buffer.seek(0)
    return buffer.read()


def player_touch_heatmap(events: pd.DataFrame, player_id: int):
    player_events = events[(events["player_id"] == player_id) & events["x"].notna() & events["y"].notna()]
    if player_events.empty:
        return None
    pitch = Pitch(pitch_type="statsbomb", line_color="white")
    fig, ax = pitch.draw(figsize=(6, 4))
    bins = pitch.bin_statistic(player_events["x"], player_events["y"], statistic="count", bins=(6, 4))
    pitch.heatmap(bins, ax=ax, cmap="Reds")
    pitch.scatter(player_events["x"], player_events["y"], ax=ax, s=5, color="white", alpha=0.5)
    ax.set_title("Touch Heatmap")
    return fig


def player_action_mix(events: pd.DataFrame, player_id: int):
    player_events = events[events["player_id"] == player_id]
    if player_events.empty:
        return None
    focus = player_events["event_type"].value_counts().head(8).reset_index()
    focus.columns = ["action", "count"]
    fig = px.pie(focus, names="action", values="count", title="Action Mix (what the player does most)")
    return fig


def similarity_reasons(player_features: pd.DataFrame, player_id: int, candidate_ids: list, player_name_map: dict) -> pd.DataFrame:
    metrics = [
        "per90_npxg",
        "per90_xa",
        "per90_progressive_passes",
        "per90_progressive_carries",
        "per90_pressures",
        "per90_tackles",
        "per90_interceptions",
        "per90_passes_final_third",
    ]
    label_map = {
        "per90_npxg": "Chance Quality",
        "per90_xa": "Chance Creation",
        "per90_progressive_passes": "Forward Passing",
        "per90_progressive_carries": "Forward Dribbling",
        "per90_pressures": "Pressing",
        "per90_tackles": "Tackling",
        "per90_interceptions": "Interceptions",
        "per90_passes_final_third": "Passes into Attack",
    }
    df = player_features[player_features["player_id"].isin([player_id] + candidate_ids)].copy()
    df = df.set_index("player_id")
    # z-score for fair comparison
    z = (df[metrics] - df[metrics].mean()) / (df[metrics].std(ddof=0) + 1e-9)
    base = z.loc[player_id]

    rows = []
    for cid in candidate_ids:
        if cid not in z.index:
            continue
        diffs = (z.loc[cid] - base).abs().sort_values()
        top = diffs.head(3).index.tolist()
        reason = ", ".join(label_map.get(m, m) for m in top)
        rows.append({
            "player_name": player_name_map.get(cid, f"Player {cid}"),
            "player_id": cid,
            "why_similar": f"Similar in: {reason}"
        })
    return pd.DataFrame(rows)


st.title("Elite Soccer Analytics — Recruitment & Tactical Intelligence")

page = st.sidebar.radio("Page", ["Recruitment Dashboard", "Premier League Hub"])

# Load data
player_features = load_csv(DATA_DIR / "player_features_rolling12mo.csv")
players = load_csv(DATA_DIR / "players.csv")
teams = load_csv(DATA_DIR / "teams.csv")
embeddings = load_embeddings(MODELS_DIR / "player_embeddings.parquet")
shortlist = load_csv(OUTPUTS_DIR / "undervalued_shortlist.csv")
events = load_csv(DATA_DIR / "events.csv")

if player_features.empty or players.empty or events.empty:
    st.warning("Some data files are missing or empty. Make sure you ran the pipeline before using the dashboard.")

player_name_map = dict(zip(players.get("player_id", []), players.get("player_name", [])))

if page == "Recruitment Dashboard":
    # KPIs
    cols = st.columns(3)
    for idx, (k, v) in enumerate(kpis(player_features).items()):
        cols[idx].metric(k, v)

    st.divider()

    # Player search
    player_ids = player_features["player_id"].unique().tolist() if not player_features.empty else []
    selected_player = st.selectbox(
        "Select Player",
        player_ids,
        format_func=lambda x: player_name_map.get(x, f"Player {x}"),
    )

    if selected_player:
        player_row = player_features[player_features["player_id"] == selected_player].iloc[0]
        role_cluster = int(player_row.get("role_cluster", -1))

        role_avg = player_features[player_features["role_cluster"] == role_cluster].mean(numeric_only=True) if not player_features.empty else pd.Series()

        st.subheader(f"Player Profile: {player_name_map.get(selected_player, 'Unknown')} (ID {selected_player})")
        tab_overview, tab_attack, tab_defend, tab_pass = st.tabs(["Overview", "Attacking", "Defending", "Passing"])

        with tab_overview:
            st.plotly_chart(radar_chart(player_row, role_avg), use_container_width=True)
            st.caption("Radar: player vs the average of their role. Further out means more of that skill per full match.")

            with st.expander("What do these stats mean? (simple explanations)"):
                st.markdown(
                    """
                    **Per 90** means per full match (90 minutes), so players are comparable even if they played different minutes.

                    **Non-penalty xG (per 90):** Quality of chances a player gets, excluding penalties.
                    **Expected Assists (per 90):** Quality of chances a player creates for others.
                    **Progressive Passes (per 90):** Passes that move the ball meaningfully toward goal.
                    **Progressive Carries (per 90):** Dribbles that move the ball meaningfully toward goal.
                    **Pressures (per 90):** Times the player closes down an opponent in possession.
                    **Tackles (per 90):** Times the player wins the ball from an opponent.
                    **Interceptions (per 90):** Times the player cuts out a pass.
                    **Passes into Final Third (per 90):** Passes that enter the attacking third.
                    """
                )

            if not player_features.empty:
                scatter = player_features.copy()
                scatter["player_name"] = scatter["player_id"].map(player_name_map)
                scatter["Chance Creation (per match)"] = scatter["per90_xa"]
                scatter["Chance Quality (per match)"] = scatter["per90_npxg"]
                fig = px.scatter(
                    scatter,
                    x="Chance Creation (per match)",
                    y="Chance Quality (per match)",
                    hover_name="player_name",
                    title="Chance Creation vs Chance Quality",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Right = creates more chances for teammates. Up = gets better scoring chances.")

            if not player_features.empty and "role_cluster" in player_features.columns:
                cluster_counts = player_features["role_cluster"].value_counts().reset_index()
                cluster_counts.columns = ["role_cluster", "count"]
                fig = px.bar(cluster_counts, x="role_cluster", y="count", title="Player Style Groups (how many players per role type)")
                fig.update_xaxes(title_text="Role type (learned from playing style)")
                fig.update_yaxes(title_text="Number of players")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Groups are learned from player actions, not fixed positions.")

        with tab_attack:
            player_shots = events[(events["event_type"] == "Shot") & (events["player_id"] == selected_player)]
            if not player_shots.empty:
                pitch = Pitch(pitch_type="statsbomb", line_color="white")
                fig, ax = pitch.draw(figsize=(6, 4))
                pitch.scatter(player_shots["x"], player_shots["y"], ax=ax, c=player_shots["shot_statsbomb_xg"], cmap="Reds", s=30)
                ax.set_title("Shot Map (darker = higher chance of scoring)")
                st.pyplot(fig)
                st.caption("Each dot is a shot. Darker dots mean better chances.")

                xg_flow = player_shots.groupby("minute")["shot_statsbomb_xg"].sum().reset_index()
                fig = px.line(xg_flow, x="minute", y="shot_statsbomb_xg", title="Chance Quality Over the Match")
                fig.update_xaxes(title_text="Match minute")
                fig.update_yaxes(title_text="Total chance quality (xG)")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Shows when the player got their best chances.")

        with tab_defend:
            defend_metrics = ["per90_pressures", "per90_tackles", "per90_interceptions"]
            if all(m in player_row.index for m in defend_metrics):
                vals = [player_row[m] for m in defend_metrics]
                labels = ["Pressures", "Tackles", "Interceptions"]
                fig = px.bar(x=labels, y=vals, title="Defensive Actions (per full match)")
                fig.update_xaxes(title_text="Defensive action")
                fig.update_yaxes(title_text="Times per full match")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("How often the player wins the ball or disrupts opponents.")

            heatmap_fig = player_touch_heatmap(events, selected_player)
            if heatmap_fig is not None:
                st.pyplot(heatmap_fig)
                st.caption("Where the player touches the ball most often.")

        with tab_pass:
            pass_metrics = ["per90_progressive_passes", "per90_progressive_carries", "per90_passes_final_third"]
            if all(m in player_row.index for m in pass_metrics):
                vals = [player_row[m] for m in pass_metrics]
                labels = ["Progressive Passes", "Progressive Carries", "Passes into Final Third"]
                fig = px.bar(x=labels, y=vals, title="Ball Progression (per full match)")
                fig.update_xaxes(title_text="How the player moves the ball forward")
                fig.update_yaxes(title_text="Times per full match")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Measures how often the player moves the ball toward goal.")

            if not player_features.empty:
                prog = player_features.copy()
                prog["player_name"] = prog["player_id"].map(player_name_map)
                prog["Forward Passes (per match)"] = prog["per90_progressive_passes"]
                prog["Forward Dribbles (per match)"] = prog["per90_progressive_carries"]
                fig = px.scatter(
                    prog,
                    x="Forward Passes (per match)",
                    y="Forward Dribbles (per match)",
                    hover_name="player_name",
                    title="Passing vs Dribbling to Move Forward",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Right = more forward passes. Up = more forward dribbles.")

        # UMAP embedding
        if not embeddings.empty:
            fig = px.scatter(embeddings, x="emb_0", y="emb_1", color="role_cluster", title="Player Similarity Map (closer = more similar)")
            fig.update_xaxes(title_text="Playing style dimension 1")
            fig.update_yaxes(title_text="Playing style dimension 2")
            highlight = embeddings[embeddings["player_id"] == selected_player]
            if not highlight.empty:
                fig.add_scatter(x=highlight["emb_0"], y=highlight["emb_1"], mode="markers", marker=dict(size=14, color="black"), name="Selected")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Players closer together play more similarly.")

        # Similar players
        if not embeddings.empty:
            emb = embeddings.set_index("player_id")
            if selected_player in emb.index:
                v = emb.loc[selected_player][[c for c in emb.columns if c.startswith("emb_")]].values.astype(float)
                mat = emb[[c for c in emb.columns if c.startswith("emb_")]].values.astype(float)
                sims = (mat @ v) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(v) + 1e-9)
                emb["similarity"] = sims
                top = emb.sort_values("similarity", ascending=False).head(6).reset_index()
                top["player_name"] = top["player_id"].map(player_name_map)
                st.subheader("Most Similar Players")
                reasons = similarity_reasons(
                    player_features,
                    selected_player,
                    top["player_id"].tolist()[1:],
                    player_name_map,
                )
                display = top.merge(reasons, on=["player_id", "player_name"], how="left")
                st.dataframe(display[["player_name", "similarity", "why_similar"]])
        
        # New visual: Action mix + shot quality distribution
        action_fig = player_action_mix(events, selected_player)
        if action_fig is not None:
            st.plotly_chart(action_fig, use_container_width=True)
            st.caption("Simple breakdown of what the player does most on the ball.")

        if "shot_statsbomb_xg" in events.columns:
            player_shots = events[(events["event_type"] == "Shot") & (events["player_id"] == selected_player)]
            if not player_shots.empty:
                fig = px.histogram(player_shots, x="shot_statsbomb_xg", nbins=10, title="Shot Quality Distribution")
                fig.update_xaxes(title_text="Chance quality of each shot")
                fig.update_yaxes(title_text="Number of shots")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Shows whether shots are mostly low‑quality or high‑quality.")

        # PDF report
        pdf_bytes = generate_pdf_report(selected_player, player_row)
        st.download_button("Download Player PDF", data=pdf_bytes, file_name=f"player_{selected_player}_report.pdf")

    st.divider()

    # Undervalued shortlist download
    if not shortlist.empty:
        st.subheader("Undervalued Player Shortlist")
        st.dataframe(shortlist.head(20))
        st.download_button("Download Shortlist CSV", data=shortlist.to_csv(index=False), file_name="undervalued_shortlist.csv")

else:
    st.subheader("Premier League Style Stats Hub")

    if events.empty:
        st.info("No event data found. Run the pipeline to populate events.")
    else:
        shots = events[events["event_type"] == "Shot"].copy()
        passes = events[events["event_type"] == "Pass"].copy()

        # KPIs
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Matches", int(events["match_id"].nunique()))
        kpi_cols[1].metric("Players", int(players["player_id"].nunique()) if not players.empty else 0)
        goals = int((shots.get("shot_outcome") == "Goal").sum()) if "shot_outcome" in shots.columns else 0
        kpi_cols[2].metric("Goals", goals)
        total_xg = float(shots.get("shot_statsbomb_xg", pd.Series([0])).sum())
        kpi_cols[3].metric("Total xG", round(total_xg, 2))
        avg_xg = float(shots.get("shot_statsbomb_xg", pd.Series([0])).mean())
        kpi_cols[4].metric("Avg xG/Shot", round(avg_xg, 3))

        # Enrich with names
        shots = shots.merge(players, on="player_id", how="left")
        if not teams.empty:
            shots = shots.merge(teams, on="team_id", how="left")

        # Top scorers
        if "shot_outcome" in shots.columns:
            top_scorers = (
                shots[shots["shot_outcome"] == "Goal"]
                .groupby("player_name")
                .size()
                .reset_index(name="goals")
                .sort_values("goals", ascending=False)
                .head(10)
            )
            st.plotly_chart(px.bar(top_scorers, x="player_name", y="goals", title="Top Scorers"), use_container_width=True)

        # Top xG
        top_xg = (
            shots.groupby("player_name")["shot_statsbomb_xg"]
            .sum()
            .reset_index()
            .sort_values("shot_statsbomb_xg", ascending=False)
            .head(10)
        )
        st.plotly_chart(px.bar(top_xg, x="player_name", y="shot_statsbomb_xg", title="Top xG"), use_container_width=True)

        # Team xG
        if "team_name" in shots.columns:
            team_xg = (
                shots.groupby("team_name")["shot_statsbomb_xg"]
                .sum()
                .reset_index()
                .sort_values("shot_statsbomb_xg", ascending=False)
                .head(12)
            )
            st.plotly_chart(px.bar(team_xg, x="team_name", y="shot_statsbomb_xg", title="Team xG"), use_container_width=True)

        # Player leaderboard table
        leaderboard = (
            shots.groupby("player_name")
            .agg(goals=("shot_outcome", lambda x: (x == "Goal").sum()), shots=("event_id", "count"), xg=("shot_statsbomb_xg", "sum"))
            .reset_index()
            .sort_values(["goals", "xg"], ascending=False)
            .head(20)
        )
        st.dataframe(leaderboard)
