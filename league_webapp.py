import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_consistent_team_name(name):
    name = name.strip()
    if name == "Lambchester":
        return "Lambchester fc"
    if name == "VilaVelebita":
        return "Vila Velebita"
    return name

def parse_match_data_final_fix(raw_data):
    """Final fixed version - handles data without double newlines"""
    matches = []
    teams_set = set()
    
    # Correct the known team name issue
    raw_data_corrected = raw_data.replace("Lambchester 26 - 51 Nes opet", "Lambchester fc 26 - 51 Nes opet")
    
    lines = raw_data_corrected.strip().split('\n')
    current_round = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a round header
        round_match = re.match(r"(\d{1,2})\s*kolo", line)
        if round_match:
            current_round = int(round_match.group(1))
            print(f"DEBUG: Found round {current_round}")
            continue
        
        # Skip non-match lines
        if "Konacna tablica" in line or "Head to head" in line:
            continue
        
        # Try to parse as a match
        match_pattern = re.compile(r"^(.*?)\s+(\d+)\s*-\s*(\d+)\s+(.*)$")
        match_result = match_pattern.match(line)
        
        if match_result and current_round is not None:
            team1_name = get_consistent_team_name(match_result.group(1))
            score1 = int(match_result.group(2))
            score2 = int(match_result.group(3))
            team2_name = get_consistent_team_name(match_result.group(4))

            matches.append({
                'round': current_round,
                'team1': team1_name,
                'score1': score1,
                'team2': team2_name,
                'score2': score2
            })
            teams_set.add(team1_name)
            teams_set.add(team2_name)
    
    print(f"DEBUG: Total matches parsed: {len(matches)}")
    rounds_found = set(m['round'] for m in matches)
    print(f"DEBUG: Rounds found: {sorted(rounds_found)}")
    
    return matches, sorted(list(teams_set))

def calculate_standings_and_progressions(matches, team_names):
    num_rounds = 0
    if matches:
        num_rounds = max(m['round'] for m in matches) 
    else:
        return pd.DataFrame(), {}, {}

    standings = {
        team: {'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0, 'MP': 0}
        for team in team_names
    }
    
    # Track cumulative points and goals through all rounds
    points_progression = {team: [] for team in team_names}
    gf_progression = {team: [] for team in team_names}
    
    # Initialize current totals
    current_pts = {team: 0 for team in team_names}
    current_gf = {team: 0 for team in team_names}

    for r_idx in range(1, num_rounds + 1):
        round_matches = [m for m in matches if m['round'] == r_idx]
        
        for match in round_matches:
            t1, s1, t2, s2 = match['team1'], match['score1'], match['team2'], match['score2']

            standings[t1]['GF'] += s1; standings[t1]['GA'] += s2; standings[t1]['MP'] += 1
            standings[t2]['GF'] += s2; standings[t2]['GA'] += s1; standings[t2]['MP'] += 1
            
            current_gf[t1] += s1
            current_gf[t2] += s2
            
            if s1 > s2: 
                standings[t1]['W'] += 1; standings[t1]['Pts'] += 3
                standings[t2]['L'] += 1
                current_pts[t1] += 3
            elif s2 > s1: 
                standings[t2]['W'] += 1; standings[t2]['Pts'] += 3
                standings[t1]['L'] += 1
                current_pts[t2] += 3
            else: 
                standings[t1]['D'] += 1; standings[t1]['Pts'] += 1
                standings[t2]['D'] += 1; standings[t2]['Pts'] += 1
                current_pts[t1] += 1
                current_pts[t2] += 1
        
        # Record progression for this round
        for team in team_names:
            points_progression[team].append(current_pts[team])
            gf_progression[team].append(current_gf[team])
            
    df_standings = pd.DataFrame.from_dict(standings, orient='index')
    if not df_standings.empty:
        df_standings = df_standings[['MP', 'W', 'D', 'L', 'GF', 'GA', 'Pts']]
    
    return df_standings, points_progression, gf_progression


def sort_final_table(df_standings, all_matches):
    if df_standings.empty:
        return df_standings

    def get_h2h_gf_sum(team_name, tied_group_teams, matches_list):
        if len(tied_group_teams) <= 1:
            return 0 
        
        h2h_sum = 0
        relevant_opponents = [opp for opp in tied_group_teams if opp != team_name]

        for match in matches_list:
            t1, s1, t2, s2 = match['team1'], match['score1'], match['team2'], match['score2']
            if t1 == team_name and t2 in relevant_opponents:
                h2h_sum += s1
            elif t2 == team_name and t1 in relevant_opponents:
                h2h_sum += s2
        return h2h_sum
    
    df_sorted = df_standings.sort_values(by=['Pts', 'GF'], ascending=[False, False])
    cached_h2h_scores = {}

    def get_sort_key(team_name_series): 
        team_name = team_name_series.name 
        pts = df_sorted.loc[team_name, 'Pts']
        overall_gf = df_sorted.loc[team_name, 'GF']
        
        tied_group_df = df_sorted[df_sorted['Pts'] == pts]
        h2h_score_for_sort = 0

        if len(tied_group_df) > 1:
            tied_group_teams = tuple(sorted(tied_group_df.index.tolist()))
            
            if tied_group_teams not in cached_h2h_scores:
                group_h2h_scores = {}
                for t_in_group in tied_group_teams:
                    group_h2h_scores[t_in_group] = get_h2h_gf_sum(t_in_group, list(tied_group_teams), all_matches)
                cached_h2h_scores[tied_group_teams] = group_h2h_scores
            
            h2h_score_for_sort = cached_h2h_scores[tied_group_teams].get(team_name, 0)
            
        return (pts, h2h_score_for_sort, overall_gf)

    df_sorted['sort_key_tuple'] = df_sorted.apply(get_sort_key, axis=1)
    df_final_sorted = df_sorted.sort_values(by='sort_key_tuple', ascending=False).drop(columns=['sort_key_tuple'])
        
    return df_final_sorted

def calculate_team_positions_progression(matches, team_names):
    """Calculate team positions progression throughout all rounds"""
    positions_progression = {team: [] for team in team_names}
    num_rounds = max(m['round'] for m in matches) if matches else 0

    # Initialize standings
    standings = {team: {'Pts': 0, 'GF': 0, 'GA': 0} for team in team_names}

    for r_idx in range(1, num_rounds + 1):
        round_matches = [m for m in matches if m['round'] == r_idx]
        
        # Update standings based on current round matches
        for match in round_matches:
            t1, s1, t2, s2 = match['team1'], match['score1'], match['team2'], match['score2']
            standings[t1]['GF'] += s1
            standings[t1]['GA'] += s2
            standings[t2]['GF'] += s2
            standings[t2]['GA'] += s1
            
            if s1 > s2:
                standings[t1]['Pts'] += 3
            elif s2 > s1:
                standings[t2]['Pts'] += 3
            else:
                standings[t1]['Pts'] += 1
                standings[t2]['Pts'] += 1
        
        # Sort teams by points and goal difference
        sorted_teams = sorted(standings.items(), 
                            key=lambda item: (item[1]['Pts'], item[1]['GF'] - item[1]['GA']), 
                            reverse=True)
        
        # Record positions for each team
        for position, (team_name, _) in enumerate(sorted_teams, start=1):
            positions_progression[team_name].append(position)

    return positions_progression
# --- Raw Data Input ---
raw_data_input = """
1 kolo
FC Fantazia 45 - 61 Arsenal osvaja
Lambchester fc 37 - 39 Come On Jolene
Vila Velebita 42 - 31 Debeli
MuscleTeam 34 - 33 Nes opet
2 kolo
Arsenal osvaja 48 - 62 MuscleTeam
Nes opet 54 - 23 Vila Velebita
Debeli 29 - 46 Lambchester fc
Come On Jolene 65 - 65 FC Fantazia
3 kolo 
Lambchester fc 38 - 42 Arsenal osvaja
Vila Velebita 28 - 31 FC Fantazia
MuscleTeam 35 - 61 Come On Jolene
Nes opet 35 - 39 Debeli
4 kolo
Arsenal osvaja 28 - 22 Nes opet
Debeli 55 - 52 MuscleTeam
Come On Jolene 39 - 46 Vila Velebita
FC Fantazia 52 - 23 Lambchester fc
5 kolo
Vila Velebita 38 - 24 Arsenal osvaja
MuscleTeam 36 - 40 Lambchester fc
Nes opet 63 - 37 FC Fantazia
Debeli 63 - 46 Come On Jolene
6 kolo
Arsenal osvaja 55 - 25 Debeli
Come On Jolene 33 - 24 Nes opet
FC Fantazia 20 - 65 MuscleTeam
Lambchester fc 41 - 45 Vila Velebita
7 kolo
Come On Jolene 38 - 35 Arsenal osvaja
FC Fantazia 51 - 57 Debeli
Lambchester fc 29 - 36 Nes opet
Vila Velebita 31 - 40 MuscleTeam
8 kolo
FC Fantazia 43 - 36 Arsenal osvaja
Lambchester fc 31 - 23 Come On Jolene
Vila Velebita 38 - 47 Debeli
MuscleTeam 30 - 38 Nes opet
9 kolo
Arsenal osvaja 42 - 34 MuscleTeam
Nes opet 32 - 28 Vila Velebita
Debeli 57 - 36 Lambchester fc
Come On Jolene 33 - 50 FC Fantazia
10 kolo
Lambchester fc 37 - 35 Arsenal osvaja
Vila Velebita 44 - 29 FC Fantazia
MuscleTeam 34 - 41 Come On Jolene
Nes opet 49 - 29 Debeli
11 kolo
Arsenal osvaja 32 - 30 Nes opet
Debeli 69 - 51 MuscleTeam
Come On Jolene 36 - 31 Vila Velebita
FC Fantazia 46 - 29 Lambchester fc
12 kolo
Vila Velebita 34 - 31 Arsenal osvaja
MuscleTeam 28 - 29 Lambchester fc
Nes opet 60 - 78 FC Fantazia
Debeli 50 - 28 Come On Jolene
13 kolo
Arsenal osvaja 61 - 48 Debeli
Come On Jolene 40 - 41 Nes opet
FC Fantazia 67 - 33 MuscleTeam
Lambchester fc 26 - 64 Vila Velebita
14 kolo
Come On Jolene 43 - 36 Arsenal osvaja
FC Fantazia 48 - 48 Debeli
Lambchester fc 29 - 34 Nes opet
Vila Velebita 60 - 29 MuscleTeam
15 kolo
FC Fantazia 19 - 17 Arsenal osvaja
Lambchester fc 35 - 36 Come On Jolene
Vila Velebita 28 - 53 Debeli
MuscleTeam 46 - 41 Nes opet
16 kolo
Arsenal osvaja 37 - 42 MuscleTeam
Nes opet 54 - 63 Vila Velebita
Debeli 45 - 35 Lambchester fc
Come On Jolene 22 - 35 FC Fantazia
17 kolo
Lambchester fc 62 - 72 Arsenal osvaja
Vila Velebita 72 - 37 FC Fantazia
MuscleTeam 28 - 35 Come On Jolene
Nes opet 50 - 27 Debeli
18 kolo
Arsenal osvaja 39 - 38 Nes opet
Debeli 31 - 32 MuscleTeam
Come On Jolene 50 - 59 Vila Velebita
FC Fantazia 40 - 31 Lambchester fc
19 kolo
Vila Velebita 45 - 39 Arsenal osvaja
MuscleTeam 36 - 49 Lambchester fc
Nes opet 51 - 44 FC Fantazia
Debeli 55 - 54 Come On Jolene
20 kolo
Arsenal osvaja 44 - 70 Debeli
Come On Jolene 46 - 40 Nes opet
FC Fantazia 32 - 49 MuscleTeam
Lambchester fc 42 - 42 Vila Velebita
21 kolo
Come On Jolene 47 - 36 Arsenal osvaja
FC Fantazia 36 - 38 Debeli
Lambchester fc 32 - 63 Nes opet
Vila Velebita 59 - 55 MuscleTeam
22 kolo
FC Fantazia 42 - 64 Arsenal osvaja
Lambchester fc 31 - 34 Come On Jolene
Vila Velebita 52 - 34 Debeli
MuscleTeam 78 - 34 Nes opet
23 kolo
Arsenal osvaja 54 - 29 MuscleTeam
Nes opet 37 - 80 Vila Velebita
Debeli 51 - 38 Lambchester fc
Come On Jolene 44 - 36 FC Fantazia
24 kolo
Lambchester fc 41 - 68 Arsenal osvaja
Vila Velebita 48 - 31 FC Fantazia
MuscleTeam 60 - 66 Come On Jolene
Nes opet 62 - 54 Debeli
25 kolo
Arsenal osvaja 55 - 37 Nes opet
Debeli 42 - 40 MuscleTeam
Come On Jolene 27 - 44 Vila Velebita
FC Fantazia 43 - 67 Lambchester fc
26 kolo
Vila Velebita 61 - 49 Arsenal osvaja
MuscleTeam 36 - 23 Lambchester fc
Nes opet 31 - 46 FC Fantazia
Debeli 61 - 48 Come On Jolene
27 kolo
Arsenal osvaja 67 - 51 Debeli
Come On Jolene 38 - 39 Nes opet
FC Fantazia 30 - 51 MuscleTeam
Lambchester fc 37 - 34 Vila Velebita
28 kolo
Come On Jolene 22 - 54 Arsenal osvaja
FC Fantazia 35 - 33 Debeli
Lambchester fc 32 - 53 Nes opet
Vila Velebita 41 - 50 MuscleTeam
29 kolo
FC Fantazia 25 - 22 Arsenal osvaja
Lambchester fc 32 - 62 Come On Jolene
Vila Velebita 16 - 62 Debeli
MuscleTeam 38 - 31 Nes opet
30 kolo
Arsenal osvaja 46 - 38 MuscleTeam
Nes opet 28 - 34 Vila Velebita
Debeli 48 - 41 Lambchester fc
Come On Jolene 51 - 41 FC Fantazia
31 kolo
Lambchester fc 40 - 35 Arsenal osvaja
Vila Velebita 55 - 35 FC Fantazia
MuscleTeam 21 - 43 Come On Jolene
Nes opet 49 - 40 Debeli
32 kolo
Arsenal osvaja 32 - 29 Nes opet
Debeli 34 - 30 MuscleTeam
Come On Jolene 27 - 60 Vila Velebita
FC Fantazia 77 - 92 Lambchester fc
33 kolo
Vila Velebita 54 - 53 Arsenal osvaja
MuscleTeam 26 - 85 Lambchester fc
Nes opet 59 - 37 FC Fantazia
Debeli 56 - 39 Come On Jolene
34 kolo
Arsenal osvaja 42 - 38 Debeli
Come On Jolene 42 - 31 Nes opet
FC Fantazia 69 - 42 MuscleTeam
Lambchester fc 57 - 43 Vila Velebita
35 kolo
Come On Jolene 31 - 23 Arsenal osvaja
FC Fantazia 35 - 39 Debeli
Lambchester fc 26 - 51 Nes opet 
Vila Velebita 31 - 60 MuscleTeam
36 kolo
FC Fantazia 30 - 40 Arsenal osvaja
Lambchester fc 36 - 51 Come On Jolene
Vila Velebita 55 - 25 Debeli
MuscleTeam 53 - 62 Nes opet
37 kolo
Arsenal osvaja 32 - 63 MuscleTeam
Nes opet 52 - 52 Vila Velebita
Debeli 53 - 32 Lambchester fc
Come On Jolene 53 - 19 FC Fantazia
38 kolo
Lambchester fc 38 - 22 Arsenal osvaja
Vila Velebita 27 - 40 FC Fantazia
MuscleTeam 34 - 42 Come On Jolene
Nes opet 33 - 26 Debeli
"""

# --- Streamlit App ---
st.set_page_config(layout="wide") 
st.title("Analiza Lige i Vizualizacija")


if raw_data_input:
    all_matches, team_names_list = parse_match_data_final_fix(raw_data_input)
    print("=== DEBUGGING MATCHES DATA ===")
    print(f"Debug: all_matches length = {len(all_matches)}")
    if all_matches:
        print(f"Debug: max round = {max(m['round'] for m in all_matches)}")
        rounds_found = set(m['round'] for m in all_matches)
        print(f"Debug: rounds found = {sorted(rounds_found)}")
        print(f"Debug: first match = {all_matches[0]}")
        print(f"Debug: last match = {all_matches[-1]}")
    else:
        print("Debug: No matches found!")
        print(f"=== END DEBUG ===\n")
    if not all_matches:
        st.error("Nema podataka o utakmicama za obradu. Provjerite unos.")
    else:
        df_standings_raw, pts_prog, gf_prog = calculate_standings_and_progressions(all_matches, team_names_list)
        
        if df_standings_raw.empty:
            st.error("Nije moguće izračunati tablicu. Provjerite podatke o utakmicama.")
        else:
            final_table = sort_final_table(df_standings_raw, all_matches)
            num_rounds_actual = max(m['round'] for m in all_matches)
            rounds_axis = np.arange(1, num_rounds_actual + 1)
            
            # Calculate positions progression
            positions_prog = calculate_team_positions_progression(all_matches, team_names_list)

            st.header("Konačna Tablica Lige")
            st.dataframe(final_table[['MP', 'W', 'D', 'L', 'GF', 'GA', 'Pts']].style.format({"GF": "{:.0f}", "GA": "{:.0f}", "Pts": "{:.0f}"}))

            # --- Plotting ---
            sns.set_theme(style="whitegrid", palette="muted") 
            
            plot_figsize = (16, 8)
            title_fontsize = 15
            label_fontsize = 11
            tick_fontsize = 9
            legend_fontsize = 9
            line_thickness = 2

            st.header("Grafički Prikazi")

            # 1. Final League Standings (Points)
            st.subheader("1. Konačni Poredak (Bodovi)")
            fig1, ax1 = plt.subplots(figsize=plot_figsize)
            bar_plot = sns.barplot(x=final_table.index, y='Pts', data=final_table, palette="viridis", ax=ax1, hue=final_table.index, dodge=False, legend=False)
            ax1.set_title('Konačni Poredak Lige (Bodovi)', fontsize=title_fontsize)
            ax1.set_xlabel('Ekipa', fontsize=label_fontsize)
            ax1.set_ylabel('Ukupno Bodova', fontsize=label_fontsize)
            ax1.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
            ax1.tick_params(axis='y', labelsize=tick_fontsize)
            for p in bar_plot.patches:
                bar_plot.annotate(format(p.get_height(), '.0f'), 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 8), 
                               textcoords = 'offset points', fontsize=tick_fontsize-1)
            plt.tight_layout()
            st.pyplot(fig1)

            # 2. Position Progression Line Chart
            st.subheader("2. Kretanje Pozicija u Tablici Kroz Sezonu")

            # Calculate positions progression
            positions_prog = calculate_team_positions_progression(all_matches, team_names_list)

            # Check if we have position data
            if not positions_prog or not any(len(positions_prog[team]) > 0 for team in team_names_list):
                st.error("Nema podataka o pozicijama za grafički prikaz")
            else:
                # Check data integrity
                valid_teams = []
                for team in team_names_list:
                    if team in positions_prog and len(positions_prog[team]) > 0:
                        valid_teams.append(team)
                        st.write(f"{team}: {len(positions_prog[team])} rounds of data")
                
                if not valid_teams:
                    st.error("Nema validnih podataka o pozicijama za bilo koju ekipu")
                else:
                    fig2, ax2 = plt.subplots(figsize=(16, 8))
                    
                    # Create distinct colors for each team
                    colors = sns.color_palette("husl", len(valid_teams))
                    
                    # Plot lines for each team
                    for idx, team in enumerate(sorted(valid_teams)):
                        positions = positions_prog[team]
                        if len(positions) > 0:
                            rounds = list(range(1, len(positions) + 1))
                            
                            ax2.plot(rounds, positions, 
                                    linewidth=2, 
                                    linestyle='-', 
                                    label=team,
                                    color=colors[idx],
                                    marker='o',
                                    markersize=3,
                                    alpha=0.8)
                    
                    # Chart formatting
                    ax2.set_title('Kretanje Pozicija u Tablici Kroz Sezonu', fontsize=16, pad=20)
                    ax2.set_xlabel('Kolo', fontsize=14)
                    ax2.set_ylabel('Pozicija', fontsize=14)
                    
                    # Set appropriate axis limits
                    max_rounds = max(len(positions_prog[team]) for team in valid_teams)
                    ax2.set_xlim(1, max_rounds)
                    ax2.set_ylim(0.5, len(team_names_list) + 0.5)
                    
                    # Grid and ticks
                    ax2.grid(True, alpha=0.3)
                    # Better x-axis tick spacing - show every 5th round plus round 1 and final round
                    if max_rounds <= 20:
                        # For shorter seasons, show every 2 rounds
                        tick_interval = 2
                    else:
                        # For longer seasons, show every 5 rounds
                        tick_interval = 5
                        
                        x_ticks = [1]  # Always include round 1
                    x_ticks.extend(range(tick_interval, max_rounds, tick_interval))  # Every 5th (or 2nd) round
                    if max_rounds not in x_ticks:  # Always include the final round
                        x_ticks.append(max_rounds)

                    ax2.set_xticks(sorted(x_ticks))
                    ax2.set_yticks(range(1, len(team_names_list) + 1))
                    # Invert y-axis so position 1 is at the top
                    ax2.invert_yaxis()
                    
                    # Legend
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)

            # 3. Win/Draw/Loss Pie Charts
            st.subheader("3. Omjer Pobjeda/Neriješeno/Poraza po Ekipama")
            num_teams = len(final_table.index)
            num_cols = min(3, num_teams)
            
            cols = st.columns(num_cols) 
            col_idx = 0
            for team_name_idx in final_table.index:
                with cols[col_idx % num_cols]:
                    team_stats = final_table.loc[team_name_idx]
                    wdl_data = [team_stats['W'], team_stats['D'], team_stats['L']]
                    wdl_labels = ['Pobjede', 'Neriješeno', 'Porazi']
                    
                    non_zero_data = [(val, lab) for val, lab in zip(wdl_data, wdl_labels) if val > 0]
                    if not non_zero_data: 
                        st.write(f"{team_name_idx}: Nema podataka W/D/L.")
                        col_idx += 1
                        continue

                    plot_data = [item[0] for item in non_zero_data]
                    plot_labels = [item[1] for item in non_zero_data]

                    fig_pie, ax_pie = plt.subplots(figsize=(3.5, 2.8))
                    ax_pie.pie(plot_data, labels=plot_labels, autopct='%1.1f%%', startangle=90, pctdistance=0.80, 
                               colors=sns.color_palette("Pastel2", len(plot_data)))
                    ax_pie.set_title(f"{team_name_idx}", fontsize=label_fontsize+1)
                    ax_pie.axis('equal')
                    plt.tight_layout(pad=0.1)
                    st.pyplot(fig_pie)
                col_idx += 1

            # 4. Average Goals For and Against Per Game
            st.subheader("4. Prosjek Golova Za i Protiv Po Utakmici")
            if 'MP' in final_table.columns and final_table['MP'].sum() > 0: 
                final_table['Avg_GF_Per_Game'] = final_table.apply(lambda row: row['GF'] / row['MP'] if row['MP'] > 0 else 0, axis=1)
                final_table['Avg_GA_Per_Game'] = final_table.apply(lambda row: row['GA'] / row['MP'] if row['MP'] > 0 else 0, axis=1)

                fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Goals For per game
                df_avg_gf = final_table.sort_values(by='Avg_GF_Per_Game', ascending=False)
                sns.barplot(x=df_avg_gf.index, y='Avg_GF_Per_Game', data=df_avg_gf, palette="Greens_r", ax=ax6a, hue=df_avg_gf.index, dodge=False, legend=False)
                ax6a.set_title('Prosjek Postignutih Bodova Po Utakmici', fontsize=title_fontsize)
                ax6a.set_xlabel('Ekipa', fontsize=label_fontsize)
                ax6a.set_ylabel('Prosjek Golova Za', fontsize=label_fontsize)
                ax6a.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
                ax6a.tick_params(axis='y', labelsize=tick_fontsize)
                for p in ax6a.patches:
                    ax6a.annotate(format(p.get_height(), '.1f'), 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 8), 
                               textcoords = 'offset points', fontsize=tick_fontsize-1)
                
                # Goals Against per game
                df_avg_ga = final_table.sort_values(by='Avg_GA_Per_Game', ascending=True)
                sns.barplot(x=df_avg_ga.index, y='Avg_GA_Per_Game', data=df_avg_ga, palette="Reds", ax=ax6b, hue=df_avg_ga.index, dodge=False, legend=False)
                ax6b.set_title('Prosjek Protivnickih Bodova Po Utakmici', fontsize=title_fontsize)
                ax6b.set_xlabel('Ekipa', fontsize=label_fontsize)
                ax6b.set_ylabel('Prosjek Golova Protiv', fontsize=label_fontsize)
                ax6b.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
                ax6b.tick_params(axis='y', labelsize=tick_fontsize)
                for p in ax6b.patches:
                    ax6b.annotate(format(p.get_height(), '.1f'), 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha = 'center', va = 'center', 
                               xytext = (0, 8), 
                               textcoords = 'offset points', fontsize=tick_fontsize-1)
                
                plt.tight_layout()
                st.pyplot(fig6)
            else:
                st.write("Nije moguće izračunati prosjek golova - nedostaju podaci o odigranim utakmicama (MP).")

            # 5. Goal Difference Analysis
            st.subheader("5. Analiza Gol-Razlike")
            if not final_table.empty:
                final_table['GD'] = final_table['GF'] - final_table['GA']
                df_gd = final_table.sort_values(by='GD', ascending=False)

                fig7, ax7 = plt.subplots(figsize=plot_figsize)
                colors = ['green' if x >= 0 else 'red' for x in df_gd['GD']]
                bars = ax7.bar(df_gd.index, df_gd['GD'], color=colors, alpha=0.7)
                ax7.set_title('Gol-Razlika Po Ekipama', fontsize=title_fontsize)
                ax7.set_xlabel('Ekipa', fontsize=label_fontsize)
                ax7.set_ylabel('Gol-Razlika (GF - GA)', fontsize=label_fontsize)
                ax7.tick_params(axis='x', rotation=45, labelsize=tick_fontsize)
                ax7.tick_params(axis='y', labelsize=tick_fontsize)
                ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                
                for bar, value in zip(bars, df_gd['GD']):
                    height = bar.get_height()
                    ax7.annotate(f'{value:+.0f}', 
                               (bar.get_x() + bar.get_width() / 2., height), 
                               ha='center', va='bottom' if height >= 0 else 'top', 
                               xytext=(0, 5 if height >= 0 else -5), 
                               textcoords='offset points', fontsize=tick_fontsize-1)
                
                plt.tight_layout()
                st.pyplot(fig7)

            # 6. Head-to-Head Win Matrix (FIXED)
            st.subheader("6. Matrica Pobjeda (Head-to-Head)")
            
            # Create head-to-head matrix
            h2h_numeric = pd.DataFrame(0.0, index=team_names_list, columns=team_names_list)
            
            for match in all_matches:
                t1, s1, t2, s2 = match['team1'], match['score1'], match['team2'], match['score2']
                if t1 in team_names_list and t2 in team_names_list:
                    if s1 > s2:  # t1 wins
                        h2h_numeric.loc[t1, t2] += 1
                    elif s2 > s1:  # t2 wins
                        h2h_numeric.loc[t2, t1] += 1
                    # Draw case: no wins added
            
            fig8, ax8 = plt.subplots(figsize=(10, 8))
            # FIXED: Changed fmt='d' to fmt='.0f' to handle float values
            sns.heatmap(h2h_numeric, annot=True, cmap='RdYlGn', center=1,
                       fmt='.0f', ax=ax8, cbar_kws={'label': 'Broj Pobjeda'})
            ax8.set_title('Matrica Pobjeda (Red = Domaćin, Kolona = Gost)', fontsize=title_fontsize)
            ax8.set_xlabel('Protivnik', fontsize=label_fontsize)
            ax8.set_ylabel('Ekipa', fontsize=label_fontsize)
            plt.tight_layout()
            st.pyplot(fig8)

            # 7. Head-to-Head Details (text-based)
            st.subheader("7. Detalji Međusobnih Susreta (Head-to-Head)")
            tied_groups = final_table[final_table.duplicated(subset=['Pts'], keep=False)].groupby('Pts')
            if not tied_groups.groups:
                st.write("Nema ekipa s istim brojem bodova koje zahtijevaju H2H analizu.")
            else:
                for pts_value, group in tied_groups:
                    tied_teams_list = list(group.index)
                    if len(tied_teams_list) > 1:
                        st.write(f"Ekipe s **{pts_value}** bodova: {', '.join(tied_teams_list)}")
                        h2h_details = {}
                        
                        for team_in_group in tied_teams_list:
                            h2h_details[team_in_group] = 0

                        for match in all_matches:
                            t1, s1, t2, s2 = match['team1'], match['score1'], match['team2'], match['score2']
                            if t1 in tied_teams_list and t2 in tied_teams_list:
                                h2h_details[t1] += s1
                                h2h_details[t2] += s2
                        
                        st.write("Ukupni H2H skorovi (golovi) unutar grupe:")
                        sorted_h2h_teams = sorted(h2h_details.items(), key=lambda item: item[1], reverse=True)
                        for team_h2h, score_h2h in sorted_h2h_teams:
                            st.write(f"- {team_h2h}: {score_h2h}")
                        st.markdown("---")
else:
    st.info("Molimo unesite podatke o utakmicama da biste vidjeli analizu.")