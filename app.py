from flask import Flask, request, render_template
from nba_api.stats.static import teams
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import numpy as np

app = Flask(__name__)

# Load and preprocess the new CSV
def load_data():
    df = pd.read_csv('nba_game_data_api_robust.csv')  # Adjust to your CSV name
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y', errors='coerce')
    median_pts = df['PTS'].median()
    df['OVER_UNDER'] = df['PTS'].apply(lambda x: 'Over' if x > median_pts else 'Under')
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split('vs.')[1].strip() if 'vs.' in x else x.split('@')[1].strip())
    opponent_stats = df.groupby('team_name')[['DREB', 'BLK', 'STL', 'PF']].mean().reset_index()
    opponent_stats.columns = ['OPPONENT', 'OPP_DREB', 'OPP_BLK', 'OPP_STL', 'OPP_PF']
    df = df.merge(opponent_stats, on='OPPONENT', how='left')
    return df

# Load data once
teams_df = load_data()
features = ['W', 'L', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 
            'HOME', 'MIN', 'OPP_DREB', 'OPP_BLK', 'OPP_STL', 'OPP_PF']

# Split data for Team-Based Models
home_df = teams_df[teams_df['MATCHUP'].str.contains('vs.')]
away_df = teams_df[teams_df['MATCHUP'].str.contains('@')]

# Train Original Model (for overall and individual predictions)
X = teams_df[features]
y = teams_df['OVER_UNDER']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [10, 20, None]}
grid_search_original = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search_original.fit(X_train, y_train)
model_original = grid_search_original.best_estimator_
print(f"Original Model - Best params: {grid_search_original.best_params_}")
print(f"Original Model - Accuracy: {accuracy_score(y_test, model_original.predict(X_test)):.2f}")

# Train Team-Based Model for Home Games
X_home = home_df[features]
y_home = home_df['OVER_UNDER']
X_home_train, X_home_test, y_home_train, y_home_test = train_test_split(X_home, y_home, test_size=0.2, random_state=42)
grid_search_team_home = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search_team_home.fit(X_home_train, y_home_train)
model_team_home = grid_search_team_home.best_estimator_
print(f"Team-Based Model (Home) - Best params: {grid_search_team_home.best_params_}")
print(f"Team-Based Model (Home) - Accuracy: {accuracy_score(y_home_test, model_team_home.predict(X_home_test)):.2f}")

# Train Team-Based Model for Away Games
X_away = away_df[features]
y_away = away_df['OVER_UNDER']
X_away_train, X_away_test, y_away_train, y_away_test = train_test_split(X_away, y_away, test_size=0.2, random_state=42)
grid_search_team_away = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search_team_away.fit(X_away_train, y_away_train)  # Fixed: y_away_test -> y_away_train
model_team_away = grid_search_team_away.best_estimator_
print(f"Team-Based Model (Away) - Best params: {grid_search_team_away.best_params_}")
print(f"Team-Based Model (Away) - Accuracy: {accuracy_score(y_away_test, model_team_away.predict(X_away_test)):.2f}")

# Prediction 1: Original Model (overall and individual predictions)
def predict_over_under_original(team1, team2, date, over_under_line, line_home, line_away):
    game_date = pd.to_datetime(date)
    
    recent_games_team1 = teams_df[(teams_df['team_name'] == team1) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(15)
    avg_pts_team1 = recent_games_team1['PTS'].mean() or teams_df[teams_df['team_name'] == team1]['PTS'].mean()
    recent_games_team2 = teams_df[(teams_df['team_name'] == team2) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(15)
    avg_pts_team2 = recent_games_team2['PTS'].mean() or teams_df[teams_df['team_name'] == team2]['PTS'].mean()
    total_avg_pts = avg_pts_team1 + avg_pts_team2
    recent_games = pd.concat([recent_games_team1, recent_games_team2])
    game_stats = recent_games[features].mean().to_frame().T if not recent_games.empty else teams_df[features].mean().to_frame().T
    overall_pred = model_original.predict(game_stats)[0]
    overall_probs = model_original.predict_proba(game_stats)[0]
    overall_prob_dict = dict(zip(model_original.classes_, overall_probs))
    overall_prob = overall_prob_dict[overall_pred] * 100
    overall_final_pred = overall_pred if (overall_pred == 'Over' and total_avg_pts > float(over_under_line)) or (overall_pred == 'Under' and total_avg_pts <= float(over_under_line)) else 'Push'
    
    team1_stats = recent_games_team1[features].mean().to_frame().T if not recent_games_team1.empty else teams_df[features].mean().to_frame().T
    team1_raw_pred = model_original.predict(team1_stats)[0]
    team1_std_pts = recent_games_team1['PTS'].std() or 10
    team1_final_pred = team1_raw_pred if (team1_raw_pred == 'Over' and avg_pts_team1 > float(line_home)) or (team1_raw_pred == 'Under' and avg_pts_team1 <= float(line_home)) else 'Push'
    team1_prob = (1 - norm.cdf(float(line_home), avg_pts_team1, team1_std_pts)) * 100 if team1_raw_pred == 'Over' else norm.cdf(float(line_home), avg_pts_team1, team1_std_pts) * 100
    
    team2_stats = recent_games_team2[features].mean().to_frame().T if not recent_games_team2.empty else teams_df[features].mean().to_frame().T
    team2_raw_pred = model_original.predict(team2_stats)[0]
    team2_std_pts = recent_games_team2['PTS'].std() or 10
    team2_final_pred = team2_raw_pred if (team2_raw_pred == 'Over' and avg_pts_team2 > float(line_away)) or (team2_raw_pred == 'Under' and avg_pts_team2 <= float(line_away)) else 'Push'
    team2_prob = (1 - norm.cdf(float(line_away), avg_pts_team2, team2_std_pts)) * 100 if team2_raw_pred == 'Over' else norm.cdf(float(line_away), avg_pts_team2, team2_std_pts) * 100
    
    return (overall_final_pred, total_avg_pts, overall_prob), (team1_raw_pred, team1_final_pred, team1_prob, avg_pts_team1), (team2_raw_pred, team2_final_pred, team2_prob, avg_pts_team2)

# Prediction for individual team over/under (Team-Based Model) - Home/Away Split
def predict_team_over_under(team, date, team_line, is_home_team=True):
    game_date = pd.to_datetime(date)
    if is_home_team:
        recent_games = teams_df[(teams_df['team_name'] == team) & (teams_df['MATCHUP'].str.contains('vs.')) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(15)
        model = model_team_home
    else:
        recent_games = teams_df[(teams_df['team_name'] == team) & (teams_df['MATCHUP'].str.contains('@')) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(15)
        model = model_team_away
    
    avg_pts = recent_games['PTS'].mean() or teams_df[teams_df['team_name'] == team]['PTS'].mean()
    std_pts = recent_games['PTS'].std() or 10
    game_stats = recent_games[features].mean().to_frame().T if not recent_games.empty else teams_df[features].mean().to_frame().T
    
    raw_prediction = model.predict(game_stats)[0]
    final_prediction = raw_prediction if (raw_prediction == 'Over' and avg_pts > float(team_line)) or (raw_prediction == 'Under' and avg_pts <= float(team_line)) else 'Push'
    prob = (1 - norm.cdf(float(team_line), avg_pts, std_pts)) * 100 if raw_prediction == 'Over' else norm.cdf(float(team_line), avg_pts, std_pts) * 100
    
    return raw_prediction, final_prediction, prob, avg_pts

# Prediction 2: Points-Based (overall line with confidence)
def predict_over_under_points(team1, team2, date, over_under_line):
    game_date = pd.to_datetime(date)
    recent_games_team1 = teams_df[(teams_df['team_name'] == team1) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(10)
    avg_pts_team1 = recent_games_team1['PTS'].mean() or teams_df[teams_df['team_name'] == team1]['PTS'].mean()
    std_pts_team1 = recent_games_team1['PTS'].std() or 10
    recent_games_team2 = teams_df[(teams_df['team_name'] == team2) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(10)
    avg_pts_team2 = recent_games_team2['PTS'].mean() or teams_df[teams_df['team_name'] == team2]['PTS'].mean()
    std_pts_team2 = recent_games_team2['PTS'].std() or 10
    
    total_avg_pts = avg_pts_team1 + avg_pts_team2
    total_std_pts = np.sqrt(std_pts_team1**2 + std_pts_team2**2)
    prediction = 'Over' if total_avg_pts > float(over_under_line) else 'Under'
    confidence = (1 - norm.cdf(float(over_under_line), total_avg_pts, total_std_pts)) * 100 if prediction == 'Over' else norm.cdf(float(over_under_line), total_avg_pts, total_std_pts) * 100
    
    return prediction, total_avg_pts, confidence

# Prediction 3: Last 5 Home/Away (overall line with confidence)
def predict_over_under_last_5_home_away(team_home, team_away, date, over_under_line, over_under_line_home, over_under_line_away):
    game_date = pd.to_datetime(date)
    recent_home_games = teams_df[(teams_df['team_name'] == team_home) & (teams_df['MATCHUP'].str.contains('vs.')) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(5)
    avg_pts_home = recent_home_games['PTS'].mean() or teams_df[teams_df['team_name'] == team_home]['PTS'].mean()
    std_pts_home = recent_home_games['PTS'].std() or 10
    recent_away_games = teams_df[(teams_df['team_name'] == team_away) & (teams_df['MATCHUP'].str.contains('@')) & (teams_df['GAME_DATE'] <= game_date)].sort_values('GAME_DATE', ascending=False).head(5)
    avg_pts_away = recent_away_games['PTS'].mean() or teams_df[teams_df['team_name'] == team_away]['PTS'].mean()
    std_pts_away = recent_away_games['PTS'].std() or 10
    
    total_avg_pts = avg_pts_home + avg_pts_away
    total_std_pts = np.sqrt(std_pts_home**2 + std_pts_away**2)
    prediction = 'Over' if total_avg_pts > float(over_under_line) else 'Under'
    confidence = (1 - norm.cdf(float(over_under_line), total_avg_pts, total_std_pts)) * 100 if prediction == 'Over' else norm.cdf(float(over_under_line), total_avg_pts, total_std_pts) * 100
    
    return prediction, avg_pts_home, avg_pts_away, total_avg_pts, confidence

@app.route('/', methods=['GET', 'POST'])
def home():
    team_names = [team['full_name'] for team in teams.get_teams()]
    if request.method == 'POST':
        team1 = request.form['home_team']
        team2 = request.form['away_team']
        date = request.form['date']
        line = request.form['line']
        line_home = request.form['line_home']
        line_away = request.form['line_away']
        try:
            (overall_pred, total_pts1, overall_prob), (team1_orig_raw, team1_orig_final, team1_orig_prob, team1_avg_pts), (team2_orig_raw, team2_orig_final, team2_orig_prob, team2_avg_pts) = predict_over_under_original(team1, team2, date, line, line_home, line_away)
            
            pred2, total_pts2, conf2 = predict_over_under_points(team1, team2, date, line)
            pred3, avg_pts_home, avg_pts_away, total_pts3, conf3 = predict_over_under_last_5_home_away(team1, team2, date, line, line_home, line_away)
            
            team1_team_raw, team1_team_final, team1_team_prob, team1_team_avg_pts = predict_team_over_under(team1, date, line_home, is_home_team=True)
            team2_team_raw, team2_team_final, team2_team_prob, team2_team_avg_pts = predict_team_over_under(team2, date, line_away, is_home_team=False)
            
            results = {
                'original_overall': f"Original Model ({team1} vs. {team2}): {overall_pred}, Total Points: {total_pts1:.2f}, Confidence: {overall_prob:.1f}%",
                'points': f"Points-Based ({team1} vs. {team2}): {pred2}, Total Points: {total_pts2:.2f}, Confidence is {conf2:.1f}%",
                'home_away': f"Home/Away ({team1} vs. {team2}): {pred3}, Home Avg: {avg_pts_home:.2f}, Away Avg: {avg_pts_away:.2f}, Total: {total_pts3:.2f}, Confidence is {conf3:.1f}%",
                'original_team1': f"Original Model ({team1}): {team1} - {line_home}, Raw: {team1_orig_raw}, Final: {team1_orig_final}, Confidence is {team1_orig_prob:.1f}%, Avg Points: {team1_avg_pts:.2f}",
                'original_team2': f"Original Model ({team2}): {team2} - {line_away}, Raw: {team2_orig_raw}, Final: {team2_orig_final}, Confidence is {team2_orig_prob:.1f}%, Avg Points: {team2_avg_pts:.2f}",
                'team_based_team1': f"Team-Based Model ({team1}): {team1} - {line_home}, Raw: {team1_team_raw}, Final: {team1_team_final}, Confidence is {team1_team_prob:.1f}%, Avg Points: {team1_team_avg_pts:.2f}",
                'team_based_team2': f"Team-Based Model ({team2}): {team2} - {line_away}, Raw: {team2_team_raw}, Final: {team2_team_final}, Confidence is {team2_team_prob:.1f}%, Avg Points: {team2_team_avg_pts:.2f}"
            }
        except Exception as e:
            results = {'error': f"Error: {str(e)}"}
        return render_template('index.html', teams=team_names, results=results)
    return render_template('index.html', teams=team_names, results=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)