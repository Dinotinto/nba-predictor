# NBA Over/Under Predictor

A small Flask web app that estimates whether an NBA game, and each team in it, will go over or under a points line. You pick the home and away teams, the game date and the lines, and the app compares several prediction methods side by side.

## How it works

The app uses 2024-25 regular-season team game logs (`nba_game_data_api_robust.csv`, about 2,100 games) pulled with the [`nba_api`](https://github.com/swar/nba_api) package. If the CSV is missing, it fetches the logs from the NBA stats API at start-up.

For each request it runs four methods and shows the results together:

| Method | What it does |
|---|---|
| **Original model** | A scikit-learn random-forest classifier scores both teams' averaged stats over their last 15 games, then checks the call against the line. If the model and the scoring average disagree, the result is a **Push**. |
| **Points-based** | Adds each team's average points over the last 10 games and turns the gap to the line into a confidence using a normal distribution. |
| **Home/Away** | The same idea, using only the home team's last 5 home games and the away team's last 5 away games. |
| **Team-based models** | Separate random-forest models for home and away form, giving an Over/Under call and confidence for each team's own line. |

Features include shooting percentages, rebounds, assists, steals, blocks, turnovers, fouls, home/away, and the opponent's average defensive stats. The trained models are stored as `.joblib` files and loaded at start-up, so the app doesn't retrain on each deploy.

## Stack

Python, Flask, pandas, NumPy, scikit-learn, SciPy, `nba_api`, Bootstrap, and Gunicorn for deployment on Render.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000. To serve it the way it's deployed:

```bash
gunicorn app:app
```

## Limitations

- The data covers the 2024-25 season only and is a snapshot, so it needs refreshing to stay current.
- The model labels are Over/Under relative to the median team score in the data, not historical betting lines.
- The confidence figures are simple statistical estimates, not calibrated probabilities.

This is a personal project for exploring sports data and modelling. It isn't betting advice.

Copyright © 2025 Adeolu Idowu. All rights reserved.
