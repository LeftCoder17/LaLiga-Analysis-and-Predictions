import sqlite3
import pandas as pd
from datetime import datetime, date

import settings


def load_matchday(season, division, matchday):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql(f"""
            SELECT * FROM Matches
                WHERE season = '{season}'
                AND division = {division}
                AND matchday = {matchday}
        """, conn)
    if data.empty:
        raise ValueError("There is no matchday data for the values given")
    return data


def load_historical_data(seasons):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        if seasons == "all":
            data = pd.read_sql("SELECT * FROM Matches", conn)
        else:
            data = pd.read_sql(f"""
                SELECT * FROM Matches
                    WHERE season IN {tuple(seasons)}
            """, conn)
    if data.empty:
        raise ValueError(f"No data for seasons {seasons}")
    return data


def save_predictions(predictions):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        predictions = predictions.drop(['weekday', 'last_5_results_local', 'last_5_results_rival',
                                        'prob_1', 'prob_2', 'prob_X'], axis=1)
        predictions.rename(columns={'team': 'home_team', 'rival': 'away_team'}, inplace=True)
        predictions.to_sql(name="Predictions", con=conn, if_exists="append", index=False)


# Aux function
def encoder_teams(seasons):
    data = load_historical_data("all")
    if len(seasons) == 1:
        data = data.loc[data['season'] == seasons]
    # We just need the teams from the training and the prediction
    team_names = data['home_team'].unique()
    team_ids = [i for i in range(len(team_names))]
    return team_names, team_ids


def weekday_match(string_date):
    month, day, year = map(int, string_date.split('/'))
    year += 2000
    n_weekday = date(year, month, day).weekday()
    return n_weekday


def determine_winner(row):
    if row['local_goals'] > row['visitor_goals']:
        return 'Local'
    elif row['local_goals'] < row['visitor_goals']:
        return 'Visitor'
    else:
        return 'Tie'


def match_result(row):
    if row['W_match'] is True:
        return 'W'
    elif row['L_match'] is True:
        return 'L'
    elif row['T_match'] is True:
        return 'T'


def numerical_time(time_string):
    if time_string is None:
        time_string = "12:00 AM"
    time_parts = datetime.strptime(time_string, "%I:%M %p")
    time_num = time_parts.hour + time_parts.minute / 60
    return time_num


def numerical_last_results_local(last_results_list):
    total = 0
    for result in last_results_list:
        if result == 'W':
            total += 3
        elif result == 'T':
            total += 1
        else:
            total += -3
    return total


# Preparing datasets functions
def preparing_training_dataset(seasons):
    data = load_historical_data(seasons)
    data[['local_goals', 'visitor_goals']] = data['score'].str.split(':', expand=True).astype(float)
    data['winner'] = data.apply(determine_winner, axis=1)
    data['goal_difference'] = data['local_goals'] - data['visitor_goals']
    data['weekday'] = data['date'].apply(weekday_match)

    home_team_df = data.copy()
    home_team_df['team'], home_team_df['rival'], home_team_df['condition'] = home_team_df['home_team'], home_team_df['away_team'], 'local'
    away_team_df = data.copy()
    away_team_df['team'], away_team_df['rival'], away_team_df['condition'] = away_team_df['away_team'], away_team_df['home_team'], 'visitor'

    matches_data = pd.concat([home_team_df, away_team_df], ignore_index=True).sort_values(by=['season', 'division', 'team', 'matchday'])
    matches_data.reset_index(drop=True, inplace=True)

    matches_data['GF_match'] = matches_data.apply(lambda row: row['local_goals'] if row['condition'] == 'local' else row['visitor_goals'], axis=1)
    matches_data['GA_match'] = matches_data.apply(lambda row: row['visitor_goals'] if row['condition'] == 'local' else row['local_goals'], axis=1)
    matches_data['GD_match'] = matches_data['GF_match'] - matches_data['GA_match']
    matches_data['W_match'] = ((matches_data['winner'] == 'Local') & (matches_data['condition'] == 'local')) | ((matches_data['winner'] == 'Visitor') & (matches_data['condition'] == 'visitor'))
    matches_data['L_match'] = ((matches_data['winner'] == 'Visitor') & (matches_data['condition'] == 'local')) | ((matches_data['winner'] == 'Local') & (matches_data['condition'] == 'visitor'))
    matches_data['T_match'] = (matches_data['winner'] == 'Tie')
    matches_data['match_result'] = matches_data.apply(match_result, axis=1)
    matches_data['last_match_result_1'] = matches_data.groupby(['season', 'team'])['match_result'].shift(1)
    matches_data['last_match_result_2'] = matches_data.groupby(['season', 'team'])['match_result'].shift(2)
    matches_data['last_match_result_3'] = matches_data.groupby(['season', 'team'])['match_result'].shift(3)
    matches_data['last_match_result_4'] = matches_data.groupby(['season', 'team'])['match_result'].shift(4)
    matches_data['last_match_result_5'] = matches_data.groupby(['season', 'team'])['match_result'].shift(5)
    matches_data['last_5_results_local'] = matches_data[['last_match_result_1', 'last_match_result_2', 'last_match_result_3',
                                                        'last_match_result_4', 'last_match_result_5']].values.tolist()

    matches_data['GF'] = matches_data.groupby(['season', 'team'])['GF_match'].cumsum()
    matches_data['GA'] = matches_data.groupby(['season', 'team'])['GA_match'].cumsum()
    matches_data['W'] = matches_data.groupby(['season', 'team'])['W_match'].cumsum()
    matches_data['L'] = matches_data.groupby(['season', 'team'])['L_match'].cumsum()
    matches_data['T'] = matches_data.groupby(['season', 'team'])['T_match'].cumsum()

    final_ranking = matches_data.reset_index()
    final_ranking['GD'] = final_ranking['GF'] - final_ranking['GA']
    final_ranking['Pts'] = final_ranking['W'] * 3 + final_ranking['T']
    final_ranking = final_ranking.sort_values(by=['season', 'division', 'matchday', 'Pts', 'GD'], ascending=[False, True, True, False, False])
    final_ranking.reset_index(drop=True, inplace=True)
    final_ranking['rank'] = final_ranking.groupby(['season', 'division', 'matchday'])['Pts'].rank(ascending=False, method='first').astype(int)

    final_ranking = final_ranking[['season', 'division', 'matchday','date', 'weekday', 'time', 'rank', 'team', 'rival', 'condition',
                                'winner', 'GF_match', 'GA_match', 'GD_match', 'GF', 'GA', 'GD', 'W', 'L', 'T', 'Pts', 'last_5_results_local']]

    # Prepare numerical data
    seasons_string = final_ranking['season'].unique()
    first_season = int(seasons_string[-1].split('-')[0])
    last_season = int(seasons_string[0].split('-')[0])
    seasons_list = list(reversed(range(first_season, last_season + 1)))
    final_ranking['season'] = final_ranking['season'].replace(seasons_string, seasons_list)
    final_ranking['time'] = final_ranking['time'].apply(numerical_time)

    team_names, team_ids = encoder_teams(seasons)
    final_ranking[['team', 'rival']] = final_ranking[['team', 'rival']].replace(team_names, team_ids)
    # winner --> 0: Local, 1: Tie, 2:Visitor
    final_ranking['winner'] = final_ranking['winner'].replace(['Local', 'Tie', 'Visitor'], [0, 1, 2])

    final_ranking['last_5_results_local'] = final_ranking['last_5_results_local'].apply(numerical_last_results_local)
    mapping = final_ranking.groupby(['season', 'matchday']).apply(lambda x: dict(zip(x['team'], x['last_5_results_local']))).to_dict()
    final_ranking['last_5_results_rival'] = final_ranking.apply(lambda row: mapping.get((row['season'], row['matchday']), {}).get(row['rival'], None), axis=1)

    # The matches are repeated: Drop when condition is Visitor
    final_ranking = final_ranking.loc[final_ranking['condition'] == 'local']

    return final_ranking


def preparing_predicting_dataset(season_p, division_p, matchday_p):
    data = load_historical_data("all")
    data = data.loc[data['season'] == season_p]
    data[['local_goals', 'visitor_goals']] = data['score'].str.split(':', expand=True).astype(float)
    data['winner'] = data.apply(determine_winner, axis=1)
    data['goal_difference'] = data['local_goals'] - data['visitor_goals']
    data['weekday'] = data['date'].apply(weekday_match)

    home_team_df = data.copy()
    home_team_df['team'], home_team_df['rival'], home_team_df['condition'] = home_team_df['home_team'], home_team_df['away_team'], 'local'
    away_team_df = data.copy()
    away_team_df['team'], away_team_df['rival'], away_team_df['condition'] = away_team_df['away_team'], away_team_df['home_team'], 'visitor'

    matches_data = pd.concat([home_team_df, away_team_df], ignore_index=True).sort_values(by=['season', 'division', 'team', 'matchday'])
    matches_data.reset_index(drop=True, inplace=True)

    matches_data['GF_match'] = matches_data.apply(lambda row: row['local_goals'] if row['condition'] == 'local' else row['visitor_goals'], axis=1)
    matches_data['GA_match'] = matches_data.apply(lambda row: row['visitor_goals'] if row['condition'] == 'local' else row['local_goals'], axis=1)
    matches_data['GD_match'] = matches_data['GF_match'] - matches_data['GA_match']
    matches_data['W_match'] = ((matches_data['winner'] == 'Local') & (matches_data['condition'] == 'local')) | ((matches_data['winner'] == 'Visitor') & (matches_data['condition'] == 'visitor'))
    matches_data['L_match'] = ((matches_data['winner'] == 'Visitor') & (matches_data['condition'] == 'local')) | ((matches_data['winner'] == 'Local') & (matches_data['condition'] == 'visitor'))
    matches_data['T_match'] = (matches_data['winner'] == 'Tie')
    matches_data['match_result'] = matches_data.apply(match_result, axis=1)
    matches_data['last_match_result_1'] = matches_data.groupby(['season', 'team'])['match_result'].shift(1)
    matches_data['last_match_result_2'] = matches_data.groupby(['season', 'team'])['match_result'].shift(2)
    matches_data['last_match_result_3'] = matches_data.groupby(['season', 'team'])['match_result'].shift(3)
    matches_data['last_match_result_4'] = matches_data.groupby(['season', 'team'])['match_result'].shift(4)
    matches_data['last_match_result_5'] = matches_data.groupby(['season', 'team'])['match_result'].shift(5)
    matches_data['last_5_results_local'] = matches_data[['last_match_result_1', 'last_match_result_2', 'last_match_result_3',
                                                        'last_match_result_4', 'last_match_result_5']].values.tolist()

    matches_data['GF'] = matches_data.groupby(['season', 'team'])['GF_match'].cumsum()
    matches_data['GA'] = matches_data.groupby(['season', 'team'])['GA_match'].cumsum()
    matches_data['W'] = matches_data.groupby(['season', 'team'])['W_match'].cumsum()
    matches_data['L'] = matches_data.groupby(['season', 'team'])['L_match'].cumsum()
    matches_data['T'] = matches_data.groupby(['season', 'team'])['T_match'].cumsum()

    final_ranking = matches_data.reset_index()
    final_ranking['GD'] = final_ranking['GF'] - final_ranking['GA']
    final_ranking['Pts'] = final_ranking['W'] * 3 + final_ranking['T']
    final_ranking = final_ranking.sort_values(by=['season', 'division', 'matchday', 'Pts', 'GD'], ascending=[False, True, True, False, False])
    final_ranking.reset_index(drop=True, inplace=True)
    final_ranking['rank'] = final_ranking.groupby(['season', 'division', 'matchday'])['Pts'].rank(ascending=False, method='first').astype(int)

    final_ranking = final_ranking[['season', 'division', 'matchday', 'weekday', 'time', 'rank', 'team', 'rival', 'condition',
                                'winner', 'GF_match', 'GA_match', 'GD_match', 'GF', 'GA', 'GD', 'W', 'L', 'T', 'Pts', 'last_5_results_local']]
    # Prepare numerical data
    final_ranking['season'] = final_ranking['season'].replace(season_p, int(season_p.split('-')[0]))

    final_ranking['time'] = final_ranking['time'].apply(numerical_time)

    team_names, team_ids = encoder_teams(season_p)
    final_ranking[['team', 'rival']] = final_ranking[['team', 'rival']].replace(team_names, team_ids)
    # winner --> 0: Local, 1: Tie, 2:Visitor
    final_ranking['winner'] = final_ranking['winner'].replace(['Local', 'Tie', 'Visitor'], [0, 1, 2])

    final_ranking['last_5_results_local'] = final_ranking['last_5_results_local'].apply(numerical_last_results_local)
    mapping = final_ranking.groupby(['season', 'matchday']).apply(lambda x: dict(zip(x['team'], x['last_5_results_local']))).to_dict()
    final_ranking['last_5_results_rival'] = final_ranking.apply(lambda row: mapping.get((row['season'], row['matchday']), {}).get(row['rival'], None), axis=1)
    # The matches are repeated: Drop when condition is Visitor
    final_ranking = final_ranking.loc[final_ranking['condition'] == 'local']

    # Select the matchday group to predict
    matchday_dataset = final_ranking.loc[(final_ranking['division'] == division_p)
                                        & (final_ranking['matchday'] == matchday_p)]

    return matchday_dataset
