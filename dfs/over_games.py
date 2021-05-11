import pandas as pd
import numpy as np

from get_fball_data import GetFballData
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

class OverOdds():
    """
    Class for predicting the percent odds that a game goes over the vegas total.
    """
    def __init__(self, week, season=2020, pbp=None):
        if pbp is not None:
            self.pbp = pbp
        else:
            gfd = GetFballData()
            self.pbp = gfd.get_pbp(2006, season)
        self.season = season
        self.week = week
        
        neutral_plays = self.pbp.query("((score_differential <= 10) & (score_differential >= -10) & (game_seconds_remaining >= 2100)) | ((score_differential <= 8) & (score_differential >= -8) & (qtr == 3))")

        tm_drive_pace = neutral_plays.groupby(['posteam', 'season', 'week', 'drive', 
                                               'drive_time_of_possession', 'drive_play_count']).agg({'epa':'sum'}) \
            .reset_index()
        tm_drive_pace['seconds'] = [(int(a)*60) + int(b) for a, b in tm_drive_pace['drive_time_of_possession'].str.split(':')]
        
        self.tm_wk_pace = tm_drive_pace.groupby(['posteam', 'season', 'week']).agg({'drive_play_count':'sum',
                                                                               'seconds':'sum'}) \
            .reset_index() \
                .query("drive_play_count > 0") \
                    .rename(columns={'drive_play_count':'plays'}) \
                        .reset_index(drop=True)

    def rolls_royce(self, col_name, anderson, groupie, og_df, shit=True):
        rolled_series = og_df.groupby(groupie)[col_name].rolling(anderson).mean().reset_index(drop=True)
        
        if shit:
            new_df = og_df[[groupie]]
            new_df[f'{col_name}_roll'] = rolled_series
            new_df[f'{col_name}_roll'] = new_df.groupby(groupie)[f'{col_name}_roll'].shift(1)
            
            return new_df[f'{col_name}_roll']
        
        else:
            return rolled_series
        
    def pace_data(self, roll, betting_data=False):
        """
        Parameters
        ----------
        roll : integer, number of games to include in the rolling mean.
        betting_data : dictionary with teams, spreads, totals, h/a, & opponent.

        Returns
        -------
        tm_games : TYPE
            DESCRIPTION.

        """
        if not betting_data:
            self.tm_wk_pace['play_roll'] = self.rolls_royce('plays', roll, 'posteam', self.tm_wk_pace)
            self.tm_wk_pace['secs_roll'] = self.rolls_royce('seconds', roll, 'posteam', self.tm_wk_pace)
            self.tm_wk_pace['secs_play_roll'] = self.tm_wk_pace['secs_roll'] / self.tm_wk_pace['play_roll']
            
            tm_games = self.pbp.groupby(['home_team', 'away_team', 'season', 'week', 'total_line']).agg({'home_score':'max',
                                                                                                      'away_score':'max',}) \
                .reset_index() \
                    .merge(self.tm_wk_pace, how='left', left_on=['home_team', 'season', 'week'],
                           right_on=['posteam', 'season', 'week']) \
                        .merge(self.tm_wk_pace, how='left', left_on=['away_team', 'season', 'week'],
                           right_on=['posteam', 'season', 'week']) \
                            .sort_values(['season', 'week']) \
                                .reset_index(drop=True)
            tm_games['over'] = np.where(tm_games['total_line'] < (tm_games['home_score'] + tm_games['away_score']), 1, 0)
            tm_games.dropna(inplace=True)
            tm_games.reset_index(drop=True, inplace=True)
            
        else:
            self.tm_wk_pace['play_roll'] = self.rolls_royce('plays', roll, 'posteam', self.tm_wk_pace, shit=False)
            self.tm_wk_pace['secs_roll'] = self.rolls_royce('seconds', roll, 'posteam', self.tm_wk_pace, shit=False)
            self.tm_wk_pace['secs_play_roll'] = self.tm_wk_pace['secs_roll'] / self.tm_wk_pace['play_roll']
            self.tm_wk_pace.drop_duplicates(subset='posteam', keep='last', inplace=True)
            
            tm_games = pd.DataFrame()
            ht, at, s, w, tote = [], [], [], [], []
            for i in betting_data:
                if betting_data[i][2] == 1:
                    ht.append(i)
                    at.append(betting_data[i][3])
                else:
                    at.append(i)
                    ht.append(betting_data[i][3])
                s.append(self.season)
                w.append(self.week)
                tote.append(betting_data[i][1])
            tm_games['home_team'] = ht
            tm_games['away_team'] = at
            tm_games['season'] = s
            tm_games['week'] = w
            tm_games['total_line'] = tote
            tm_games.query("total_line != 0", inplace=True)
            tm_games.drop_duplicates(inplace=True)
            
            tm_games = tm_games.merge(self.tm_wk_pace[['posteam', 'secs_play_roll']], how='left', 
                                      left_on=['home_team'],
                                      right_on=['posteam']) \
                .merge(self.tm_wk_pace[['posteam', 'secs_play_roll']], how='left', 
                       left_on=['away_team'],
                       right_on=['posteam'])
            tm_games.drop(['posteam_x', 'posteam_y'], axis=1, inplace=True)
        
        return tm_games
    
    def predict_overs(self, roll, betting_data):
        tm_games = self.pace_data(roll=roll)
        
        X = tm_games.drop(['home_team', 'away_team', 'home_score', 'away_score', 
                           'posteam_x', 'posteam_y', 'plays_x', 'plays_y', 'seconds_x', 
                           'seconds_y', 'play_roll_x', 'play_roll_y', 'secs_roll_x', 
                           'secs_roll_y', 'over'], axis=1)
        y = tm_games['over']
        
        np.random.seed(69)
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        np.random.seed(69)
        over_model_prod = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, gamma=4, gpu_id=-1,
                      importance_type='gain', interaction_constraints='',
                      learning_rate=0.0024091452883402854, max_delta_step=0,
                      max_depth=6, min_child_weight=1, min_split_loss=4, missing=np.nan,
                      monotone_constraints='()', n_estimators=100, n_jobs=0,
                      num_parallel_tree=1, objective='binary:logistic', random_state=0,
                      reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                      tree_method='exact', validate_parameters=1, verbosity=None)
        over_model_prod.fit(X, y)
        
        new_pace = self.pace_data(roll=roll, betting_data=betting_data)
        X_new = new_pace.drop(['home_team', 'away_team'], axis=1)
        X_new = scaler.transform(X_new)

        over_preds = over_model_prod.predict_proba(X_new)
        
        new_pace['over_odds'] = over_preds[:, 1]

        return new_pace