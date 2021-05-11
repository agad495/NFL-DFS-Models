import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, VotingRegressor
from get_fball_data import GetFballData

class DFS_Models():
    """
    Class containing models for Weekly DFS player projections (NFL).
    """
    def __init__(self, season, pbp=None):
        gfd = GetFballData()
        self.season = season
        if pbp is not None:
            self.pbp = pbp
        else:
            self.pbp = gfd.get_pbp(2006, season)

    def rolls_royce(self, col_name, anderson, og_df, groupie, shit=True, mp=1):
        rolled_series = og_df.groupby(groupie)[col_name].rolling(anderson, min_periods=mp).mean().reset_index(drop=True)
        
        if shit:
            new_df = og_df[[groupie]]
            new_df[f'{col_name}_roll'] = rolled_series
            new_df[f'{col_name}_roll'] = new_df.groupby(groupie)[f'{col_name}_roll'].shift(1)
            
            return new_df[f'{col_name}_roll']
        
        else:
            return rolled_series

    def get_wr_data(self, roll, current, betting_data=None, no_roll=None):
        passing_pbp = self.pbp.query("pass_attempt == 1")
        passing_pbp['rz_tgt'] = np.where(passing_pbp['yardline_100'] - passing_pbp['air_yards'] <= 20, 1, 0)
        
        team_tgts = passing_pbp.groupby(['posteam', 'season', 'week']).agg({'pass_attempt': 'sum',
                                                                    'air_yards': 'sum'}) \
            .rename(columns={'pass_attempt': 'tm_tgts', 'air_yards': 'tm_air'}) \
                .reset_index()
        
        td_tgts = passing_pbp.groupby(['receiver', 'receiver_id', 'posteam', 'season', 'week', 'home_team']).agg({'pass_attempt': 'sum',
                                                                                                                     'complete_pass': 'sum',
                                                                                                                     'epa': 'sum',
                                                                                                                     'yards_gained': 'sum',
                                                                                                                     'air_yards': 'sum',
                                                                                                                     'pass_touchdown': 'sum',
                                                                                                                     'td_prob': 'sum',
                                                                                                                     'yards_after_catch': 'sum',
                                                                                                                     'success': 'sum',
                                                                                                                     'shotgun': 'sum',
                                                                                                                     'no_huddle':'sum',
                                                                                                                     'spread_line': 'mean',
                                                                                                                     'total_line': 'mean',
                                                                                                                     'rz_tgt': 'sum'}) \
            .reset_index() \
                .merge(team_tgts, left_on=['posteam', 'season', 'week'],
                       right_on=['posteam', 'season', 'week'],
                       how='left')
                
        td_tgts['100plus_rec'] = np.where(td_tgts['yards_gained'] >= 100, 1, 0)
        td_tgts['ppr'] = td_tgts['complete_pass'] + (td_tgts['yards_gained'] * 0.1) + (td_tgts['pass_touchdown'] * 6)
        td_tgts.sort_values(['receiver_id', 'season', 'week'], inplace=True)
        td_tgts.reset_index(drop=True, inplace=True)
        
        if no_roll:
            return td_tgts
                
        if current:
            recs_new = td_tgts[['receiver', 'receiver_id', 'posteam', 'season', 'week', 'ppr', 'spread_line', 'total_line']]
            recs_new['tgt_roll'] = self.rolls_royce('pass_attempt', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['rec_roll'] = self.rolls_royce('complete_pass', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['epa_roll'] = self.rolls_royce('epa', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['yds_roll'] = self.rolls_royce('yards_gained', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['air_roll'] = self.rolls_royce('air_yards', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['yac_roll'] = self.rolls_royce('yards_after_catch', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['tds_roll'] = self.rolls_royce('pass_touchdown', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['tdp_roll'] = self.rolls_royce('td_prob', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['suc_roll'] = self.rolls_royce('success', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['ppr_roll'] = self.rolls_royce('ppr', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['ppr_roll_4'] = self.rolls_royce('ppr', 4, td_tgts, 'receiver_id', shit=False)
            recs_new['tm_tgt_roll'] = self.rolls_royce('tm_tgts', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['tm_air_roll'] = self.rolls_royce('tm_air', roll, td_tgts, 'receiver_id', shit=False)
            recs_new['sr_roll'] = recs_new['suc_roll'] / recs_new['tgt_roll']
            recs_new['tgt_share_roll'] = recs_new['tgt_roll'] / recs_new['tm_tgt_roll']
            recs_new['air_share_roll'] = recs_new['air_roll'] / recs_new['tm_air_roll']
            recs_new['wopr_roll'] = (recs_new['tgt_share_roll'] * 1.5) + (recs_new['air_share_roll'] * 0.7)
            recs_new.dropna(inplace=True)
            
            recs_new.query("season == 2020", inplace=True)
            recs_new.drop_duplicates(subset='receiver_id', keep='last', inplace=True)
            
            for i in betting_data:
                recs_new['spread_line'] = np.where(recs_new['posteam'] == i, betting_data[i][0], recs_new['spread_line'])
                recs_new['total_line'] = np.where(recs_new['posteam'] == i, betting_data[i][1], recs_new['total_line'])
            recs_new['implied_tt'] = (recs_new['total_line'] / 2) - (recs_new['spread_line'] / 2)
                        
            return recs_new
            
        else:
            recs = td_tgts[['receiver', 'receiver_id', 'season', 'week', 'ppr', 'spread_line', 'total_line']]
            recs['tgt_roll'] = self.rolls_royce('pass_attempt', roll, td_tgts, 'receiver_id')
            recs['rec_roll'] = self.rolls_royce('complete_pass', roll, td_tgts, 'receiver_id')
            recs['epa_roll'] = self.rolls_royce('epa', roll, td_tgts, 'receiver_id')
            recs['yds_roll'] = self.rolls_royce('yards_gained', roll, td_tgts, 'receiver_id')
            recs['air_roll'] = self.rolls_royce('air_yards', roll, td_tgts, 'receiver_id')
            recs['yac_roll'] = self.rolls_royce('yards_after_catch', roll, td_tgts, 'receiver_id')
            recs['tds_roll'] = self.rolls_royce('pass_touchdown', roll, td_tgts, 'receiver_id')
            recs['tdp_roll'] = self.rolls_royce('td_prob', roll, td_tgts, 'receiver_id')
            recs['suc_roll'] = self.rolls_royce('success', roll, td_tgts, 'receiver_id')
            recs['ppr_roll'] = self.rolls_royce('ppr', roll, td_tgts, 'receiver_id')
            recs['tm_tgt_roll'] = self.rolls_royce('tm_tgts', roll, td_tgts, 'receiver_id')
            recs['tm_air_roll'] = self.rolls_royce('tm_air', roll, td_tgts, 'receiver_id')
            recs['sr_roll'] = recs['suc_roll'] / recs['tgt_roll']
            recs['tgt_share_roll'] = recs['tgt_roll'] / recs['tm_tgt_roll']
            recs['air_share_roll'] = recs['air_roll'] / recs['tm_air_roll']
            recs['wopr_roll'] = (recs['tgt_share_roll'] * 1.5) + (recs['air_share_roll'] * 0.7)
            recs['implied_tt'] = (recs['total_line'] / 2) - (recs['spread_line'] / 2)
            recs.dropna(inplace=True)
    
            return recs
    
    def wr_model_ppr(self, roll, betting_data):
        recs = self.get_wr_data(roll=roll, current=False)

        X = recs.drop(['receiver', 'receiver_id', 'season', 'week', 'ppr'], axis=1)
        y = recs['ppr']

        np.random.seed(69)
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)

        reg1 = ElasticNet(alpha=0.0028281965345676027, copy_X=True, fit_intercept=True,
                   l1_ratio=0.013081763357145266, max_iter=100000, normalize=False,
                   positive=False, precompute=False, random_state=None,
                   selection='cyclic', tol=0.0001, warm_start=False)
        reg2 = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                            max_depth=8, max_features=0.958484466900927,
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=9,
                            min_weight_fraction_leaf=0.0, n_estimators=97, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,
                            warm_start=False)
        ereg = VotingRegressor(estimators=[('ela', reg1), ('etr', reg2)])
        
        np.random.seed(69)
        wr_model_prod = ereg
        wr_model_prod.fit(X_train, y)

        recs_new = self.get_wr_data(roll=roll, current=True, betting_data=betting_data)
        
        X_new = recs_new.drop(['receiver', 'receiver_id', 'posteam', 'season', 'week', 'ppr', 'ppr_roll_4'], axis=1)
        X_new = scaler.transform(X_new)
        
        np.random.seed(69)
        wr_preds_new = wr_model_prod.predict(X_new)
        recs_new['ppr_rec_preds'] = wr_preds_new
        recs_new['ppr_rec_diff'] = recs_new['ppr_rec_preds'] - recs_new['ppr_roll']
        recs_new['ppr_rec_diff_4'] = recs_new['ppr_rec_preds'] - recs_new['ppr_roll_4']
        
        return recs_new

    def get_rb_data(self, roll, current, betting_data=None, no_roll=None):
        rushing_pbp = self.pbp.query("rush_attempt == 1")
        
        team_rush = rushing_pbp.groupby(['posteam', 'season', 'week']).agg({'rush_attempt': 'sum'}) \
            .rename(columns={'rush_attempt': 'tm_rushes'}) \
                .reset_index()
        
        td_rush = rushing_pbp.groupby(['rusher', 'rusher_id', 'posteam', 'season', 'week', 'home_team']).agg({'rush_attempt': 'sum',
                                                                                                                 'epa': 'sum',
                                                                                                                 'yards_gained': 'sum',
                                                                                                                 'rush_touchdown': 'sum',
                                                                                                                 'td_prob': 'sum',
                                                                                                                 'success': 'sum',
                                                                                                                 'shotgun': 'sum',
                                                                                                                 'no_huddle':'sum',
                                                                                                                 'spread_line': 'mean',
                                                                                                                 'total_line': 'mean'}) \
            .reset_index() \
                .merge(team_rush, left_on=['posteam', 'season', 'week'],
                       right_on=['posteam', 'season', 'week'],
                       how='left')
        
        td_rush['100plus_rush'] = np.where(td_rush['yards_gained'] >= 100, 1, 0)
        td_rush['ppr'] = (td_rush['yards_gained'] * 0.1) + (td_rush['rush_touchdown'] * 6)
        td_rush['home'] = np.where(td_rush['home_team'] == td_rush['posteam'], 1, 0)
        td_rush.sort_values(['rusher_id', 'season', 'week'], inplace=True)
        td_rush.reset_index(drop=True, inplace=True)
        
        if no_roll:
            return td_rush
                
        if current:
            rush_new = td_rush[['rusher', 'rusher_id', 'posteam', 'season', 'week', 'ppr', 'home']]
            rush_new['rush_roll'] = self.rolls_royce('rush_attempt', roll, td_rush, 'rusher_id', shit=False)
            rush_new['epa_roll'] = self.rolls_royce('epa', roll, td_rush, 'rusher_id', shit=False)
            rush_new['yds_roll'] = self.rolls_royce('yards_gained', roll, td_rush, 'rusher_id', shit=False)
            rush_new['tds_roll'] = self.rolls_royce('rush_touchdown', roll, td_rush, 'rusher_id', shit=False)
            rush_new['tdp_roll'] = self.rolls_royce('td_prob', roll, td_rush, 'rusher_id', shit=False)
            rush_new['suc_roll'] = self.rolls_royce('success', roll, td_rush, 'rusher_id', shit=False)
            rush_new['gun_roll'] = self.rolls_royce('shotgun', roll, td_rush, 'rusher_id', shit=False)
            rush_new['ppr_roll'] = self.rolls_royce('ppr', roll, td_rush, 'rusher_id', shit=False)
            rush_new['ppr_roll_4'] = self.rolls_royce('ppr', 4, td_rush, 'rusher_id', shit=False)
            rush_new['tm_rush_roll'] = self.rolls_royce('tm_rushes', roll, td_rush, 'rusher_id', shit=False)
            rush_new['rush_share_roll'] = rush_new['rush_roll'] / rush_new['tm_rush_roll']
            rush_new.dropna(inplace=True)
            
            rush_new.query("season == 2020", inplace=True)
            rush_new.drop_duplicates(subset='rusher_id', keep='last', inplace=True)
                                    
            for i in betting_data:
                rush_new['home'] = np.where(rush_new['posteam'] == i, betting_data[i][2], rush_new['home'])

            return rush_new
            
        else:
            rush = td_rush[['rusher', 'rusher_id', 'season', 'week', 'ppr', 'home']]
            rush['rush_roll'] = self.rolls_royce('rush_attempt', roll, td_rush, 'rusher_id')
            rush['epa_roll'] = self.rolls_royce('epa', roll, td_rush, 'rusher_id')
            rush['yds_roll'] = self.rolls_royce('yards_gained', roll, td_rush, 'rusher_id')
            rush['tds_roll'] = self.rolls_royce('rush_touchdown', roll, td_rush, 'rusher_id')
            rush['tdp_roll'] = self.rolls_royce('td_prob', roll, td_rush, 'rusher_id')
            rush['suc_roll'] = self.rolls_royce('success', roll, td_rush, 'rusher_id')
            rush['gun_roll'] = self.rolls_royce('shotgun', roll, td_rush, 'rusher_id')
            rush['ppr_roll'] = self.rolls_royce('ppr', roll, td_rush, 'rusher_id')
            rush['tm_rush_roll'] = self.rolls_royce('tm_rushes', roll, td_rush, 'rusher_id')
            rush['rush_share_roll'] = rush['rush_roll'] / rush['tm_rush_roll']
            rush.dropna(inplace=True)
    
            return rush
        
    def rb_model_ppr(self, roll, betting_data):
        rush = self.get_rb_data(roll=roll, current=False)

        X = rush.drop(['rusher', 'rusher_id', 'season', 'week', 'ppr', 'tm_rush_roll',
                       'epa_roll', 'week', 'season'], axis=1)
        y = rush['ppr']

        np.random.seed(69)
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)

        reg1 = ElasticNet(alpha=0.002027265056082726, copy_X=True, fit_intercept=True,
                   l1_ratio=0.011884219247245734, max_iter=100000, normalize=False,
                   positive=False, precompute=False, random_state=None,
                   selection='cyclic', tol=0.0001, warm_start=False)
        reg2 = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                            max_depth=8, max_features=0.958484466900927,
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=9,
                            min_weight_fraction_leaf=0.0, n_estimators=97, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,
                            warm_start=False)
        ereg = VotingRegressor(estimators=[('ela', reg1), ('etr', reg2)])
        
        np.random.seed(69)
        rb_model_prod = ereg
        rb_model_prod.fit(X_train, y)

        rush_new = self.get_rb_data(roll=roll, current=True, betting_data=betting_data)
        
        X_new = rush_new.drop(['rusher', 'rusher_id', 'season', 'week', 'ppr', 'tm_rush_roll',
                       'epa_roll', 'week', 'season', 'ppr_roll_4', 'posteam'], axis=1)
        X_new = scaler.transform(X_new)
        
        np.random.seed(69)
        rb_preds_new = rb_model_prod.predict(X_new)
        rush_new['ppr_rush_preds'] = rb_preds_new
        rush_new['ppr_rush_diff'] = rush_new['ppr_rush_preds'] - rush_new['ppr_roll']
        rush_new['ppr_rush_diff_4'] = rush_new['ppr_rush_preds'] - rush_new['ppr_roll_4']
        
        return rush_new
        
    def get_qb_data(self, roll, current, betting_data=None, no_roll=None):
        passing_pbp = self.pbp.query("pass_attempt == 1")
        
        td_pass = passing_pbp.groupby(['passer', 'passer_id', 'posteam', 'season', 'week', 'home_team']).agg({'pass_attempt': 'sum',
                                                                                             'complete_pass': 'sum',
                                                                                             'epa': 'sum',
                                                                                             'yards_gained': 'sum',
                                                                                             'air_yards': 'sum',
                                                                                             'pass_touchdown': 'sum',
                                                                                             'td_prob': 'sum',
                                                                                             'yards_after_catch': 'sum',
                                                                                             'success': 'sum',
                                                                                             'shotgun': 'sum',
                                                                                             'no_huddle':'sum',
                                                                                             'spread_line': 'mean',
                                                                                             'total_line': 'mean',
                                                                                             'interception': 'sum',
                                                                                             'cpoe': 'sum'}) \
            .reset_index()
        
        td_pass['300plus_pass'] = np.where(td_pass['yards_gained'] >= 300, 1, 0)
        td_pass['ppr'] = (td_pass['yards_gained'] * 0.04) + (td_pass['pass_touchdown'] * 4) + (td_pass['interception'] * -1)
        td_pass['home'] = np.where(td_pass['home_team'] == td_pass['posteam'], 1, 0)
        td_pass.sort_values(['passer_id', 'season', 'week'], inplace=True)
        td_pass.reset_index(drop=True, inplace=True)
        
        if no_roll:
            return td_pass
                
        if current:
            pass_new = td_pass[['passer', 'passer_id', 'posteam', 'season', 'week', 'ppr', 'spread_line', 'total_line', 'home']]
            pass_new['att_roll'] = self.rolls_royce('pass_attempt', roll, td_pass, 'passer_id', shit=False)
            pass_new['rec_roll'] = self.rolls_royce('complete_pass', roll, td_pass, 'passer_id', shit=False)
            pass_new['epa_roll'] = self.rolls_royce('epa', roll, td_pass, 'passer_id', shit=False)
            pass_new['yds_roll'] = self.rolls_royce('yards_gained', roll, td_pass, 'passer_id', shit=False)
            pass_new['air_roll'] = self.rolls_royce('air_yards', roll, td_pass, 'passer_id', shit=False)
            pass_new['yac_roll'] = self.rolls_royce('yards_after_catch', roll, td_pass, 'passer_id', shit=False)
            pass_new['tds_roll'] = self.rolls_royce('pass_touchdown', roll, td_pass, 'passer_id', shit=False)
            pass_new['tdp_roll'] = self.rolls_royce('td_prob', roll, td_pass, 'passer_id', shit=False)
            pass_new['suc_roll'] = self.rolls_royce('success', roll, td_pass, 'passer_id', shit=False)
            pass_new['cpoe_roll'] = self.rolls_royce('cpoe', roll, td_pass, 'passer_id', shit=False)
            pass_new['ppr_roll'] = self.rolls_royce('ppr', roll, td_pass, 'passer_id', shit=False)
            pass_new['ppr_roll_4'] = self.rolls_royce('ppr', 4, td_pass, 'passer_id', shit=False)
            pass_new['sr_roll'] = pass_new['suc_roll'] / pass_new['att_roll']
            pass_new['epa_rate_roll'] = pass_new['epa_roll'] / pass_new['att_roll']
            pass_new['adot_roll'] = pass_new['air_roll'] / pass_new['att_roll']
            pass_new.dropna(inplace=True)
            
            pass_new.query("season == 2020", inplace=True)
            pass_new.drop_duplicates(subset='passer_id', keep='last', inplace=True)
            
            for i in betting_data:
                pass_new['spread_line'] = np.where(pass_new['posteam'] == i, betting_data[i][0], pass_new['spread_line'])
                pass_new['total_line'] = np.where(pass_new['posteam'] == i, betting_data[i][1], pass_new['total_line'])
                pass_new['home'] = np.where(pass_new['posteam'] == i, betting_data[i][2], pass_new['home'])
            pass_new['implied_tt'] = (pass_new['total_line'] / 2) - (pass_new['spread_line'] / 2)
                        
            return pass_new
            
        else:
            passer = td_pass[['passer', 'passer_id', 'posteam', 'season', 'week', 'ppr', 'spread_line', 'total_line', 'home']]
            passer['att_roll'] = self.rolls_royce('pass_attempt', roll, td_pass, 'passer_id')
            passer['rec_roll'] = self.rolls_royce('complete_pass', roll, td_pass, 'passer_id')
            passer['epa_roll'] = self.rolls_royce('epa', roll, td_pass, 'passer_id')
            passer['yds_roll'] = self.rolls_royce('yards_gained', roll, td_pass, 'passer_id')
            passer['air_roll'] = self.rolls_royce('air_yards', roll, td_pass, 'passer_id')
            passer['yac_roll'] = self.rolls_royce('yards_after_catch', roll, td_pass, 'passer_id')
            passer['tds_roll'] = self.rolls_royce('pass_touchdown', roll, td_pass, 'passer_id')
            passer['tdp_roll'] = self.rolls_royce('td_prob', roll, td_pass, 'passer_id')
            passer['suc_roll'] = self.rolls_royce('success', roll, td_pass, 'passer_id')
            passer['cpoe_roll'] = self.rolls_royce('cpoe', roll, td_pass, 'passer_id')
            passer['ppr_roll'] = self.rolls_royce('ppr', roll, td_pass, 'passer_id')
            passer['ppr_roll_4'] = self.rolls_royce('ppr', 4, td_pass, 'passer_id')
            passer['sr_roll'] = passer['suc_roll'] / passer['att_roll']
            passer['epa_rate_roll'] = passer['epa_roll'] / passer['att_roll']
            passer['adot_roll'] = passer['air_roll'] / passer['att_roll']
            passer.dropna(inplace=True)
    
            return passer

    def qb_model_ppr(self, roll, betting_data):
        passer = self.get_qb_data(roll=roll, current=False)

        X = passer.drop(['passer', 'passer_id', 'posteam', 'season', 'week', 'ppr'], axis=1)
        y = passer['ppr']

        np.random.seed(69)
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)
        
        np.random.seed(69)
        qb_model_prod = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                                    max_depth=8, max_features=0.958484466900927,
                                    max_leaf_nodes=None, max_samples=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=9,
                                    min_weight_fraction_leaf=0.0, n_estimators=97, n_jobs=None,
                                    oob_score=False, random_state=None, verbose=0,
                                    warm_start=False)
        qb_model_prod.fit(X_train, y)

        pass_new = self.get_qb_data(roll=roll, current=True, betting_data=betting_data)
        
        X_new = pass_new.drop(['passer', 'passer_id', 'posteam', 'season', 'week', 'ppr', 'ppr_roll_4'], axis=1)
        X_new = scaler.transform(X_new)
        
        np.random.seed(69)
        qb_preds_new = qb_model_prod.predict(X_new)
        pass_new['ppr_pass_preds'] = qb_preds_new
        pass_new['ppr_pass_diff'] = pass_new['ppr_pass_preds'] - pass_new['ppr_roll']
        pass_new['ppr_pass_diff_4'] = pass_new['ppr_pass_preds'] - pass_new['ppr_roll_4']
        
        return pass_new

    def get_def_data(self, current, betting_data=None):
        self.pbp['datda'] = np.where((self.pbp['return_touchdown'] == 1) & (self.pbp['play_type'] == 'kickoff'), 
                                     self.pbp['defteam'], 'Nah')
        self.pbp['defteam'] = np.where((self.pbp['return_touchdown'] == 1) & (self.pbp['play_type'] == 'kickoff'), 
                                       self.pbp['posteam'], self.pbp['defteam'])
        self.pbp['posteam'] = np.where((self.pbp['return_touchdown'] == 1) & (self.pbp['play_type'] == 'kickoff'), 
                                       self.pbp['datda'], self.pbp['posteam'])
        self.pbp['return_touchdown'] = np.where((self.pbp['field_goal_result'] == 'blocked') & (self.pbp['touchdown'] == 1), 
                                                1, self.pbp['return_touchdown'])
        self.pbp['return_touchdown'] = np.where((self.pbp['punt_blocked'] == 1) & (self.pbp['touchdown'] == 1), 
                                                1, self.pbp['return_touchdown'])
        self.pbp['blocked_kick'] = np.where((self.pbp['punt_blocked'] == 1) | (self.pbp['field_goal_result'] == 'blocked'), 
                                            1, 0)
        self.pbp['defensive_two_point_conv'].fillna(0, inplace=True)

        
        team_scores = self.pbp.groupby(['defteam', 'posteam', 'season', 'week', 'home_team', 'spread_line', 'total_line']).agg({'posteam_score_post': 'max',
                                                                                                                          'defteam_score_post': 'max',
                                                                                                                          'sack':'sum',
                                                                                                                          'interception':'sum',
                                                                                                                          'fumble':'sum',
                                                                                                                          'fumble_lost':'sum',
                                                                                                                          'return_touchdown':'sum',
                                                                                                                          'safety':'sum',
                                                                                                                          'blocked_kick':'sum',
                                                                                                                          'defensive_two_point_conv':'sum',
                                                                                                                          'epa':'sum',
                                                                                                                          'yards_gained':'sum',
                                                                                                                          'pass_attempt':'sum',
                                                                                                                          'rush_attempt':'sum',
                                                                                                                          'play':'sum',
                                                                                                                          'success':'sum',
                                                                                                                          'td_prob':'sum',
                                                                                                                          'cpoe':'sum'}) \
            .reset_index()
        team_scores['home'] = np.where(team_scores['posteam'] == team_scores['home_team'], 1, 0)
        team_scores['spread_line'] = np.where(team_scores['home'] == 1, -1 * team_scores['spread_line'], team_scores['spread_line'])
        team_scores['pa_ppr'] = np.where(team_scores['posteam_score_post'] == 0, 10,
                                         np.where((team_scores['posteam_score_post'] > 0) & (team_scores['posteam_score_post'] <= 6), 7,
                                                  np.where((team_scores['posteam_score_post'] > 6) & (team_scores['posteam_score_post'] <= 13), 4,
                                                           np.where((team_scores['posteam_score_post'] > 13) & (team_scores['posteam_score_post'] <= 20), 1,
                                                                    np.where((team_scores['posteam_score_post'] > 20) & (team_scores['posteam_score_post'] <= 27), 0,
                                                                             np.where((team_scores['posteam_score_post'] > 27) & (team_scores['posteam_score_post'] <= 34), -1,
                                                                                      np.where(team_scores['posteam_score_post'] > 34, -4, 10000)))))))
        team_scores['ppr'] = (team_scores['sack'] + (team_scores['interception']*2) +
                                (team_scores['fumble_lost']*2) + (team_scores['return_touchdown']*6) +
                                (team_scores['safety']*2) + (team_scores['blocked_kick']*2) +
                                team_scores['pa_ppr'] + (team_scores['defensive_two_point_conv']*2))
                
        if current:
            def_new = team_scores[['defteam', 'posteam', 'season', 'week', 'ppr', 'spread_line', 'total_line', 'home']]
            def_new.dropna(inplace=True)
            
            def_new.query("season == 2020", inplace=True)
            def_new.drop_duplicates(subset='defteam', keep='last', inplace=True)
            
            for i in betting_data:
                def_new['spread_line'] = np.where(def_new['defteam'] == i, betting_data[i][0]*-1, def_new['spread_line'])
                def_new['total_line'] = np.where(def_new['defteam'] == i, betting_data[i][1], def_new['total_line'])
                def_new['home'] = np.where(def_new['defteam'] == i, abs(betting_data[i][2]-1), def_new['home'])
            def_new['implied_tt'] = (def_new['total_line'] / 2) - (def_new['spread_line'] / 2)
                        
            return def_new
            
        else:
            def_new = team_scores[['defteam', 'posteam', 'season', 'week', 'ppr', 'spread_line', 'total_line', 'home']]
            def_new['implied_tt'] = (def_new['total_line'] / 2) - (def_new['spread_line'] / 2)
            def_new.dropna(inplace=True)
    
            return def_new

    def def_model_ppr(self, betting_data):
        def_teams = self.get_def_data(current=False)

        X = def_teams.drop(['defteam', 'posteam', 'season', 'week', 'ppr'], axis=1)
        y = def_teams['ppr']

        np.random.seed(69)
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)
        
        np.random.seed(69)
        def_model_prod = ElasticNet(alpha=0.009669496558390946, copy_X=True, fit_intercept=True,
                                    l1_ratio=0.010046423641363409, max_iter=100000, normalize=False,
                                    positive=False, precompute=False, random_state=None,
                                    selection='cyclic', tol=0.0001, warm_start=False)
        def_model_prod.fit(X_train, y)

        def_new = self.get_def_data(current=True, betting_data=betting_data)
        
        X_new = def_new.drop(['defteam', 'posteam', 'season', 'week', 'ppr'], axis=1)
        X_new = scaler.transform(X_new)
        
        np.random.seed(69)
        def_preds_new = def_model_prod.predict(X_new)
        def_new['ppr_def_preds'] = def_preds_new
        
        return def_new.drop('posteam', axis=1)

    def get_flex_data(self, roll, current, betting_data=None):
        td_tgts = self.get_wr_data(roll=16, current=False, no_roll=True)
        td_tgts['ppr'] = td_tgts['complete_pass'] + (td_tgts['yards_gained'] * 0.1) + (td_tgts['pass_touchdown'] * 6) + (td_tgts['100plus_rec'] * 3)
        td_tgts.rename(columns={'pass_attempt':'targets',
                                'complete_pass':'receptions',
                                'epa':'rec_epa',
                                'yards_gained':'rec_yards',
                                'air_yards':'rec_air_yards',
                                'yards_after_catch':'rec_yac',
                                'pass_touchdown':'rec_td',
                                'ppr':'rec_ppr'}, inplace=True)
        td_rush = self.get_rb_data(roll=16, current=False, no_roll=True)
        td_rush['ppr'] = (td_rush['yards_gained'] * 0.1) + (td_rush['rush_touchdown'] * 6) + (td_rush['100plus_rush'] * 3)
        td_rush.rename(columns={'epa':'rush_epa',
                                'yards_gained':'rush_yards',
                                'rush_touchdown':'rush_td',
                                'ppr':'rush_ppr'}, inplace=True)
        td_pass = self.get_qb_data(roll=16, current=False, no_roll=True)
        td_pass['ppr'] = (td_pass['yards_gained'] * 0.04) + (td_pass['pass_touchdown'] * 4) + (td_pass['interception'] * -1) + (td_pass['300plus_pass'] * 3)
        td_pass.rename(columns={'epa':'pass_epa',
                                 'yards_gained':'pass_yards',
                                 'air_yards':'pass_air_yards',
                                 'pass_touchdown':'pass_td',
                                 'yards_after_catch':'pass_yac',
                                 'ppr':'pass_ppr'}, inplace=True)
            
        full_flex = td_tgts.merge(td_rush, how='outer', 
                          left_on=['receiver', 'receiver_id', 'posteam', 'season', 'week', 'home_team'],
                          right_on=['rusher', 'rusher_id', 'posteam', 'season', 'week', 'home_team'])
        full_flex['player'] = np.where(full_flex['receiver'].isnull(), full_flex['rusher'], full_flex['receiver'])
        full_flex['player_id'] = np.where(full_flex['receiver_id'].isnull(), full_flex['rusher_id'], full_flex['receiver_id'])
        full_flex = full_flex.merge(td_pass, how='outer', 
                                  left_on=['player', 'player_id', 'posteam', 'season', 'week', 'home_team'],
                                  right_on=['passer', 'passer_id', 'posteam', 'season', 'week', 'home_team'])
        full_flex['player'] = np.where(full_flex['player'].isnull(), full_flex['passer'], full_flex['player'])
        full_flex['player_id'] = np.where(full_flex['player_id'].isnull(), full_flex['passer_id'], full_flex['player_id'])
        full_flex['spread_line'] = np.where(full_flex['spread_line'] < abs(full_flex['spread_line_x']), full_flex['spread_line_x'], full_flex['spread_line'])
        full_flex['spread_line'] = np.where(full_flex['spread_line'] < abs(full_flex['spread_line_y']), full_flex['spread_line_y'], full_flex['spread_line'])
        full_flex['total_line'] = np.where(full_flex['total_line'] < abs(full_flex['total_line_x']), full_flex['total_line_x'], full_flex['total_line'])
        full_flex['total_line'] = np.where(full_flex['total_line'] < abs(full_flex['total_line_y']), full_flex['total_line_y'], full_flex['total_line'])
        full_flex.fillna(0, inplace=True)
        full_flex['td_prob'] = full_flex['td_prob'] + full_flex['td_prob_x'] + full_flex['td_prob_y']
        full_flex['success'] = full_flex['success'] + full_flex['success_x'] + full_flex['success_y']
        full_flex['home'] = np.where(full_flex['home_team'] == full_flex['posteam'], 1, 0)
        full_flex['no_huddle'] = full_flex['no_huddle'] + full_flex['no_huddle_x'] + full_flex['no_huddle_y']
        full_flex['total_ppr'] = full_flex['pass_ppr'] + full_flex['rush_ppr'] + full_flex['rec_ppr']
        full_flex.sort_values(['player_id', 'season', 'week'], inplace=True)
        full_flex.reset_index(drop=True, inplace=True)

        if current:
            fakeball = full_flex[['player', 'player_id', 'posteam', 'season', 'week', 'total_ppr', 'spread_line', 'total_line', 'home']]
            rolled_cols = ['pass_attempt', 'complete_pass', 'pass_epa', 'pass_yards', 'pass_air_yards',
                           'pass_td', 'td_prob', 'pass_yac', 'success', 'no_huddle', 'interception', 'cpoe',
                           'pass_ppr', 'rush_ppr', 'rec_ppr', 'total_ppr', 'targets', 'receptions',
                           'rec_epa', 'rec_yards', 'rec_air_yards', 'rec_td', 'rec_yac', 'rz_tgt',
                           'tm_tgts', 'tm_air', 'rush_attempt', 'rush_epa', 'rush_yards', 'rush_td',
                           'tm_rushes', '100plus_rush', '100plus_rec', '300plus_pass']
            for i in rolled_cols:
                fakeball[f'{i}_roll'] = self.rolls_royce(i, roll, full_flex, 'player_id', shit=False, mp=4)
            fakeball['cmp_pct'] = np.where(fakeball['pass_attempt_roll'] != 0, fakeball['complete_pass_roll'] / fakeball['pass_attempt_roll'], 0)
            fakeball['rec_pct'] = np.where(fakeball['targets_roll'] != 0, fakeball['receptions_roll'] / fakeball['targets_roll'], 0)
            fakeball['epa_per_pass'] = np.where(fakeball['pass_attempt_roll'] != 0, fakeball['pass_epa_roll'] / fakeball['pass_attempt_roll'], 0)
            fakeball['epa_per_rush'] = np.where(fakeball['rush_attempt_roll'] != 0, fakeball['rush_epa_roll'] / fakeball['rush_attempt_roll'], 0)
            fakeball['epa_per_target'] = np.where(fakeball['targets_roll'] != 0, fakeball['rec_epa_roll'] / fakeball['targets_roll'], 0)
            fakeball['pass_adot'] = np.where(fakeball['pass_attempt_roll'] != 0, fakeball['pass_air_yards_roll'] / fakeball['pass_attempt_roll'], 0)
            fakeball['rec_adot'] = np.where(fakeball['targets_roll'] != 0, fakeball['rec_air_yards_roll'] / fakeball['targets_roll'], 0)
            fakeball['sr'] = fakeball['success_roll'] / (fakeball['targets_roll'] + fakeball['pass_attempt_roll'] + fakeball['rush_attempt_roll'])
            fakeball['rz_tgt_rate'] = np.where(fakeball['targets_roll'] != 0, fakeball['rz_tgt_roll'] / fakeball['targets_roll'], 0)
            fakeball['tgt_share'] = np.where(fakeball['targets_roll'] != 0, fakeball['targets_roll'] / fakeball['tm_tgts_roll'], 0)
            fakeball['air_share'] = np.where(fakeball['targets_roll'] != 0, fakeball['rec_air_yards_roll'] / fakeball['tm_air_roll'], 0)
            fakeball['wopr'] = (fakeball['tgt_share'] * 1.5) + (fakeball['air_share'] * 0.7)
            fakeball['ppr25'] = np.where(fakeball['total_ppr'] >= 25, 1, 0)
            fakeball['ppr30'] = np.where(fakeball['total_ppr'] >= 30, 1, 0)
            fakeball.dropna(inplace=True)
            
            fakeball.query("season == 2020", inplace=True)
            fakeball.drop_duplicates(subset='player_id', keep='last', inplace=True)

            for i in betting_data:
                fakeball['spread_line'] = np.where(fakeball['posteam'] == i, betting_data[i][0], fakeball['spread_line'])
                fakeball['total_line'] = np.where(fakeball['posteam'] == i, betting_data[i][1], fakeball['total_line'])
                fakeball['home'] = np.where(fakeball['posteam'] == i, betting_data[i][2], fakeball['home'])
            
        
        else:
            fakeball = full_flex[['player', 'player_id', 'season', 'week', 'total_ppr', 'spread_line', 'total_line', 'home']]
            rolled_cols = ['pass_attempt', 'complete_pass', 'pass_epa', 'pass_yards', 'pass_air_yards',
                           'pass_td', 'td_prob', 'pass_yac', 'success', 'no_huddle', 'interception', 'cpoe',
                           'pass_ppr', 'rush_ppr', 'rec_ppr', 'total_ppr', 'targets', 'receptions',
                           'rec_epa', 'rec_yards', 'rec_air_yards', 'rec_td', 'rec_yac', 'rz_tgt',
                           'tm_tgts', 'tm_air', 'rush_attempt', 'rush_epa', 'rush_yards', 'rush_td',
                           'tm_rushes', '100plus_rush', '100plus_rec', '300plus_pass']
            for i in rolled_cols:
                fakeball[f'{i}_roll'] = self.rolls_royce(i, roll, full_flex, 'player_id', mp=4)
            fakeball['cmp_pct'] = np.where(fakeball['pass_attempt_roll'] != 0, fakeball['complete_pass_roll'] / fakeball['pass_attempt_roll'], 0)
            fakeball['rec_pct'] = np.where(fakeball['targets_roll'] != 0, fakeball['receptions_roll'] / fakeball['targets_roll'], 0)
            fakeball['epa_per_pass'] = np.where(fakeball['pass_attempt_roll'] != 0, fakeball['pass_epa_roll'] / fakeball['pass_attempt_roll'], 0)
            fakeball['epa_per_rush'] = np.where(fakeball['rush_attempt_roll'] != 0, fakeball['rush_epa_roll'] / fakeball['rush_attempt_roll'], 0)
            fakeball['epa_per_target'] = np.where(fakeball['targets_roll'] != 0, fakeball['rec_epa_roll'] / fakeball['targets_roll'], 0)
            fakeball['pass_adot'] = np.where(fakeball['pass_attempt_roll'] != 0, fakeball['pass_air_yards_roll'] / fakeball['pass_attempt_roll'], 0)
            fakeball['rec_adot'] = np.where(fakeball['targets_roll'] != 0, fakeball['rec_air_yards_roll'] / fakeball['targets_roll'], 0)
            fakeball['sr'] = fakeball['success_roll'] / (fakeball['targets_roll'] + fakeball['pass_attempt_roll'] + fakeball['rush_attempt_roll'])
            fakeball['rz_tgt_rate'] = np.where(fakeball['targets_roll'] != 0, fakeball['rz_tgt_roll'] / fakeball['targets_roll'], 0)
            fakeball['tgt_share'] = np.where(fakeball['targets_roll'] != 0, fakeball['targets_roll'] / fakeball['tm_tgts_roll'], 0)
            fakeball['air_share'] = np.where(fakeball['targets_roll'] != 0, fakeball['rec_air_yards_roll'] / fakeball['tm_air_roll'], 0)
            fakeball['wopr'] = (fakeball['tgt_share'] * 1.5) + (fakeball['air_share'] * 0.7)
            fakeball['ppr25'] = np.where(fakeball['total_ppr'] >= 25, 1, 0)
            fakeball['ppr30'] = np.where(fakeball['total_ppr'] >= 30, 1, 0)
            fakeball.dropna(inplace=True)
            
        
        return fakeball

    def smash_model_ppr(self, roll, betting_data):
        flexball = self.get_flex_data(roll=roll, current=False)

        X = flexball.drop(['player', 'player_id', 'season', 'week', 'total_ppr', 'tm_tgts_roll',
                           'tm_air_roll', 'ppr25', 'ppr30'], axis=1)
        y = flexball['ppr25']

        np.random.seed(69)
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)
        
        np.random.seed(69)
        smash_model_prod = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                                                 criterion='gini', max_depth=9,
                                                 max_features=0.1911271103547656, max_leaf_nodes=None,
                                                 max_samples=None, min_impurity_decrease=0.0,
                                                 min_impurity_split=None, min_samples_leaf=1,
                                                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                                                 n_estimators=89, n_jobs=None, oob_score=False,
                                                 random_state=None, verbose=0, warm_start=False)
        smash_model_prod.fit(X_train, y)

        flex_new = self.get_flex_data(roll=roll, current=True, betting_data=betting_data)
        
        X_new = flex_new.drop(['player', 'player_id', 'posteam', 'season', 'week', 'total_ppr', 'tm_tgts_roll',
                               'tm_air_roll', 'ppr25', 'ppr30'], axis=1)
        X_new = scaler.transform(X_new)
        
        np.random.seed(69)
        smash_preds_new = smash_model_prod.predict_proba(X_new)
        flex_new['ppr25_odds'] = smash_preds_new[:, 1]
        flex_new = flex_new[['player', 'player_id', 'posteam', 'ppr25_odds']]
        
        return flex_new
