# NFL DFS Models
Set of models (using scikit-learn) to predict weekly fantasy football scoring in DraftKings' PPR format.

## Use:
```python
import pandas as pd
import numpy as np

from dfs_models import DFS_Models
from dfs_optimizer import FragilistaDFS
from over_games import OverOdds
from get_fball_data import GetFballData 
from dksb_scraping import ScrapeDK

gfd = GetFballData()
pbp = gfd.get_pbp(2006, 2020)

dfs = DFS_Models(season=2020, pbp=pbp)
scrape_lines = ScrapeDK()
betting_data = scrape_lines.nfl()

wrs = dfs.wr_model_ppr(roll=8, betting_data = betting_data)
wrs.sort_values('ppr_rec_preds', inplace=True, ascending=False)

rbs = dfs.rb_model_ppr(roll=8, betting_data=betting_data)
rbs.sort_values('ppr_rush_preds', inplace=True, ascending=False)

qbs = dfs.qb_model_ppr(roll=12, betting_data=betting_data)
qbs.sort_values('ppr_pass_preds', inplace=True, ascending=False)

defs = dfs.def_model_ppr(betting_data=betting_data)
defs.sort_values('ppr_def_preds', inplace=True, ascending=False)

smash_spots = dfs.smash_model_ppr(roll=8, betting_data=betting_data)
smash_spots.sort_values('ppr25_odds', inplace=True, ascending=False)

totes = OverOdds(week=12, season=2020, pbp=pbp)
overit = totes.predict_overs(roll=8, betting_data=betting_data)

flex = wrs[['receiver', 'receiver_id', 'posteam', 'ppr_roll_4', 'ppr_rec_preds', 'ppr_rec_diff_4']]. \
    merge(rbs[['rusher', 'rusher_id', 'posteam', 'ppr_roll_4', 'ppr_rush_preds', 'ppr_rush_diff_4']],
          left_on=['receiver_id'], right_on=['rusher_id'], how='outer'). \
        merge(qbs[['passer', 'passer_id', 'posteam', 'ppr_roll_4', 'ppr_pass_preds', 'ppr_pass_diff_4']],
              left_on=['rusher_id'], right_on=['passer_id'], how='outer'). \
            fillna(0)
flex['receiver'] = np.where((flex['receiver'] == 0) & (flex['rusher'] != 0),
                            flex['rusher'],
                            np.where((flex['receiver'] == 0) & (flex['passer'] != 0),
                                     flex['passer'], flex['receiver']))
flex['receiver_id'] = np.where((flex['receiver_id'] == 0) & (flex['rusher_id'] != 0),
                            flex['rusher_id'],
                            np.where((flex['receiver_id'] == 0) & (flex['passer_id'] != 0),
                                     flex['passer_id'], flex['receiver_id']))
flex['posteam_x'] = np.where((flex['posteam_x'] == 0) & (flex['posteam_y'] != 0),
                            flex['posteam_y'],
                            np.where((flex['posteam_x'] == 0) & (flex['posteam'] != 0),
                                     flex['posteam'], flex['posteam_x']))
flex.drop(['rusher', 'rusher_id', 'passer', 'passer_id', 'posteam', 'posteam_y'], axis=1, inplace=True)
flex = flex.merge(smash_spots[['player', 'player_id', 'posteam', 'ppr25_odds']], 
                  left_on=['receiver_id'], right_on=['player_id'], how='outer')
flex['ppr25_odds'].fillna(0, inplace=True)
flex['ppr_roll_4'] = flex['ppr_roll_4_x'] + flex['ppr_roll_4_y'] + flex['ppr_roll_4']
flex['ppr_preds'] = flex['ppr_rec_preds'] + flex['ppr_rush_preds'] + flex['ppr_pass_preds']
flex['ppr_4_diff'] = flex['ppr_rec_diff_4'] + flex['ppr_rush_diff_4'] + flex['ppr_pass_diff_4']
flex.sort_values('ppr_preds', inplace=True, ascending=False)
flex['player'] = np.where(flex['player'].isnull(), flex['receiver'], flex['player'])
flex['player_id'] = np.where(flex['player_id'].isnull(), flex['receiver_id'], flex['player_id'])
flex['posteam'] = np.where(flex['posteam'].isnull(), flex['posteam_x'], flex['posteam'])
flex.drop(['posteam_x', 'receiver', 'receiver_id'], axis=1, inplace=True)
flex = flex.groupby(['player', 'player_id', 'posteam']).agg({'ppr_rec_preds':'sum',
                                                             'ppr_rec_diff_4':'sum',
                                                             'ppr_rush_preds':'sum',
                                                             'ppr_rush_diff_4':'sum',
                                                             'ppr_pass_preds':'sum',
                                                             'ppr_pass_diff_4':'sum',
                                                             'ppr25_odds':'mean',
                                                             'ppr_preds':'sum',
                                                             'ppr_roll_4':'sum',
                                                             'ppr_4_diff':'sum'}).reset_index()
```

The "flex" dataframe then contains the following:
* Median PPR projections:
  * Passing (ppr_pass_preds)
  * Rushing (ppr_rush_preds)
  * Receiving (ppr_rec_preds)
  * Total (ppr_preds)
* Average PPR scoring over player's previous 4 games:
  * Total (ppr_roll_4)
* Difference between PPR scoring in previous 4 games and median PPR projections:
  * Passing (ppr_pass_diff_4)
  * Rushing (ppr_rush_diff_4)
  * Receiving (ppr_rec_diff_4)
  * Total (ppr_4_diff)
* Probability of gaining 25+ PPR points:
  * Total (ppr25_odds)
