import pandas as pd
import numpy as np

class GetFballData():
    
    def get_pbp(self, start, end):
        YEARS = [i for i in range(start, (end+1))]
        
        data = pd.DataFrame()
        
        for i in YEARS:  
            #low_memory=False eliminates a warning
            i_data = pd.read_csv('https://github.com/guga31bb/nflfastR-data/blob/master/data/' \
                                 'play_by_play_' + str(i) + '.csv.gz?raw=True',
                                 compression='gzip', low_memory=False)
        
            #sort=True eliminates a warning and alphabetically sorts columns
            data = data.append(i_data, sort=True)
        
        #Give each row a unique index
        data.reset_index(drop=True, inplace=True)
        
        return data
    
    def get_rosters(self):
        rosters = pd.read_csv('https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/roster-data/roster.csv')
        
        rosters['season_start'] = pd.to_datetime(rosters.loc[:,'team.season'], format='%Y') + pd.DateOffset(days=242)
        rosters['teamPlayers.birthDate'] = pd.to_datetime(rosters.loc[:,'teamPlayers.birthDate'], format='%m/%d/%Y')
        rosters['age'] = ((rosters['season_start'] - rosters['teamPlayers.birthDate']) / np.timedelta64(1, 'Y')) // 1
        names = []
        for f, l in zip(rosters['teamPlayers.firstName'], rosters['teamPlayers.lastName']):
            name = f"{f[0]}.{l}"
            names.append(name)
        rosters['name'] = names
        
        return rosters