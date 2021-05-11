import pandas as pd
import numpy as np
import requests
import re

from bs4 import BeautifulSoup

class ScrapeDK():
    def soup_setup(self, url):
        
        response = requests.get(url)
        print(response)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        return soup
    
    def nfl(self):
    
        soup_today = self.soup_setup('https://sportsbook.draftkings.com/leagues/football/3?category=game-lines&subcategory=game')
        
        games = {}
        for i in range(len(soup_today.findAll('span', {'class':'event-cell__name'}))):
            tm = soup_today.findAll('span', {'class':'event-cell__name'})[i]
            dkRe = re.search('>.+<', str(tm))
            tm = dkRe.group().replace('>', '').replace('<', '')
            ny_la = ['NY Giants', 'NY Jets', 'LA Chargers']
            if tm in ny_la:
                tm = tm[:4]
                tm = tm.replace(' ', '')
            else:
                tm = tm.split(sep=' ')[0]
            
            games[i] = [tm]
            
        x = 0
        y = 0
        for i in range(len(soup_today.findAll('span', {'class':'sportsbook-outcome-cell__line'}))):
            if i % 2 != 0:
                sp1 = soup_today.findAll('span', {'class':'sportsbook-outcome-cell__line'})[i-1]
                dkRe = re.search('>.+<', str(sp1))
                sp1 = float(dkRe.group().replace('>', '').replace('<', '').replace('+', ''))
                
                sp2 = soup_today.findAll('span', {'class':'sportsbook-outcome-cell__line'})[i]
                dkRe = re.search('>.+<', str(sp2))
                sp2 = float(dkRe.group().replace('>', '').replace('<', '').replace('+', ''))
                
                if (sp1 < 0) | (sp2 < 0):    
                    games[x].append(sp1)
                    games[x+1].append(sp2)
                    x += 2
                    
                else:    
                    games[y].append(sp1)
                    games[y+1].append(sp2)
                    games[y].append(0)
                    games[y+1].append(1)
                    games[y].append(games[y+1][0])
                    games[y+1].append(games[y][0])
                    y += 2
                    
        for i in range(len(games)):
            new_key = games[i][0]
            games[new_key] = games.pop(i)
            games[new_key].pop(0)
        
        tm_list = []
        for i in games:
            tm_list.append(i)
        for x in tm_list:
            if len(games[x]) != 4:
                games.pop(x)
        
        return games
    
    def epl(self, games=1):
        
        soup_today = self.soup_setup('https://sportsbook.draftkings.com/leagues/soccer/53591936?category=game-lines&subcategory=money-line-(regular-time)')
        
        tms = []
        odds = []
        for i in range(3*games):
            dkRe = re.search('>.+<', str(soup_today.findAll('span', {'class':'sportsbook-outcome-cell__label'})[i]))
            sauce = dkRe.group().replace('>', '').replace('<', '')
            tms.append(sauce)
            dkRe = re.search('>.+<', str(soup_today.findAll('span', {'class':'sportsbook-odds american default-color'})[i]))
            brisket = float(dkRe.group().replace('>', '').replace('<', '').replace('+', ''))
            odds.append(brisket)
        
        tm_names = {'Manchester United':'Manchester Utd', 'Sheffield United':'Sheffield Utd',
                    'Newcastle':'Newcastle Utd', 'West Bromwich': 'West Brom'}
        tms = [tm_names[i] if i in tm_names else i for i in tms]
        odds = [round(1+(i/100),2) if i > 0 else round(1-(100/i),2) for i in odds]
        
        ale = []
        for i in range(0, len(tms), 3):
            ale.append([tms[i], tms[i+2], odds[i], odds[i+1], odds[i+2]])
            
        return ale
    
    def nba(self, games=None):
        
        soup_today = self.soup_setup('https://sportsbook.draftkings.com/leagues/basketball/103?category=game-lines&subcategory=game')
        
        tms = []
        odds = []
        spreads = []
        totals = []
        if games:
            x = games*2
            y = games*4
        else:
            x = len(soup_today.findAll('span', {'class':'event-cell__name'}))
            y = len(soup_today.findAll('span', {'class':'sportsbook-outcome-cell__line'}))
        for i in range(x):
            dkRe = re.search('>.+<', str(soup_today.findAll('span', {'class':'event-cell__name'})[i]))
            salmon = dkRe.group().replace('>', '').replace('<', '')
            salmon = salmon.replace(' ', '')[:3]
            if salmon == 'CHA':
                salmon = 'CHO'
            tms.append(salmon)
        for i in range(y):
            dkRe = re.search('>.+<', str(soup_today.findAll('span', {'class':'sportsbook-outcome-cell__line'})[i]))
            tuna = float(dkRe.group().replace('>', '').replace('<', '').replace('+', ''))
            if tuna < 100:
                spreads.append(tuna)
            else:
                totals.append(tuna) 
        for i in range(len(spreads)):
            if i % 2 == 0:
                odds.append([spreads[i], totals[i], 0])
            else:
                odds.append([spreads[i], totals[i], 1])

        trout = {}
        for x in range(len(tms)):
            trout[tms[x]] = odds[x]
            
        return trout
        