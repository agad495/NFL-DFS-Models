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
        tm_soup = soup_today.findAll('span', {'class':'event-cell__name'})
        for i in range(len(tm_soup)):
            tm = tm_soup[i].text
            ny_la = ['NY Giants', 'NY Jets', 'LA Chargers']
            if tm in ny_la:
                tm = tm[:4]
                tm = tm.replace(' ', '')
            else:
                tm = tm.split(sep=' ')[0]
            
            games[i] = tm
            
        x = 0
        y = 0
        odds_soup = soup_today.findAll('span', {'class':'sportsbook-outcome-cell__line'})
        for i in range(len(odds_soup)):
            if i % 2 != 0:
                sp1 = float(odds_soup[i-1].text.replace('+', ''))
                
                sp2 = float(odds_soup[i].text.replace('+', ''))
                
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
    
