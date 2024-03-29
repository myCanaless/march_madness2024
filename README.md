# 2024 March Madness
Creating models to forecast the outcome of the men's 2024 collegiate basketball tournaments.
## Dataset Description
Datasets that was given: 

[Kaggle's - 2024 March Madness](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/overview)
```sh
data_path = '../data'
os.listdir(data_path)
```

```sh
['2024_tourney_seeds.csv',
 'Cities.csv',
 'Conferences.csv',
 'MConferenceTourneyGames.csv',
 'MGameCities.csv',
 'MMasseyOrdinals.csv',
 'MNCAATourneyCompactResults.csv',
 'MNCAATourneyDetailedResults.csv',
 'MNCAATourneySeedRoundSlots.csv',
 'MNCAATourneySeeds.csv',
 'MNCAATourneySlots.csv',
 'MRegularSeasonCompactResults.csv',
 'MRegularSeasonDetailedResults.csv',
 'MSeasons.csv',
 'MSecondaryTourneyCompactResults.csv',
 'MSecondaryTourneyTeams.csv',
 'MTeamCoaches.csv',
 'MTeamConferences.csv',
 'MTeams.csv',
 'MTeamSpellings.csv',
 'sample_submission.csv',
 'WGameCities.csv',
 'WNCAATourneyCompactResults.csv',
 'WNCAATourneyDetailedResults.csv',
 'WNCAATourneySeeds.csv',
 'WNCAATourneySlots.csv',
 'WRegularSeasonCompactResults.csv',
 'WRegularSeasonDetailedResults.csv',
 'WSeasons.csv',
 'WTeamConferences.csv',
 'WTeams.csv',
 'WTeamSpellings.csv']
```
Datasets that I would be using: 
* [MRegularSeasonCompactResults](../data/MRegularSeasonCompactResults.csv) - Game results from the regular season

* [MRegularSeasonDetailedResults](../data/MRegularSeasonDetailedResults.csv) - Detailed Results

* [MNCAATourneyCompactResults](../data/MNCAATourneyCompactResults.csv) - Game results from past tournaments 

* [MNCAATourneySeeds](../data/MNCAATourneySeeds.csv) - Seeding of past tournaments

* [2024_tourney_seeds](../data/2024_tourney_seeds.csv) - 2024 seeds 

* [MTeams](../data/MTeams.csv) - Men's Team

```sh
season_results.columns
```
```sh
Index(['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
       'League'],
      dtype='object')
```
- Season - this is the year of the associated entry in MSeasons.csv or WSeasons.csv, namely the year in which the final tournament occurs.

- DayNum - this integer always ranges from 0 to 132, and tells you what day the game was played on. 

- WTeamID - this identifies the id number of the team that won the game, as listed in the "MTeams.csv" or "WTeams.csv" file. 
- WScore - this identifies the number of points scored by the winning team.
- LTeamID - this identifies the id number of the team that lost the game.
- LScore - this identifies the number of points scored by the losing team. Thus you can be confident that WScore will be greater than LScore for all games listed.
- WLoc - this identifies the "location" of the winning team. 
- NumOT - this indicates the number of overtime periods in the game, an integer 0 or higher.
- WFGM - field goals made (by the winning team)
- WFGA - field goals attempted (by the winning team)
- WFGM3 - three pointers made (by the winning team)
- WFGA3 - three pointers attempted (by the winning team)
- WFTM - free throws made (by the winning team)
- WFTA - free throws attempted (by the winning team)
- WOR - offensive rebounds (pulled by the winning team)
- WDR - defensive rebounds (pulled by the winning team)
- WAst - assists (by the winning team)
- WTO - turnovers committed (by the winning team)
- WStl - steals (accomplished by the winning team)
- WBlk - blocks (accomplished by the winning team)
- WPF - personal fouls committed (by the winning team)
- (and then the same set of stats from the perspective of the losing team: LFGM is the number of field goals made by the losing team, and so on up to LPF).

## Analysis
Creating a metrics of game results as "W" or "L" 
```sh
team_results = pd.concat(
[season_results[["Season", "League","DayNum", "WTeamID", "WScore", "LScore"]].assign(GameResult="W")
.rename(columns={"WTeamID": "TeamID", "WScore": "TeamScore", "LScore": "OppScore"}),
    
season_results[["Season", "League","DayNum", "LTeamID", "WScore", "LScore"]].assign(GameResult="L")
.rename(columns={"LTeamID": "TeamID", "LScore": "TeamScore", "WScore": "OppScore"}),]
).reset_index(drop=True)

----------------------------------------------------------------------------------------------

team_results.sample(10, random_state=99)
```
```sh
	Season	League	DayNum	TeamID	TeamScore	OppScore	GameResult
	2019	M	    101	    1410	59	        44	        W
	2014	M	    41	    1461	61	        64	        L
	2014	M	    33	    1420	68	        70	        L
	2017	M	    61	    1223	51	        61	        L
	2021	M	    124	    1455	80	        63	        W
	2023	M	    59	    1158	89	        61	        W
	2015	M	    116	    1156	53	        56	        L
	2010	M	    16	    1428	68	        67	        W
	2009	M	    47	    1253	61	        75	        L
	2003	M	    96	    1225	44	        52	        L
```
Using merge() and groupby() of the TeamIDs and TeamNames

```sh
team_performance = team_results.groupby(['Season','TeamID','GameResult']).size().unstack(fill_value=0)
team_performance = team_performance[['W','L']]
team_performance = team_performance.merge(teams,on='TeamID')
team_performance = team_performance.groupby('TeamName').agg({'W':'sum','L':'sum'}).reset_index()
```

```sh
TeamName	    W	L
Abilene Chr	    136	152
Air Force	    282	352
Akron	        440	235
Alabama         417	270
Alabama A&M	    189	406
...	...	...	... ... ...
Wright St	    382	285
Wyoming	        310	342
Xavier	        473	219
Yale	        323	255
Youngstown St	231	397
```
Top 10 of Most Wins and Losses:

![Top 10 of Most Wins and Lost](<docs/Top10 of Most Wins and Loss.png>)

For 2024 Season, the Top 10 of Most Wins:

![2024 Most wins](<docs/Top 10 Most Wins in 2024.png>)

The Number 1 Chalk Seed for each region: 

```sh
seeds_2024.query('Chalk_Seed == 1')
```
```sh
TeamName	    Chalk_Seed	Region
Connecticut	    1	        W
North Carolina	1	        X
Purdue	        1	        Y
Houston	        1	        Z
```
### Using Advanced Basketball Statistics

- Effective Field Goal Percentage - eFG%
- True Shooting Percentage - TS%

![Effective Field Goal Percentage](<docs/Effective Field Goal Percentage.jpg>)

![True Shooting Percentage](<docs/True Shooting Percentage.jpg>)

Calculating the total points given by Kaggle:

![Total Points](<docs/Total Points Calculations.jpg>)

```sh
season_results[['Season','League','DayNum','WTeamID','LTeamID','ScoreDiff','TS%_Diff','eFG%_Diff']]
```

```sh
Season	League	DayNum	WTeamID	LTeamID	ScoreDiff	TS%_Diff	eFG%_Diff
2003	M	    10	    1104	1328	6	        0.021201	0.057417
2003	M	    10	    1272	1393	7	        0.081874	0.080886
2003	M	    11	    1266	1437	12	        0.148889	0.160841
2003	M	    11	    1296	1457	6	        0.092575	0.084586
2003	M	    11	    1400	1208	6	        0.096529	0.105500
...	...	...	...	...	...	...	...	... ... ... ... ... ... ... ... ... ... 
2024	M	    114	    1454	1237	5	        0.055799	0.022289
2024	M	    114	    1455	1412	8	        0.111259	0.126706
2024	M	    114	    1459	1359	22	        0.158614	0.232512
2024	M	    114	    1462	1177	33	        0.114577	0.143693
2024	M	    114	    1464	1178	14	        0.122873	0.141778

112504 rows × 8 columns
```
Getting the total wins and losses as well as the margin wins and losses by using size() and mean() 

```sh
Season	TeamID	Wins	Losses
2003	1102	12.0	16.0
2003	1103	13.0	14.0
2003	1104	17.0	11.0
2003	1105	7.0	    19.0
2003	1106	13.0	15.0
...	...	...	...	... ... ...
2015	1212	0.0	    27.0
2015	1363	0.0	    28.0
2021	1152	0.0	    9.0
2022	1175	0.0	    25.0
2022	1249	0.0	    27.0
7617 rows × 4 columns
```
Getting the Win/Loss ratios and Overall Score

```sh
Season	TeamID	Wins	AvgWinScoreDiff	Losses	AvgLossScoreDiff	WinLossRatio	AvgScoreDiff
2003	1102	12	    15.583333	    16.0	-11.250000	        0.705882	    0.250000
2003	1103	13	    9.384615	    14.0	-7.500000	        0.866667	    0.629630
2003	1104	17	    13.176471	    11.0	-9.454545	        1.416667	    4.285714
2003	1105	7	    13.000000	    19.0	-11.473684	        0.350000	    -4.884615
2003	1106	13	    10.384615	    15.0	-9.266667	        0.812500	    -0.142857
...	...	...	...	...	...	...	...	... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
2024	1474	10	    8.900000	    18.0	-14.166667	        0.526316	    -5.928571
2024	1475	5	    8.000000	    21.0	-11.952381	        0.227273	    -8.115385
2024	1476	3	    10.666667	    25.0	-16.480000	        0.115385	    -13.571429
2024	1477	7	    6.714286	    18.0	-17.222222	        0.368421	    -10.520000
2024	1478	9	    15.888889	    16.0	-14.250000	        0.529412	    -3.400000
7609 rows × 8 columns
```
Getting the WinLoss Ratio 

```sh
df['WinLossRatioDiff'] = df['WinLossRatio_W'] - df['WinLossRatio_L']
df['AvgScoreDiff'] = df['AvgScoreDiff_W'] - df['AvgScoreDiff_L']
```
Inverting the data frame so the model gets examples to learn from 

```sh
new_df = df.copy()
df_inverted = df.copy()
df_inverted['WinLossRatioDiff'] = -df['WinLossRatioDiff'] 
df_inverted['AvgScoreDiff'] = -df['AvgScoreDiff']  
df_inverted['Outcome'] = 0
new_df['Outcome'] = 1
df = pd.concat([new_df, df_inverted], ignore_index=True)
```


### Features to use for model

```sh
features = [
    'DayNum', 'TS%_Diff', 'eFG%_Diff',
    'WinLossRatio_W', 'WinLossRatio_L',
    'AvgScoreDiff_W', 'AvgScoreDiff_L',
    'WinLossRatioDiff', 'AvgScoreDiff'
    ]
```

## Model

```sh
X = df[['WinLossRatioDiff', 'AvgScoreDiff']]
y = df['Outcome'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- Random Forest Classifier
```sh
Random Forest Accuracy: 0.6674
```
- Logistic Regression
```sh
Logistic Regression Accuracy: 0.7321
```
- XGBoost Classifier
```sh
XGBoost Accuracy: 0.7356
```

## Notes

Did not complete bracket simulation in time but will use this as base for future use.

## Reference  

* [Kaggle - ML March Madness](https://www.kaggle.com/code/robikscube/machine-learning-bracket-gpu-powered)
* [Kaggle - Bracket Simulation](https://www.kaggle.com/code/lennarthaupts/simulate-n-brackets) - for future use 
* [Kaggle - XgBoost](https://www.kaggle.com/code/khoongweihao/ncaam2020-2021-xgboost-lightgbm-k-fold)
* [Kaggle - 538 ratings](https://www.kaggle.com/datasets/raddar/ncaa-men-538-team-ratings/data) - for future use
* [Github - March Madness](https://github.com/harvitronix/kaggle-march-madness-machine-learning)

