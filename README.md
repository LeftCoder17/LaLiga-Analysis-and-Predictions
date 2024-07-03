# LaLiga_Analysis_and_Predictions
## Description
Python program used to analyse the historical results of LaLiga, the spanish football league. Besides, a machine learning model is created that uses the results from the analysis to train a model with Random Forest Classifier. This model will try to predict whether the result of a match will be a victory for the local, the visitor or a tie.

The programs are based in Python version 3.9.13, and below there are the instructions to run it properly with the same libraries and their versions.


## Repository structure
```
LaLiga_Analysis_and_Predictions/
  ├─── analysis/				# Jupyter Notebooks used to explore the data
  │          ...
  ├─── logs/					# Logs of the program are written
  │          ...
  ├─── models/					# The place were trained models are stored
  │          ...
  ├─── quiniela/				# Main Python package
  │          ...
  ├─── reports/					# The place where some XLSX files with ranking are stored and the HTML version of the notebook from the analysis
  │          ...
  ├─── .gitignore
  ├─── cli.py					# Main executable. Entrypoint for CLI
  ├─── laliga.sqlite			# The database
  ├─── README.md
  ├─── requirements.txt			# List of libraries needed to run the project
  └─── settings.py				# General parameters of the program
```


## How to run it
First of all install the dependences (```pip install -r requirements.txt```), then run the cli.py code on the terminal:

```
foo@bar:~$ python cli.py train --training_seasons 2010:2020
Model succesfully trained and saved in ./models/my_quiniela.model
foo@bar:~$ python cli.py predict 2021-2022 1 2
Matchday 2 - LaLiga - Division 1 - Season 2021-2022
=============================================================
       Atlético Madrid      vs         Elche CF         --> Pred: 1 | Prob: 1: 0.43, X: 0.30, 2: 0.27
        Real Sociedad       vs      Rayo Vallecano      --> Pred: 1 | Prob: 1: 0.42, X: 0.30, 2: 0.28
           Athletic         vs        Barcelona         --> Pred: 2 | Prob: 1: 0.34, X: 0.26, 2: 0.40
          CA Osasuna        vs      Celta de Vigo       --> Pred: 1 | Prob: 1: 0.41, X: 0.30, 2: 0.28
           Espanyol         vs        Villarreal        --> Pred: 1 | Prob: 1: 0.44, X: 0.28, 2: 0.29
          Granada CF        vs         Valencia         --> Pred: 1 | Prob: 1: 0.38, X: 0.30, 2: 0.32
           Levante          vs       Real Madrid        --> Pred: 2 | Prob: 1: 0.31, X: 0.29, 2: 0.40
          Real Betis        vs         Cádiz CF         --> Pred: 1 | Prob: 1: 0.46, X: 0.31, 2: 0.24
            Getafe          vs        Sevilla FC        --> Pred: 1 | Prob: 1: 0.38, X: 0.30, 2: 0.31
            Alavés          vs       RCD Mallorca       --> Pred: 1 | Prob: 1: 0.44, X: 0.31, 2: 0.25
```



The first line is used to train the model with the range of seasons desired. It is also possible to write the seasons separated with ',' such as: '2004-2005,2005-2006'
This would be the same as writing '2004:2006' 
Use 'all' to train with all seasons available in database. It is normal if the model needs a few seconds to train.

Once the model is trained, we can try to predict the results from a determined season, division and matchday (write them in this order), as in the example of the second command.

The output of this "predict" command shows, not only the title 
"Matchday 3 - LaLiga ....", but also all the matches, line by line, with its respectively predictions and probabilities. The left team is the local team, while the team at the right is the visitor team. There are 3 possible outcomes for each prediction:
- 1 --> The local team wins
- X --> There is a tie
- 2 --> The visitor team wins

Also, every prediction is followed by the probability of each outcome to happen, in order to help whoever needs this program to decide with more information which outcome is better to bet. Maybe, these probabilities are helpful if are used with knowledge about the current match, as for example, which players are injured. 


## Data
The data is provided as a SQLite3 database that is inside the ZIP file. This database contains the following tables:

   * ```Matches```: All the matches played between seasons 1928-1929 and 2021-2022 with the date and score. Columns are ```season```,	```division```, ```matchday```, ```date```, ```time```, ```home_team```, ```away_team```, ```score```. Have in mind there is no time information for many of them and also that it contains matches still not played from current season.
   * ```Predictions```: The table for you to insert your predictions. It is initially empty. Columns are ```season```,	 ```timestamp```, ```division```, ```matchday```, ```home_team```, ```away_team```, ```prediction```, ```confidence```.

The data source is [Transfermarkt](https://www.transfermarkt.com/), and it was scraped using Python's library BeautifulSoup4.
