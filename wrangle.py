import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from itertools import product

from acquire import tome_prep
from acquire import acquire_shots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------------------------------------------------
# Processing Functions
# ------------------------------------------------------------------------------------------------

def game_shots(df):
    '''
    Creates a running total of each 3-point attempt and result for each player, for each game they play
    '''
    # Initialize game_id with the first player-game indexed in the dataframe
    game_id = 22100014

    # Need their player_id as well to reset games_counter below
    player_id = 203992

    # Create lists to hold the running counts per player-game - these will become the columns
    count_hold_3pm = []
    count_hold_3pa = []
    # games counter will change only when game changes
    games_counter = []

    # Initialize counters for player-game 3pa and 3pm
    counter_3pm = 0
    counter_3pa = 0
    counter_games_played = 0

    # Loop through each df row.  Since it goes through all of a player's games first, if the game_id is new it is a new player-game
    for row in df.index:
        # If game is new, reset the counters
        if df.game_id[row] != game_id:
            counter_3pm = 0
            counter_3pa = 0
            # If player is new, reset this counter
            if df.player_id[row] != player_id:
                counter_games_played = 0
                player_id = df.player_id[row]
            # Else, player is not new, and they are playing the next game
            else:
                counter_games_played += 1
        # Make or miss the attempt counter goes up one
        counter_3pa += 1
        # If they make the shot, increase the made counter
        if df['shot_result'][row] == 'Made Shot':
            counter_3pm += 1
        # Append the holder lists
        games_counter.append(counter_games_played)
        count_hold_3pm.append(counter_3pm)
        count_hold_3pa.append(counter_3pa)
        # Set the game_id for next loop comparison
        game_id = df.game_id[row]

    df['games_played'] = games_counter
    df['game_3pa'] = count_hold_3pa
    df['game_3pm'] = count_hold_3pm 
    df['game_3miss'] = df['game_3pa'] - df['game_3pm']

    return df

def season_shots(df):
    '''
    Create cumulative 3 point shot information columns for each player-season
    '''
    # Initialize game_id with the first player-game indexed in the dataframe
    player_id = 203992

    # Create lists to hold the running counts per player-game - these will become the columns
    to_date_season_3pm_hold = []
    to_date_season_3pa_hold = []

    # Initialize counters for player-game 3pa and 3pm
    counter_3pm = 0
    counter_3pa = 0

    # Loop through each df row.  Since it goes through all of a player's games first, if the game_id is new it is a new player-game
    for row in df.index:
        # If game is new, reset the counters
        if df.player_id[row] != player_id:
            counter_3pm = 0
            counter_3pa = 0
        counter_3pa += 1
        if df['shot_result'][row] == 'Made Shot':
            counter_3pm += 1
        to_date_season_3pm_hold.append(counter_3pm)
        to_date_season_3pa_hold.append(counter_3pa)
        player_id = df.player_id[row]
        
    df['cum_3pa'] = to_date_season_3pa_hold
    df['cum_3pm'] = to_date_season_3pm_hold 
    df['cum_3miss'] = df['cum_3pa'] - df['cum_3pm']

    return df

def create_metrics(df):
    '''
    Using the new 3 point shot columns create above, we can create player metrics,
     from the basic 3pt pecentage, to more complex metrics.
    '''
    # Simple 3pt percentage
    df['cum_3pct'] = df.cum_3pm/df.cum_3pa

    # Three metric v1 is made^2 / attempts
    df['tm_v1'] = (df.cum_3pm**2 / df.cum_3pa)/(df.games_played + 1)

    # Three metric v2 is made * [(1 - (made/attempts)) / 2 + (made / attempts)
    df['tm_v2'] = (df.cum_3pm * (( 1 - (df.cum_3pm/df.cum_3pa))/2 + (df.cum_3pm/df.cum_3pa))) / (df.games_played + 1)

    # Three metric v3 is made^2/miss
    df['tm_v3'] = (df.cum_3pm**2 / df.cum_3miss) / (df.games_played + 1)
    df['tm_v3'] = np.where(df['tm_v3'] == np.inf, 0, df['tm_v3'])

    return df

def create_distance(df):
    '''
    Brings back the distance column using pythagorean math, and is more accurate than original
    '''
    df['distance'] = ((df.loc_x/10)**2 + (df.loc_y/10)**2)**(1/2)

    return df

def create_game_event(df):
    '''
    Brings back GAME_EVENT_ID as game_event_id, needed for Tableau graphics
    '''
    # Get an original GAME_EVENT_ID column from acquire shots, then reduce the df down to it and join target columns
    df_shots, df_outlier_3pt = acquire_shots()
    game_events = df_shots[['GAME_ID','abs_time','GAME_EVENT_ID']]

    # Merge on the target columns
    df = df.merge(game_events, how = 'inner', left_on = ('game_id','abs_time'), right_on = ('GAME_ID','abs_time'))

    # Drops and renames
    df.drop(columns = 'GAME_ID', inplace = True)
    df.rename(columns = {'GAME_EVENT_ID':'game_event_id'}, inplace = True)

    # Since we have it, returns df_outlier_3pt
    return df, df_outlier_3pt

def encoder(train, validate, test):
    '''
    Encodes categoricals
    '''
    # Encode target for each subset
    train['shot_made_flag'] = np.where(train.shot_result == 'Made Shot',1,0)
    validate['shot_made_flag'] = np.where(validate.shot_result == 'Made Shot',1,0)
    test['shot_made_flag'] = np.where(test.shot_result == 'Made Shot',1,0)

    # Encode these columns
    encode_cols = ['home','zone','shot_type','period']

    train_encoded = pd.get_dummies(train, columns = encode_cols)
    validate_encoded = pd.get_dummies(validate, columns = encode_cols)
    test_encoded = pd.get_dummies(test, columns = encode_cols)

    return train_encoded, validate_encoded, test_encoded

def wrangle_prep():
    '''
    Combines all wrangle functions together.  Done before bivariate EDA and modeling.
    Returns the original dataframe, the outlier 3pt shots (for reference), an unencoded but split X_train_exp
    for analysis, and encoded and scaled X and y for train, validate and test sets.
    '''
    # Acquires the dataset
    df = tome_prep()
    # Prep and modify
    df = game_shots(df)
    df = season_shots(df)
    df = create_metrics(df)
    df = create_distance(df)
    df , df_outlier_3pt = create_game_event(df)
    # Split (stratify on target = 'shot_result')
    train, validate, test = splitter(df, target = 'shot_result')
    # Breaks out X_train, with categoricals unencoded
    X_train_exp = train
    # Encode cats, scale numericals, and then seperate into X and y
    train, validate, test = encoder(train, validate,test)
    train_scaled, validate_scaled, test_scaled = scaling_minmax(train, validate, test)
    X_train, y_train, X_validate, y_validate, X_test, y_test = seperate_X_y(train_scaled, validate_scaled, test_scaled) 

    return df, df_outlier_3pt, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

# ------------------------------------------------------------------------------------------------
# Supporting Functions
# ------------------------------------------------------------------------------------------------

def scaling_minmax(train, validate, test):

    '''
    This function takes in a data set that is split, makes a copy and uses the min max scaler to scale all three data sets.
    Additionally it adds the columns names on the scaled data and returns trained scaled data, validate scaled data and test scale data.
    '''
    # Columns to scale - only those with values that range greater than 0-10ish
    columns_to_scale = ['abs_time', 'play_time', 'since_rest', 'loc_x', 'loc_y', 'score_margin','points','cum_3pa', 'cum_3pm', 'cum_3miss','distance']
    #copying the dataframes for distinguishing between scaled and unscaled data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # defining the minmax scaler 
    scaler = MinMaxScaler()
    
    #scaling the trained data and giving the scaled data column names 
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    #scaling the validate data and giving the scaled data column names 
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    
    #scaling the test data and giving the scaled data column names 
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    #returns three dataframes; train_scaled, validate_scaled, test_scaled
    return train_scaled, validate_scaled, test_scaled

def seperate_X_y(train_scaled, validate_scaled, test_scaled):
    '''
    '''
    
    target = 'shot_result'

    X_drop_columns_list = ['player', 'player_id', 'team', 'team_id', 'game_id','loc_x', 'loc_y','shot_result',
                        'games_played', 'game_3pa', 'game_3pm', 'game_3miss', 'cum_3pa', 'cum_3pm', 'cum_3miss',
                        'game_event_id', 'shot_made_flag','tm_v1','tm_v3']

    X_train = train_scaled.drop(columns = X_drop_columns_list)
    y_train = train_scaled[target]

    X_validate = validate_scaled.drop(columns = X_drop_columns_list)
    y_validate = validate_scaled[target]

    X_test = test_scaled.drop(columns = X_drop_columns_list)
    y_test = test_scaled[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def splitter(df, target = 'None', train_split_1 = .8, train_split_2 = .7, random_state = 123):
    '''
    Splits a dataset into train, validate and test dataframes.
    Optional target, with default splits of 56% 'Train' (80% * 70%), 20% 'Test', 24% Validate (80% * 30%)
    Default random seed/state of 123
    '''
    if target == 'None':
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state)
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state)
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        print('You did not stratify.  If looking to stratify, ensure to add argument: "target = variable to stratify on".')
        return train, validate, test
    else: 
        train, test = train_test_split(df, train_size = train_split_1, random_state = random_state, stratify = df[target])
        train, validate = train_test_split(train, train_size = train_split_2, random_state = random_state, stratify = train[target])
        print(f'Train = {train.shape[0]} rows ({100*(train_split_1*train_split_2):.1f}%) | Validate = {validate.shape[0]} rows ({100*(train_split_1*(1-train_split_2)):.1f}%) | Test = {test.shape[0]} rows ({100*(1-train_split_1):.1f}%)')
        return train, validate, test  