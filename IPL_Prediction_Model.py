import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')



# Load 2023 dataset
def load_data():
    file_path1 = "/Users/aryanbansal/Downloads/archive-2/IPL23dataset.csv"
    df1 = pd.read_csv(file_path1)
# Add a year column
    df1['Year'] = 2023
    
# Load 2024 dataset
    file_path2 = "/Users/aryanbansal/Downloads/archive-2/ipl_complete_data_2024.csv"
    df2 = pd.read_csv(file_path2)
# Rename columns for consistency
    df2 = df2.rename(columns={
        'Team 1': 'Team1',
        'Team 2': 'Team2',
        'winner': 'Winner'
    })
# Add a year column
    df2['Year'] = 2024
    
# Load 2025 dataset
    file_path3 = "/Users/aryanbansal/Downloads/archive-2/ipl_2025_latest_data.csv" 
    df3 = pd.read_csv(file_path3)
# Rename columns for consistency
    df3 = df3.rename(columns={
        'Team 1': 'Team1',
        'Team 2': 'Team2',
        'winner': 'Winner'
    })
# Add a year column
    df3['Year'] = 2025
    
    return df1, df2, df3






# Normalize team names to handle inconsistencies
def normalize_team_names(df):
    # Create a mapping of team names
    team_mapping = {
        'MUMBAI INDIANS': 'Mumbai Indians',
        'CHENNAI SUPER KINGS': 'Chennai Super Kings',
        'ROYAL CHALLENGERS BENGALURU': 'Royal Challengers Bengaluru',
        'ROYAL CHALLENGERS BANGALORE': 'Royal Challengers Bengaluru',
        'GUJARAT TITANS': 'Gujarat Titans',
        'LUCKNOW SUPER GIANTS': 'Lucknow Super Giants',
        'KOLKATA KNIGHT RIDERS': 'Kolkata Knight Riders',
        'PUNJAB KINGS': 'Punjab Kings',
        'RAJASTHAN ROYALS': 'Rajasthan Royals',
        'DELHI CAPITALS': 'Delhi Capitals',
        'SUNRISERS HYDERABAD': 'Sunrisers Hyderabad'
    }
    
    # Function to normalize names
    def normalize_name(name):
        if isinstance(name, str):
            # Check for exact match in keys
            if name in team_mapping:
                return team_mapping[name]
            # Check for case-insensitive match
            for key, value in team_mapping.items():
                if name.upper() == key.upper():
                    return value
        return name
    
    # Apply normalization to team columns
    if 'Team1' in df.columns:
        df['Team1'] = df['Team1'].apply(normalize_name)
    if 'Team2' in df.columns:
        df['Team2'] = df['Team2'].apply(normalize_name)
    if 'Winner' in df.columns:
        df['Winner'] = df['Winner'].apply(normalize_name)
    if 'Winning Team' in df.columns:
        df['Winner'] = df['Winning Team'].apply(normalize_name)
    
    return df

# Process datasets and prepare features
def prepare_features(df1, df2, df3):
    # Normalize team names
    df1 = normalize_team_names(df1)
    df2 = normalize_team_names(df2)
    df3 = normalize_team_names(df3)
    
    # Add target variable to each dataset
    if 'Winner' not in df1.columns and 'Winning Team' in df1.columns:
        df1['Winner'] = df1['Winning Team']
    df1['Team1_Won'] = np.where(df1['Team1'] == df1['Winner'], 1, 0)
    
    df2['Team1_Won'] = np.where(df2['Team1'] == df2['Winner'], 1, 0)
    df3['Team1_Won'] = np.where(df3['Team1'] == df3['Winner'], 1, 0)
    
    # Combine datasets
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Calculate team win rates
    team_stats = {}
    for _, match in combined_df.iterrows():
        team1 = match['Team1']
        team2 = match['Team2']
        
        # Initialize team records if not exist
        if team1 not in team_stats:
            team_stats[team1] = {'matches': 0, 'wins': 0}
        if team2 not in team_stats:
            team_stats[team2] = {'matches': 0, 'wins': 0}
        
        # Update counts
        team_stats[team1]['matches'] += 1
        team_stats[team2]['matches'] += 1
        
        # Update wins based on winner
        if 'Team1_Won' in match:
            if match['Team1_Won'] == 1:
                team_stats[team1]['wins'] += 1
            else:
                team_stats[team2]['wins'] += 1
    
    # Calculate win rates
    for team in team_stats:
        if team_stats[team]['matches'] > 0:
            team_stats[team]['win_rate'] = team_stats[team]['wins'] / team_stats[team]['matches']
        else:
            team_stats[team]['win_rate'] = 0.5
    
    # Create feature vectors
    features = []
    for _, match in combined_df.iterrows():
        team1 = match['Team1']
        team2 = match['Team2']
        
        # Get team stats (default to 0.5 if not found)
        team1_win_rate = team_stats.get(team1, {'win_rate': 0.5})['win_rate']
        team2_win_rate = team_stats.get(team2, {'win_rate': 0.5})['win_rate']
        
        # Feature vector
        feature_vector = {
            'team1': team1,
            'team2': team2,
            'venue': match.get('Venue', 'Unknown'),
            'team1_win_rate': team1_win_rate,
            'team2_win_rate': team2_win_rate,
            'win_rate_diff': team1_win_rate - team2_win_rate,
            'team1_is_home': 1 if 'Venue' in match and team1 in match['Venue'] else 0
        }
        
        # Add target if available
        if 'Team1_Won' in match:
            feature_vector['target'] = match['Team1_Won']
        
        features.append(feature_vector)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    # Convert categorical variables to numeric using one-hot encoding
    features_df = pd.get_dummies(features_df, columns=['team1', 'team2', 'venue'], drop_first=True)
    
    return features_df, team_stats

# Train the prediction model
def train_model(features_df):
    # Split features and target
    if 'target' not in features_df.columns:
        print("Error: Target variable not found")
        return None, None, None
    
    X = features_df.drop('target', axis=1)
    y = features_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, scaler, X.columns.tolist()

# Function to predict match outcome
def predict_match(team1, team2, venue, model, scaler, columns, team_stats):
    # Create a feature vector for the new match
    team1_win_rate = team_stats.get(team1, {'win_rate': 0.5})['win_rate']
    team2_win_rate = team_stats.get(team2, {'win_rate': 0.5})['win_rate']
    
    # Basic features
    new_match = {
        'team1_win_rate': team1_win_rate,
        'team2_win_rate': team2_win_rate,
        'win_rate_diff': team1_win_rate - team2_win_rate,
        'team1_is_home': 1 if team1 in venue else 0
    }
    
    # Create one-hot encoded columns
    for col in columns:
        if col.startswith('team1_') and col[6:] == team1:
            new_match[col] = 1
        elif col.startswith('team2_') and col[6:] == team2:
            new_match[col] = 1
        elif col.startswith('venue_') and col[6:] in venue:
            new_match[col] = 1
        elif col not in new_match:
            new_match[col] = 0
    
    # Create DataFrame and ensure all columns exist
    new_df = pd.DataFrame([new_match])
    for col in columns:
        if col not in new_df.columns:
            new_df[col] = 0
            
    # Reorder columns to match training data
    new_df = new_df[columns]
    
    # Scale features
    new_df_scaled = scaler.transform(new_df)
    
    # Predict
    pred_proba = model.predict_proba(new_df_scaled)[0]
    
    # Return prediction
    if pred_proba[1] > 0.5:
        return {
            'predicted_winner': team1,
            'win_probability': pred_proba[1],
            'team1_win_prob': pred_proba[1],
            'team2_win_prob': pred_proba[0]
        }
    else:
        return {
            'predicted_winner': team2,
            'win_probability': pred_proba[0],
            'team1_win_prob': pred_proba[1],
            'team2_win_prob': pred_proba[0]
        }

# Main function to build and run the model
def main():
    # Load data
    print("Loading data...")
    df1, df2, df3 = load_data()
    
    # Prepare features
    print("Preparing features...")
    features_df, team_stats = prepare_features(df1, df2, df3)
    
    # Train model
    print("Training model...")
    model, scaler, columns = train_model(features_df)
    
    if model is None:
        print("Error: Model training failed")
        return
    
    # Save model components
    model_components = {
        'model': model,
        'scaler': scaler,
        'columns': columns,
        'team_stats': team_stats
    }
    
    model_path = "/Users/aryanbansal/Desktop/Ipl_Mode/ipl_model.pkl"  
    with open('ipl_model.pkl', 'wb') as f:
        pickle.dump(model_components, f)
    
    print("Model saved to ipl_model.pkl")
    
    # Example prediction
    team1 = "Mumbai Indians"
    team2 = "Chennai Super Kings"
    venue = "Wankhede Stadium"
    
    prediction = predict_match(team1, team2, venue, model, scaler, columns, team_stats)
    
    print(f"\nPrediction for {team1} vs {team2} at {venue}:")
    print(f"Predicted winner: {prediction['predicted_winner']}")
    print(f"Win probability: {prediction['win_probability']:.2f}")
    print(f"{team1} win probability: {prediction['team1_win_prob']:.2f}")
    print(f"{team2} win probability: {prediction['team2_win_prob']:.2f}")

if __name__ == "__main__":
    main()



