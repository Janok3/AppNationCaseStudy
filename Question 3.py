""" 
SQL Querry:
SELECT 
    campaign_id,
    channel,
    impressions,
    clicks,
    conversions,
    spend,
    revenue,
    -- Calculate CTR
    (clicks / impressions) AS ctr,
    -- Calculate Conversion Rate
    (conversions / clicks) AS conversion_rate,
    -- Calculate ROI
    ((revenue - spend) / spend) AS roi,
    -- Determine if profitable
    CASE 
        WHEN revenue > spend THEN TRUE
        ELSE FALSE
    END AS is_profitable
FROM 
    Question3_CampaignData; 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
data = pd.read_csv('Question 3 Campaign Data.csv')

# Calculate features
data['ctr'] = data['clicks'] / data['impressions']
data['conversion_rate'] = data['conversions'] / data['clicks']
data['roi'] = (data['revenue'] - data['spend']) / data['spend']
data['is_profitable'] = data['revenue'] > data['spend']

# Prepare features and target
X = data[['ctr', 'conversion_rate', 'roi']]
y = data['is_profitable']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")