# Annapurti Grain Consumption Forecasting - Complete Guide

## Table of Contents
1. [What This System Does](#1-what-this-system-does)
2. [The Complete Data Flow](#2-the-complete-data-flow)
3. [Input Data Format](#3-input-data-format)
4. [Output Data Format](#4-output-data-format)
5. [Libraries Used and Why](#5-libraries-used-and-why)
6. [Step-by-Step Algorithm Explanation](#6-step-by-step-algorithm-explanation)
7. [How to Use with New Data](#7-how-to-use-with-new-data)
8. [Understanding the Predictions](#8-understanding-the-predictions)

---

## 1. What This System Does

### The Problem
You have a grain distribution system (like a ration shop) where:
- Beneficiaries have **smart cards**
- They visit **Fair Price Shops (FPS)** to collect grain
- Different **commodities** (rice, wheat, sugar) are distributed
- You need to **predict how much grain will be needed NEXT MONTH**

### The Solution
This system uses **Machine Learning** to:
1. Learn patterns from historical transactions
2. Predict future grain consumption
3. Support predictions at different levels (per card, per shop, per commodity)

### Why This Matters
- **Supply Planning**: Know how much grain to stock at each FPS
- **Logistics**: Plan transportation and storage
- **Budget**: Forecast expenses

---

## 2. The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT DATA                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Transaction Data    2. Member Data       3. Card Data               │
│  (Who bought what,      (Family members,     (Card type,                │
│   when, how much)        age, gender)         household size)           │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: DATA LOADING                              │
│  • Read CSV files                                                        │
│  • Parse dates                                                           │
│  • Clean data                                                            │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 2: CREATE MONTHLY PANEL                         │
│  Transform: Transaction rows → One row per (card × month)               │
│                                                                          │
│  Before:                         After:                                  │
│  card_no  | date    | qty       card_no | month   | total_qty           │
│  C001     | Jan-15  | 10   →    C001    | Jan-25  | 35                  │
│  C001     | Jan-20  | 25        C001    | Feb-25  | 20                  │
│  C001     | Feb-10  | 20        C001    | Mar-25  | 0 (no transaction)  │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: FEATURE ENGINEERING                           │
│  Create features (input variables) for prediction:                       │
│                                                                          │
│  • Lag features: What was consumption 1, 2, 3 months ago?               │
│  • Rolling averages: What's the average over last 3/6 months?           │
│  • Temporal features: Which month? Festival season?                      │
│  • Demographics: How many family members? Ages?                          │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 4: TRAIN ML MODEL                              │
│  • Split data: Past months for training, latest for testing            │
│  • Train LightGBM (Gradient Boosting) model                             │
│  • Evaluate accuracy                                                     │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      STEP 5: PREDICT NEXT MONTH                          │
│  Use trained model to predict consumption for next month                │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                         │
│  CSV file with predictions:                                              │
│  card_no      | predicted_qty_kg | prediction_month                     │
│  02061012291  | 35.5             | 2026-04                              │
│  07020211490  | 22.3             | 2026-04                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Input Data Format

### File 1: Transaction Data (`Annpurti Txn Data.csv`)

This is your MAIN data file containing all grain distribution records.

| Column | Meaning | Example |
|--------|---------|---------|
| `TXN_NO` | Unique transaction ID | 41300250908153243 |
| `card_no` | Smart card number | 02061012291 |
| `MEMBER_ID` | Which family member collected | 3 |
| `MEMBERNAME_ENG` | Member name | RANJITA NATH |
| `TXNDATETIME2` | When transaction happened | 2025-09-08 15:32:43 |
| `home_fps` | Home Fair Price Shop ID | 6257 |
| `card_category` | Card type (A/PH/S) | A |
| `ALLOTMENT_MONTH` | **Target month** for grain | 2025-10-01 |
| `COMMODITY_CODE` | Type of grain (41=Rice, 43=Wheat) | 41 |
| `qty` | Quantity in KG | 10.00 |

**Important**: `ALLOTMENT_MONTH` is the month FOR WHICH the grain is allocated, not when it was collected. A person can collect multiple months' worth in one visit.

### File 2: Member Details (`Annpurti Txn Member Details Data.csv`)

Family composition for each card.

| Column | Meaning | Example |
|--------|---------|---------|
| `card_no` | Smart card number | 12061410125 |
| `MEMBERNAME_ENG` | Member name | TARA BEWA |
| `slno` | Serial number in family | 1 |
| `age` | Age in years | 85 |
| `gender` | Male/Female | Female |
| `relation` | Relation to head | Head |
| `member_id` | Unique member ID | 5131872 |

### File 3: Card Details (`Annpurti Txn Card Details Data.csv`)

One row per smart card with metadata.

| Column | Meaning | Example |
|--------|---------|---------|
| `card_no` | Smart card number | 19135420241 |
| `FAMILY_HEAD_ENG` | Head of family name | S MINAKHSI |
| `name` | FPS name | 1913P334-DEBA PRASAD SAHOO |
| `CARD_TYPE` | Priority category | PH |
| `NO_OF_MEMBERS` | Household size | 2 |

**Card Types**:
- `A` = Antyodaya (Poorest of poor) - Gets ~35 kg/month
- `PH` = Priority Household - Gets ~15-20 kg/month
- `S` = State Priority - Gets ~15 kg/month

---

## 4. Output Data Format

### Card-Level Predictions (`predictions_card_level.csv`)

```csv
card_no,predicted_qty_kg,prediction_month
1010910154,0.28670738467547224,2026-04
1010910280,10.631309114627129,2026-04
1021310207,34.23230292521974,2026-04
```

| Column | Meaning |
|--------|---------|
| `card_no` | The smart card number |
| `predicted_qty_kg` | Predicted grain consumption in KG for next month |
| `prediction_month` | Which month this prediction is for |

### FPS-Level Predictions (`predictions_fps_level.csv`)

```csv
home_fps,predicted_qty_kg,prediction_month
1120,793.858514,2026-04
39394,765.142974,2026-04
```

This tells you: "FPS shop #1120 will need approximately 794 kg of grain in April 2026"

---

## 5. Libraries Used and Why

### pandas (Data Manipulation)
```python
import pandas as pd
```
**What it does**: Handles tabular data (like Excel spreadsheets)
- Read CSV files: `pd.read_csv('file.csv')`
- Filter rows: `df[df['qty'] > 0]`
- Group and aggregate: `df.groupby('card_no')['qty'].sum()`
- Join tables: `df1.merge(df2, on='card_no')`

**Example in our code**:
```python
# Read transaction data
df = pd.read_csv('Annpurti Txn Data.csv')

# Calculate total quantity per card per month
monthly = df.groupby(['card_no', 'allotment_yearmonth']).agg(
    total_qty=('qty', 'sum')
)
```

### numpy (Numerical Computing)
```python
import numpy as np
```
**What it does**: Fast mathematical operations on arrays
- `np.mean()` - Calculate average
- `np.sin()`, `np.cos()` - Trigonometric functions (for encoding months cyclically)
- `np.maximum(predictions, 0)` - Ensure no negative predictions

**Example in our code**:
```python
# Encode month as cyclical feature (so January is close to December)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

### scikit-learn (Machine Learning Basics)
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
```
**What it does**: Standard ML algorithms and evaluation metrics

**Metrics we use**:
- `MAE` (Mean Absolute Error): Average error in KG
- `RMSE` (Root Mean Square Error): Penalizes large errors more
- `R²` (R-squared): How much variance is explained (1.0 = perfect)

### LightGBM (Gradient Boosting Model)
```python
from lightgbm import LGBMRegressor
```
**What it does**: Fast, accurate prediction model

**Why LightGBM?**
1. Handles missing values automatically
2. Works well with mixed feature types
3. Fast training
4. Good accuracy for tabular data

---

## 6. Step-by-Step Algorithm Explanation

### Step 1: Data Loading

```python
def load_transaction_data():
    # Read CSV file
    df = pd.read_csv('Annpurti Txn Data.csv')

    # Convert text dates to actual datetime objects
    df['ALLOTMENT_MONTH'] = pd.to_datetime(df['ALLOTMENT_MONTH'])

    # Extract useful parts
    df['allotment_year'] = df['ALLOTMENT_MONTH'].dt.year   # 2025
    df['allotment_month'] = df['ALLOTMENT_MONTH'].dt.month # 1-12

    return df
```

**What happens**: Raw CSV text → Structured DataFrame with proper data types

### Step 2: Create Monthly Panel

**Problem**: Transaction data has multiple rows per card per month. We need ONE row per card per month.

```
BEFORE (Transaction level):
card_no     | ALLOTMENT_MONTH | qty
C001        | 2025-01         | 10
C001        | 2025-01         | 25   ← Same card, same month, 2 transactions
C001        | 2025-02         | 20
C002        | 2025-01         | 15

AFTER (Panel format):
card_no     | allotment_yearmonth | total_qty
C001        | 2025-01             | 35    ← Summed
C001        | 2025-02             | 20
C001        | 2025-03             | 0     ← No transaction, filled with 0
C002        | 2025-01             | 15
C002        | 2025-02             | 0     ← No transaction, filled with 0
```

```python
def create_monthly_consumption_panel(txn_df, group_cols=['card_no']):
    # Sum quantity per card per month
    monthly = txn_df.groupby(['card_no', 'allotment_yearmonth']).agg(
        total_qty=('qty', 'sum')
    )

    # Create ALL possible (card × month) combinations
    # Fill missing with 0 (no consumption that month)
    # This is important! ML needs complete data

    return panel
```

### Step 3: Feature Engineering

**What are features?** Input variables that help predict the target.

#### Lag Features (What happened in the past?)

```python
# lag_1 = consumption 1 month ago
# lag_2 = consumption 2 months ago
# etc.

df['lag_1'] = df.groupby('card_no')['total_qty'].shift(1)
```

**Example**:
```
card_no | month   | total_qty | lag_1 | lag_2
C001    | Jan-25  | 35        | NaN   | NaN    ← No previous data
C001    | Feb-25  | 20        | 35    | NaN
C001    | Mar-25  | 30        | 20    | 35
C001    | Apr-25  | ?         | 30    | 20     ← Use this to predict
```

**Why?** If someone consumed 30 kg last month, they'll likely consume around 30 kg this month too.

#### Rolling Statistics (What's the trend?)

```python
# Average of last 3 months
df['rolling_mean_3m'] = df.groupby('card_no')['total_qty'].transform(
    lambda x: x.shift(1).rolling(3).mean()
)
```

**Example**:
```
month   | total_qty | rolling_mean_3m
Jan-25  | 35        | NaN
Feb-25  | 20        | NaN
Mar-25  | 30        | NaN
Apr-25  | 25        | 28.3  ← (35+20+30)/3
May-25  | 28        | 25.0  ← (20+30+25)/3
```

**Why?** Smooths out random fluctuations, shows the underlying trend.

#### Temporal Features (When is it?)

```python
# Which month (1-12)?
df['month'] = df['allotment_yearmonth'].apply(lambda x: x.month)

# Cyclical encoding (so December and January are "close")
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Is it festival season? (October-November)
df['is_festival_season'] = df['month'].isin([10, 11]).astype(int)
```

**Why cyclical encoding?**
Without it: January=1, December=12 → looks far apart
With sin/cos: Both have similar values → model knows they're adjacent

#### Demographic Features (Who is the household?)

```python
# From member data, calculate per card:
member_agg = member_df.groupby('card_no').agg(
    num_members=('member_id', 'count'),      # How many people?
    avg_age=('age', 'mean'),                 # Average age
    num_children=('age', lambda x: (x < 18).sum()),  # Kids count
    num_elderly=('age', lambda x: (x > 60).sum())    # Elderly count
)
```

**Why?** Larger families consume more. Elderly may have different patterns.

### Step 4: Train the Model

#### What is Gradient Boosting?

Imagine you're trying to predict grain consumption:

1. **First tree** makes a rough prediction (maybe just the average)
2. **Second tree** looks at the ERRORS from tree 1 and tries to fix them
3. **Third tree** fixes errors from tree 1+2
4. ... continue for 100 trees

```
Actual: 35 kg

Tree 1 predicts: 25 kg → Error = 10 kg
Tree 2 fixes:    +8 kg → Error = 2 kg
Tree 3 fixes:    +1.5 kg → Error = 0.5 kg
...
Final prediction: 34.5 kg (very close!)
```

```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=100,      # Number of trees
    learning_rate=0.1,     # How much each tree contributes
    max_depth=6,           # How complex each tree can be
)

# X = features (lag_1, lag_2, num_members, etc.)
# y = target (total_qty)
model.fit(X_train, y_train)
```

#### Train/Test Split

```
Timeline: Jan-25 ... Feb-26 ... Mar-26

Training data: Jan-25 to Feb-26 (learn patterns)
Test data: Mar-26 (evaluate accuracy)
Prediction: Apr-26 (what we actually want)
```

**Why not use all data for training?**
We need to test on data the model hasn't seen to know if it will work in the future.

### Step 5: Generate Predictions

```python
# Get latest month's data for each card
latest_data = panel_df[panel_df['allotment_yearmonth'] == max_month]

# Extract features
X_pred = latest_data[feature_columns]

# Predict
predictions = model.predict(X_pred)

# No negative consumption allowed
predictions = np.maximum(predictions, 0)
```

---

## 7. How to Use with New Data

### Option A: Update Existing Data

1. **Add new transactions** to `Annpurti Txn Data.csv`
   - Keep the same column format
   - Add rows for new months

2. **Run the forecaster**:
```python
from grain_forecaster import AnnapurtiForecaster

forecaster = AnnapurtiForecaster()
forecaster.load_data()
forecaster.prepare_features(aggregation=['card_no'])
forecaster.train()
predictions = forecaster.predict_next_month()
predictions.to_csv('new_predictions.csv', index=False)
```

### Option B: Use Different Data Files

```python
from grain_forecaster import AnnapurtiForecaster
from pathlib import Path

# Point to your data directory
forecaster = AnnapurtiForecaster(data_dir=Path("path/to/your/data"))

# Files expected in that directory:
# - Annpurti Txn Data.csv
# - Annpurti Txn Member Details Data.csv
# - Annpurti Txn Card Details Data.csv

forecaster.load_data()
forecaster.prepare_features(aggregation=['card_no'])
forecaster.train()
predictions = forecaster.predict_next_month()
```

### Option C: Different Aggregation Levels

```python
# Per Fair Price Shop
forecaster.prepare_features(aggregation=['home_fps'])

# Per Commodity
forecaster.prepare_features(aggregation=['COMMODITY_CODE'])

# Per FPS × Commodity (most detailed for supply planning)
forecaster.prepare_features(aggregation=['home_fps', 'COMMODITY_CODE'])
```

---

## 8. Understanding the Predictions

### Output File Structure

```csv
card_no,predicted_qty_kg,prediction_month
1010910154,0.28670738467547224,2026-04
1021310207,34.23230292521974,2026-04
```

**Reading this**:
- Card `1010910154` is predicted to consume **0.29 kg** in April 2026 (probably inactive)
- Card `1021310207` is predicted to consume **34.23 kg** in April 2026 (likely Antyodaya card)

### Model Accuracy Metrics

When you train, you see:
```
Model Performance:
  MAE:  1.98 kg      ← Average error is about 2 kg
  RMSE: 4.03 kg      ← Typical error range
  MAPE: 20.3%        ← On average, predictions are 20% off
  R²:   0.854        ← Model explains 85% of the variation
```

**Is 85% R² good?**
Yes! For demand forecasting, anything above 70% is considered good.

### Feature Importance

```
Feature              | Importance
---------------------|------------
avg_age              | 347    ← Household age matters most!
month_sin            | 293    ← Seasonality is important
month_cos            | 283
cumulative_avg_qty   | 260    ← Historical average matters
lag_1                | 141    ← Last month's consumption
```

**Interpretation**: The model found that:
1. Household age composition is the strongest predictor
2. Which month it is (seasonality) matters a lot
3. Historical consumption patterns are important
4. Last month's consumption is a good indicator

---

## Quick Start Commands

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm

# Run full forecasting
cd d:\Projects\Annapurti_Prediction_ALG
python grain_forecaster.py

# Or use interactively
python
>>> from grain_forecaster import AnnapurtiForecaster
>>> f = AnnapurtiForecaster()
>>> f.load_data()
>>> f.prepare_features(aggregation=['card_no'])
>>> f.train()
>>> predictions = f.predict_next_month()
>>> predictions.to_csv('my_predictions.csv')
```

---

## Summary

| Step | What Happens | Output |
|------|--------------|--------|
| 1. Load Data | Read 3 CSV files | DataFrames in memory |
| 2. Create Panel | Transform to (entity × month) format | One row per card per month |
| 3. Engineer Features | Create lag, rolling, temporal, demographic features | 25 input variables |
| 4. Train Model | LightGBM learns patterns | Trained model |
| 5. Predict | Apply model to latest data | CSV with predictions |

The predictions tell you: **"Card X will likely need Y kg of grain in month Z"**

Aggregate to FPS level to know: **"Shop X will need Y kg total in month Z"**
