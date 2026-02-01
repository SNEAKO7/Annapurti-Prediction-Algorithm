"""
Annapurti Grain Consumption Forecaster
=======================================

Production-grade forecasting system for next-month grain consumption prediction.

Problem Formulation:
-------------------
- Target: Total grain quantity (kg) to be distributed in the NEXT month
- Granularity: Monthly (prediction horizon = 1 month ahead)
- Aggregation levels supported:
    * Per smart card (card_no)
    * Per Fair Price Shop (home_fps)
    * Per commodity type (COMMODITY_CODE)
    * Combinations: FPS × Commodity, Card × Commodity, etc.

Data Sources:
-------------
1. Transaction Data - Historical grain distribution records
2. Member Details - Household demographics per card
3. Card Details - Smart card metadata (type, household size)

Key Assumptions:
----------------
1. ALLOTMENT_MONTH represents the target consumption month, not transaction date
2. A card can collect multiple months' worth of grain in a single transaction
3. Quantity is directly proportional to household size and card type entitlements
4. Seasonality patterns exist at monthly level
5. Missing months may indicate: inactive cards, migration, or supply issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("Historical_Data")

TRANSACTION_FILE = "Annpurti Txn Data.csv"
MEMBER_FILE = "Annpurti Txn Member Details Data.csv"
CARD_FILE = "Annpurti Txn Card Details Data.csv"

# Commodity code mapping (derived from data patterns)
COMMODITY_NAMES = {
    41: "Rice",
    43: "Wheat",
    45: "Sugar/Other"
}

# Card category entitlements (kg per member per month - approximate from data)
# These will be refined from actual data patterns
CARD_CATEGORIES = {
    "A": "Antyodaya (Poorest)",
    "PH": "Priority Household",
    "S": "State Priority"
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_transaction_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and parse transaction data with proper date handling."""
    df = pd.read_csv(data_dir / TRANSACTION_FILE)

    # Parse datetime columns
    df['TXNDATETIME2'] = pd.to_datetime(df['TXNDATETIME2'])
    df['ALLOTMENT_MONTH'] = pd.to_datetime(df['ALLOTMENT_MONTH'])

    # Extract useful date features from allotment month
    df['allotment_year'] = df['ALLOTMENT_MONTH'].dt.year
    df['allotment_month'] = df['ALLOTMENT_MONTH'].dt.month
    df['allotment_yearmonth'] = df['ALLOTMENT_MONTH'].dt.to_period('M')

    # Transaction timing features
    df['txn_year'] = df['TXNDATETIME2'].dt.year
    df['txn_month'] = df['TXNDATETIME2'].dt.month
    df['txn_day_of_week'] = df['TXNDATETIME2'].dt.dayofweek

    # Convert qty to numeric
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')

    # Ensure home_fps is string for consistent grouping
    df['home_fps'] = df['home_fps'].astype(str)

    print(f"Loaded {len(df):,} transaction records")
    print(f"Date range: {df['ALLOTMENT_MONTH'].min()} to {df['ALLOTMENT_MONTH'].max()}")
    print(f"Unique cards: {df['card_no'].nunique():,}")

    return df


def load_member_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load member details and aggregate to card level."""
    df = pd.read_csv(data_dir / MEMBER_FILE)

    # Convert age to numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    print(f"Loaded {len(df):,} member records for {df['card_no'].nunique():,} cards")

    return df


def load_card_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load card metadata."""
    df = pd.read_csv(data_dir / CARD_FILE)

    # Convert NO_OF_MEMBERS to numeric
    df['NO_OF_MEMBERS'] = pd.to_numeric(df['NO_OF_MEMBERS'], errors='coerce')

    print(f"Loaded {len(df):,} card records")

    return df


# =============================================================================
# FEATURE ENGINEERING - MEMBER LEVEL AGGREGATIONS
# =============================================================================

def aggregate_member_features(member_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate member-level data to card-level demographic features.

    Features created:
    - num_members: Count of household members
    - avg_age: Average age of household
    - num_children: Members under 18
    - num_adults: Members 18-60
    - num_elderly: Members over 60
    - gender_ratio: Proportion of females
    """
    agg_features = member_df.groupby('card_no').agg(
        num_members=('member_id', 'count'),
        avg_age=('age', 'mean'),
        min_age=('age', 'min'),
        max_age=('age', 'max'),
        num_females=('gender', lambda x: (x == 'Female').sum()),
        num_males=('gender', lambda x: (x == 'Male').sum())
    ).reset_index()

    # Derived features
    agg_features['gender_ratio'] = agg_features['num_females'] / agg_features['num_members']

    # Age group counts (re-calculate from raw data)
    age_groups = member_df.groupby('card_no').apply(
        lambda x: pd.Series({
            'num_children': (x['age'] < 18).sum(),
            'num_adults': ((x['age'] >= 18) & (x['age'] <= 60)).sum(),
            'num_elderly': (x['age'] > 60).sum()
        })
    ).reset_index()

    agg_features = agg_features.merge(age_groups, on='card_no', how='left')

    return agg_features


# =============================================================================
# FEATURE ENGINEERING - TIME SERIES FEATURES
# =============================================================================

def create_monthly_consumption_panel(
    txn_df: pd.DataFrame,
    group_cols: List[str] = ['card_no']
) -> pd.DataFrame:
    """
    Transform transaction data into a monthly panel dataset.

    This is the core transformation for supervised learning:
    - Each row = one entity (card, FPS, etc.) for one month
    - Handles missing months by filling with zeros (no consumption)

    Args:
        txn_df: Transaction DataFrame
        group_cols: Columns to group by (e.g., ['card_no'], ['home_fps', 'COMMODITY_CODE'])

    Returns:
        Panel DataFrame with one row per entity-month combination
    """
    # Aggregate by group columns and allotment month
    agg_cols = group_cols + ['allotment_yearmonth']

    monthly = txn_df.groupby(agg_cols).agg(
        total_qty=('qty', 'sum'),
        num_transactions=('TXN_NO', 'nunique'),
        num_commodities=('COMMODITY_CODE', 'nunique')
    ).reset_index()

    # Get full date range
    all_months = pd.period_range(
        txn_df['allotment_yearmonth'].min(),
        txn_df['allotment_yearmonth'].max(),
        freq='M'
    )

    # Get all unique entities
    unique_entities = txn_df[group_cols].drop_duplicates()

    # Create complete panel (all entities × all months)
    from itertools import product

    entity_tuples = [tuple(x) for x in unique_entities.values]
    full_index = pd.DataFrame(
        list(product(entity_tuples, all_months)),
        columns=['entity', 'allotment_yearmonth']
    )

    # Unpack entity tuple back to columns
    for i, col in enumerate(group_cols):
        full_index[col] = full_index['entity'].apply(lambda x: x[i] if isinstance(x, tuple) else x)
    full_index = full_index.drop('entity', axis=1)

    # Merge with actual data
    panel = full_index.merge(monthly, on=agg_cols, how='left')

    # Fill missing with 0 (no consumption that month)
    panel['total_qty'] = panel['total_qty'].fillna(0)
    panel['num_transactions'] = panel['num_transactions'].fillna(0)
    panel['num_commodities'] = panel['num_commodities'].fillna(0)

    return panel


def add_lag_features(
    panel_df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = 'total_qty',
    lags: List[int] = [1, 2, 3, 6, 12]
) -> pd.DataFrame:
    """
    Add lagged consumption features for time series forecasting.

    Features:
    - lag_N: Consumption N months ago
    - rolling_mean_3m: 3-month rolling average
    - rolling_mean_6m: 6-month rolling average
    - rolling_std_3m: 3-month rolling std (volatility)
    """
    df = panel_df.copy()
    df = df.sort_values(group_cols + ['allotment_yearmonth'])

    # Create lag features
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)

    # Rolling statistics
    for window in [3, 6]:
        df[f'rolling_mean_{window}m'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}m'] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Consumption trend (comparing to same month last year if available)
    df['lag_12'] = df.groupby(group_cols)[target_col].shift(12)
    df['yoy_change'] = df[target_col] - df['lag_12']

    return df


def add_temporal_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features for seasonality modeling."""
    df = panel_df.copy()

    # Extract month and year from period
    df['month'] = df['allotment_yearmonth'].apply(lambda x: x.month)
    df['year'] = df['allotment_yearmonth'].apply(lambda x: x.year)

    # Cyclical encoding for month (captures seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Quarter
    df['quarter'] = (df['month'] - 1) // 3 + 1

    # Is festival season (approximate - Oct-Nov for Diwali/Durga Puja)
    df['is_festival_season'] = df['month'].isin([10, 11]).astype(int)

    # Is lean season (typically June-August in Odisha - monsoon)
    df['is_monsoon'] = df['month'].isin([6, 7, 8]).astype(int)

    return df


# =============================================================================
# FEATURE ENGINEERING - ENTITY BEHAVIORAL FEATURES
# =============================================================================

def add_behavioral_features(
    panel_df: pd.DataFrame,
    txn_df: pd.DataFrame,
    group_cols: List[str]
) -> pd.DataFrame:
    """
    Add behavioral features based on historical patterns.

    Features:
    - active_months: How many months entity has been active
    - first_seen_months_ago: Tenure
    - avg_monthly_consumption: Historical average
    - consumption_consistency: Std/Mean ratio
    - pct_months_active: Proportion of months with non-zero consumption
    """
    df = panel_df.copy()

    # Calculate cumulative features (using only past data to avoid leakage)
    df = df.sort_values(group_cols + ['allotment_yearmonth'])

    # Cumulative count of active months
    df['cumulative_active_months'] = df.groupby(group_cols)['total_qty'].transform(
        lambda x: (x.shift(1) > 0).cumsum()
    )

    # Cumulative average (excluding current month)
    df['cumulative_avg_qty'] = df.groupby(group_cols)['total_qty'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Months since first transaction
    df['months_since_start'] = df.groupby(group_cols).cumcount()

    # Activity rate (proportion of past months with consumption)
    df['activity_rate'] = df['cumulative_active_months'] / (df['months_since_start'] + 1)

    return df


# =============================================================================
# MODEL BUILDING
# =============================================================================

def prepare_training_data(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'total_qty',
    min_history_months: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare train/test split maintaining temporal order.

    Uses the last available month as test set.
    Filters out entities with insufficient history.
    """
    df = panel_df.copy()

    # Remove rows with NaN features (due to lag computation)
    df = df.dropna(subset=feature_cols)

    # Get the latest month for testing
    max_month = df['allotment_yearmonth'].max()

    # Train: all months except the last
    # Test: last month only
    train_mask = df['allotment_yearmonth'] < max_month
    test_mask = df['allotment_yearmonth'] == max_month

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    return X_train, X_test, y_train, y_test


def build_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Build baseline models for comparison.

    Baselines:
    1. Last value (naive): predict last month's consumption
    2. Historical mean: predict average of all past months
    3. Same month last year: seasonal naive
    """
    baselines = {
        'last_value': lambda x: x['lag_1'] if 'lag_1' in x.index else 0,
        'rolling_mean': lambda x: x['rolling_mean_3m'] if 'rolling_mean_3m' in x.index else 0,
    }

    return baselines


def build_ml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'gradient_boosting'
) -> Any:
    """
    Build ML model for consumption prediction.

    Supports:
    - gradient_boosting: LightGBM or XGBoost
    - random_forest: Sklearn RandomForest
    - linear: Ridge regression with regularization
    """
    if model_type == 'gradient_boosting':
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

    elif model_type == 'linear':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit model
    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.

    Metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Square Error
    - MAPE: Mean Absolute Percentage Error (for non-zero values)
    - R2: Coefficient of determination
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    # Clip negative predictions to 0 (consumption can't be negative)
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE only for non-zero actuals
    non_zero_mask = y_test > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
    else:
        mape = np.nan

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

    return metrics


# =============================================================================
# MULTI-LEVEL AGGREGATION FORECASTING
# =============================================================================

class AnnapurtiForecaster:
    """
    Production forecaster supporting multiple aggregation levels.

    Usage:
        forecaster = AnnapurtiForecaster()
        forecaster.load_data()
        forecaster.prepare_features(aggregation=['card_no'])
        forecaster.train()
        predictions = forecaster.predict_next_month()
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.txn_df = None
        self.member_df = None
        self.card_df = None
        self.panel_df = None
        self.model = None
        self.feature_cols = None
        self.aggregation_cols = None

    def load_data(self) -> None:
        """Load all data sources."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        self.txn_df = load_transaction_data(self.data_dir)
        self.member_df = load_member_data(self.data_dir)
        self.card_df = load_card_data(self.data_dir)

        # Create member aggregations
        self.member_agg = aggregate_member_features(self.member_df)

        print(f"\nData loaded successfully!")

    def prepare_features(
        self,
        aggregation: List[str] = ['card_no'],
        include_demographics: bool = True
    ) -> None:
        """
        Prepare feature matrix for the specified aggregation level.

        Args:
            aggregation: Columns to group by. Options:
                - ['card_no'] - per smart card
                - ['home_fps'] - per Fair Price Shop
                - ['COMMODITY_CODE'] - per commodity
                - ['home_fps', 'COMMODITY_CODE'] - FPS × commodity
                - ['card_no', 'COMMODITY_CODE'] - card × commodity
            include_demographics: Whether to include household demographic features
        """
        print("\n" + "=" * 60)
        print(f"PREPARING FEATURES - Aggregation: {aggregation}")
        print("=" * 60)

        self.aggregation_cols = aggregation

        # Step 1: Create monthly panel
        print("\nStep 1: Creating monthly consumption panel...")
        self.panel_df = create_monthly_consumption_panel(self.txn_df, aggregation)
        print(f"  Panel shape: {self.panel_df.shape}")

        # Step 2: Add lag features
        print("\nStep 2: Adding lag features...")
        self.panel_df = add_lag_features(
            self.panel_df,
            aggregation,
            lags=[1, 2, 3, 6, 12]
        )

        # Step 3: Add temporal features
        print("\nStep 3: Adding temporal features...")
        self.panel_df = add_temporal_features(self.panel_df)

        # Step 4: Add behavioral features
        print("\nStep 4: Adding behavioral features...")
        self.panel_df = add_behavioral_features(
            self.panel_df,
            self.txn_df,
            aggregation
        )

        # Step 5: Add demographic features (if card-level aggregation)
        if include_demographics and 'card_no' in aggregation:
            print("\nStep 5: Adding demographic features...")

            # Merge member aggregations
            self.panel_df = self.panel_df.merge(
                self.member_agg,
                on='card_no',
                how='left'
            )

            # Merge card details
            self.panel_df = self.panel_df.merge(
                self.card_df[['card_no', 'CARD_TYPE', 'NO_OF_MEMBERS']],
                on='card_no',
                how='left'
            )

            # Encode card type
            self.panel_df['card_type_encoded'] = self.panel_df['CARD_TYPE'].map(
                {'A': 0, 'PH': 1, 'S': 2}
            ).fillna(1)  # Default to PH

        # Define feature columns
        self.feature_cols = [
            # Lag features
            'lag_1', 'lag_2', 'lag_3', 'lag_6',
            # Rolling features
            'rolling_mean_3m', 'rolling_mean_6m',
            'rolling_std_3m', 'rolling_std_6m',
            # Temporal features
            'month_sin', 'month_cos', 'quarter',
            'is_festival_season', 'is_monsoon',
            # Behavioral features
            'cumulative_active_months', 'cumulative_avg_qty',
            'months_since_start', 'activity_rate'
        ]

        # Add demographic features if available
        if 'num_members' in self.panel_df.columns:
            self.feature_cols.extend([
                'num_members', 'avg_age', 'gender_ratio',
                'num_children', 'num_adults', 'num_elderly',
                'card_type_encoded', 'NO_OF_MEMBERS'
            ])

        # Filter to available features only
        self.feature_cols = [f for f in self.feature_cols if f in self.panel_df.columns]

        print(f"\nFeature columns ({len(self.feature_cols)}): {self.feature_cols}")
        print(f"Final panel shape: {self.panel_df.shape}")

    def train(self, model_type: str = 'gradient_boosting') -> Dict[str, float]:
        """
        Train the forecasting model.

        Returns:
            Dictionary of evaluation metrics on held-out test set
        """
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)

        # Prepare train/test split
        X_train, X_test, y_train, y_test = prepare_training_data(
            self.panel_df,
            self.feature_cols
        )

        # Build and train model
        print(f"\nTraining {model_type} model...")
        self.model = build_ml_model(X_train, y_train, model_type)

        # Evaluate
        print("\nEvaluating on test set...")
        metrics = evaluate_model(self.model, X_test, y_test)

        print("\nModel Performance:")
        print(f"  MAE:  {metrics['MAE']:.2f} kg")
        print(f"  RMSE: {metrics['RMSE']:.2f} kg")
        print(f"  MAPE: {metrics['MAPE']:.1f}%")
        print(f"  R²:   {metrics['R2']:.3f}")

        return metrics

    def predict_next_month(self) -> pd.DataFrame:
        """
        Generate predictions for the next month.

        Returns:
            DataFrame with entity identifiers and predicted consumption
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        print("\n" + "=" * 60)
        print("GENERATING NEXT-MONTH PREDICTIONS")
        print("=" * 60)

        # Get the latest month in data
        max_month = self.panel_df['allotment_yearmonth'].max()
        next_month = max_month + 1

        print(f"Predicting for: {next_month}")

        # Get latest data for each entity (to use for lag features)
        latest_data = self.panel_df[
            self.panel_df['allotment_yearmonth'] == max_month
        ].copy()

        # Prepare features for prediction
        X_pred = latest_data[self.feature_cols].copy()

        # Handle any remaining NaN values
        X_pred = X_pred.fillna(0)

        # Generate predictions
        predictions = self.model.predict(X_pred)
        predictions = np.maximum(predictions, 0)  # No negative consumption

        # Create output dataframe
        result = latest_data[self.aggregation_cols].copy()
        result['predicted_qty_kg'] = predictions
        result['prediction_month'] = str(next_month)

        print(f"\nGenerated {len(result):,} predictions")
        print(f"Total predicted consumption: {predictions.sum():,.0f} kg")
        print(f"Average per entity: {predictions.mean():.1f} kg")

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            print("Feature importance not available for this model type")
            return None

    def predict_at_aggregation(
        self,
        aggregation: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate predictions to a different level.

        For example, if model was trained at card level,
        this can aggregate to FPS level.
        """
        predictions = self.predict_next_month()

        if aggregation == self.aggregation_cols:
            return predictions

        # Need to join with transaction data to get mapping
        # then aggregate
        agg_result = predictions.merge(
            self.txn_df[self.aggregation_cols + aggregation].drop_duplicates(),
            on=self.aggregation_cols,
            how='left'
        ).groupby(aggregation).agg(
            predicted_qty_kg=('predicted_qty_kg', 'sum'),
            num_entities=('predicted_qty_kg', 'count')
        ).reset_index()

        return agg_result


# =============================================================================
# HANDLING EDGE CASES
# =============================================================================

def handle_new_cards(
    panel_df: pd.DataFrame,
    card_df: pd.DataFrame,
    member_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Handle new cards with no transaction history.

    Strategy: Use similar card profiles to estimate expected consumption.
    """
    # Cards with history
    cards_with_history = set(panel_df['card_no'].unique())

    # All registered cards
    all_cards = set(card_df['card_no'].unique())

    # New cards (no history)
    new_cards = all_cards - cards_with_history

    if len(new_cards) == 0:
        return pd.DataFrame()

    print(f"Found {len(new_cards)} new cards without transaction history")

    # For new cards, estimate based on card type and household size
    new_card_df = card_df[card_df['card_no'].isin(new_cards)].copy()

    # Calculate average consumption by card type and household size
    avg_by_profile = panel_df.groupby(['CARD_TYPE', 'NO_OF_MEMBERS'])['total_qty'].mean().reset_index()
    avg_by_profile.columns = ['CARD_TYPE', 'NO_OF_MEMBERS', 'estimated_qty']

    # Merge to get estimates
    new_card_df = new_card_df.merge(avg_by_profile, on=['CARD_TYPE', 'NO_OF_MEMBERS'], how='left')

    # Fallback: overall average by card type
    avg_by_type = panel_df.groupby('CARD_TYPE')['total_qty'].mean()
    new_card_df['estimated_qty'] = new_card_df.apply(
        lambda x: x['estimated_qty'] if pd.notna(x['estimated_qty'])
                  else avg_by_type.get(x['CARD_TYPE'], panel_df['total_qty'].mean()),
        axis=1
    )

    return new_card_df[['card_no', 'CARD_TYPE', 'NO_OF_MEMBERS', 'estimated_qty']]


def handle_inactive_cards(
    panel_df: pd.DataFrame,
    inactive_threshold_months: int = 3
) -> pd.DataFrame:
    """
    Identify cards that have become inactive (no transactions for N months).

    These may need special handling:
    - Migration
    - Death
    - System issues
    """
    max_month = panel_df['allotment_yearmonth'].max()

    # Get last active month for each card
    last_active = panel_df[panel_df['total_qty'] > 0].groupby('card_no')['allotment_yearmonth'].max().reset_index()
    last_active.columns = ['card_no', 'last_active_month']

    # Calculate months since last activity
    last_active['months_inactive'] = last_active['last_active_month'].apply(
        lambda x: (max_month - x).n
    )

    # Flag inactive cards
    inactive_cards = last_active[last_active['months_inactive'] >= inactive_threshold_months]

    print(f"Found {len(inactive_cards)} cards inactive for {inactive_threshold_months}+ months")

    return inactive_cards


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution demonstrating the full forecasting workflow."""

    print("=" * 70)
    print("ANNAPURTI GRAIN CONSUMPTION FORECASTING SYSTEM")
    print("=" * 70)

    # Initialize forecaster
    forecaster = AnnapurtiForecaster()

    # Load data
    forecaster.load_data()

    # Explore the data
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    txn = forecaster.txn_df

    print("\n1. Transaction Volume by Month:")
    monthly_volume = txn.groupby('allotment_yearmonth').agg(
        total_qty=('qty', 'sum'),
        num_transactions=('TXN_NO', 'nunique'),
        num_cards=('card_no', 'nunique')
    )
    print(monthly_volume)

    print("\n2. Consumption by Card Category:")
    print(txn.groupby('card_category')['qty'].agg(['sum', 'mean', 'count']))

    print("\n3. Consumption by Commodity:")
    print(txn.groupby('COMMODITY_CODE')['qty'].agg(['sum', 'mean', 'count']))

    # Prepare features at card level
    print("\n" + "=" * 60)
    print("CARD-LEVEL FORECASTING")
    print("=" * 60)

    forecaster.prepare_features(aggregation=['card_no'])

    # Train model
    metrics = forecaster.train(model_type='gradient_boosting')

    # Get feature importance
    importance = forecaster.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Important Features:")
        print(importance.head(10))

    # Generate predictions
    predictions = forecaster.predict_next_month()
    print("\nSample Predictions (Top 10 by quantity):")
    print(predictions.nlargest(10, 'predicted_qty_kg'))

    # Save predictions
    output_path = Path("predictions_next_month.csv")
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Demonstrate FPS-level aggregation
    print("\n" + "=" * 60)
    print("FPS-LEVEL FORECASTING")
    print("=" * 60)

    forecaster_fps = AnnapurtiForecaster()
    forecaster_fps.load_data()
    forecaster_fps.prepare_features(aggregation=['home_fps'])
    forecaster_fps.train()
    fps_predictions = forecaster_fps.predict_next_month()

    print("\nTop 10 FPS by Predicted Demand:")
    print(fps_predictions.nlargest(10, 'predicted_qty_kg'))

    # Commodity × FPS level
    print("\n" + "=" * 60)
    print("FPS × COMMODITY FORECASTING")
    print("=" * 60)

    forecaster_multi = AnnapurtiForecaster()
    forecaster_multi.load_data()
    forecaster_multi.prepare_features(aggregation=['home_fps', 'COMMODITY_CODE'])
    forecaster_multi.train()

    return forecaster


if __name__ == "__main__":
    forecaster = main()
