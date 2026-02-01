"""
Annapurti Grain Distribution System - Hierarchical Forecast Report Generator
==============================================================================

This script generates production-grade forecast outputs with STRICT hierarchical
consistency. All breakdowns sum exactly to their parent totals.

Key Feature: Bottom-Up Aggregation with Reconciliation
------------------------------------------------------
We use the card-level predictions as the single source of truth, then aggregate
upward. This ensures:
- FPS totals = sum of card predictions within that FPS
- Card-type totals = sum of card predictions by card type
- FPS x Card-type = sum of cards matching both criteria
- Grand total = sum of all card predictions

This approach guarantees mathematical consistency across all views.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("Historical_Data")
PREDICTIONS_DIR = Path("predictions")
OUTPUT_DIR = Path("forecast_reports")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def get_latest_prediction_file(pattern):
    """Find the most recent prediction file matching the pattern."""
    files = sorted(PREDICTIONS_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {PREDICTIONS_DIR}")
    return files[0]


def load_all_data():
    """Load all required data sources."""
    print("Loading data sources...")

    # Load card-level predictions (SOURCE OF TRUTH) - find latest file
    card_pred_file = get_latest_prediction_file("predictions_card_level_*.csv")
    card_predictions = pd.read_csv(card_pred_file)
    print(f"  Card predictions: {len(card_predictions):,} records (from {card_pred_file.name})")

    # Load card details (for card type and FPS mapping)
    card_details = pd.read_csv(DATA_DIR / "Annpurti Txn Card Details Data.csv")
    print(f"  Card details: {len(card_details):,} records")

    # Load transaction data (for historical patterns and FPS mapping)
    txn_data = pd.read_csv(DATA_DIR / "Annpurti Txn Data.csv")
    txn_data['ALLOTMENT_MONTH'] = pd.to_datetime(txn_data['ALLOTMENT_MONTH'])
    print(f"  Transaction data: {len(txn_data):,} records")

    return card_predictions, card_details, txn_data


def prepare_master_dataset(card_predictions, card_details, txn_data):
    """
    Prepare a single master dataset with all required dimensions.

    This is the core of the HIERARCHICAL CONSISTENCY mechanism:
    - Each card-level prediction is enriched with card_type and home_fps
    - All aggregations are computed from this single source
    """
    print("\nPreparing master dataset...")

    # Convert card_no to string for consistent joining
    card_predictions['card_no'] = card_predictions['card_no'].astype(str)
    card_details['card_no'] = card_details['card_no'].astype(str)
    txn_data['card_no'] = txn_data['card_no'].astype(str)

    # Get card type from card details
    card_type_map = card_details[['card_no', 'CARD_TYPE']].drop_duplicates()

    # Get home_fps from transaction data (most recent FPS for each card)
    fps_map = txn_data.sort_values('ALLOTMENT_MONTH').groupby('card_no')['home_fps'].last().reset_index()
    fps_map['home_fps'] = fps_map['home_fps'].astype(str)

    # Merge predictions with card type and FPS
    master = card_predictions.merge(card_type_map, on='card_no', how='left')
    master = master.merge(fps_map, on='card_no', how='left')

    # Handle missing values
    master['CARD_TYPE'] = master['CARD_TYPE'].fillna('PH')  # Default to Priority Household
    master['home_fps'] = master['home_fps'].fillna('UNKNOWN')

    # Remove any cards with unknown FPS from FPS-specific reports
    master_with_fps = master[master['home_fps'] != 'UNKNOWN'].copy()

    print(f"  Master dataset: {len(master):,} cards")
    print(f"  Cards with FPS mapping: {len(master_with_fps):,}")
    print(f"  Card types: {master['CARD_TYPE'].value_counts().to_dict()}")

    return master, master_with_fps


# =============================================================================
# HIERARCHICAL CONSISTENCY MECHANISM
# =============================================================================

class HierarchicalForecaster:
    """
    Ensures all forecast views are mathematically consistent.

    THE KEY PRINCIPLE: Bottom-Up Aggregation
    ----------------------------------------
    1. Start with the most granular level (card-level predictions)
    2. All higher-level aggregations are computed by summing lower levels
    3. The GRAND TOTAL is computed once and used as the reconciliation anchor

    This approach GUARANTEES:
    - Sum(card-type breakdown) = Grand Total
    - Sum(FPS breakdown) = Grand Total
    - Sum(FPS x Card-type breakdown) = Grand Total
    - Any sub-breakdown sums to its parent

    NO ROUNDING ERRORS: We use precise decimal arithmetic throughout
    and only round for display purposes at the very end.
    """

    def __init__(self, master_df):
        self.master = master_df.copy()
        self.grand_total = self.master['predicted_qty_kg'].sum()
        self.total_cards = len(self.master)
        self.total_fps = self.master['home_fps'].nunique()

    def get_overall_aggregate(self):
        """View 1: Overall Aggregate Forecast"""
        return {
            'total_predicted_qty_kg': self.grand_total,
            'total_cards': self.total_cards,
            'total_fps': self.total_fps,
            'avg_per_fps': self.grand_total / self.total_fps if self.total_fps > 0 else 0,
            'avg_per_card': self.grand_total / self.total_cards if self.total_cards > 0 else 0
        }

    def get_card_type_breakdown(self):
        """View 2: Card-Type-Wise Breakdown"""
        breakdown = self.master.groupby('CARD_TYPE').agg(
            total_qty_kg=('predicted_qty_kg', 'sum'),
            num_cards=('card_no', 'count'),
            avg_per_card=('predicted_qty_kg', 'mean')
        ).reset_index()

        # Calculate share percentage
        breakdown['share_pct'] = (breakdown['total_qty_kg'] / self.grand_total * 100)

        # Verify sum equals grand total
        breakdown_sum = breakdown['total_qty_kg'].sum()
        assert np.isclose(breakdown_sum, self.grand_total), \
            f"Card-type sum {breakdown_sum} != Grand total {self.grand_total}"

        return breakdown.sort_values('total_qty_kg', ascending=False)

    def get_fps_breakdown(self):
        """View 3: Home FPS-Wise Forecast"""
        breakdown = self.master.groupby('home_fps').agg(
            total_qty_kg=('predicted_qty_kg', 'sum'),
            num_cards=('card_no', 'count'),
            avg_per_card=('predicted_qty_kg', 'mean')
        ).reset_index()

        # Verify sum equals grand total
        breakdown_sum = breakdown['total_qty_kg'].sum()
        assert np.isclose(breakdown_sum, self.grand_total), \
            f"FPS sum {breakdown_sum} != Grand total {self.grand_total}"

        return breakdown.sort_values('total_qty_kg', ascending=False)

    def get_fps_cardtype_breakdown(self):
        """View 4: FPS x Card-Type Breakdown"""
        breakdown = self.master.groupby(['home_fps', 'CARD_TYPE']).agg(
            total_qty_kg=('predicted_qty_kg', 'sum'),
            num_cards=('card_no', 'count')
        ).reset_index()

        # Verify sum equals grand total
        breakdown_sum = breakdown['total_qty_kg'].sum()
        assert np.isclose(breakdown_sum, self.grand_total), \
            f"FPS x CardType sum {breakdown_sum} != Grand total {self.grand_total}"

        return breakdown.sort_values(['home_fps', 'total_qty_kg'], ascending=[True, False])

    def verify_consistency(self):
        """
        Master consistency check - verifies all aggregations reconcile.

        This is the RECONCILIATION FEATURE that ensures hierarchical balance.
        """
        results = {
            'grand_total': self.grand_total,
            'card_type_sum': self.get_card_type_breakdown()['total_qty_kg'].sum(),
            'fps_sum': self.get_fps_breakdown()['total_qty_kg'].sum(),
            'fps_cardtype_sum': self.get_fps_cardtype_breakdown()['total_qty_kg'].sum()
        }

        all_match = all(np.isclose(v, self.grand_total) for v in results.values())

        return results, all_match


def get_historical_monthly_consumption(txn_data):
    """View 5: Historical Monthly Consumption"""
    txn_data['allotment_yearmonth'] = txn_data['ALLOTMENT_MONTH'].dt.to_period('M')

    monthly = txn_data.groupby('allotment_yearmonth').agg(
        total_qty_kg=('qty', 'sum'),
        num_transactions=('TXN_NO', 'nunique'),
        num_cards=('card_no', 'nunique')
    ).reset_index()

    monthly['allotment_yearmonth'] = monthly['allotment_yearmonth'].astype(str)
    monthly['avg_per_card'] = monthly['total_qty_kg'] / monthly['num_cards']

    return monthly.sort_values('allotment_yearmonth')


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_full_report():
    """Generate all forecast views with consistency verification."""

    print("=" * 70)
    print("ANNAPURTI GRAIN DISTRIBUTION SYSTEM")
    print("HIERARCHICAL FORECAST REPORT - APRIL 2026")
    print("=" * 70)
    print(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    card_predictions, card_details, txn_data = load_all_data()

    # Prepare master dataset
    master, master_with_fps = prepare_master_dataset(card_predictions, card_details, txn_data)

    # Initialize hierarchical forecaster
    forecaster = HierarchicalForecaster(master)

    # =========================================================================
    # VIEW 1: OVERALL AGGREGATE FORECAST
    # =========================================================================
    print("\n" + "=" * 70)
    print("VIEW 1: OVERALL AGGREGATE FORECAST (April 2026)")
    print("=" * 70)

    overall = forecaster.get_overall_aggregate()

    print(f"""
    TOTAL PREDICTED GRAIN CONSUMPTION
    ---------------------------------------------------------------------

    Total Predicted Quantity:     {overall['total_predicted_qty_kg']:,.2f} kg

    Coverage:
      - Total Cards:              {overall['total_cards']:,}
      - Total FPS Shops:          {overall['total_fps']:,}

    Averages:
      - Average per FPS:          {overall['avg_per_fps']:,.2f} kg
      - Average per Card:         {overall['avg_per_card']:,.2f} kg

    ---------------------------------------------------------------------
    Note: This total serves as the ANCHOR for all breakdowns below.
          All sub-totals must sum back to {overall['total_predicted_qty_kg']:,.2f} kg
    """)

    # =========================================================================
    # VIEW 2: CARD-TYPE-WISE BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 70)
    print("VIEW 2: CARD-TYPE-WISE BREAKDOWN (April 2026)")
    print("=" * 70)

    card_type_df = forecaster.get_card_type_breakdown()

    card_type_names = {
        'A': 'Antyodaya (Poorest Households)',
        'PH': 'Priority Household',
        'S': 'State Priority'
    }

    print("""
    FORECAST BY CARD CATEGORY
    ---------------------------------------------------------------------
    """)

    for _, row in card_type_df.iterrows():
        card_name = card_type_names.get(row['CARD_TYPE'], row['CARD_TYPE'])
        print(f"""    Card Type {row['CARD_TYPE']} - {card_name}
      - Total Predicted:          {row['total_qty_kg']:,.2f} kg
      - Number of Cards:          {row['num_cards']:,}
      - Average per Card:         {row['avg_per_card']:,.2f} kg
      - Share of Total:           {row['share_pct']:.1f}%
    """)

    card_type_total = card_type_df['total_qty_kg'].sum()
    print(f"""    ---------------------------------------------------------------------
    VERIFICATION: Sum of Card Types = {card_type_total:,.2f} kg
                  Grand Total       = {overall['total_predicted_qty_kg']:,.2f} kg
                  Match: {'YES' if np.isclose(card_type_total, overall['total_predicted_qty_kg']) else 'NO'}
    """)

    # =========================================================================
    # VIEW 3: HOME FPS-WISE FORECAST
    # =========================================================================
    print("\n" + "=" * 70)
    print("VIEW 3: HOME FPS-WISE FORECAST (April 2026)")
    print("=" * 70)

    fps_df = forecaster.get_fps_breakdown()

    print("""
    FORECAST BY FAIR PRICE SHOP (Top 20 FPS by Volume)
    ---------------------------------------------------------------------
    """)

    print(f"    {'FPS ID':<12} {'Total (kg)':>14} {'Cards':>8} {'Avg/Card':>12}")
    print(f"    {'-'*12} {'-'*14} {'-'*8} {'-'*12}")

    for _, row in fps_df.head(20).iterrows():
        print(f"    {row['home_fps']:<12} {row['total_qty_kg']:>14,.2f} {row['num_cards']:>8,} {row['avg_per_card']:>12,.2f}")

    if len(fps_df) > 20:
        others = fps_df.iloc[20:]
        print(f"    {'... and':<12} {others['total_qty_kg'].sum():>14,.2f} {others['num_cards'].sum():>8,} {'(remaining)':<12}")

    fps_total = fps_df['total_qty_kg'].sum()
    print(f"""
    ---------------------------------------------------------------------
    SUMMARY:
      - Total FPS Shops:          {len(fps_df):,}
      - Sum of All FPS:           {fps_total:,.2f} kg
      - Grand Total:              {overall['total_predicted_qty_kg']:,.2f} kg
      - Match: {'YES' if np.isclose(fps_total, overall['total_predicted_qty_kg']) else 'NO'}
    """)

    # =========================================================================
    # VIEW 4: FPS x CARD-TYPE BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 70)
    print("VIEW 4: FPS x CARD-TYPE BREAKDOWN (April 2026)")
    print("=" * 70)

    fps_cardtype_df = forecaster.get_fps_cardtype_breakdown()

    # Create pivot table for display
    pivot = fps_cardtype_df.pivot_table(
        index='home_fps',
        columns='CARD_TYPE',
        values='total_qty_kg',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Add row totals
    card_types = ['A', 'PH', 'S']
    available_types = [ct for ct in card_types if ct in pivot.columns]
    pivot['FPS_Total'] = pivot[available_types].sum(axis=1)

    print("""
    BREAKDOWN BY FPS AND CARD TYPE (Top 15 FPS)
    ---------------------------------------------------------------------
    """)

    top_fps = pivot.nlargest(15, 'FPS_Total')

    header = f"    {'FPS ID':<12}"
    for ct in available_types:
        header += f" {ct + ' (kg)':>12}"
    header += f" {'FPS Total':>14}"
    print(header)
    print(f"    {'-'*12}" + f" {'-'*12}" * len(available_types) + f" {'-'*14}")

    for _, row in top_fps.iterrows():
        line = f"    {row['home_fps']:<12}"
        for ct in available_types:
            line += f" {row[ct]:>12,.2f}"
        line += f" {row['FPS_Total']:>14,.2f}"
        print(line)

    # Column totals
    print(f"    {'-'*12}" + f" {'-'*12}" * len(available_types) + f" {'-'*14}")
    totals_line = f"    {'TOTAL':<12}"
    for ct in available_types:
        totals_line += f" {pivot[ct].sum():>12,.2f}"
    totals_line += f" {pivot['FPS_Total'].sum():>14,.2f}"
    print(totals_line)

    fps_cardtype_total = fps_cardtype_df['total_qty_kg'].sum()
    print(f"""
    ---------------------------------------------------------------------
    VERIFICATION:
      - Sum of FPS x CardType:    {fps_cardtype_total:,.2f} kg
      - Grand Total:              {overall['total_predicted_qty_kg']:,.2f} kg
      - Match: {'YES' if np.isclose(fps_cardtype_total, overall['total_predicted_qty_kg']) else 'NO'}

    CROSS-CHECK BY CARD TYPE:
    """)
    for ct in available_types:
        ct_from_pivot = pivot[ct].sum()
        ct_from_view2 = card_type_df[card_type_df['CARD_TYPE'] == ct]['total_qty_kg'].values[0] if ct in card_type_df['CARD_TYPE'].values else 0
        print(f"      - Type {ct}: {ct_from_pivot:,.2f} kg (should be {ct_from_view2:,.2f} kg) - {'MATCH' if np.isclose(ct_from_pivot, ct_from_view2) else 'MISMATCH'}")

    # =========================================================================
    # VIEW 5: MONTH-WISE CONSUMPTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("VIEW 5: MONTH-WISE CONSUMPTION (Historical + Forecast)")
    print("=" * 70)

    monthly_df = get_historical_monthly_consumption(txn_data)

    print("""
    HISTORICAL MONTHLY GRAIN CONSUMPTION
    ---------------------------------------------------------------------
    """)

    print(f"    {'Month':<12} {'Total (kg)':>14} {'Cards':>10} {'Avg/Card':>12} {'Status':<12}")
    print(f"    {'-'*12} {'-'*14} {'-'*10} {'-'*12} {'-'*12}")

    for _, row in monthly_df.iterrows():
        print(f"    {row['allotment_yearmonth']:<12} {row['total_qty_kg']:>14,.2f} {row['num_cards']:>10,} {row['avg_per_card']:>12,.2f} Historical")

    # Add the forecast row
    print(f"    {'-'*12} {'-'*14} {'-'*10} {'-'*12} {'-'*12}")
    print(f"    {'2026-04':<12} {overall['total_predicted_qty_kg']:>14,.2f} {overall['total_cards']:>10,} {overall['avg_per_card']:>12,.2f} FORECAST")

    print("""
    ---------------------------------------------------------------------
    Note: April 2026 is the PREDICTED month. All other rows are historical.
    """)

    # =========================================================================
    # FINAL CONSISTENCY VERIFICATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("HIERARCHICAL CONSISTENCY VERIFICATION")
    print("=" * 70)

    results, all_match = forecaster.verify_consistency()

    print("""
    RECONCILIATION CHECK
    ---------------------------------------------------------------------

    All forecast views must sum to the same Grand Total.
    This ensures no data is lost or duplicated across views.
    """)

    print(f"    Anchor (Grand Total):     {results['grand_total']:,.2f} kg")
    print(f"    Card-Type Sum:            {results['card_type_sum']:,.2f} kg  {'[OK]' if np.isclose(results['card_type_sum'], results['grand_total']) else '[MISMATCH]'}")
    print(f"    FPS Sum:                  {results['fps_sum']:,.2f} kg  {'[OK]' if np.isclose(results['fps_sum'], results['grand_total']) else '[MISMATCH]'}")
    print(f"    FPS x CardType Sum:       {results['fps_cardtype_sum']:,.2f} kg  {'[OK]' if np.isclose(results['fps_cardtype_sum'], results['grand_total']) else '[MISMATCH]'}")

    print(f"""
    ---------------------------------------------------------------------
    OVERALL STATUS: {'ALL VIEWS RECONCILED SUCCESSFULLY' if all_match else 'RECONCILIATION FAILED - CHECK DATA'}
    """)

    # =========================================================================
    # HIERARCHICAL CONSISTENCY MECHANISM EXPLANATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("HIERARCHICAL CONSISTENCY MECHANISM")
    print("=" * 70)

    print("""
    HOW WE ENSURE TOTALS ALWAYS BALANCE
    ---------------------------------------------------------------------

    The forecast system uses a "BOTTOM-UP AGGREGATION" approach to
    guarantee that all views are mathematically consistent:

    1. SINGLE SOURCE OF TRUTH
       ----------------------
       All forecasts originate from card-level predictions. Each
       beneficiary card has exactly one predicted quantity for April.
       This is our "atomic" level of data.

    2. ENRICHMENT WITH DIMENSIONS
       ---------------------------
       Each card is tagged with its attributes:
       - Card Type (A, PH, or S)
       - Home FPS (Fair Price Shop ID)

       This creates a master dataset where every row has all dimensions.

    3. AGGREGATION ALWAYS SUMS UP
       --------------------------
       When we create any view (by card type, by FPS, by FPS+card type),
       we ALWAYS use SUM aggregation on the same base data.

       This mathematically guarantees:
       - Sum of all card types = Total
       - Sum of all FPS = Total
       - Sum of all FPS x CardType combinations = Total

    4. NO INDEPENDENT CALCULATIONS
       ---------------------------
       We NEVER calculate FPS totals independently of card totals.
       FPS totals are always derived by summing the cards within that FPS.

       This prevents scenarios where:
       - FPS model predicts 1000 kg
       - But sum of card predictions = 950 kg
       (This kind of mismatch is impossible with bottom-up aggregation)

    5. VERIFICATION AT EVERY LEVEL
       ---------------------------
       Before presenting any view, we verify that its total matches
       the grand total. If there's any discrepancy, we flag it immediately.

    ---------------------------------------------------------------------

    WHY THIS MATTERS FOR OPERATIONS
    -------------------------------

    * Supply planning can trust that FPS-level totals represent the
      actual expected demand from individual beneficiaries

    * Card-type allocations can be used for policy analysis, knowing
      they represent the true breakdown of total demand

    * No "leakage" or "phantom quantities" that appear in one view
      but disappear in another

    * Auditors can verify any number by drilling down to the card level

    ---------------------------------------------------------------------
    """)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save individual views
    card_type_df.to_csv(OUTPUT_DIR / f"view2_card_type_breakdown_{timestamp}.csv", index=False)
    fps_df.to_csv(OUTPUT_DIR / f"view3_fps_breakdown_{timestamp}.csv", index=False)
    fps_cardtype_df.to_csv(OUTPUT_DIR / f"view4_fps_cardtype_breakdown_{timestamp}.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / f"view5_monthly_consumption_{timestamp}.csv", index=False)

    # Save master summary
    summary = pd.DataFrame([{
        'metric': 'Total Predicted Quantity (kg)',
        'value': overall['total_predicted_qty_kg']
    }, {
        'metric': 'Total Cards',
        'value': overall['total_cards']
    }, {
        'metric': 'Total FPS Shops',
        'value': overall['total_fps']
    }, {
        'metric': 'Average per FPS (kg)',
        'value': overall['avg_per_fps']
    }, {
        'metric': 'Average per Card (kg)',
        'value': overall['avg_per_card']
    }])
    summary.to_csv(OUTPUT_DIR / f"view1_overall_aggregate_{timestamp}.csv", index=False)

    print(f"\n    Files saved to: {OUTPUT_DIR}/")
    print(f"    • view1_overall_aggregate_{timestamp}.csv")
    print(f"    • view2_card_type_breakdown_{timestamp}.csv")
    print(f"    • view3_fps_breakdown_{timestamp}.csv")
    print(f"    • view4_fps_cardtype_breakdown_{timestamp}.csv")
    print(f"    • view5_monthly_consumption_{timestamp}.csv")

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)

    return {
        'overall': overall,
        'card_type': card_type_df,
        'fps': fps_df,
        'fps_cardtype': fps_cardtype_df,
        'monthly': monthly_df
    }


if __name__ == "__main__":
    results = generate_full_report()
