
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path("Historical_Data")
TRANSACTION_FILE = DATA_DIR / "Annpurti Txn Data.csv"
CARD_FILE = DATA_DIR / "Annpurti Txn Card Details Data.csv"

def calculate_average():
    print("Loading data...")
    
    # Load Card Data
    try:
        cards = pd.read_csv(CARD_FILE)
        # Assuming there might be duplicates, let's keep the last one or drop duplicates
        cards = cards[['card_no', 'CARD_TYPE']].drop_duplicates()
        print(f"Loaded {len(cards)} card records.")
    except Exception as e:
        print(f"Error loading card data: {e}")
        return

    # Load Transaction Data
    try:
        txns = pd.read_csv(TRANSACTION_FILE)
        # Ensure qty is numeric
        txns['qty'] = pd.to_numeric(txns['qty'], errors='coerce')
        # Ensure ALLOTMENT_MONTH is handled if we want monthly average
        txns['ALLOTMENT_MONTH'] = pd.to_datetime(txns['ALLOTMENT_MONTH'])
        print(f"Loaded {len(txns)} transaction records.")
    except Exception as e:
        print(f"Error loading txn data: {e}")
        return

    # Merge
    print("Merging data...")
    merged = txns.merge(cards, on='card_no', how='left')
    
    # Check for missing card types
    missing_types = merged['CARD_TYPE'].isna().sum()
    if missing_types > 0:
        print(f"Warning: {missing_types} transactions have no associated card type.")

    # Filter for Card Type A
    type_a = merged[merged['CARD_TYPE'] == 'A']
    
    if len(type_a) == 0:
        print("No transactions found for Card Type 'A'.")
        print("Available Card Types:", merged['CARD_TYPE'].unique())
        return

    print(f"Found {len(type_a)} transactions for Card Type 'A'.")

    # Calculate Average Monthly Consumption per Card
    # Group by Card and Month -> Sum Quantity
    monthly_consumption = type_a.groupby(['card_no', 'ALLOTMENT_MONTH'])['qty'].sum().reset_index()
    
    # Calculate statistics
    avg_monthly = monthly_consumption['qty'].mean()
    median_monthly = monthly_consumption['qty'].median()
    std_monthly = monthly_consumption['qty'].std()
    
    with open('avg_output.txt', 'w') as f:
        f.write("-" * 30 + "\n")
        f.write("STATISTICS FOR CARD TYPE 'A'\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Monthly Consumption: {avg_monthly:.2f} kg\n")
        f.write(f"Median Monthly Consumption:  {median_monthly:.2f} kg\n")
        f.write(f"Std Dev of Consumption:      {std_monthly:.2f} kg\n")
        f.write(f"Number of Unique Cards:      {monthly_consumption['card_no'].nunique()}\n")
        f.write("-" * 30 + "\n")
    print("Analysis complete. Results written to avg_output.txt")

if __name__ == "__main__":
    calculate_average()
