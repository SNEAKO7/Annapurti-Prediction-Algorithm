"""
Simple script to run Annapurti grain forecasting.

Usage:
    python run_forecast.py                    # Card-level predictions
    python run_forecast.py --level fps        # FPS-level predictions
    python run_forecast.py --level commodity  # Commodity-level predictions
    python run_forecast.py --level fps_commodity  # FPS × Commodity
"""

import argparse
from pathlib import Path
from datetime import datetime
from grain_forecaster import AnnapurtiForecaster
from generate_forecast_report import generate_full_report


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Annapurti Grain Consumption Forecaster')
    parser.add_argument('--level', choices=['card', 'fps', 'commodity', 'fps_commodity'],
                        default='card', help='Aggregation level for predictions')
    parser.add_argument('--data-dir', type=str, default='Historical_Data',
                        help='Directory containing the CSV data files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (if not specified, saves to predictions/ folder with timestamp)')
    args = parser.parse_args()

    # Map level to aggregation columns
    aggregation_map = {
        'card': ['card_no'],
        'fps': ['home_fps'],
        'commodity': ['COMMODITY_CODE'],
        'fps_commodity': ['home_fps', 'COMMODITY_CODE']
    }

    aggregation = aggregation_map[args.level]

    # Create predictions folder if it doesn't exist
    predictions_dir = Path('predictions')
    predictions_dir.mkdir(exist_ok=True)

    # Default output filename with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = predictions_dir / f'predictions_{args.level}_level_{timestamp}.csv'
    else:
        # If user provided custom path, use it as-is
        args.output = Path(args.output)

    print("=" * 70)
    print("ANNAPURTI GRAIN CONSUMPTION FORECASTER")
    print("=" * 70)
    print(f"\nAggregation level: {args.level}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output}")

    # Initialize and run forecaster
    forecaster = AnnapurtiForecaster(data_dir=Path(args.data_dir))

    print("\n[1/4] Loading data...")
    forecaster.load_data()

    print("\n[2/4] Preparing features...")
    forecaster.prepare_features(aggregation=aggregation)

    print("\n[3/4] Training model...")
    metrics = forecaster.train()

    print("\n[4/4] Generating predictions...")
    predictions = forecaster.predict_next_month()

    # Save predictions
    predictions.to_csv(args.output, index=False)
    print(f"\n[OK] Predictions saved to: {args.output}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entities: {len(predictions):,}")
    print(f"Total predicted consumption: {predictions['predicted_qty_kg'].sum():,.0f} kg")
    print(f"Average per entity: {predictions['predicted_qty_kg'].mean():.1f} kg")
    print(f"Prediction month: {predictions['prediction_month'].iloc[0]}")

    print("\nTop 10 by predicted demand:")
    print(predictions.nlargest(10, 'predicted_qty_kg').to_string(index=False))

    # Feature importance
    importance = forecaster.get_feature_importance()
    if importance is not None:
        print("\nTop 5 important features:")
        print(importance.head(5).to_string(index=False))

    # Generate Forecast Reports
    print("\n[Optional] Generating Forecast Reports...")
    try:
        generate_full_report()
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")

    return predictions


if __name__ == "__main__":
    main()
