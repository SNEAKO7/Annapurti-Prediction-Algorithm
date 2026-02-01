# Annapurti Grain Consumption Forecasting

A production-grade machine learning system to predict next-month grain consumption for the Annapurti distribution system. This project uses historical transaction data to forecast demand at the Smart Card, Fair Price Shop (FPS), and Commodity levels.

## Key Features
- **Multi-level Forecasting**: generate predictions for individual beneficiaries, specific shops, or commodity types.
- **Advanced Machine Learning**: Uses LightGBM (Gradient Boosting) for high-accuracy demand forecasting.
- **Robust Feature Engineering**: Incorporates lag features, rolling statistics, temporal seasonality, and household demographics.
- **Production Ready**: Includes modular code structure, data validation, and easy-to-use CLI.

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Run the forecaster from the command line:

```bash
# Default: Predict for all smart cards
python run_forecast.py

# Predict for each Fair Price Shop
python run_forecast.py --level fps

# Predict per Commodity
python run_forecast.py --level commodity
```

For a detailed explanation of the algorithm, data flow, and output formats, please refer to the [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md).
