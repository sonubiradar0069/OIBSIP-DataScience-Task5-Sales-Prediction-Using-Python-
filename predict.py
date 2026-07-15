import os
import argparse
import joblib
import pandas as pd

def predict_sales():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Predict product sales using trained ML model.")
    parser.add_argument("--tv", type=float, required=True, help="TV advertising budget (in thousands of dollars)")
    parser.add_argument("--radio", type=float, required=True, help="Radio advertising budget (in thousands of dollars)")
    parser.add_argument("--newspaper", type=float, required=True, help="Newspaper advertising budget (in thousands of dollars)")
    
    args = parser.parse_args()

    # Load model
    model_path = os.path.join("models", "sales_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at '{model_path}'. Please run 'python src/train.py' first.")
        return

    # Check for negative values
    if args.tv < 0 or args.radio < 0 or args.newspaper < 0:
        print("Warning: Advertising budgets should be positive values.")

    # Load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create df for prediction
    input_data = pd.DataFrame([{
        'TV': args.tv,
        'Radio': args.radio,
        'Newspaper': args.newspaper
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display prediction result beautifully
    print("=" * 50)
    print("           SALES PREDICTION ENGINE")
    print("=" * 50)
    print(f"Input Budgets:")
    print(f"  - TV Advertising:        ${args.tv:,.2f}k")
    print(f"  - Radio Advertising:     ${args.radio:,.2f}k")
    print(f"  - Newspaper Advertising: ${args.newspaper:,.2f}k")
    print("-" * 50)
    print(f"  => Predicted Sales:      {prediction:,.2f} thousand units")
    print("=" * 50)

if __name__ == "__main__":
    predict_sales()
