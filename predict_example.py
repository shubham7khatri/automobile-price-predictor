"""Example of using the saved model in another program.

Keep car_utils.py in the same folder (or on your Python path) as the
saved model needs it to load.
"""
import joblib
import pandas as pd

model = joblib.load('car_price_model.joblib')

# Raw values, exactly like the CSV ('?' for unknown is fine)
new_car = pd.DataFrame([{
    'symboling': 1, 'normalized-losses': '?', 'make': 'mazda',
    'fuel-type': 'gas', 'aspiration': 'std', 'num-of-doors': 'two',
    'body-style': 'hatchback', 'drive-wheels': 'fwd',
    'engine-location': 'front', 'wheel-base': 93.1, 'length': 159.1,
    'width': 64.2, 'height': 54.1, 'curb-weight': 1900,
    'engine-type': 'ohc', 'num-of-cylinders': 'four', 'engine-size': 91,
    'fuel-system': '2bbl', 'bore': 3.03, 'stroke': 3.15,
    'compression-ratio': 9, 'horsepower': 68, 'peak-rpm': 5000,
    'city-mpg': 31, 'highway-mpg': 38,
}])

print(f'Predicted price: ${model.predict(new_car)[0]:,.0f}')
