import pandas as pd

data = pd.read_csv("sales_data.txt")

expected = ['Product', 'Price', 'Quantity', 'Tax']
for col in expected:
    if col not in data.columns:
        raise NameError(f"Missing expected column: {col}")

data['Price'] = pd.to_numeric(data['Price'],errors='coerce')
data['Quantity'] = pd.to_numeric(data['Quantity'],errors='coerce')
data['Revenue Before Tax'] = data['Price']*data['Quantity']
data['Revenue After Tax'] = data['Revenue Before Tax'] * (1 - data['Tax'])
data[['Revenue Before Tax', 'Revenue After Tax']] = data[['Revenue Before Tax', 'Revenue After Tax']].fillna(0).round(2)
filter_100 = data[data['Revenue Before Tax']>100]
data['Product'] = data['Product'].apply(lambda x: x.upper())

print(data.info())

