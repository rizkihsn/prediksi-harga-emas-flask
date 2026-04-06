import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io

# ================== DATASET ==================
csv_data = """Date,Harga_Emas,Jumlah_Pelanggan
2010,12663555,12093
2011,13882342.5,12094
2012,15974156.3,12095
2013,14658765,12096
2014,14936310,12097
2015,14612100,12098
2016,15438136.4,12099
2017,17515642.5,12100
2018,18392020,12101
2019,21028516.9,12102
2020,26520781.4,12103
2021,26404909.4,12104
"""

df = pd.read_csv(io.StringIO(csv_data))

# ================== TRAINING MODEL (REGRESI BERGANDA) ==================
X = df[['Date', 'Jumlah_Pelanggan']]
y = df['Harga_Emas']

model = LinearRegression()
model.fit(X, y)

# ================== EVALUASI ==================
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

intercept = model.intercept_
coef_date = model.coef_[0]
coef_pelanggan = model.coef_[1]

# ================== PREDIKSI MASA DEPAN ==================
future_years = list(range(2025, 2031))

# Asumsi jumlah pelanggan naik linear seperti data sebelumnya
last_pelanggan = df['Jumlah_Pelanggan'].iloc[-1]
future_pelanggan = [last_pelanggan + i for i in range(1, len(future_years)+1)]

future_preds = []

for year, pelanggan in zip(future_years, future_pelanggan):
    input_df = pd.DataFrame([[year, pelanggan]], 
                            columns=['Date', 'Jumlah_Pelanggan'])
    pred = model.predict(input_df)[0]
    future_preds.append(round(pred, 2))

# ================== OUTPUT KE TERMINAL ==================
if __name__ == "__main__":
    print("="*60)
    print("HASIL VALIDASI MODEL REGRESI LINEAR BERGANDA")
    print("="*60)

    print(f"MAE  : {mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"R²   : {r2:.4f}")

    print("\nKoefisien Model:")
    print(f"Intercept          : {intercept:,.2f}")
    print(f"Koefisien Tahun    : {coef_date:,.2f}")
    print(f"Koefisien Pelanggan: {coef_pelanggan:,.2f}")

    print("\nPersamaan Regresi:")
    print(f"Harga = {intercept:,.2f} + ({coef_date:,.2f} × Tahun) + ({coef_pelanggan:,.2f} × Pelanggan)")

    print("\nInterpretasi:")
    print(f"- Setiap kenaikan 1 tahun, harga naik sekitar Rp {coef_date:,.2f}")
    print(f"- Setiap tambahan 1 pelanggan, harga naik sekitar Rp {coef_pelanggan:,.2f}")

    print("\nPrediksi Masa Depan (2025-2030):")
    for year, pelanggan, pred in zip(future_years, future_pelanggan, future_preds):
        print(f"Tahun {year} | Pelanggan {pelanggan} → Rp {pred:,.2f}")

    print("="*60)

    # ================== GRAFIK ==================
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Data aktual
    plt.scatter(df['Date'], df['Harga_Emas'], 
                 color='gold',label='Data Aktual', s=70)

    # Garis regresi
    plt.plot(df['Date'], model.predict(X), 
             color='red', label='Garis Regresi', linewidth=2.5)

    plt.xlabel('Tahun')
    plt.ylabel('Harga Emas (Rp)')
    plt.title('Tren & Prediksi Harga Emas')
    plt.legend()
    plt.grid(True)

    plt.show()