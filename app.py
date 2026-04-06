from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
import os

# Import dari model.py
from model import (df, mae, mse, r2, intercept, 
                   coef_date, coef_pelanggan,
                   future_years, future_preds, model)

app = Flask(__name__)

# ================== CONFIG UPLOAD ==================
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================== HALAMAN UTAMA ==================
@app.route('/', methods=['GET', 'POST'])
def index():
    global model, df

    prediction = None
    year_input = None
    pelanggan_input = None
    error = None

    if request.method == 'POST':
        try:
            year_input = int(request.form['year'])
            pelanggan_input = int(request.form['pelanggan'])

            if 2000 <= year_input <= 2050:
                input_df = pd.DataFrame([[year_input, pelanggan_input]],
                                        columns=['Date', 'Jumlah_Pelanggan'])
                pred_value = model.predict(input_df)[0]
                prediction = round(pred_value, 2)
            else:
                error = "Tahun harus antara 2000 - 2050"
        except:
            error = "Masukkan input yang valid!"

    # ================== TABEL ==================
    df_display = df.copy()

    def format_harga(x):
        if float(x).is_integer():
            return f"Rp {int(x):,}"
        else:
            return f"Rp {x:,.2f}"

    df_display['Harga_Emas'] = df_display['Harga_Emas'].apply(format_harga)
    df_display['Jumlah_Pelanggan'] = df_display['Jumlah_Pelanggan'].apply(lambda x: f"{int(x):,}")

    df_display = df_display.rename(columns={
        'Date': 'Tahun',
        'Harga_Emas': 'Harga Emas (Rp)',
        'Jumlah_Pelanggan': 'Jumlah Pelanggan'
    })

    table_html = df_display.to_html(
        classes="table table-striped table-hover table-bordered align-middle",
        index=False,
        border=0
    )

    # ================== GRAFIK ==================
    plt.figure(figsize=(10, 6))

    # Data aktual
    plt.scatter(df['Date'], df['Harga_Emas'], color='gold', label='Data Aktual')

    # Garis regresi
    plt.plot(df['Date'], model.predict(df[['Date', 'Jumlah_Pelanggan']]),
             color='red', label='Garis Regresi', linewidth=2.5)

    # Titik prediksi
    if prediction is not None:
        plt.scatter([year_input], [prediction],
                    s=150, marker='*', color='green',
                    label=f'Prediksi {year_input}')

    plt.xlabel('Tahun')
    plt.ylabel('Harga Emas (Rp)')
    plt.title('Tren & Prediksi Harga Emas')
    plt.legend()
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # ================== RENDER ==================
    return render_template('index.html',
                           table_html=table_html,
                           mae=round(mae, 2),
                           mse=round(mse, 2),
                           r2=round(r2, 4),
                           intercept=round(intercept, 2),
                           coef_date=round(coef_date, 2),
                           coef_pelanggan=round(coef_pelanggan, 2),
                           prediction=prediction,
                           year=year_input,
                           pelanggan=pelanggan_input,
                           plot_url=plot_url,
                           error=error,
                           future_years=future_years,
                           future_preds=future_preds,
                           zip=zip)

# ================== UPLOAD DATASET ==================
@app.route('/upload', methods=['POST'])
def upload():
    global model, df

    file = request.files['file']

    if file and file.filename != '':
        try:
            # simpan file ke folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # baca file
            df = pd.read_csv(filepath)

            # pastikan format kolom
            df.columns = ['Date', 'Harga_Emas', 'Jumlah_Pelanggan']

            X = df[['Date', 'Jumlah_Pelanggan']]
            y = df['Harga_Emas']

            model = LinearRegression()
            model.fit(X, y)

            return "✅ Dataset berhasil diupload & model diperbarui!"
        except:
            return "Format dataset salah! Pastikan kolom: Date, Harga_Emas, Jumlah_Pelanggan"

    return "Upload gagal!"

# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True)