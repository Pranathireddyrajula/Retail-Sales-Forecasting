import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.title("Retail Sales Forecasting ")
st.write("""
Upload any retail sales CSV and forecast future weekly sales for all stores/products.
""")

# 1️⃣ Upload CSV
uploaded_file = st.file_uploader("Upload your sales CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # remove spaces in column names

    # Column mapping
    date_col = st.selectbox("Date column", df.columns)
    sales_col = st.selectbox("Sales column", df.columns)
    store_col = st.selectbox("Store/Product column", df.columns)

    # Optional features
    optional_features = ['Holiday_Flag','Temperature','Fuel_Price','CPI','Unemployment']
    feature_cols = []
    for f in optional_features:
        if f in df.columns:
            feature_cols.append(f)
        else:
            df[f] = 0
            feature_cols.append(f)

    # Rename columns
    df.rename(columns={date_col:'Date', sales_col:'Sales', store_col:'Store'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Store','Date'])

    # Load trained XGBoost model
    model = joblib.load("xgb_sales_model.joblib")

    # Forecast horizon
    horizon = st.slider("Forecast horizon (weeks)", 4, 52, 12)

    # Initialize forecast list
    all_forecasts = []

    stores = df['Store'].unique()

    # Forecast loop for all stores
    for store in stores:
        df_store = df[df['Store']==store].copy()
        if df_store.shape[0] < 4:
            st.warning(f"Store {store} has less than 4 historical rows. Skipping.")
            continue

        last_rows = df_store[['Sales'] + feature_cols + ['Date']].tail(4).copy()
        last_rows.reset_index(drop=True, inplace=True)

        # Predict future weeks
        for i in range(horizon):
            lag_1 = last_rows.iloc[-1]['Sales']
            lag_2 = last_rows.iloc[-2]['Sales']
            lag_4 = last_rows.iloc[-4]['Sales']
            rolling_4 = last_rows['Sales'].tail(4).mean()

            next_date = last_rows.iloc[-1]['Date'] + pd.Timedelta(weeks=1)
            week_of_year = next_date.isocalendar().week
            month = next_date.month
            year = next_date.year

            # External features
            Holiday_Flag = last_rows['Holiday_Flag'].iloc[-1]
            Temperature = last_rows['Temperature'].iloc[-1]
            Fuel_Price = last_rows['Fuel_Price'].iloc[-1]
            CPI = last_rows['CPI'].iloc[-1]
            Unemployment = last_rows['Unemployment'].iloc[-1]

            X_pred = pd.DataFrame({
                'lag_1':[lag_1],'lag_2':[lag_2],'lag_4':[lag_4],
                'rolling_4':[rolling_4],'week_of_year':[week_of_year],
                'month':[month],'year':[year],
                'Holiday_Flag':[Holiday_Flag],'Temperature':[Temperature],
                'Fuel_Price':[Fuel_Price],'CPI':[CPI],'Unemployment':[Unemployment]
            })

            pred_sales = model.predict(X_pred)[0]
            all_forecasts.append({'Store': store, 'Date': next_date, 'Predicted_Sales': pred_sales})

            new_row = pd.DataFrame({
                'Date':[next_date],'Sales':[pred_sales],
                'Holiday_Flag':[Holiday_Flag],'Temperature':[Temperature],
                'Fuel_Price':[Fuel_Price],'CPI':[CPI],'Unemployment':[Unemployment]
            })
            last_rows = pd.concat([last_rows, new_row], ignore_index=True)

    # Convert to DataFrame
    forecast_df = pd.DataFrame(all_forecasts)

    # 6️⃣ KPI calculation
    if not forecast_df.empty:
        total_forecast = forecast_df['Predicted_Sales'].sum()
        avg_weekly = forecast_df['Predicted_Sales'].mean()
        peak_forecast = forecast_df.loc[forecast_df['Predicted_Sales'].idxmax()]['Predicted_Sales']
    else:
        total_forecast = avg_weekly = peak_forecast = 0

    st.subheader("Forecast KPIs")
    st.metric("Total Forecasted Sales", f"{total_forecast:,.0f}")
    st.metric("Average Weekly Sales", f"{avg_weekly:,.0f}")
    st.metric("Peak Forecasted Week Sales", f"{peak_forecast:,.0f}")

    # 6.5️⃣ Collapsible Predicted Sales Table
    with st.expander("View Predicted Sales Table"):
        if not forecast_df.empty:
            st.dataframe(forecast_df)
        else:
            st.write("No forecast data available.")

    # 7️⃣ Historical + Forecast plot
    plot_df = pd.DataFrame()
    for store in stores:
        hist = df[df['Store']==store][['Date','Sales']].copy()
        hist['Type'] = 'Historical'
        hist['Store'] = store

        fut = forecast_df[forecast_df['Store']==store][['Date','Predicted_Sales']].copy()
        fut = fut.rename(columns={'Predicted_Sales':'Sales'})
        fut['Type'] = 'Forecasted'
        fut['Store'] = store

        combined = pd.concat([hist,fut])
        plot_df = pd.concat([plot_df, combined])

    fig = px.line(plot_df, x='Date', y='Sales', color='Store', line_dash='Type',
                  labels={'Sales':'Weekly Sales','Type':'Data Type'}, title='Historical + Forecasted Sales')
    st.plotly_chart(fig, use_container_width=True)

    # 8️⃣ Feature importance chart
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance (XGBoost)")
        feat_imp = pd.DataFrame({'Feature': model.get_booster().feature_names, 'Importance': model.feature_importances_})
        fig2 = px.bar(feat_imp.sort_values('Importance', ascending=False), x='Feature', y='Importance', title="Feature Importance")
        st.plotly_chart(fig2, use_container_width=True)

    # 9️⃣ Download forecast CSV
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button("Download Forecast CSV", data=csv, file_name="forecast_all_stores.csv", mime="text/csv")
