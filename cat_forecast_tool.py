import streamlit as st

def run(pid: int):
    import streamlit as st
    
    if st.button("← Back to dashboard"):
        st.session_state.current_project = None        # clear router flags
        st.session_state.current_type    = None
        st.rerun()


    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    # from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    from plotly.subplots import make_subplots
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    from streamlit_option_menu import option_menu
    from statsmodels.tsa.seasonal import STL
    from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    import xgboost as xgb
    from prophet import Prophet
    import statsmodels.regression.linear_model as sm
    from statsmodels.tools import add_constant
    import io
    import pickle
    import os
    import hashlib

    import plotly.graph_objects as go
    from statsmodels.tsa.seasonal import seasonal_decompose
    import statsmodels.api as sm
    # import pmdarima as pm

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.base import BaseEstimator, RegressorMixin




    def detect_date_column(df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                return col
        return None

    # selected=option_menu(
    #     menu_title="",
    #     options=["PRE-PROCESS","EXPLORE","ENGINEER","MODEL","EVALUATE"],
    #     icons = ["sliders",         # PRE-PROCESS – tuning, adjustments
    #          "search",          # EXPLORE – data exploration
    #          "tools",           # ENGINEER – feature engineering
    #          "cpu",             # MODEL – training a model
    #          "bar-chart"]   ,    
    #     # icons=["database","diagram-3","clipboard-data"],
    #     orientation="horizontal"
    # )


    # Custom CSS for styling
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color:rgb(253, 241, 171); /* Ochre */
            color: black;
        }
        .stButton>button {
            background-color: #F0E68C; /* Ochre */
            color: black;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FFD700; /* Darker ochre */
            color: white;
        }
        .stSlider>div, .stSelectbox, .stMultiselect {
            color: black;
            border-color: black;
        }
        .stRadio div[role="radiogroup"] > label > div:first-child {
            background-color: #F0E68C; /* Ochre */
            color: black;
        }
        .stCheckbox label {
            color: black;
        }
        hr.thick {
            border: 2px solid black;
        }
        hr.thin {
            border: 1px solid gray;
        }
        </style>
    """, unsafe_allow_html=True)

    # def fit_models(data, target_col, date_col, future_periods, frequency="M"):
    #     """Fit multiple forecasting models and return forecasts."""
    #     forecasts = {}

    #     # Determine seasonal periods based on frequency
    #     if frequency == "D":  # Daily
    #         seasonal_periods = 7  # Weekly seasonality
    #     elif frequency == "W":  # Weekly
    #         seasonal_periods = 52  # Yearly seasonality
    #     elif frequency == "Q":  # Quarterly
    #         seasonal_periods = 4  # Yearly seasonality
    #     elif frequency == "Y":  # Yearly
    #         seasonal_periods = 1  # No seasonality
    #     else:  # Default to monthly
    #         seasonal_periods = 12  # Yearly seasonality

    #     # Holt-Winters (Triple Exponential Smoothing)
    #     if len(data) >= 2 * seasonal_periods:  # Ensure enough data for seasonality
    #         hw_model = ExponentialSmoothing(data[target_col], trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
    #         hw_forecast = hw_model.forecast(steps=future_periods)
    #     else:
    #         hw_forecast = [None] * future_periods

    #     # Moving Average Forecast
    #     sma_forecast = data[target_col].rolling(window=3, min_periods=1).mean().tolist() + [data[target_col].iloc[-3:].mean()] * future_periods

    #     # Exponential Smoothing (ETS)
    #     if len(data) >= 2 * seasonal_periods:
    #         ets_model = ExponentialSmoothing(data[target_col], seasonal="add", seasonal_periods=seasonal_periods).fit()
    #         ets_forecast = ets_model.forecast(steps=future_periods)
    #     else:
    #         ets_forecast = [None] * future_periods

    #     # ARIMA Model (AutoARIMA)
    #     arima_model = auto_arima(data[target_col], seasonal=False, trace=True)
    #     arima_forecast = arima_model.predict(n_periods=future_periods)
    #     arima_fitted = arima_model.predict_in_sample(start=1, end=len(data))  # Skip the first prediction

    #     # SARIMA Model (AutoARIMA)
    #     sarima_model = auto_arima(data[target_col], seasonal=True, m=seasonal_periods, trace=True)
    #     sarima_forecast = sarima_model.predict(n_periods=future_periods)
    #     sarima_fitted = sarima_model.predict_in_sample(start=1, end=len(data))  # Skip the first prediction

    #     # Prophet Model
    #     df_prophet = data.reset_index().rename(columns={date_col: "ds", target_col: "y"})
    #     prophet_model = Prophet()
    #     prophet_model.fit(df_prophet)
    #     future_prophet = prophet_model.make_future_dataframe(periods=future_periods, freq=frequency)
    #     prophet_forecast_df = prophet_model.predict(future_prophet)
    #     prophet_fitted = prophet_forecast_df["yhat"].values[: len(data)]  # Past fitted values
    #     prophet_forecast = prophet_forecast_df["yhat"].values[-future_periods:]  # Future forecasts

    #     # Store results
    #     forecast_length = len(data) + future_periods
    #     df_forecast = pd.DataFrame({
    #         "Date": list(data.index) + list(pd.date_range(start=data.index[-1] + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1), periods=future_periods, freq=frequency)),
    #         "Actual": list(data[target_col]) + [None] * future_periods,
    #         "SMA": sma_forecast[:forecast_length],
    #         "Holt-Winters": (list(hw_model.fittedvalues) if len(data) >= 2 * seasonal_periods else []) + list(hw_forecast),
    #         "ETS": (list(ets_model.fittedvalues) if len(data) >= 2 * seasonal_periods else []) + list(ets_forecast),
    #         "ARIMA": list(arima_fitted) + list(arima_forecast),
    #         "SARIMA": list(sarima_fitted) + list(sarima_forecast),
    #         "Prophet": np.concatenate((prophet_fitted, prophet_forecast)),
    #     })

    #     # Calculate accuracy (1 - MAPE)
    #     accuracy = {}
    #     for model in ["SMA", "Holt-Winters", "ETS", "ARIMA", "SARIMA", "Prophet"]:
    #         if model in df_forecast.columns:
    #             actual = df_forecast["Actual"].dropna()
    #             predicted = df_forecast[model].dropna()
    #             if len(actual) > 0 and len(predicted) > 0:
    #                 mape = mean_absolute_percentage_error(actual, predicted[:len(actual)])
    #                 accuracy[model] = 1 - mape
    #             else:
    #                 accuracy[model] = None

    #     return df_forecast, accuracy


    def fit_models(data, target_col, date_col, future_periods, frequency="M"):
        """Fit multiple forecasting models and return forecasts."""
        forecasts = {}

        # Determine seasonal periods based on frequency
        if frequency == "D":  # Daily
            seasonal_periods = 7  # Weekly seasonality
        elif frequency == "W":  # Weekly
            seasonal_periods = 52  # Yearly seasonality
        elif frequency == "Q":  # Quarterly
            seasonal_periods = 4  # Yearly seasonality
        elif frequency == "Y":  # Yearly
            seasonal_periods = 1  # No seasonality
        else:  # Default to monthly
            seasonal_periods = 12  # Yearly seasonality

        # Holt-Winters (Triple Exponential Smoothing)
        if len(data) >= 2 * seasonal_periods:  # Ensure enough data for seasonality
            hw_model = ExponentialSmoothing(data[target_col], trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
            hw_forecast = hw_model.forecast(steps=future_periods)
        else:
            hw_forecast = [None] * future_periods

        # # Moving Average Forecast
        sma_forecast = data[target_col].rolling(window=3, min_periods=1).mean().tolist() + [data[target_col].iloc[-3:].mean()] * future_periods

        # Exponential Smoothing (ETS)
        if len(data) >= 2 * seasonal_periods:
            ets_model = ExponentialSmoothing(data[target_col], seasonal="add", seasonal_periods=seasonal_periods).fit()
            ets_forecast = ets_model.forecast(steps=future_periods)
        else:
            ets_forecast = [None] * future_periods

        # ARIMA Model (AutoARIMA)
        try:
            arima_model = auto_arima(data[target_col], seasonal=False, trace=True)
            arima_forecast = arima_model.predict(n_periods=future_periods)
            arima_fitted = arima_model.predict_in_sample(start=1, end=len(data))  # Skip the first prediction
        except Exception as e:
            print(f"ARIMA failed for group: {e}")
            arima_forecast = [None] * future_periods
            arima_fitted = [None] * len(data)

        # SARIMA Model (AutoARIMA)
        try:
            if len(data) >= 2 * seasonal_periods:  # Only fit SARIMA if there's enough data
                sarima_model = auto_arima(data[target_col], seasonal=True, m=seasonal_periods, trace=True)
                sarima_forecast = sarima_model.predict(n_periods=future_periods)
                sarima_fitted = sarima_model.predict_in_sample(start=1, end=len(data))  # Skip the first prediction
            else:
                sarima_forecast = [None] * future_periods
                sarima_fitted = [None] * len(data)
        except Exception as e:
            print(f"SARIMA failed for group: {e}")
            sarima_forecast = [None] * future_periods
            sarima_fitted = [None] * len(data)

        # Prophet Model
        # changepoints = ['2020-01-01', '2022-01-01']
        try:
            df_prophet = data.reset_index().rename(columns={date_col: "ds", target_col: "y"})
            prophet_model = Prophet(
                
                                    # changepoint_prior_scale=0.01,
                                    # seasonality_mode='additive', 
                                    # n_changepoints=15
                                    )
            prophet_model.fit(df_prophet)
            future_prophet = prophet_model.make_future_dataframe(periods=future_periods, freq=frequency)
            prophet_forecast_df = prophet_model.predict(future_prophet)
            prophet_fitted = prophet_forecast_df["yhat"].values[: len(data)]  # Past fitted values
            prophet_forecast = prophet_forecast_df["yhat"].values[-future_periods:]  # Future forecasts
        except Exception as e:
            print(f"Prophet failed for group: {e}")
            prophet_fitted = [None] * len(data)
            prophet_forecast = [None] * future_periods

        # Ensure all columns have the same length
        forecast_length = len(data) + future_periods
        date_column = list(data.index) + list(pd.date_range(start=data.index[-1] + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1), periods=future_periods, freq=frequency))
        actual_column = list(data[target_col]) + [None] * future_periods
        sma_column = sma_forecast[:forecast_length]
        hw_column = (list(hw_model.fittedvalues) if len(data) >= 2 * seasonal_periods else [None] * len(data)) + list(hw_forecast)
        ets_column = (list(ets_model.fittedvalues) if len(data) >= 2 * seasonal_periods else [None] * len(data)) + list(ets_forecast)
        arima_column = list(arima_fitted) + list(arima_forecast)
        sarima_column = list(sarima_fitted) + list(sarima_forecast)
        prophet_column = list(prophet_fitted) + list(prophet_forecast)

        # Store results
        df_forecast = pd.DataFrame({
            "Date": date_column,
            "Actual": actual_column,
            "SMA": sma_column,
            "Holt-Winters": hw_column,
            "ETS": ets_column,
            # "ARIMA": arima_column,
            "SARIMA": sarima_column,
            "Prophet": prophet_column,
        })

        # Calculate accuracy (1 - MAPE)
        accuracy = {}
        for model in ["SMA", "Holt-Winters", "ETS", "ARIMA", "SARIMA", "Prophet"]:
            if model in df_forecast.columns:
                actual = df_forecast["Actual"].dropna()
                predicted = df_forecast[model].dropna()
                if len(actual) > 0 and len(predicted) > 0:
                    mape = mean_absolute_percentage_error(actual, predicted[:len(actual)])
                    accuracy[model] = 1 - mape
                else:
                    accuracy[model] = None

        return df_forecast, accuracy


    # def calculate_annual_growth_single(forecast_results, forecast_horizon, start_year=2020, fiscal_start_month=1):
    #     """Calculate annual growth rates for each segment and model."""
    #     growth_results = {}

    #     for segment, df_forecast in forecast_results.items():
    #         models = ["Prophet", "Holt-Winters", "SARIMA", "ARIMA", "ETS"]  # Add all models here
    #         growth_results[segment] = {}

    #         for model in models:
    #             # Extract actual and forecasted data
    #             actual_dates = df_forecast["Date"][: len(df_forecast) - forecast_horizon]  # Past data
    #             actual_volume = df_forecast["Actual"][: len(df_forecast) - forecast_horizon]  # Past volume
    #             future_dates = df_forecast["Date"][-forecast_horizon:]  # Future dates (36 months)
    #             model_forecast = df_forecast[model][-forecast_horizon:]  # Model forecast
                
    #             # Combine actual and predicted data into a single DataFrame
    #             combined_df = pd.DataFrame({
    #                 "date": pd.concat([actual_dates, future_dates], ignore_index=True),
    #                 "volume": pd.concat([actual_volume, model_forecast], ignore_index=True),
    #             })

    #             # Ensure index alignment by resetting it before filtering
    #             combined_df = combined_df.reset_index(drop=True)

    #             # Apply the mask on the DataFrame instead of individual Series
    #             combined_df = combined_df[combined_df['date'].dt.year >= start_year - 1]  # Include previous year for fiscal year

    #             # Convert 'date' to datetime and set as index
    #             combined_df["date"] = pd.to_datetime(combined_df["date"])
    #             combined_df.set_index("date", inplace=True)

    #             # Adjust dates to align with the fiscal year start month
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year

    #             # Group by fiscal year and sum the volume
    #             annual_df = combined_df.groupby('fiscal_year')['volume'].mean().reset_index()

    #             # Calculate annual growth rate (YoY comparison)
    #             annual_df[f"growth_rate_{model}"] = annual_df["volume"].pct_change(periods=1)

    #             # Store results
    #             growth_results[segment][model] = annual_df

    #     return growth_results





    def calculate_annual_growth_single(forecast_results, forecast_horizon, start_year=2017, fiscal_start_month=1, frequency="M"):
        """Calculate annual growth rates for each segment and model."""
        growth_results = {}

        for segment, df_forecast in forecast_results.items():
            models = ["Prophet", "Holt-Winters", "SARIMA", "ETS"]  # Add all models here
            growth_results[segment] = {}

            

            for model in models:
                # Extract actual and forecasted data
                actual_dates = df_forecast["Date"][: len(df_forecast) - forecast_horizon]  # Past data
                actual_volume = df_forecast["Actual"][: len(df_forecast) - forecast_horizon]  # Past volume
                future_dates = df_forecast["Date"][-forecast_horizon:]  # Future dates
                model_forecast = df_forecast[model][-forecast_horizon:]  # Model forecast

                # Combine actual and predicted data into a single DataFrame
                combined_df = pd.DataFrame({
                    "date": pd.concat([actual_dates, future_dates], ignore_index=True),
                    "volume": pd.concat([actual_volume, model_forecast], ignore_index=True),
                })

                # Ensure index alignment by resetting it before filtering
                combined_df = combined_df.reset_index(drop=True)

                # Apply the mask on the DataFrame instead of individual Series
                combined_df = combined_df[combined_df['date'].dt.year >= start_year - 1]  # Include previous year for fiscal year

                # Convert 'date' to datetime and set as index
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                combined_df.set_index("date", inplace=True)

                # Adjust dates to align with the fiscal year start month based on frequency
                if frequency == "D":  # Daily data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "W":  # Weekly data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "M":  # Monthly data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "Q":  # Quarterly data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "Y":  # Yearly data
                    combined_df['fiscal_year'] = combined_df.index.year  # No adjustment needed for yearly data

                # Group by fiscal year and calculate mean volume
                if frequency in ["D", "W", "M", "Q"]:  # For daily, weekly, monthly, or quarterly data
                    annual_df = combined_df.groupby('fiscal_year')['volume'].mean().reset_index()
                elif frequency == "Y":  # For yearly data
                    annual_df = combined_df.groupby('fiscal_year')['volume'].mean().reset_index()


                # ADDED: Shift fiscal year index by +1 (only if not January fiscal year)
                # if fiscal_start_month != 1:
                #     annual_df['fiscal_year'] = annual_df['fiscal_year'] + 1

                if not (1 <= fiscal_start_month <= 5):
                    annual_df['fiscal_year'] = annual_df['fiscal_year'] + 1


                # Calculate annual growth rate (YoY comparison)
                annual_df[f"growth_rate_{model}"] = annual_df["volume"].pct_change(periods=1)

                # Store results
                growth_results[segment][model] = annual_df

        return growth_results






    # def calculate_annual_growth(forecast_results, model_name, forecast_horizon, start_year=2020, fiscal_start_month=1, frequency="M"):
    #     """Calculate annual growth rates for each segment and model."""
    #     growth_results = {}

    #     for segment, data in forecast_results.items():
    #         # Extract actual and forecasted data

    #         st.write(data)
    #         actual_dates = pd.to_datetime(data['actual_dates'])
    #         actual_volume = data['actual_volume']
    #         future_dates = pd.date_range(start=actual_dates.max() + pd.DateOffset(months=1), periods=forecast_horizon, freq=frequency)
    #         future_forecast = data[f'{model_name}_future_forecast']

    #         # Convert DatetimeIndex to Series
    #         actual_dates_series = pd.Series(actual_dates)
    #         future_dates_series = pd.Series(future_dates)

    #         # Combine actual and predicted data into a single DataFrame
    #         combined_df = pd.DataFrame({
    #             'date': pd.concat([actual_dates_series, future_dates_series], ignore_index=True),
    #             'volume': pd.concat([pd.Series(actual_volume), pd.Series(future_forecast)], ignore_index=True)
    #         })

    #         # Ensure index alignment by resetting it before filtering
    #         combined_df = combined_df.reset_index(drop=True)

    #         # Apply the mask on the DataFrame instead of individual Series
    #         combined_df = combined_df[combined_df['date'].dt.year >= start_year - 1]  # Include previous year for fiscal year

    #         # Convert 'date' to datetime and set as index
    #         combined_df['date'] = pd.to_datetime(combined_df['date'])
    #         combined_df.set_index('date', inplace=True)

    #         # Adjust dates to align with the fiscal year start month based on frequency
    #         if frequency == "D":  # Daily data
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
    #         elif frequency == "W":  # Weekly data
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
    #         elif frequency == "M":  # Monthly data
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
    #         elif frequency == "Q":  # Quarterly data
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
    #         elif frequency == "Y":  # Yearly data
    #             combined_df['fiscal_year'] = combined_df.index.year  # No adjustment needed for yearly data

    #         # Group by fiscal year and calculate mean volume
    #         if frequency in ["D", "W", "M", "Q"]:  # For daily, weekly, monthly, or quarterly data
    #             annual_df = combined_df.groupby('fiscal_year')['volume'].mean().reset_index()
    #         elif frequency == "Y":  # For yearly data
    #             annual_df = combined_df.groupby('fiscal_year')['volume'].mean().reset_index()


    #          # ADDED: Shift fiscal year index by +1 (only if not January fiscal year)
    #         # if fiscal_start_month != 1:
    #         #     annual_df['fiscal_year'] = annual_df['fiscal_year'] + 1

    #         if not (1 <= fiscal_start_month <= 5):
    #             annual_df['fiscal_year'] = annual_df['fiscal_year'] + 1





    #         # Calculate annual growth rate (YoY comparison)
    #         annual_df[f'growth_rate_{model_name}'] = annual_df['volume'].pct_change(periods=1)

    #         # Store results
    #         growth_results[segment] = annual_df

    #     return growth_results



    def calculate_feature_growth(forecast_results, model_name, forecast_horizon, start_year=2017, fiscal_start_month=1, frequency="M"):
        """Calculate annual growth rates for each feature or component."""
        growth_results = {}

        for segment, data in forecast_results.items():
            # Extract feature or component forecasts
            if 'feature_forecasts' in data:
                forecasts = data['feature_forecasts']
            elif 'component_forecasts' in data:
                forecasts = data['component_forecasts']
            else:
                continue  # Skip if no feature or component forecasts are available

            for feature_name, feature_data in forecasts.items():
                # Extract actual and forecasted data
                actual_values = feature_data['actual_values']
                past_forecast = feature_data['past_forecast']
                future_forecast = feature_data['future_forecast']

                # Create date ranges
                actual_dates = pd.to_datetime(data['actual_dates'])
                future_dates = pd.date_range(start=actual_dates.max() + pd.DateOffset(months=1), periods=forecast_horizon, freq=frequency)

                # Combine actual and forecasted data
                combined_df = pd.DataFrame({
                    'date': pd.concat([pd.Series(actual_dates), pd.Series(future_dates)], ignore_index=True),
                    'volume': pd.concat([pd.Series(actual_values), pd.Series(past_forecast), pd.Series(future_forecast)], ignore_index=True)
                })

                # Filter data to include only the relevant years
                combined_df = combined_df[combined_df['date'].dt.year >= start_year - 1]

                # Convert 'date' to datetime and set as index
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df.set_index('date', inplace=True)

                # Adjust dates to align with the fiscal year start month
                if frequency in ["D", "W", "M", "Q"]:  # For daily, weekly, monthly, or quarterly data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "Y":  # For yearly data
                    combined_df['fiscal_year'] = combined_df.index.year

                # Group by fiscal year and calculate mean volume
                annual_df = combined_df.groupby('fiscal_year')['volume'].mean().reset_index()

                if not (1 <= fiscal_start_month <= 5):
                    annual_df['fiscal_year'] = annual_df['fiscal_year'] + 1





                # Calculate annual growth rate (YoY comparison)
                annual_df[f'growth_rate_{model_name}'] = annual_df['volume'].pct_change(periods=1)

                # Store results
                growth_results[f"{segment}_{feature_name}"] = annual_df

        return growth_results





    # def calculate_halfyearly_growth_single(forecast_results, forecast_horizon, start_year=2020, fiscal_start_month=1):
    #     """Calculate half-yearly growth rates for each segment and model."""
    #     growth_results = {}

    #     for segment, df_forecast in forecast_results.items():
    #         # Ensure df_forecast is a DataFrame
    #         if not isinstance(df_forecast, pd.DataFrame):
    #             raise ValueError(f"Expected a DataFrame for segment {segment}, but got {type(df_forecast)}")

    #         models = ["Prophet", "Holt-Winters", "SARIMA", "ARIMA", "ETS"]  # Add all models here
    #         growth_results[segment] = {}

    #         for model in models:
    #             # Check if the model column exists in the DataFrame
    #             if model not in df_forecast.columns:
    #                 print(f"Warning: Model '{model}' not found in segment '{segment}'. Skipping.")
    #                 continue

    #             # Extract actual and forecasted data
    #             actual_dates = df_forecast["Date"][: len(df_forecast) - forecast_horizon]  # Past data
    #             actual_volume = df_forecast["Actual"][: len(df_forecast) - forecast_horizon]  # Past volume
    #             future_dates = df_forecast["Date"][-forecast_horizon:]  # Future dates
    #             model_forecast = df_forecast[model][-forecast_horizon:]  # Model forecast

    #             # Combine actual and predicted data into a single DataFrame
    #             combined_df = pd.DataFrame({
    #                 "date": pd.concat([actual_dates, future_dates], ignore_index=True),
    #                 "volume": pd.concat([actual_volume, model_forecast], ignore_index=True),
    #             })

    #             # Convert 'date' to datetime and set as index
    #             combined_df["date"] = pd.to_datetime(combined_df["date"])
    #             combined_df.set_index("date", inplace=True)

    #             # Adjust dates to align with the fiscal year start month
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
    #             combined_df['fiscal_half'] = ['H1' if month >= fiscal_start_month else 'H2' for month in combined_df.index.month]

    #             # Group by fiscal year and fiscal half, then sum the volume
    #             halfyear_df = combined_df.groupby(['fiscal_year', 'fiscal_half'])['volume'].mean().reset_index()

    #             # Calculate half-yearly growth rate (YoY comparison)
    #             halfyear_df[f"growth_rate_{model}"] = halfyear_df["volume"].pct_change(periods=2)

    #             # Store results
    #             growth_results[segment][model] = halfyear_df

    #     return growth_results


    def calculate_halfyearly_growth_single(forecast_results, forecast_horizon, start_year=2017, fiscal_start_month=1, frequency="M"):
        """Calculate half-yearly growth rates for each segment and model."""
        growth_results = {}

        for segment, df_forecast in forecast_results.items():
            # Ensure df_forecast is a DataFrame
            if not isinstance(df_forecast, pd.DataFrame):
                raise ValueError(f"Expected a DataFrame for segment {segment}, but got {type(df_forecast)}")

            models = ["Prophet", "Holt-Winters", "SARIMA", "ETS"]  # Add all models here
            growth_results[segment] = {}

            for model in models:
                # Check if the model column exists in the DataFrame
                if model not in df_forecast.columns:
                    print(f"Warning: Model '{model}' not found in segment '{segment}'. Skipping.")
                    continue

                # Extract actual and forecasted data
                actual_dates = df_forecast["Date"][: len(df_forecast) - forecast_horizon]  # Past data
                actual_volume = df_forecast["Actual"][: len(df_forecast) - forecast_horizon]  # Past volume
                future_dates = df_forecast["Date"][-forecast_horizon:]  # Future dates
                model_forecast = df_forecast[model][-forecast_horizon:]  # Model forecast

                # Combine actual and predicted data into a single DataFrame
                combined_df = pd.DataFrame({
                    "date": pd.concat([actual_dates, future_dates], ignore_index=True),
                    "volume": pd.concat([actual_volume, model_forecast], ignore_index=True),
                })

                # Convert 'date' to datetime and set as index
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                combined_df.set_index("date", inplace=True)

                # Adjust dates to align with the fiscal year start month based on frequency
                if frequency in ["D", "W", "M", "Q"]:  # For daily, weekly, monthly, or quarterly data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "Y":  # For yearly data
                    combined_df['fiscal_year'] = combined_df.index.year  # No adjustment needed for yearly data

                # Determine fiscal half (H1 or H2) based on the fiscal year start month
                if frequency in ["D", "W", "M"]:  # For daily, weekly, or monthly data
                    combined_df['fiscal_half'] = [
                        'H1' if (month - fiscal_start_month) % 12 < 6 else 'H2'
                        for month in combined_df.index.month
                    ]
                # elif frequency =="M":  # For daily, weekly, or monthly data
                #     combined_df['fiscal_half'] = ['H1' if month < fiscal_start_month else 'H2' for month in combined_df.index.month]
                elif frequency == "Q":  # For quarterly data
                    combined_df['fiscal_half'] = ['H1' if quarter < 3 else 'H2' for quarter in (combined_df.index.month - 1) // 3 + 1]
                elif frequency == "Y":  # For yearly data
                    combined_df['fiscal_half'] = 'H1'  # Yearly data is treated as a single half

                # Group by fiscal year and fiscal half, then calculate mean volume
                halfyear_df = combined_df.groupby(['fiscal_year', 'fiscal_half'])['volume'].mean().reset_index()

                # ADDED: Shift fiscal year index by +1 (only if not January fiscal year)
                # if fiscal_start_month != 1:
                #     halfyear_df['fiscal_year'] = halfyear_df['fiscal_year'] + 1

                if not (1 <= fiscal_start_month <= 5):
                    halfyear_df['fiscal_year'] = halfyear_df['fiscal_year'] + 1




                # Calculate half-yearly growth rate (YoY comparison)
                halfyear_df[f"growth_rate_{model}"] = halfyear_df["volume"].pct_change(periods=2)

                # Store results
                growth_results[segment][model] = halfyear_df

        return growth_results


    # def calculate_quarterly_growth_single(forecast_results, forecast_horizon, start_year=2020, fiscal_start_month=1):
    #     """Calculate quarterly growth rates for each segment and model."""
    #     growth_results = {}

    #     for segment, df_forecast in forecast_results.items():
    #         # Ensure df_forecast is a DataFrame
    #         if not isinstance(df_forecast, pd.DataFrame):
    #             raise ValueError(f"Expected a DataFrame for segment {segment}, but got {type(df_forecast)}")

    #         models = ["Prophet", "Holt-Winters", "SARIMA", "ARIMA", "ETS"]  # Add all models here
    #         growth_results[segment] = {}

    #         for model in models:
    #             # Check if the model column exists in the DataFrame
    #             if model not in df_forecast.columns:
    #                 print(f"Warning: Model '{model}' not found in segment '{segment}'. Skipping.")
    #                 continue

    #             # Extract actual and forecasted data
    #             actual_dates = df_forecast["Date"][: len(df_forecast) - forecast_horizon]  # Past data
    #             actual_volume = df_forecast["Actual"][: len(df_forecast) - forecast_horizon]  # Past volume
    #             future_dates = df_forecast["Date"][-forecast_horizon:]  # Future dates
    #             model_forecast = df_forecast[model][-forecast_horizon:]  # Model forecast

    #             # Combine actual and predicted data into a single DataFrame
    #             combined_df = pd.DataFrame({
    #                 "date": pd.concat([actual_dates, future_dates], ignore_index=True),
    #                 "volume": pd.concat([actual_volume, model_forecast], ignore_index=True),
    #             })

    #             # Convert 'date' to datetime and set as index
    #             combined_df["date"] = pd.to_datetime(combined_df["date"])
    #             combined_df.set_index("date", inplace=True)

    #             # Adjust dates to align with the fiscal year start month
    #             combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
    #             combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year

    #             # Determine the fiscal quarter based on the fiscal year start month
    #             combined_df['fiscal_quarter'] = ((combined_df.index.month - fiscal_start_month) % 12 // 3) + 1

    #             # Group by fiscal year and fiscal quarter, then sum the volume
    #             quarterly_df = combined_df.groupby(['fiscal_year', 'fiscal_quarter'])['volume'].mean().reset_index()

    #             # Calculate quarterly growth rate (QoQ comparison)
    #             quarterly_df[f"growth_rate_{model}"] = quarterly_df["volume"].pct_change(periods=1)

    #             # Store results
    #             growth_results[segment][model] = quarterly_df

    #     return growth_results


    def calculate_quarterly_growth_single(forecast_results, forecast_horizon, start_year=2017, fiscal_start_month=1, frequency="M"):
        """Calculate quarterly growth rates for each segment and model."""
        growth_results = {}

        for segment, df_forecast in forecast_results.items():
            # Ensure df_forecast is a DataFrame
            if not isinstance(df_forecast, pd.DataFrame):
                raise ValueError(f"Expected a DataFrame for segment {segment}, but got {type(df_forecast)}")

            models = ["Prophet", "Holt-Winters", "SARIMA", "ETS"]  # Add all models here
            growth_results[segment] = {}

            for model in models:
                # Check if the model column exists in the DataFrame
                if model not in df_forecast.columns:
                    print(f"Warning: Model '{model}' not found in segment '{segment}'. Skipping.")
                    continue

                # Extract actual and forecasted data
                actual_dates = df_forecast["Date"][: len(df_forecast) - forecast_horizon]  # Past data
                actual_volume = df_forecast["Actual"][: len(df_forecast) - forecast_horizon]  # Past volume
                future_dates = df_forecast["Date"][-forecast_horizon:]  # Future dates
                model_forecast = df_forecast[model][-forecast_horizon:]  # Model forecast

                # Combine actual and predicted data into a single DataFrame
                combined_df = pd.DataFrame({
                    "date": pd.concat([actual_dates, future_dates], ignore_index=True),
                    "volume": pd.concat([actual_volume, model_forecast], ignore_index=True),
                })

                # Convert 'date' to datetime and set as index
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                combined_df.set_index("date", inplace=True)

                # Adjust dates to align with the fiscal year start month based on frequency
                if frequency in ["D", "W", "M", "Q"]:  # For daily, weekly, monthly, or quarterly data
                    combined_df['fiscal_year'] = combined_df.index - pd.offsets.DateOffset(months=fiscal_start_month - 1)
                    combined_df['fiscal_year'] = combined_df['fiscal_year'].dt.year
                elif frequency == "Y":  # For yearly data
                    combined_df['fiscal_year'] = combined_df.index.year  # No adjustment needed for yearly data

                # Determine the fiscal quarter based on the fiscal year start month and frequency
                if frequency in ["D", "W", "M"]:  # For daily, weekly, or monthly data
                    combined_df['fiscal_quarter'] = ((combined_df.index.month - fiscal_start_month) % 12 // 3) + 1
                elif frequency == "Q":  # For quarterly data
                    combined_df['fiscal_quarter'] = ((combined_df.index.month - fiscal_start_month) % 12 // 3) + 1
                elif frequency == "Y":  # For yearly data
                    combined_df['fiscal_quarter'] = 1  # Yearly data is treated as a single quarter

                # Group by fiscal year and fiscal quarter, then calculate mean volume
                quarterly_df = combined_df.groupby(['fiscal_year', 'fiscal_quarter'])['volume'].mean().reset_index()

                # ADDED: Shift fiscal year index by +1 (only if not January fiscal year)
                # if fiscal_start_month != 1:
                #     quarterly_df['fiscal_year'] = quarterly_df['fiscal_year'] + 1


                if not (1 <= fiscal_start_month <= 5):
                    quarterly_df['fiscal_year'] = quarterly_df['fiscal_year'] + 1




                # Calculate quarterly growth rate (QoQ comparison)
                quarterly_df[f"growth_rate_{model}"] = quarterly_df["volume"].pct_change(periods=1)

                # Store results
                growth_results[segment][model] = quarterly_df

        return growth_results

    # def detect_frequency(date_series):
    #     """Detects frequency of a time series by calculating mode of time difference."""
    #     date_series = date_series.sort_values().drop_duplicates()
    #     diffs = date_series.diff().dropna()
        
    #     if diffs.empty:
    #         return "Unknown"

    #     mode_diff = diffs.mode()[0]

    #     if pd.Timedelta(days=25) <= mode_diff <= pd.Timedelta(days=35):
    #         return "Monthly"
    #     elif pd.Timedelta(days=350) <= mode_diff <= pd.Timedelta(days=380):
    #         return "Yearly"
    #     elif pd.Timedelta(days=6) <= mode_diff <= pd.Timedelta(days=8):
    #         return "Weekly"
    #     else:
    #         return f"Custom ({mode_diff})"

    import pandas as pd

    def detect_frequency(date_series):
        """Detects frequency of a time series by calculating mode of time difference.
        Returns: 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly', or 'Custom (X days)'
        """
        date_series = date_series.sort_values().drop_duplicates()
        diffs = date_series.diff().dropna()
        
        if diffs.empty:
            return "Unknown"

        mode_diff = diffs.mode()[0]  # Most common time difference

        # Convert to days for easier comparison
        mode_days = mode_diff.total_seconds() / (24 * 3600)

        # Check frequency ranges (allowing small variations)
        if 0.9 <= mode_days <= 1.1:  # ~1 day
            return "Daily"
        elif 6 <= mode_days <= 8:  # ~7 days
            return "Weekly"
        elif 25 <= mode_days <= 35:  # ~30/31 days
            return "Monthly"
        elif 85 <= mode_days <= 95:  # ~91 days (1 quarter)
            return "Quarterly"
        elif 350 <= mode_days <= 380:  # ~365 days
            return "Yearly"
        else:
            return f"Custom ({mode_diff})"





    # # Apply full-width tabs
    # st.markdown(
    #         """
    #         <style>
    #             div.stTabs button {
    #                 flex-grow: 1;
    #                 text-align: center;
    #             }
    #         </style>
    #         """,
    #         unsafe_allow_html=True
    #     )
            
    # Function to convert data to a specific frequency



    def main():

        import streamlit as st

        st.sidebar.title("📈 FORECASTING")

        st.sidebar.markdown('<hr class="thick">', unsafe_allow_html=True)
        with st.sidebar:
            st.header("📂 Upload Data")
            uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

        # if 'uploaded_file' not in st.session_state:
        #     st.session_state.uploaded_file = None

        # if uploaded_file != st.session_state.uploaded_file:
        #     # Clear all session_state variables if the file changes or is removed
        #     st.session_state.clear()
        #     st.session_state.uploaded_file = uploaded_file

        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
            st.session_state.selected_sheet = None

        if uploaded_file != st.session_state.uploaded_file:
            # Clear all session_state variables if the file changes or is removed
            st.session_state.clear()
            st.session_state.uploaded_file = uploaded_file
            st.session_state.selected_sheet = None

        # If an Excel file is uploaded, show sheet selection
        if uploaded_file is not None and uploaded_file.name.endswith('.xlsx'):
            import pandas as pd
            try:
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                
                if len(sheet_names) > 1:
                    with st.sidebar:
                        selected_sheet = st.sidebar.selectbox(
                            "Select a sheet",
                            sheet_names,
                            index=0,
                            key="selected_sheet"
                        )
                else:
                    st.session_state.selected_sheet = sheet_names[0]
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")

        # Apply full-width tabs
        st.markdown(
                """
                <style>
                    div.stTabs button {
                        flex-grow: 1;
                        text-align: center;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
        

        selected=option_menu(
            menu_title="",
            options=["PRE-PROCESS","EXPLORE","MODEL","EVALUATE"],
            icons = ["sliders",         # PRE-PROCESS – tuning, adjustments
                "search",          # EXPLORE – data exploration
                "tools",           # ENGINEER – feature engineering
                "diagram-3",             # MODEL – training a model
                "bar-chart"]   ,    
            # icons=["database","diagram-3","clipboard-data"],
            orientation="horizontal"
        )



        # tab1, tab2, tab3, tab14,tab15 = st.tabs(["PRE-PROCESS","EXPLORE" ,"ENGINEER", "MODEL", "EVALUATE"])

    








        # with tab1:
        if selected=="PRE-PROCESS":
            # if 'modified_data' not in st.session_state:
            #     # You can initialize it with an empty DataFrame or any default value
            #     st.session_state.modified_data = None
            # render_workflow(0)


            # if uploaded_file:
            #     if uploaded_file.name.endswith(".csv"):
            #         d0 = pd.read_csv(uploaded_file)
            #     else:
            #         d0 = pd.read_excel(uploaded_file)
            if 'd0' not in st.session_state:
                st.session_state.d0=None

            if 'date_col' not in st.session_state:
                st.session_state.date_col=None

            if uploaded_file:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        d0 = pd.read_csv(uploaded_file)
                    else:
                        # Check if sheet has been selected (for Excel files)
                        if 'selected_sheet' in st.session_state and st.session_state.selected_sheet:
                            d0 = pd.read_excel(uploaded_file, sheet_name=st.session_state.selected_sheet)
                        else:
                            # Default to first sheet if no selection was made (for single-sheet files)
                            d0 = pd.read_excel(uploaded_file)
                    
                    # # Display the loaded data
                    # st.write("### Data Preview")
                    with st.expander("Show Data"):
                        st.dataframe(d0)
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                # else:
                #     st.warning("Please upload a data file first.")

                
                
                
                date_col = detect_date_column(d0)
                if not date_col:
                    date_col = st.selectbox("📅 Select the date column", d0.columns, index=0)
                
                d0[date_col] = pd.to_datetime(d0[date_col])

                st.session_state.date_col=date_col
                st.session_state.d0=d0



                d0 = d0.sort_values(by=date_col)

                basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']


                col1, col2 = st.columns(2)

                with col1:
                    columns_with_multiple_unique_values = [
                                                            col for col in basis_columns 
                                                            if col in d0.columns and d0[col].nunique() > 0 and col != date_col
                                                        ]

                    # Allow user to select which columns to consider
                    # selected_columns = st.multiselect("COLUMNS CONSIDERED", columns_with_multiple_unique_values, default=columns_with_multiple_unique_values[0])
                    # selected_columns = [st.selectbox(
                    #     "COLUMN TO CONSIDER", 
                    #     columns_with_multiple_unique_values, 
                    #     index=0 if columns_with_multiple_unique_values else None
                    # )]

                    if 'selected_column' not in st.session_state:
                        st.session_state.selected_column = columns_with_multiple_unique_values[0] if columns_with_multiple_unique_values else None

                    selected_columns = [st.selectbox(
                        "COLUMN TO CONSIDER", 
                        columns_with_multiple_unique_values,
                        index=columns_with_multiple_unique_values.index(st.session_state.selected_column) if st.session_state.selected_column in columns_with_multiple_unique_values else 0,
                        key="selected_column"
                    )]

                


                    



            

                st.markdown('<hr class="thin">', unsafe_allow_html=True)
            

                if 'Fiscal Year' in d0.columns:
                    # st.markdown('<hr class="thin">', unsafe_allow_html=True)
                    all_features = [col for col in d0.columns if col not in [ date_col] + basis_columns + ['Fiscal Year']]

                    d0 = d0[[date_col] + selected_columns + all_features + ['Fiscal Year']]

                    d0 = d0.groupby(selected_columns + [date_col] + ['Fiscal Year'], as_index=False)[ all_features].sum()
                else:

                    all_features = [col for col in d0.columns if col not in [ date_col] + basis_columns]

                    d0 = d0[[date_col] + selected_columns + all_features ]

                    d0 = d0.groupby(selected_columns + [date_col] , as_index=False)[ all_features].sum()
                    
        
                    
                    # st.error("'Fiscal Year' column not found in the dataset.")


                st.session_state.d0=d0




                # with col2:
                    # st.markdown("###### SHOW DATA")

                    # with st.expander("Show Data"):

                    #   st.write(d0)







                tab11,tab12,tab13=st.tabs(["Vadlidation","Feature Overview","Periodicity"])

                with tab11:


                

                    col1,col2=st.columns(2)

                    with col1:
                        columns_needed = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']

                        # 1. Clean column names (remove leading/trailing spaces)
                        d0.columns = d0.columns.str.strip()
                        renamed_cols = {col: col.strip() for col in d0.columns if col != col.strip()}
                        if renamed_cols:
                            st.warning("Column names had leading/trailing spaces. They have been cleaned:")
                            st.write(renamed_cols)

                        # 2. Date column validation and conversion
                        if not pd.api.types.is_datetime64_any_dtype(d0[date_col]):
                            try:
                                # Try converting to datetime
                                d0[date_col] = pd.to_datetime(d0[date_col], errors='coerce')
                                
                                # Check if conversion was successful
                                if d0[date_col].isnull().all():
                                    st.error(f"Could not convert '{date_col}' to datetime. Please select a different column.")
                                    date_col = st.selectbox("📅 Select a valid date column", 
                                                        [col for col in d0.columns if col != date_col])
                                    d0[date_col] = pd.to_datetime(d0[date_col], errors='coerce')
                            except Exception as e:
                                st.error(f"Error converting date column: {str(e)}")
                                st.stop()

                        # 3. Check for required columns
                        st.write("#### Required Columns Validation")
                        
                        # Check if at least one column from columns_needed exists
                        found_columns = [col for col in columns_needed if col in d0.columns]
                        if found_columns:
                            st.success(f"✅ Found required columns: {', '.join(found_columns)}")
                        else:
                            st.error(f"❌ Data must contain at least one of these columns: {', '.join(columns_needed)}\n\nUpload the correct file.")
                            st.stop()
                            
                        # # Check if Fiscal Year column exists
                        # if 'Fiscal Year' in d0.columns:
                        #     st.success("✅ Found required 'Fiscal Year' column")
                        # else:
                        #     st.error("❌ Data must contain a 'Fiscal Year' column")
                        #     st.stop()

                        # if 'fiscal_start_month' not in st.session_state:
                        #     st.session_state.fiscal_start_month = 1  # e.g., January


                        # # Check if Fiscal Year column exists
                        # if 'Fiscal Year' in d0.columns:
                        #     st.success("✅ Found required 'Fiscal Year' column")
                        # else:
                        #     st.warning("⚠️ 'Fiscal Year' column not found - creating it based on date column")
                            
                        #     # Let user specify fiscal year start month (default April)
                        #     fiscal_start_month = st.number_input(
                        #         "Enter starting month of fiscal year (1-12)", 
                        #         min_value=1, 
                        #         max_value=12, 
                        #         value=4,  # default to April
                        #         help="E.g., enter 4 for April-March fiscal year"
                        #     )
                            
                        #     # Create Fiscal Year column
                        #     d0['Fiscal Year'] = d0[date_col].apply(lambda x: 
                        #         f"FY{x.year + 1}" if x.month >= fiscal_start_month else f"FY{x.year}"
                        #     )
                            
                        #     st.success(f"✅ Created 'Fiscal Year' column (FY format starting month {fiscal_start_month})")
                        #     st.write("Sample of created Fiscal Years:")
                        #     st.dataframe(d0[[date_col, 'Fiscal Year']])

                        #     st.session_state.fiscal_start_month=fiscal_start_month


                        # Check if Fiscal Year column exists
                        if 'Fiscal Year' in d0.columns:
                            st.success("✅ Found required 'Fiscal Year' column")
                        else:
                            # st.warning("⚠️ 'Fiscal Year' column not found - creating it based on date column")

                            with st.expander("⚠️ 'Fiscal Year' column not found - creating it based on date column:",expanded=True):
                            
                                # Initialize fiscal_start_month in session state if not present
                                if 'fiscal_start_month' not in st.session_state:
                                    st.session_state.fiscal_start_month = 4  # default to April
                                
                                # Create select box with month names
                                month_names = [
                                    'January (1)', 'February (2)', 'March (3)', 'April (4)',
                                    'May (5)', 'June (6)', 'July (7)', 'August (8)',
                                    'September (9)', 'October (10)', 'November (11)', 'December (12)'
                                ]
                                
                                # Get current selection index (adjusting for 1-based vs 0-based)
                                current_index = st.session_state.fiscal_start_month - 1
                                
                                # Create select box that updates session state
                                selected_month = st.selectbox(
                                    "Select starting month of fiscal year",
                                    options=month_names,
                                    index=current_index,
                                    help="The month when your fiscal year begins",
                                    key='fiscal_month_select'
                                )
                                
                                # Extract month number from selection
                                fiscal_start_month = int(selected_month.split('(')[1].replace(')', ''))
                                
                                # Update session state only if changed
                                if fiscal_start_month != st.session_state.fiscal_start_month:
                                    st.session_state.fiscal_start_month = fiscal_start_month
                                
                                # Create Fiscal Year column using the session state value with shortened FY format
                                d0['Fiscal Year'] = d0[date_col].apply(lambda x: 
                                    f"FY{(x.year + 1) % 100:02d}" if x.month >= st.session_state.fiscal_start_month else f"FY{x.year % 100:02d}"
                                )
                                
                                st.success(f"✅ Created 'Fiscal Year' column (FY format starting from {month_names[st.session_state.fiscal_start_month-1].split(' ')[0]})")
                                st.write("Sample of created Fiscal Years:")
                                st.dataframe(d0[[date_col, 'Fiscal Year']], use_container_width=True)


                            # st.write(d0[['SubCategory','date','Fiscal Year']])











                        # 4. Missing values check
                        # st.subheader("🧹 Data Quality Report")
                        
                        # Create a DataFrame for missing values report
                        missing_report = pd.DataFrame({
                            'Column': d0.columns,
                            'Data Type': d0.dtypes.astype(str),
                            'Missing Values': d0.isnull().sum(),
                            '% Missing': (d0.isnull().mean() * 100).round(2)
                        })
                        
                        # Highlight problematic columns
                        def highlight_problems(row):
                            if row['Missing Values'] > 0:
                                return ['background-color: #FFF3CD'] * len(row)  # Light yellow
                            return [''] * len(row)
                        
                        st.write("#### Missing Values Summary")
                        st.dataframe(missing_report.style.apply(highlight_problems, axis=1),use_container_width=True)
                        
                        # Special warning if date column has missing values
                        if d0[date_col].isnull().any():
                            missing_dates = d0[date_col].isnull().sum()
                            st.error(f"🚨 Critical: Date column '{date_col}' has {missing_dates} missing values. These rows will be dropped.")
                            d0 = d0.dropna(subset=[date_col])

                    with col2:

                        # 5. Additional date validations
                        st.write("#### Date Column Validation")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success(f"**Date column:** `{date_col}`")
                            st.success(f"**Date range:** {d0[date_col].min().date()} to {d0[date_col].max().date()}")
                            # st.write(f"**Total days:** {(d0[date_col].max() - d0[date_col].min()).days + 1}")
                        
                        with col2:
                            st.success(f"**Records count:** {len(d0)}")
                            duplicates = d0[date_col].duplicated().sum()
                            if duplicates > 0:
                                st.error(f"**Duplicate dates:** {duplicates}")
                            else:
                                st.success("**No duplicate dates**")
                            
                    

                    # Final check if dataframe is valid
                    if d0.empty:
                        st.error("❌ Data is empty after validations. Please check your data and try again.")
                        st.stop()

                    st.success("✅ Data validation complete. You can proceed with analysis.")


                





                with tab12:
                    import plotly.express as px
                
                    # st.header("📊 Feature Overview")
                    
                    # Section 1: Basic Statistics
                    with st.expander("📈 Basic Statistics", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("#### Numerical Features")
                            num_cols = d0.select_dtypes(include=['int64', 'float64']).columns
                            if len(num_cols) > 0:
                                st.dataframe(d0[num_cols].describe().T.style.background_gradient(cmap='Blues'))
                            else:
                                st.warning("No numerical features found")
                        
                        with col2:
                            st.write("#### Categorical Features")
                            cat_cols = d0.select_dtypes(include=['object', 'category']).columns
                            if len(cat_cols) > 0:
                                st.dataframe(d0[cat_cols].describe().T.style.background_gradient(cmap='Greens'))
                            else:
                                st.warning("No categorical features found")
                    
                    # Section 2: Missing Values Analysis
                    with st.expander("🔍 Missing Values Analysis", expanded=False):
                        missing_values = d0.isnull().sum()
                        missing_pct = (missing_values / len(d0)) * 100
                        missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_pct.round(2)})
                        missing_df = missing_df[missing_df['Missing Values'] > 0]
                        
                        if len(missing_df) > 0:
                            st.warning(f"{len(missing_df)} columns have missing values")
                            col1, col2 = st.columns([2, 3])
                            
                            with col1:
                                st.dataframe(missing_df.style.background_gradient(cmap='Reds'))
                            
                            with col2:
                                fig = px.bar(missing_df, 
                                            x=missing_df.index, 
                                            y='Percentage (%)',
                                            title='Missing Values Percentage',
                                            color='Percentage (%)',
                                            color_continuous_scale='Reds')
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.success("No missing values found in any columns!")


                    # with st.expander("🧭 Column Relationships Explorer", expanded=False):
                    #     basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                        
                    #     # Find which basis columns are in the dataset
                    #     available_basis_cols = [col for col in basis_columns if col in d0.columns]
                        
                    #     if available_basis_cols:
                    #         # st.write("### ✅ Available Basis Columns in Data")
                    #         # st.write(available_basis_cols)

                    #         col33,col34=st.columns(2)

                    #         with col33:

                    #             selected_basis_col = st.selectbox("🔹 Select a basis column", available_basis_cols)

                    #         # Filter other categorical columns to group by
                    #         other_cat_cols = [col for col in d0.select_dtypes(include=['object', 'category']).columns if col != selected_basis_col]

                            
                            
                    #         if other_cat_cols:
                    #             with col34:
                    #                 selected_group_col = st.selectbox("🔸 Group by (view unique count and values of this column)", other_cat_cols)

                    #             grouped = d0.groupby(selected_basis_col)[selected_group_col].agg([
                    #                 ('Unique Count', 'nunique'),
                    #                 ('Unique Values', lambda x: sorted(x.dropna().unique().tolist()))
                    #             ]).reset_index()

                    #             st.write(f"#### Summary of `{selected_group_col}` within `{selected_basis_col}`")
                    #             st.dataframe(grouped, use_container_width=True)

                        
                    #         else:
                    #             st.info("Not enough other categorical columns to compare.")
                    #     else:
                    #         st.warning("None of the basis columns are present in the dataset.")



                    # with st.expander("🧭 Hierarchical Column Relationships Explorer", expanded=False):
                    #     basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                        
                    #     # Find which basis columns are in the dataset
                    #     available_basis_cols = [col for col in basis_columns if col in d0.columns]
                        
                    #     if available_basis_cols:
                    #         # Initialize session state for selected columns if not exists
                    #         if 'hierarchical_cols' not in st.session_state:
                    #             st.session_state.hierarchical_cols = []
                            
                    #         # Display current hierarchy
                    #         if st.session_state.hierarchical_cols:
                    #             st.write("#### Current Hierarchy")
                    #             st.write(" → ".join([f"**{col}**" for col in st.session_state.hierarchical_cols]))
                                
                    #             # st.write("---")
                            
                    #         # Column selection UI
                    #         col1, col2 = st.columns([3, 1])
                            
                    #         with col1:
                    #             # Available columns are those not already selected
                    #             available_cols = [col for col in available_basis_cols if col not in st.session_state.hierarchical_cols]
                                
                    #             if available_cols:
                    #                 new_col = st.selectbox(
                    #                     "Select a column to add to hierarchy",
                    #                     available_cols,
                    #                     key="new_hierarchical_col"
                    #                 )
                    #             else:
                    #                 st.info("All available columns have been added to the hierarchy")
                            
                    #         with col2:
                    #             st.write("")  # For alignment
                    #             st.write("")  # For alignment
                    #             if st.button("➕ Add", disabled=not available_cols):
                    #                 st.session_state.hierarchical_cols.append(new_col)
                    #                 st.rerun()
                            
                    #         # Remove button if there are selected columns
                    #         if st.session_state.hierarchical_cols:
                    #             if st.button("➖ Remove last"):
                    #                 st.session_state.hierarchical_cols.pop()
                    #                 st.rerun()
                            
                    #         # Generate hierarchical summary if columns are selected
                    #         if st.session_state.hierarchical_cols:
                    #             current_df = d0.copy()
                                
                    #             # For each level in the hierarchy
                    #             for i, col in enumerate(st.session_state.hierarchical_cols):
                    #                 group_cols = st.session_state.hierarchical_cols[:i+1]
                                    
                    #                 # Calculate remaining columns that could be analyzed
                    #                 remaining_cols = [c for c in available_basis_cols if c not in group_cols]
                                    
                    #                 # Create consistent summary display for all levels
                    #                 st.write(f"#### Level {i+1}: Analysis by {' → '.join(group_cols)}")
                                    
                    #                 if remaining_cols:
                    #                     # Calculate summary statistics for remaining columns
                    #                     summary_stats = []
                                        
                    #                     for rem_col in remaining_cols:
                    #                         stats = current_df.groupby(group_cols)[rem_col].agg([
                    #                             ('Unique Count', 'nunique'),
                    #                             ('Unique Values', lambda x: sorted(x.dropna().unique().tolist()))
                    #                         ]).reset_index()
                    #                         stats['Analyzed Column'] = rem_col
                    #                         summary_stats.append(stats)
                                        
                    #                     # Combine all summary stats
                    #                     full_summary = pd.concat(summary_stats)
                                        
                    #                     # Reorder columns for better readability
                    #                     column_order = group_cols + ['Analyzed Column', 'Unique Count', 'Unique Values']
                    #                     full_summary = full_summary[column_order]

                                        
                                        
                    #                     # Display summary
                    #                     st.dataframe(full_summary, use_container_width=True, height=min(400, 70 + 35*len(full_summary)))
                                        
                    #                     # # Show sample records for each group
                    #                     # with st.expander(f"🔍 Sample records for each {' → '.join(group_cols)} combination"):
                    #                     #     samples = current_df.groupby(group_cols).apply(
                    #                     #         lambda x: x.head(3).drop(columns=group_cols)
                    #                     #     ).reset_index()
                    #                     #     st.dataframe(samples, use_container_width=True)
                    #                 else:
                    #                     st.info("No remaining columns to analyze at this level")
                                    
                    #                 # Update current_df for next level by keeping one row per group
                    #                 if i < len(st.session_state.hierarchical_cols) - 1:
                    #                     current_df = current_df.drop_duplicates(subset=st.session_state.hierarchical_cols[:i+2])
                                    
                    #                 st.write("---")
                    #     else:
                    #         st.warning("None of the basis columns are present in the dataset.")


                    with st.expander("🧭 Hierarchical Column Relationships Explorer", expanded=True):
                        # basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                        
                        # # Find which basis columns are in the dataset
                        # available_basis_cols = [col for col in basis_columns if col in d0.columns]
                        
                        # if available_basis_cols:
                        #     # Initialize session state
                        #     if 'hierarchical_cols' not in st.session_state:
                        #         st.session_state.hierarchical_cols = []
                        #     if 'sort_by_col' not in st.session_state:
                        #         st.session_state.sort_by_col = None
                        #     if 'sort_ascending' not in st.session_state:
                        #         st.session_state.sort_ascending = True
                            
                        #     # Display current hierarchy
                        #     if st.session_state.hierarchical_cols:
                        #         st.write("#### Current Hierarchy")
                        #         st.write(" → ".join([f"**{col}**" for col in st.session_state.hierarchical_cols]))
                        #         # st.write("---")
                            
                        #     # Column selection UI
                        #     col1, col2 = st.columns([3, 1])
                            
                        #     with col1:
                        #         available_cols = [col for col in available_basis_cols if col not in st.session_state.hierarchical_cols]
                                
                        #         if available_cols:
                        #             new_col = st.selectbox(
                        #                 "Select a column to add to hierarchy",
                        #                 available_cols,
                        #                 key="new_hierarchical_col"
                        #             )
                        #         else:
                        #             st.info("All available columns have been added to the hierarchy")
                            
                        #     with col2:
                        #         st.write("")  # For alignment
                        #         st.write("")  # For alignment
                        #         if st.button("➕ Add", disabled=not available_cols):
                        #             st.session_state.hierarchical_cols.append(new_col)
                        #             st.rerun()
                            
                        #     # Remove button if there are selected columns
                        #     if st.session_state.hierarchical_cols:
                        #         if st.button("➖ Remove last"):
                        #             st.session_state.hierarchical_cols.pop()
                        #             st.rerun()

                        #     st.markdown('<hr class="thin">', unsafe_allow_html=True)
                            
                        #     # Generate hierarchical summary if columns are selected
                        #     if st.session_state.hierarchical_cols:
                        #         # Sort controls
                        #         # with st.expander("🔢 Sorting Options", expanded=False):
                        #         sort_col1, sort_col2 = st.columns(2)
                        #         with sort_col1:
                        #             st.session_state.sort_by_col = st.selectbox(
                        #                 "Sort by column",
                        #                 st.session_state.hierarchical_cols,
                        #                 index=0,
                        #                 key="sort_by_select"
                        #             )
                        #         with sort_col2:
                        #             st.session_state.sort_ascending = st.radio(
                        #                 "Sort order",
                        #                 ["Ascending", "Descending"],
                        #                 index=0,
                        #                 key="sort_order_radio"
                        #             )
                                
                        #         current_df = d0.copy()
                                
                        #         # For each level in the hierarchy
                        #         for i, col in enumerate(st.session_state.hierarchical_cols):
                        #             group_cols = st.session_state.hierarchical_cols[:i+1]
                        #             remaining_cols = [c for c in available_basis_cols if c not in group_cols]
                                    
                        #             st.write(f"#### Level {i+1}: Analysis by {' → '.join(group_cols)}")
                                    
                        #             if remaining_cols:
                        #                 # Calculate summary statistics for remaining columns
                        #                 summary_stats = []
                                        
                        #                 for rem_col in remaining_cols:
                        #                     stats = current_df.groupby(group_cols)[rem_col].agg([
                        #                         ('Unique Count', 'nunique'),
                        #                         ('Unique Values', lambda x: sorted(x.dropna().unique().tolist()))
                        #                     ]).reset_index()
                        #                     stats['Analyzed Column'] = rem_col
                        #                     summary_stats.append(stats)
                                        
                        #                 # Combine all summary stats
                        #                 full_summary = pd.concat(summary_stats)
                                        
                        #                 # Apply sorting if sort column is in this level's group columns
                        #                 if st.session_state.sort_by_col in group_cols:
                        #                     full_summary = full_summary.sort_values(
                        #                         by=st.session_state.sort_by_col,
                        #                         ascending=(st.session_state.sort_ascending == "Ascending")
                        #                     )
                                        
                        #                 # Reorder columns for better readability
                        #                 column_order = group_cols + ['Analyzed Column', 'Unique Count', 'Unique Values']
                        #                 full_summary = full_summary[column_order]
                                        
                        #                 # Display summary with dynamic height
                        #                 st.dataframe(
                        #                     full_summary, 
                        #                     use_container_width=True,
                        #                     height=min(400, 70 + 35*len(full_summary))
                        #                 )
                        #             else:
                        #                 st.info("No remaining columns to analyze at this level")
                                    
                        #             # Update current_df for next level by keeping one row per group
                        #             if i < len(st.session_state.hierarchical_cols) - 1:
                        #                 current_df = current_df.drop_duplicates(subset=st.session_state.hierarchical_cols[:i+2])
                                    
                        #             st.write("---")
                        # else:
                        #     st.warning("None of the basis columns are present in the dataset.")




                        basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                        
                        # Find which basis columns are in the dataset
                        available_basis_cols = [col for col in basis_columns if col in d0.columns]

                        if available_basis_cols:
                            # Initialize session state
                            if 'hierarchical_cols' not in st.session_state:
                                st.session_state.hierarchical_cols = []
                            
                            # # Display current hierarchy
                            # if st.session_state.hierarchical_cols:
                            #     st.write("#### Current Hierarchy")
                            #     st.write(" → ".join([f"**{col}**" for col in st.session_state.hierarchical_cols]))

                            if st.session_state.hierarchical_cols:
                                st.markdown("#### Current Hierarchy")
                                st.markdown(
                                    f"""
                                    <div style="
                                        display: inline-block;
                                        padding: 0.6em 1.2em;
                                        margin-bottom: 1rem;
                                        background: #ffffff;
                                        border: 1px solid #d0d3d6;
                                        border-left: 5px solid #F0E68C;
                                        border-radius: 12px;
                                        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                                        font-size: 0.95rem;
                                        font-family: 'Segoe UI', sans-serif;
                                        color: #333333;
                                        white-space: nowrap;
                                    ">
                                        {" → ".join([f"<strong>{col}</strong>" for col in st.session_state.hierarchical_cols])}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )








                            
                            # Add column button
                            available_cols = [col for col in available_basis_cols if col not in st.session_state.hierarchical_cols]


                            col43,col44=st.columns([1,6])

                            with col43:
                            
                                if available_cols:
                                    if st.button("➕ Add Level"):
                                        st.session_state.hierarchical_cols.append(None)  # Add placeholder
                                        st.rerun()
                                else:
                                    st.info("All available columns have been added to the hierarchy")

                            with col44:
                            
                                # Remove button if there are selected columns
                                if st.session_state.hierarchical_cols:
                                    if st.button("➖ Remove last"):
                                        st.session_state.hierarchical_cols.pop()
                                        st.rerun()
                            
                            # Create columns for level selection
                            if st.session_state.hierarchical_cols:
                                # Determine number of columns needed (max 4 per row)
                                num_levels = len(st.session_state.hierarchical_cols)
                                cols_per_row = min(4, num_levels)
                                level_columns = st.columns(cols_per_row)
                                
                                # Create dropdowns in columns
                                for i in range(num_levels):
                                    with level_columns[i % cols_per_row]:
                                        # Get available columns for this level
                                        prev_selected = st.session_state.hierarchical_cols[:i]
                                        available_for_level = [col for col in available_basis_cols 
                                                            if col not in prev_selected]
                                        
                                        # Update the value in session state when selection changes
                                        selected = st.selectbox(
                                            f"Level {i+1}",
                                            available_for_level,
                                            index=available_for_level.index(st.session_state.hierarchical_cols[i]) 
                                                if st.session_state.hierarchical_cols[i] in available_for_level 
                                                else 0,
                                            key=f"hier_col_{i}"
                                        )
                                        st.session_state.hierarchical_cols[i] = selected
                            
                            # Only show table for the current level (last selected)
                            if st.session_state.hierarchical_cols:
                                current_level = len(st.session_state.hierarchical_cols)
                                group_cols = st.session_state.hierarchical_cols[:current_level]
                                remaining_cols = [c for c in available_basis_cols if c not in group_cols]
                                
                                st.write(f"#### Level {current_level}: Analysis by {' → '.join(group_cols)}")
                                
                                if remaining_cols:
                                    # Calculate summary statistics for remaining columns
                                    summary_stats = []
                                    
                                    for rem_col in remaining_cols:
                                        stats = d0.groupby(group_cols)[rem_col].agg([
                                            ('Unique Count', 'nunique'),
                                            ('Unique Values', lambda x: sorted(x.dropna().unique().tolist()))
                                        ]).reset_index()
                                        stats['Analyzed Column'] = rem_col
                                        summary_stats.append(stats)
                                    
                                    # Combine all summary stats
                                    full_summary = pd.concat(summary_stats)
                                    
                                    # Automatically sort by all hierarchical columns in order
                                    full_summary = full_summary.sort_values(by=group_cols)
                                    
                                    # Reorder columns for better readability
                                    column_order = group_cols + ['Analyzed Column', 'Unique Count', 'Unique Values']
                                    full_summary = full_summary[column_order]
                                    
                                    # Display summary with dynamic height
                                    st.dataframe(
                                        full_summary, 
                                        use_container_width=True,
                                        height=min(400, 70 + 35*len(full_summary))
                                    )
                                else:
                                    st.info("No remaining columns to analyze at this level")
                        else:
                            st.warning("None of the basis columns are present in the dataset.")
                            


                        












                with tab13:


    #---------------------------------------------------------WITHOUT PRESELECTION TO STOP RERUN OF THIS--------------------------------------------
                    # # Detect frequency before grouping
                    # frequency = detect_frequency(d0[date_col])
                    # # st.write(f"Detected frequency: **{frequency}**")
                    # # Custom CSS for a small, styled box
                    # st.markdown(
                    #     f"""
                    #     <style>
                    #         .small-box {{
                    #             display: inline-block;
                    #             background-color: #f0f0f5; /* Light grey background */
                    #             color: #333; /* Dark text */
                    #             font-size: 16px;
                    #             font-weight: bold;
                    #             padding: 8px 12px;
                    #             border-radius: 8px;
                    #             border: 1px solid #ccc;
                    #             box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                    #         }}
                    #     </style>
                    #     <p class="small-box">Detected frequency: {frequency}</p>
                    #     """,
                    #     unsafe_allow_html=True
                    # )

                    # st.session_state.frequency=frequency

                    # d0 = d0.sort_values(by=date_col)

                    # # Add a dropdown to select the desired frequency
                    # if frequency in ['Daily', 'Weekly','Quaterly','Monthly']:  # Daily or Weekly
                    #     frequency_options = ['None', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
                    #     selected_frequency = st.selectbox("Select the desired frequency", frequency_options)

                    #     if selected_frequency != 'None':
                    #         # Let user select grouping columns
                    #         possible_group_cols = [col for col in d0.columns if col != date_col]
                    #         basis_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']

                    #         # Get the default selection: intersection of basis_columns and possible_group_cols
                    #         default_group_cols = [col for col in basis_columns if col in possible_group_cols]

                    #         # Include 'Fiscal Year' if it's in the data
                    #         if 'Fiscal Year' in possible_group_cols:
                    #             default_group_cols.append('Fiscal Year')

                    #         col31,col32=st.columns(2)
                    #         with col31:

                    #             # Display the multiselect with those as default
                    #             selected_group_cols = st.multiselect(
                    #                 "Select columns to group by", 
                    #                 possible_group_cols,
                    #                 default=default_group_cols
                    #             )

                    #         with col32:
                            
                    #             # Let user select columns to resample as mean (others will be sum)
                    #             numeric_cols = d0.select_dtypes(include=['number']).columns.tolist()
                    #             mean_cols = st.multiselect(
                    #                 "Select numeric columns to resample as mean (others will be sum)", 
                    #                 numeric_cols,
                    #                 default=[]
                    #             )
                                
                    #         # Define the resampling frequency mapping
                    #         resample_freq = {
                    #             'Daily': 'D',
                    #             'Weekly': 'W',
                    #             'Monthly': 'M',
                    #             'Quarterly': 'Q',
                    #             'Yearly': 'Y'
                    #         }

                    #         # Resample the data for each group
                    #         resampled_data = []
                            
                    #         # If no group columns selected, treat as one group
                    #         groups = [None] if not selected_group_cols else d0.groupby(selected_group_cols)
                            
                    #         for group, group_data in groups:
                    #             if group is not None:  # If grouping
                    #                 group_data = d0[d0[selected_group_cols].apply(tuple, axis=1) == group]
                                
                    #             # Set the date column as the index for resampling
                    #             group_data = group_data.set_index(date_col)
                                
                    #             # Create aggregation dictionary
                    #             agg_dict = {}
                    #             for col in numeric_cols:
                    #                 if col in mean_cols:
                    #                     agg_dict[col] = 'mean'
                    #                 else:
                    #                     agg_dict[col] = 'sum'
                                
                    #             # Resample with different aggregations
                    #             group_data = group_data.resample(resample_freq[selected_frequency]).agg(agg_dict).reset_index()
                                
                    #             # Add the grouping columns back to the resampled data if they exist
                    #             if group is not None:
                    #                 if isinstance(group, tuple):  # Multiple group columns
                    #                     for col, value in zip(selected_group_cols, group):
                    #                         group_data[col] = value
                    #                 else:  # Single group column
                    #                     group_data[selected_group_cols[0]] = group
                                
                    #             resampled_data.append(group_data)

                    #         # Combine all resampled groups
                    #         d0 = pd.concat(resampled_data, ignore_index=True)

                    #         frequency = detect_frequency(d0[date_col])

                    #         st.markdown(
                    #             f"""
                    #             <style>
                    #                 .small-box {{
                    #                     display: inline-block;
                    #                     background-color: #f0f0f5; /* Light grey background */
                    #                     color: #333; /* Dark text */
                    #                     font-size: 16px;
                    #                     font-weight: bold;
                    #                     padding: 8px 12px;
                    #                     border-radius: 8px;
                    #                     border: 1px solid #ccc;
                    #                     box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                    #                 }}
                    #             </style>
                    #             <p class="small-box">Detected New frequency: {frequency}</p>
                    #             """,
                    #             unsafe_allow_html=True
                    #         )

                    #     st.session_state.frequency = frequency
                    #     st.dataframe(d0)



    #---------------------------------------------------------WITH PRESELECTION TO STOP RERUN OF THIS--------------------------------------------

                    # # Detect initial frequency if not already in session state
                    # if 'initial_frequency' not in st.session_state:
                    #     st.session_state.initial_frequency = detect_frequency(d0[date_col])
                    #     st.session_state.frequency = st.session_state.initial_frequency

                    # # Display the current frequency
                    # st.markdown(
                    #     f"""
                    #     <style>
                    #         .small-box {{
                    #             display: inline-block;
                    #             background-color: #f0f0f5;
                    #             color: #333;
                    #             font-size: 16px;
                    #             font-weight: bold;
                    #             padding: 8px 12px;
                    #             border-radius: 8px;
                    #             border: 1px solid #ccc;
                    #             box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                    #         }}
                    #     </style>
                    #     <p class="small-box">Current frequency: {st.session_state.frequency}</p>
                    #     """,
                    #     unsafe_allow_html=True
                    # )

                    # d0 = d0.sort_values(by=date_col)

                    # # Initialize session state for previous selections if not exists
                    # if 'prev_selections' not in st.session_state:
                    #     st.session_state.prev_selections = {
                    #         'selected_frequency': None,
                    #         'selected_group_cols': None,
                    #         'mean_cols': None
                    #     }

                    # # Add a dropdown to select the desired frequency
                    # if st.session_state.initial_frequency in ['Daily', 'Weekly', 'Quaterly', 'Monthly']:
                    #     frequency_options = ['None', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
                    #     selected_frequency = st.selectbox("Select the desired frequency", frequency_options)
                        
                    #     # Let user select grouping columns
                    #     possible_group_cols = [col for col in d0.columns if col != date_col]
                    #     basis_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                    #     default_group_cols = [col for col in basis_columns if col in possible_group_cols]
                        
                    #     if 'Fiscal Year' in possible_group_cols:
                    #         default_group_cols.append('Fiscal Year')

                    #     col31, col32 = st.columns(2)
                    #     with col31:
                    #         selected_group_cols = st.multiselect(
                    #             "Select columns to group by", 
                    #             possible_group_cols,
                    #             default=default_group_cols
                    #         )

                    #     with col32:
                    #         numeric_cols = d0.select_dtypes(include=['number']).columns.tolist()
                    #         mean_cols = st.multiselect(
                    #             "Select numeric columns to resample as mean (others will be sum)", 
                    #             numeric_cols,
                    #             default=[]
                    #         )
                        
                    #     # Check if any selection has changed
                    #     current_selections = {
                    #         'selected_frequency': selected_frequency,
                    #         'selected_group_cols': tuple(selected_group_cols) if selected_group_cols else None,
                    #         'mean_cols': tuple(mean_cols) if mean_cols else None
                    #     }
                        
                    #     selections_changed = (
                    #         current_selections['selected_frequency'] != st.session_state.prev_selections['selected_frequency'] or
                    #         current_selections['selected_group_cols'] != st.session_state.prev_selections['selected_group_cols'] or
                    #         current_selections['mean_cols'] != st.session_state.prev_selections['mean_cols']
                    #     )
                        
                    #     # Only run the resampling if selections have changed or if we haven't stored the result yet
                    #     if (selected_frequency != 'None' and (selections_changed or 'resampled_data' not in st.session_state)):
                    #         resample_freq = {
                    #             'Daily': 'D',
                    #             'Weekly': 'W',
                    #             'Monthly': 'M',
                    #             'Quarterly': 'Q',
                    #             'Yearly': 'Y'
                    #         }

                    #         resampled_data = []
                    #         groups = [None] if not selected_group_cols else d0.groupby(selected_group_cols)
                            
                    #         for group, group_data in groups:
                    #             if group is not None:
                    #                 group_data = d0[d0[selected_group_cols].apply(tuple, axis=1) == group]
                                
                    #             group_data = group_data.set_index(date_col)
                                
                    #             agg_dict = {}
                    #             for col in numeric_cols:
                    #                 if col in mean_cols:
                    #                     agg_dict[col] = 'mean'
                    #                 else:
                    #                     agg_dict[col] = 'sum'
                                
                    #             group_data = group_data.resample(resample_freq[selected_frequency]).agg(agg_dict).reset_index()
                                
                    #             if group is not None:
                    #                 if isinstance(group, tuple):
                    #                     for col, value in zip(selected_group_cols, group):
                    #                         group_data[col] = value
                    #                 else:
                    #                     group_data[selected_group_cols[0]] = group
                                
                    #             resampled_data.append(group_data)

                    #         d0 = pd.concat(resampled_data, ignore_index=True)
                            
                    #         # Store the result and new frequency in session state
                    #         st.session_state.resampled_data = d0
                    #         st.session_state.frequency = detect_frequency(d0[date_col])
                    #         st.session_state.prev_selections = current_selections
                            
                    #         st.markdown(
                    #             f"""
                    #             <style>
                    #                 .small-box {{
                    #                     display: inline-block;
                    #                     background-color: #f0f0f5;
                    #                     color: #333;
                    #                     font-size: 16px;
                    #                     font-weight: bold;
                    #                     padding: 8px 12px;
                    #                     border-radius: 8px;
                    #                     border: 1px solid #ccc;
                    #                     box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                    #                 }}
                    #             </style>
                    #             <p class="small-box">New frequency: {st.session_state.frequency}</p>
                    #             """,
                    #             unsafe_allow_html=True
                    #         )
                        
                    #     # Use the stored resampled data if available and selections haven't changed
                    #     elif 'resampled_data' in st.session_state and not selections_changed:
                    #         d0 = st.session_state.resampled_data




                    #     st.dataframe(d0)


    #-----------------------------------------------------------------------------------------------------




                    # Detect initial frequency if not already in session state
                    if 'initial_frequency' not in st.session_state:
                        st.session_state.initial_frequency = detect_frequency(d0[date_col])
                        st.session_state.frequency = st.session_state.initial_frequency

                    # st.session_state.fiscal_start_month = fiscal_start_month

                    # st.write(fiscal_start_month)

                    # Display the current frequency
                    st.markdown(
                        f"""
                        <style>
                            .small-box {{
                                display: inline-block;
                                background-color: #f0f0f5;
                                color: #333;
                                font-size: 16px;
                                font-weight: bold;
                                padding: 8px 12px;
                                border-radius: 8px;
                                border: 1px solid #ccc;
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                            }}
                        </style>
                        <p class="small-box">Current frequency: {st.session_state.frequency}</p>
                        """,
                        unsafe_allow_html=True
                    )
                    

                    d0 = d0.sort_values(by=date_col)

                    

                    # Initialize session state for previous selections if not exists
                    if 'prev_selections' not in st.session_state:
                        st.session_state.prev_selections = {
                            'selected_frequency': None,
                            'selected_group_cols': None,
                            'mean_cols': None,
                            'fiscal_start_month': st.session_state.get('fiscal_start_month', 4)  # Get from session state or default to 4
                        }

                    # Add a dropdown to select the desired frequency
                    if st.session_state.initial_frequency in ['Daily', 'Weekly', 'Quaterly', 'Monthly']:
                        frequency_options = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
                        # Set default to current frequency
                        default_freq_index = frequency_options.index(st.session_state.frequency) if st.session_state.frequency in frequency_options else 0
                        selected_frequency = st.selectbox(
                            "Select the desired frequency", 
                            frequency_options,
                            index=default_freq_index
                        )
                        
                        # Let user select grouping columns
                        possible_group_cols = [col for col in d0.columns if col != date_col]
                        basis_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                        default_group_cols = [col for col in basis_columns if col in possible_group_cols]
                        
                        if 'Fiscal Year' in possible_group_cols:
                            default_group_cols.append('Fiscal Year')

                        col31, col32 = st.columns(2)
                        with col31:
                            selected_group_cols = st.multiselect(
                                "Select columns to group by", 
                                possible_group_cols,
                                default=default_group_cols
                            )

                        with col32:
                            numeric_cols = d0.select_dtypes(include=['number']).columns.tolist()
                            mean_cols = st.multiselect(
                                "Select numeric columns to resample as mean (others will be sum)", 
                                numeric_cols,
                                default=[]
                            )

                        current_fiscal_month = st.session_state.get('fiscal_start_month', 4)
                        
                        # Check if any selection has changed
                        current_selections = {
                            'selected_frequency': selected_frequency,
                            'selected_group_cols': tuple(selected_group_cols) if selected_group_cols else None,
                            'mean_cols': tuple(mean_cols) if mean_cols else None,
                            'fiscal_start_month': current_fiscal_month
                        }
                        
                        selections_changed = (
                            current_selections['selected_frequency'] != st.session_state.prev_selections['selected_frequency'] or
                            current_selections['selected_group_cols'] != st.session_state.prev_selections['selected_group_cols'] or
                            current_selections['mean_cols'] != st.session_state.prev_selections['mean_cols'] or
                            current_selections['fiscal_start_month'] != st.session_state.prev_selections['fiscal_start_month']

                        )
                        
                        # Only run the resampling if selections have changed or if we haven't stored the result yet
                        if selections_changed or 'resampled_data' not in st.session_state:
                            resample_freq = {
                                'Daily': 'D',
                                'Weekly': 'W',
                                'Monthly': 'M',
                                'Quarterly': 'Q',
                                'Yearly': 'Y'
                            }

                            resampled_data = []
                            groups = [None] if not selected_group_cols else d0.groupby(selected_group_cols)
                            
                            for group, group_data in groups:
                                if group is not None:
                                    group_data = d0[d0[selected_group_cols].apply(tuple, axis=1) == group]
                                
                                group_data = group_data.set_index(date_col)
                                
                                agg_dict = {}
                                for col in numeric_cols:
                                    if col in mean_cols:
                                        agg_dict[col] = 'mean'
                                    else:
                                        agg_dict[col] = 'sum'
                                
                                group_data = group_data.resample(resample_freq[selected_frequency]).agg(agg_dict).reset_index()
                                
                                if group is not None:
                                    if isinstance(group, tuple):
                                        for col, value in zip(selected_group_cols, group):
                                            group_data[col] = value
                                    else:
                                        group_data[selected_group_cols[0]] = group
                                
                                resampled_data.append(group_data)

                            d0 = pd.concat(resampled_data, ignore_index=True)
                            
                            # Store the result and new frequency in session state
                            st.session_state.resampled_data = d0
                            st.session_state.frequency = detect_frequency(d0[date_col])
                            st.session_state.prev_selections = current_selections
                            
                            st.markdown(
                                f"""
                                <style>
                                    .small-box {{
                                        display: inline-block;
                                        background-color: #f0f0f5;
                                        color: #333;
                                        font-size: 16px;
                                        font-weight: bold;
                                        padding: 8px 12px;
                                        border-radius: 8px;
                                        border: 1px solid #ccc;
                                        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                                    }}
                                </style>
                                <p class="small-box">New frequency: {st.session_state.frequency}</p>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        # Use the stored resampled data if available and selections haven't changed
                        elif 'resampled_data' in st.session_state and not selections_changed:
                            d0 = st.session_state.resampled_data

                        st.dataframe(d0)
                        # st.rerun()


                    st.session_state.d0=d0





















                    # # Function to generate a hash of a DataFrame
                    # def hash_dataframe(df):
                    #     return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()

                    # # Check if d0_auto has changed
                    # if 'd0_auto_hash' not in st.session_state:
                    #     st.session_state.d0_auto_hash = hash_dataframe(d0)  # Initialize hash


                    # current_hash = hash_dataframe(d0)
                    # if current_hash != st.session_state.d0_auto_hash:
                    #     st.session_state.d0_auto_hash = current_hash  # Update hash
                    #     st.session_state.modified_data = d0.copy()  # Update modified_data with latest d0_auto

                    # if 'modified_data' not in st.session_state:
                    #     st.session_state.modified_data = d0_auto.copy()
                        
                    

                    
                    st.markdown('<hr class="thick">', unsafe_allow_html=True)

                    
                    


        # with tab2:
        if selected=="EXPLORE":
            if uploaded_file:

                if 'd0'in st.session_state:
                    d0=st.session_state.d0

                if 'date_col'in st.session_state:
                    date_col=st.session_state.date_col

                


            #    # Add filtering section before Section 3
            #     with st.expander("🔍 Filter Data", expanded=False):
                # Get available basis columns from the dataset
                # available_basis_columns = [col for col in ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 
                #                                         'Brand', 'PPG', 'Variant', 'PackType', 'PackSize'] 
                #                         if col in d0.columns]
                
                # if available_basis_columns:
                #     st.write("Filter data by one or more dimensions:")
                #     filters = {}
                #     cols = st.columns(2)  # Create 2 columns for filter UI
                    
                #     for i, col in enumerate(available_basis_columns):
                #         with cols[i % 2]:  # Alternate between columns
                #             unique_vals = d0[col].unique()
                #             if len(unique_vals) > 20:
                #                 selected = st.multiselect(
                #                     f"Select {col}",
                #                     options=unique_vals,
                #                     default=None,
                #                     help=f"Select one or more {col} values"
                #                 )
                #             else:
                #                 selected = st.multiselect(
                #                     f"Select {col}",
                #                     options=unique_vals,
                #                     default=unique_vals if len(unique_vals) <= 5 else None,
                #                     help=f"Select one or more {col} values"
                #                 )
                #             if selected:
                #                 filters[col] = selected
                    
                #     # Create a filtered copy of the dataframe for distribution plots only
                #     if filters:
                #         mask = pd.Series(True, index=d0.index)
                #         for col, values in filters.items():
                #             mask &= d0[col].isin(values)
                #         d0_filtered = d0[mask].copy()
                #         # st.success(f"Filtered to {len(d0_filtered)} records (applies only to Distribution Plots)")
                #         st.success(f"Filtered to {len(d0_filtered)} records.")
                #     else:
                #         d0_filtered = d0.copy()
                # else:
                #     st.info("No standard filter columns available in this dataset")
                #     d0_filtered = d0.copy()



                # basis_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']
                # time_related_columns = ['Year', 'Month', 'Week', 'date', 'Date']


                # excluded_columns = basis_columns + time_related_columns
                # filtered_num_cols = [col for col in num_cols if col not in excluded_columns]


                # # Section 3: Distribution Plots
                # with st.expander("📊 Feature Distributions", expanded=False):
                #     selected_col = st.selectbox("Select a feature to visualize", filtered_num_cols)
                    
                #     if pd.api.types.is_numeric_dtype(d0_filtered[selected_col]):
                #         col1, col2 = st.columns(2)
                        
                #         with col1:
                #             # Histogram
                #             fig = px.histogram(d0_filtered, x=selected_col, 
                #                             title=f'Distribution of {selected_col}',
                #                             nbins=50)
                #             st.plotly_chart(fig, use_container_width=True)
                        
                #         with col2:
                #             # Box plot
                #             fig = px.box(d0_filtered, y=selected_col, 
                #                     title=f'Box Plot of {selected_col}')
                #             st.plotly_chart(fig, use_container_width=True)
                #     else:
                #         # For categorical features
                #         value_counts = d0_filtered[selected_col].value_counts().nlargest(20)
                #         fig = px.bar(value_counts, 
                #                     title=f'Value Counts for {selected_col}',
                #                     labels={'index': selected_col, 'value': 'Count'})
                #         st.plotly_chart(fig, use_container_width=True)
                
                # # Section 4: Time Series Overview (if date column exists)
                # if 'date_col' in locals() or 'date_col' in globals():
                #     with st.expander("⏳ Time Series Overview", expanded=False):
                #         numeric_cols = d0_filtered.select_dtypes(include=['int64', 'float64']).columns
                #         filtered_num_cols = [col for col in numeric_cols if col not in excluded_columns]
                        
                        
                #         if len(filtered_num_cols) > 0:
                #             selected_metric = st.selectbox("Select metric for time series", filtered_num_cols)
                            
                #             # Resample frequency selector
                #             freq = st.radio("Resample frequency", 
                #                         ['D', 'W', 'M', 'Q', 'Y'], 
                #                         index=2, horizontal=True,
                #                         format_func=lambda x: {
                #                             'D': 'Daily',
                #                             'W': 'Weekly',
                #                             'M': 'Monthly',
                #                             'Q': 'Quarterly',
                #                             'Y': 'Yearly'
                #                         }[x])
                            
                #             ts_df = d0_filtered.set_index(date_col)[selected_metric].resample(freq).mean()
                            
                #             fig = px.line(ts_df, 
                #                         title=f'{selected_metric} over Time ({freq})',
                #                         labels={'value': selected_metric})
                #             st.plotly_chart(fig, use_container_width=True)
                #         else:
                #             st.warning("No numerical features available for time series analysis")
                
                #     # Section 5: Enhanced Correlation Analysis
                #     if len(num_cols) > 1:
                #         with st.expander("📌 Correlation Analysis", expanded=True):


                #             # Multiselect for correlation analysis
                #             selected_cols = st.multiselect(
                #                 "Select columns for correlation analysis",
                #                 options=filtered_num_cols,
                #                 default=filtered_num_cols[:min(5, len(filtered_num_cols))]  # Show first 5 by default
                #             )
                            
                #             if len(selected_cols) < 2:
                #                 st.warning("Please select at least 2 columns for correlation analysis")
                #             else:
                #                 # Correlation matrix for selected columns
                #                 # st.subheader("Correlation Matrix")
                #                 corr_matrix = d0_filtered[selected_cols].corr()
                                
                #                 fig = px.imshow(corr_matrix,
                #                             text_auto=True,
                #                             aspect="auto",
                #                             color_continuous_scale='RdBu',
                #                             title=f'Correlation Matrix ({len(selected_cols)} selected columns)',
                #                             width=600, height=600)
                #                 st.plotly_chart(fig, use_container_width=True)
                                
                        
                #                 # Top correlations table
                #                 st.markdown('<hr class="thin">', unsafe_allow_html=True)
                #                 st.write("#### Top Correlations")
                #                 corr_series = corr_matrix.unstack().sort_values(ascending=False)
                #                 corr_series = corr_series[corr_series < 1]  # Remove self-correlations
                #                 top_corrs = pd.concat([corr_series.head(5), corr_series.tail(5)])
                                
                #                 st.dataframe(
                #                     top_corrs.to_frame('Correlation')
                #                     .style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
                #                     .format("{:.2f}"),use_container_width=True
                #                 )



                # Define basis columns
                basis_columns = ['Market', 'Channel', 'Region', 'Category', 'SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']

                # Create two columns for the filter UI
                col1, col2 = st.columns(2)

                with col1:

                    # with col1:
                    # Get columns with multiple unique values (excluding date column)
                    columns_with_multiple_unique_valuess = [
                        col for col in basis_columns 
                        if col in d0.columns and d0[col].nunique() > 0 and col != date_col
                    ]
                    
                    # Allow user to select which columns to consider
                    columns_selected = st.multiselect(
                        "COLUMNS CONSIDERED", 
                        columns_with_multiple_unique_valuess, 
                        default=columns_with_multiple_unique_valuess[0] if columns_with_multiple_unique_valuess else None,key='feature_explore'
                    )

                    






                # with col2:
                #     # Get possible target columns (excluding date and basis columns)
                #     possible_target_col = [col for col in d0.columns if col not in [date_col] + basis_columns]
                #     target_col = st.selectbox("SELECT WHAT TO FORECAST", possible_target_col)

                st.markdown('<hr class="thin">', unsafe_allow_html=True)
                import plotly.express as px

                # Check if Fiscal Year exists in the dataset
                if 'Fiscal Year' in d0.columns:
                    # First, ensure all grouping columns are 1-dimensional
                    all_features = [col for col in d0.columns if col not in [date_col] + basis_columns + ['Fiscal Year'] + ['Year','Month','Week']]
                    for col in columns_selected + [date_col, 'Fiscal Year']:
                        if col in d0.columns:
                            # Convert any list-like or array-like values to strings
                            if any(isinstance(x, (list, tuple, np.ndarray)) for x in d0[col]):
                                d0[col] = d0[col].astype(str)
                                
                    # Now perform the groupby operation
                    d0_auto = d0[[date_col] + columns_selected + all_features + ['Fiscal Year']]

                    # Ensure we only include columns that exist in the dataframe
                    groupby_cols = [col for col in columns_selected + [date_col, 'Fiscal Year'] if col in d0_auto.columns]
                    sum_cols = [col for col in all_features if col in d0_auto.columns]

                    d0_auto = d0_auto.groupby(groupby_cols, as_index=False)[sum_cols].sum()
                                    
                    # If columns are selected for grouping, show group selector

                    
                    if columns_selected:
                        with col2:
                            # Group data by selected columns
                            grouped_data = d0_auto.groupby(columns_selected)
                            
                            # Get the list of groups
                            group_names = list(grouped_data.groups.keys())
                            
                            # Let user select a specific group
                            selected_group = st.selectbox(
                                f"Select the group", 
                                group_names, 
                                key="group_selection_for_explore"
                            )
                        
                        # Get data for the selected group and set date as index
                        group_data = grouped_data.get_group(selected_group).set_index(date_col)
                        
                        # Now you can use group_data for visualization and analysis
                        # Section 3: Distribution Plots
                        with st.expander("📊 Feature Distributions", expanded=False):
                            # Filter numeric columns (excluding basis and time-related columns)
                            filtered_num_cols = [col for col in all_features if pd.api.types.is_numeric_dtype(group_data[col])]
                            
                            if filtered_num_cols:
                                selected_col = st.selectbox("Select a feature to visualize", filtered_num_cols)
                                
                                col3, col4 = st.columns(2)
                                
                                with col3:
                                    # Histogram
                                    fig = px.histogram(group_data, x=selected_col, 
                                                    title=f'Distribution of {selected_col}',
                                                    nbins=50)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col4:
                                    # Box plot
                                    fig = px.box(group_data, y=selected_col, 
                                            title=f'Box Plot of {selected_col}')
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No numeric features available for distribution analysis")

                        # Section 4: Time Series Analysis
                        with st.expander("⏳ Time Series Overview", expanded=False):
                            if len(filtered_num_cols) > 0:
                                selected_metric = st.selectbox("Select metric for time series", filtered_num_cols)
                                
                                # Resample frequency selector
                                freq = st.radio("Resample frequency", 
                                            ['D', 'W', 'M', 'Q', 'Y'], 
                                            index=2, horizontal=True,
                                            format_func=lambda x: {
                                                'D': 'Daily',
                                                'W': 'Weekly',
                                                'M': 'Monthly',
                                                'Q': 'Quarterly',
                                                'Y': 'Yearly'
                                            }[x])
                                
                                # Resample the time series data
                                ts_df = group_data[selected_metric].resample(freq).mean()
                                
                                # Plot the time series
                                fig = px.line(ts_df, 
                                            title=f'{selected_metric} over Time ({freq})',
                                            labels={'value': selected_metric})
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No numerical features available for time series analysis")

                        # Section 5: Correlation Analysis
                        if len(filtered_num_cols) > 1:
                            with st.expander("📌 Correlation Analysis", expanded=True):
                                # Multiselect for correlation analysis
                                selected_cols = st.multiselect(
                                    "Select columns for correlation analysis",
                                    options=  filtered_num_cols,
                                    default=filtered_num_cols[:min(4, len(filtered_num_cols))]  # Show target + first 4 by default
                                )
                                
                                if len(selected_cols) >= 2:
                                    # Correlation matrix for selected columns
                                    corr_matrix = group_data[selected_cols].corr()
                                    
                                    fig = px.imshow(corr_matrix,
                                                text_auto=True,
                                                aspect="auto",
                                                color_continuous_scale='RdBu',
                                                title=f'Correlation Matrix ({len(selected_cols)} selected columns)',
                                                width=600, height=600)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Top correlations table
                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)
                                    st.write("#### Top Correlations")
                                    corr_series = corr_matrix.unstack().sort_values(ascending=False)
                                    corr_series = corr_series[corr_series < 1]  # Remove self-correlations
                                    top_corrs = pd.concat([corr_series.head(5), corr_series.tail(5)])
                                    
                                    st.dataframe(
                                        top_corrs.to_frame('Correlation')
                                        .style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
                                        .format("{:.2f}"),use_container_width=True
                                    )
                                else:
                                    st.warning("Please select at least 2 columns for correlation analysis")
                else:
                    st.error("'Fiscal Year' column not found in the dataset.")


                st.session_state.modified_data=d0


            



                # with st.expander("🪞 Dual Axis Comparison", expanded=False):
                #     numeric_cols = d0.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    
                #     if len(numeric_cols) >= 2:
                #         col1 = st.selectbox("Select feature for Y-axis (left)", numeric_cols, key="left_axis")
                #         col2 = st.selectbox("Select feature for Y-axis (right)", [col for col in numeric_cols if col != col1], key="right_axis")
                #         x_axis = st.selectbox("Select X-axis (time or category)", d0.columns, key="x_axis_dual")

                #         if pd.api.types.is_numeric_dtype(d0[x_axis]):
                #             sorted_df = d0.sort_values(by=x_axis)
                #         else:
                #             sorted_df = d0.copy()

                #         fig = make_subplots(specs=[[{"secondary_y": True}]])

                #         fig.add_trace(
                #             go.Scatter(x=sorted_df[x_axis], y=sorted_df[col1], name=col1, line=dict(color='royalblue')),
                #             secondary_y=False,
                #         )

                #         fig.add_trace(
                #             go.Scatter(x=sorted_df[x_axis], y=sorted_df[col2], name=col2, line=dict(color='darkorange')),
                #             secondary_y=True,
                #         )

                #         fig.update_layout(
                #             title_text=f"{col1} vs {col2} over {x_axis}",
                #             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                #             margin=dict(t=40, b=30),
                #         )

                #         fig.update_xaxes(title_text=x_axis)
                #         fig.update_yaxes(title_text=col1, secondary_y=False)
                #         fig.update_yaxes(title_text=col2, secondary_y=True)

                #         st.plotly_chart(fig, use_container_width=True)
                #     else:
                #         st.warning("Need at least 2 numerical columns to create a dual axis plot.")




        # if selected=="DATA":  
        # with tab3:
        # if selected=="ENGINEER":
            # render_workflow(1)
            # show_workflow("ENGINEER")
                
            # if uploaded_file:
            #     if 'd0'in st.session_state:
            #         d0=st.session_state.d0

            #     if 'date_col'in st.session_state:
            #         date_col=st.session_state.date_col
            # # if uploaded_file:
            # #     if uploaded_file.name.endswith(".csv"):
            # #         d0 = pd.read_csv(uploaded_file)
            # #     else:
            # #         d0 = pd.read_excel(uploaded_file)

            # if uploaded_file:
            #     try:
            #         if uploaded_file.name.endswith(".csv"):
            #             d0 = pd.read_csv(uploaded_file)
            #         else:
            #             # Check if sheet has been selected (for Excel files)
            #             if 'selected_sheet' in st.session_state and st.session_state.selected_sheet:
            #                 d0 = pd.read_excel(uploaded_file, sheet_name=st.session_state.selected_sheet)
            #             else:
            #                 # Default to first sheet if no selection was made (for single-sheet files)
            #                 d0 = pd.read_excel(uploaded_file)
                    
            #         # # Display the loaded data
            #         # st.write("### Data Preview")
                # with st.expander("Show Data"):
                #     st.dataframe(d0)
                    
            #     except Exception as e:
            #         st.error(f"Error reading file: {str(e)}")
            # # else:
            # #     st.warning("Please upload a data file first.")
                
                
            #     date_col = detect_date_column(d0)
            #     if not date_col:
            #         date_col = st.selectbox("📅 Select the date column", d0.columns, index=0)
                
            #     d0[date_col] = pd.to_datetime(d0[date_col])

                # # Detect frequency before grouping
                # frequency = detect_frequency(d0[date_col])
                # # st.write(f"Detected frequency: **{frequency}**")
                # # Custom CSS for a small, styled box
                # st.markdown(
                #     f"""
                #     <style>
                #         .small-box {{
                #             display: inline-block;
                #             background-color: #f0f0f5; /* Light grey background */
                #             color: #333; /* Dark text */
                #             font-size: 16px;
                #             font-weight: bold;
                #             padding: 8px 12px;
                #             border-radius: 8px;
                #             border: 1px solid #ccc;
                #             box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                #         }}
                #     </style>
                #     <p class="small-box">Detected frequency: {frequency}</p>
                #     """,
                #     unsafe_allow_html=True
                # )

                # st.session_state.frequency=frequency

                # # with col1:
                # with st.expander("Show Data"):
                #     st.dataframe(d0)

                

                # d0 = d0.sort_values(by=date_col)

                # basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']


                # col1, col2 = st.columns(2)

                # with col1:
                #     columns_with_multiple_unique_values = [
                #                                             col for col in basis_columns 
                #                                             if col in d0.columns and d0[col].nunique() > 0 and col != date_col
                #                                         ]

                #     # Allow user to select which columns to consider
                #     # selected_columns = st.multiselect("COLUMNS CONSIDERED", columns_with_multiple_unique_values, default=columns_with_multiple_unique_values[0])
                #     # selected_columns = [st.selectbox(
                #     #     "COLUMN TO CONSIDER", 
                #     #     columns_with_multiple_unique_values, 
                #     #     index=0 if columns_with_multiple_unique_values else None
                #     # )]

                #     if 'selected_column' not in st.session_state:
                #         st.session_state.selected_column = columns_with_multiple_unique_values[0] if columns_with_multiple_unique_values else None

                #     selected_columns = [st.selectbox(
                #         "COLUMN TO CONSIDER", 
                #         columns_with_multiple_unique_values,
                #         index=columns_with_multiple_unique_values.index(st.session_state.selected_column) if st.session_state.selected_column in columns_with_multiple_unique_values else 0,
                #         key="selected_column"
                #     )]


                    



                # # with col2:
                # #     possible_target_col = [col for col in d0.columns if col not in [date_col] + basis_columns + ['Year','Month','Week']]
                #     # target_col = st.selectbox("SELECT WHAT TO FORECAST", possible_target_col)

                # st.markdown('<hr class="thin">', unsafe_allow_html=True)
                # # all_features = [col for col in d0.columns if col not in [target_col, date_col]+basis_columns+['Fiscal Year']]

                # # d0_auto = d0[[date_col] + selected_columns + [target_col]+all_features+['Fiscal Year']]
                

                # # d0_auto=d0_auto.groupby(selected_columns+[date_col]+['Fiscal Year'], as_index=False)[[target_col]+all_features].sum()

                # if 'Fiscal Year' in d0.columns:
                #     # st.markdown('<hr class="thin">', unsafe_allow_html=True)
                #     all_features = [col for col in d0.columns if col not in [ date_col] + basis_columns + ['Fiscal Year']]

                #     d0_auto = d0[[date_col] + selected_columns + all_features + ['Fiscal Year']]

                #     d0_auto = d0_auto.groupby(selected_columns + [date_col] + ['Fiscal Year'], as_index=False)[ all_features].sum()
                # else:
                #     st.error("'Fiscal Year' column not found in the dataset.")

                # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                # # st.write(d0_auto)
                # d0_auto = d0_auto.sort_values(by=date_col)

                # with st.expander('Convert Periodicity!'):

                #     # Add a dropdown to select the desired frequency
                #     if frequency in ['Daily', 'Weekly','Quaterly','Monthly']:  # Daily or Weekly
                #         frequency_options = ['None', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
                #         selected_frequency = st.selectbox("Select the desired frequency", frequency_options)

                #         if selected_frequency != 'None':
                #             # Define the resampling frequency mapping
                #             resample_freq = {
                #                 'Daily': 'D',
                #                 'Weekly': 'W',
                #                 'Monthly': 'M',
                #                 'Quarterly': 'Q',
                #                 'Yearly': 'Y'
                #             }

                #             # Resample the data for each group
                #             resampled_data = []
                #             for group, group_data in d0_auto.groupby(selected_columns):
                #                 # Set the date column as the index for resampling
                #                 group_data = group_data.set_index(date_col)
                #                 # Resample and sum the data
                #                 group_data = group_data.resample(resample_freq[selected_frequency]).sum().reset_index()
                #                 # Add the grouping columns back to the resampled data
                #                 for col, value in zip(selected_columns, group):
                #                     group_data[col] = value
                #                 resampled_data.append(group_data)

                #             # Combine all resampled groups
                #             d0_auto = pd.concat(resampled_data, ignore_index=True)

                #         # st.write(d0_auto)

                #         frequency = detect_frequency(d0_auto[date_col])

                #         st.markdown(
                #             f"""
                #             <style>
                #                 .small-box {{
                #                     display: inline-block;
                #                     background-color: #f0f0f5; /* Light grey background */
                #                     color: #333; /* Dark text */
                #                     font-size: 16px;
                #                     font-weight: bold;
                #                     padding: 8px 12px;
                #                     border-radius: 8px;
                #                     border: 1px solid #ccc;
                #                     box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                #                 }}
                #             </style>
                #             <p class="small-box">Detected New frequency: {frequency}</p>
                #             """,
                #             unsafe_allow_html=True
                #         )

                #         st.session_state.frequency=frequency

                #         # with col1:
                        
                #         st.dataframe(d0_auto)
                # # # Add frequency conversion option
                # # st.markdown('<hr class="thin">', unsafe_allow_html=True)
                # # st.subheader("Frequency Conversion")
                # # frequency_options = ["Monthly", "Quarterly", "Yearly"]
                # # selected_frequency = st.selectbox("Convert data to:", frequency_options)

                # # # Convert data to the selected frequency
                # # if selected_frequency:
                # #     d0_auto = convert_frequency(d0_auto, date_col, target_col, selected_columns, selected_frequency)

                # #     st.write(f"Data converted to {selected_frequency} frequency:")
                # #     st.dataframe(d0_auto)

                # #     # Store the converted data in session state for later use
                # #     # st.session_state.converted_data = d0_auto

                # # Function to generate a hash of a DataFrame
                # def hash_dataframe(df):
                #     return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()

                # # Check if d0_auto has changed
                # if 'd0_auto_hash' not in st.session_state:
                #     st.session_state.d0_auto_hash = hash_dataframe(d0_auto)  # Initialize hash


                # current_hash = hash_dataframe(d0_auto)
                # if current_hash != st.session_state.d0_auto_hash:
                #     st.session_state.d0_auto_hash = current_hash  # Update hash
                #     st.session_state.modified_data = d0_auto.copy()  # Update modified_data with latest d0_auto

                # if 'modified_data' not in st.session_state:
                # st.session_state.modified_data = d0_auto.copy()
                    
                

                
                # # st.markdown('<hr class="thick">', unsafe_allow_html=True)

                # import plotly.express as px

                

                # # Create tabs for different functionalities
                # tab4, tab5, tab6 = st.tabs(["Trends", "Transformation","Create"])

                










                    # with st.expander("Scatter Plot:"):

                    #     col18,col19=st.columns(2)

                    #     with col18:

                    #         if selected_columns:
                    #             # Group data by selected columns
                    #             grouped_data = d0_auto.groupby(selected_columns)

                    #             # Get the list of groups
                    #             group_names = list(grouped_data.groups.keys())

                        

                    #             selected_group = st.selectbox(f"Select the group", group_names,key="selection_for_corr")

                    #             group_data = grouped_data.get_group(selected_group).set_index(date_col)

                    #     with col19:

                        
                    #     # Select a variable for y-axis
                    #         y_var = st.selectbox("Select the variable for Y-axis", all_features, index=0)

                    #     # Detect the frequency of the time series
                        
                        
                    #     # Calculate correlation
                    #     correlation = group_data[[target_col, y_var]].corr().loc[target_col, y_var]
                        
                    #     # Plot the data using Plotly
                    #     fig = px.scatter(
                    #         group_data,
                    #         x=target_col,
                    #         y=y_var,
                    #         title=f"{target_col} vs {y_var}",
                    #         labels={target_col: f"{target_col}", y_var: y_var},
                    #         trendline="ols"  # Add a trendline for better visualization
                    #     )
                        
                    #     # Update layout
                    #     fig.update_layout(
                    #         xaxis_title=f"{target_col}",
                    #         yaxis_title=y_var,
                    #         hovermode="x unified"
                    #     )
                        
                    #     st.plotly_chart(fig, use_container_width=True)

                    #     st.markdown(
                    #         f"""
                    #         <div style="
                    #             border: 1px solid blue; 
                    #             padding: 10px; 
                    #             border-radius: 5px; 
                    #             background-color: #f9f9f9; 
                    #             font-weight: bold; 
                    #             text-align: left;
                    #             font-size: 18px;">
                    #             Correlation between {target_col} and {y_var}: {correlation:.2f}
                    #         </div>
                    #         """,
                    #         unsafe_allow_html=True
                    #     )

                    # # st.markdown('<hr class="thin">', unsafe_allow_html=True)


                    # # Correlation Heatmap Section
                    # # st.markdown("---")
                    # # st.subheader("Correlation Heatmap")

                    # with st.expander("Correlation Heatmap:",expanded=True):

                    #     # Create multiselect for variables to include in correlation
                    #     selected_vars = st.multiselect(
                    #         "Select variables for correlation analysis",
                    #         options=all_features + [target_col],
                    #         default=[target_col, y_var]  # Default to already selected variables
                    #     )

                    #     if len(selected_vars) >= 2:
                    #         # Calculate correlation matrix
                    #         corr_matrix = group_data[selected_vars].corr()
                            
                    #         # Create heatmap
                    #         fig = go.Figure(data=go.Heatmap(
                    #             z=corr_matrix.values,
                    #             x=corr_matrix.columns,
                    #             y=corr_matrix.index,
                    #             colorscale='RdBu',
                    #             zmin=-1,
                    #             zmax=1,
                    #             text=corr_matrix.round(2).values,
                    #             texttemplate="%{text}",
                    #             hoverinfo="x+y+z"
                    #         ))
                            
                    #         # Update layout
                    #         fig.update_layout(
                    #             title="Correlation Heatmap",
                    #             xaxis_title="Variables",
                    #             yaxis_title="Variables",
                    #             width=800,
                    #             height=600
                    #         )
                            
                    #         st.plotly_chart(fig, use_container_width=True)


                    # # with st.expander("Correlation Matrix:"):

                    
                    # #     # Display correlation matrix as table
                    # #     st.markdown("**Correlation Matrix:**")


                    # #     st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1).format("{:.2f}"),use_container_width=True)
                        
                    # #     # Interpretation guide
                    # #     st.markdown("""
                    # #     <div style="background-color:#f0f2f6; padding:15px; border-radius:5px; margin-top:10px;">
                    # #         <strong>Correlation Interpretation Guide:</strong>
                    # #         <ul>
                    # #             <li>1.0: Perfect positive correlation</li>
                    # #             <li>0.8-1.0: Very strong positive relationship</li>
                    # #             <li>0.6-0.8: Strong positive relationship</li>
                    # #             <li>0.4-0.6: Moderate positive relationship</li>
                    # #             <li>0.2-0.4: Weak positive relationship</li>
                    # #             <li>0.0-0.2: Very weak or no relationship</li>
                    # #             <li>Negative values indicate inverse relationships</li>
                    # #         </ul>
                    # #     </div>
                    # #     """, unsafe_allow_html=True)
                    # # # else:
                    # # #     st.warning("Please select at least 2 variables for correlation analysis")

                    # # st.markdown('<hr class="thin">', unsafe_allow_html=True)


                    # # # Calculate and display the correlation matrix heatmap for the selected group
                    # # st.subheader(f"Correlation Matrix Heatmap for Group: {selected_group}")

                    # # # Initialize an empty DataFrame to store correlation values for the selected group
                    # # correlation_matrix = pd.DataFrame(index=[selected_group], columns=all_features)

                    # # # Calculate correlations for the selected group and all y_vars
                    # # for y_var in all_features:
                    # #     correlation_matrix.loc[selected_group, y_var] = group_data[[target_col, y_var]].corr().loc[target_col, y_var]

                    # # # Display the correlation matrix
                    # # st.dataframe(correlation_matrix)

                    # # # Visualize the correlation matrix as a heatmap
                    # # fig_heatmap = px.imshow(
                    # #     correlation_matrix.astype(float),
                    # #     labels=dict(x="Y-axis Variable", y="Group", color="Correlation"),
                    # #     x=all_features,
                    # #     y=[selected_group],  # Only show the selected group
                    # #     color_continuous_scale='Viridis',
                    # #     title=f"Correlation Matrix Heatmap for Group: {selected_group}"
                    # # )
                    # # fig_heatmap.update_layout(
                    # #     xaxis_title="Y-axis Variable",
                    # #     yaxis_title="Group",
                    # #     coloraxis_colorbar=dict(title="Correlation")
                    # # )
                    # # st.plotly_chart(fig_heatmap, use_container_width=True)


    #             with tab4:
    #                 # st.header("Trend, Seasonality, and Residuals Analysis")

    #                 # col28,col29=st.columns(2)

    #                 # with col28:

    #                 #     col20,col21=st.columns(2)

    #                 #     with col20:

    #                 #         if selected_columns:
    #                 #             # Group data by selected columns
    #                 #             grouped_data = d0_auto.groupby(selected_columns)

    #                 #             # Get the list of groups
    #                 #             group_names = list(grouped_data.groups.keys())

                        

    #                 #             selected_group = st.selectbox(f"Select the group", group_names,key="selection_for_trend")

    #                 #             group_data = grouped_data.get_group(selected_group).set_index(date_col)

    #                 #     with col21:

    #                 #         # Select the variable to analyze
    #                 #         analysis_var = st.selectbox("Select the variable to analyze", [target_col] + all_features, index=0)

    #                 #     # Perform STL decomposition
    #                 #     stl = sm.tsa.STL(group_data[analysis_var], period=12)
    #                 #     decomposition = stl.fit()

    #                 #     # Create a DataFrame for decomposition results
    #                 #     decomposition_df = pd.DataFrame({
    #                 #         'Date': group_data.index,
    #                 #         'Observed': group_data[analysis_var],  # STL does not return observed, so we use the original series
    #                 #         'Trend': decomposition.trend,
    #                 #         'Seasonal': decomposition.seasonal,
    #                 #         'Residual': decomposition.resid
    #                 #     }).set_index('Date')

    #                 #     # Plot Observed and Trend in one graph
    #                 #     fig1 = go.Figure()

    #                 #     fig1.add_trace(go.Scatter(
    #                 #         x=decomposition_df.index,
    #                 #         y=decomposition_df['Observed'],
    #                 #         name='Observed',
    #                 #         line=dict(color='blue')
    #                 #     ))

    #                 #     fig1.add_trace(go.Scatter(
    #                 #         x=decomposition_df.index,
    #                 #         y=decomposition_df['Trend'],
    #                 #         name='Trend',
    #                 #         line=dict(color='green')
    #                 #     ))

    #                 #     fig1.update_layout(
    #                 #         title=f"Observed and Trend of {analysis_var}",
    #                 #         xaxis=dict(title="Date"),
    #                 #         yaxis=dict(title="Value"),
    #                 #         legend=dict(x=0.1, y=1.1),
    #                 #         hovermode="x unified"
    #                 #     )

    #                 #     st.plotly_chart(fig1, use_container_width=True)
    #                 #     st.markdown('<hr class="thin">', unsafe_allow_html=True)

    #                 #     # Plot Seasonality separately
    #                 #     fig2 = go.Figure()

    #                 #     fig2.add_trace(go.Scatter(
    #                 #         x=decomposition_df.index,
    #                 #         y=decomposition_df['Seasonal'],
    #                 #         name='Seasonality',
    #                 #         line=dict(color='red')
    #                 #     ))

    #                 #     fig2.update_layout(
    #                 #         title=f"Seasonality of {analysis_var}",
    #                 #         xaxis=dict(title="Date"),
    #                 #         yaxis=dict(title="Value"),
    #                 #         hovermode="x unified"
    #                 #     )

    #                 #     st.plotly_chart(fig2, use_container_width=True)
    #                 #     st.markdown('<hr class="thin">', unsafe_allow_html=True)

    #                 #     # Plot Residuals separately
    #                 #     fig3 = go.Figure()

    #                 #     fig3.add_trace(go.Scatter(
    #                 #         x=decomposition_df.index,
    #                 #         y=decomposition_df['Residual'],
    #                 #         name='Residuals',
    #                 #         line=dict(color='purple')
    #                 #     ))

    #                 #     fig3.update_layout(
    #                 #         title=f"Residuals of {analysis_var}",
    #                 #         xaxis=dict(title="Date"),
    #                 #         yaxis=dict(title="Value"),
    #                 #         hovermode="x unified"
    #                 #     )

    #                 #     st.plotly_chart(fig3, use_container_width=True)

    # #---------------------------------------------------------------------------------------------------- SEASONAL DECOMPOSE

    #                 # # Assuming d0_auto, selected_columns, date_col, target_col, and all_features are already defined


    #                 if 'modified_data' in st.session_state:
    #                     d0_auto=st.session_state.modified_data 

    

    #                 col28, col29 = st.columns(2)

    #                 with col28:
    #                     col20, col21 = st.columns(2)

    #                     with col20:
    #                         if selected_columns:
    #                             # Group data by selected columns
    #                             grouped_data = d0_auto.groupby(selected_columns)

    #                             # Get the list of groups
    #                             group_names = list(grouped_data.groups.keys())

    #                             selected_group = st.selectbox(f"Select the group", group_names, key="selection_for_trend")

    #                             group_data = grouped_data.get_group(selected_group).set_index(date_col)

    #                     with col21:
    #                         # Select the variable to analyze
    #                         analysis_var = st.selectbox("Select the variable to analyze", [target_col] + all_features, index=0)

    #                     # Perform seasonal decomposition
    #                     decomposition = sm.tsa.seasonal_decompose(group_data[analysis_var], model='additive', period=12)
                        
    #                     # Perform STL decomposition for trend extraction
    #                     stl = STL(group_data[analysis_var], period=12)
    #                     stl_result = stl.fit()
                        
    #                     # Create a DataFrame for decomposition results
    #                     decomposition_df = pd.DataFrame({
    #                         'Date': group_data.index,
    #                         'Observed': decomposition.observed,
    #                         'Trend': stl_result.trend,  # Using trend from STL
    #                         'Seasonal': decomposition.seasonal,  # Using seasonality from seasonal_decompose
    #                         'Residual': decomposition.resid  # Using residual from seasonal_decompose
    #                     }).set_index('Date')

    #                     # Plot Observed and Trend in one graph
    #                     fig1 = go.Figure()

    #                     fig1.add_trace(go.Scatter(
    #                         x=decomposition_df.index,
    #                         y=decomposition_df['Observed'],
    #                         name='Observed',
    #                         line=dict(color='blue')
    #                     ))

    #                     fig1.add_trace(go.Scatter(
    #                         x=decomposition_df.index,
    #                         y=decomposition_df['Trend'],
    #                         name='Trend',
    #                         line=dict(color='green')
    #                     ))

    #                     fig1.update_layout(
    #                         title=f"Observed and Trend of {analysis_var}",
    #                         xaxis=dict(title="Date"),
    #                         yaxis=dict(title="Value"),
    #                         legend=dict(x=0.1, y=1.1),
    #                         hovermode="x unified"
    #                     )

    #                     st.plotly_chart(fig1, use_container_width=True)
    #                     st.markdown('<hr class="thin">', unsafe_allow_html=True)

    #                     # Plot Seasonality separately
    #                     fig2 = go.Figure()

    #                     fig2.add_trace(go.Scatter(
    #                         x=decomposition_df.index,
    #                         y=decomposition_df['Seasonal'],
    #                         name='Seasonality',
    #                         line=dict(color='red')
    #                     ))

    #                     fig2.update_layout(
    #                         title=f"Seasonality of {analysis_var}",
    #                         xaxis=dict(title="Date"),
    #                         yaxis=dict(title="Value"),
    #                         hovermode="x unified"
    #                     )

    #                     st.plotly_chart(fig2, use_container_width=True)
    #                     st.markdown('<hr class="thin">', unsafe_allow_html=True)

    #                     # Plot Residuals separately
    #                     fig3 = go.Figure()

    #                     fig3.add_trace(go.Scatter(
    #                         x=decomposition_df.index,
    #                         y=decomposition_df['Residual'],
    #                         name='Residuals',
    #                         line=dict(color='purple')
    #                     ))

    #                     fig3.update_layout(
    #                         title=f"Residuals of {analysis_var}",
    #                         xaxis=dict(title="Date"),
    #                         yaxis=dict(title="Value"),
    #                         hovermode="x unified"
    #                     )

    #                     st.plotly_chart(fig3, use_container_width=True)

    #      #---------------------------------------------------------------------------------------------------- SEASONAL DECOMPOSE
                

    #                 with col29:
                        
    #                     # if 'modified_data' not in st.session_state:
    #                     #     st.session_state.modified_data = d0_auto.copy()  

    #                     # # Add option to append trend and seasonality as new features
    #                     # trend_col = f'{analysis_var}_Trend'
    #                     # seasonality_col = f'{analysis_var}_Seasonality'

    #                     # # Create a temporary DataFrame for preview
    #                     # preview_data = st.session_state.modified_data.copy()

    #                     # # Iterate over all groups and calculate trend and seasonality for preview
    #                     # for group_name, group_data in grouped_data:
    #                     #     group_data = group_data.set_index(date_col)

    #                     #     # Perform STL decomposition for the current group
    #                     #     stl = sm.tsa.STL(group_data[analysis_var], period=12)
    #                     #     decomposition = stl.fit()

    #                     #     # Create a DataFrame for decomposition results
    #                     #     group_data_reset = group_data.reset_index()

    #                     #     # Ensure group_name is a tuple for comparison
    #                     #     if not isinstance(group_name, tuple):
    #                     #         group_name = (group_name,)

    #                     #     # Filter rows in preview_data that match the current group
    #                     #     group_mask = (preview_data[selected_columns].apply(tuple, axis=1) == group_name)

    #                     #     # Check if the lengths match
    #                     #     if group_mask.sum() != len(decomposition.trend):
    #                     #         # st.error(f"Length mismatch for group {group_name}. Expected {group_mask.sum()} rows, got {len(decomposition.trend)}.")
    #                     #         continue

    #                     #     # Assign trend and seasonality values to the preview dataset
    #                     #     preview_data.loc[group_mask, trend_col] = decomposition.trend.values
    #                     #     preview_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

    #                     # # Display the preview of the data with trend and seasonality columns
    #                     # st.info(f"Add Trend and Seasonality as features for every group based on '{analysis_var}:")
    #                     # st.dataframe(preview_data)

    #                     # # Add a SAVE button
    #                     # if st.button("SAVE"):
    #                     #     # Ensure the columns exist in modified_data
    #                     #     if trend_col not in st.session_state.modified_data.columns:
    #                     #         st.session_state.modified_data[trend_col] = None  # Initialize the column if it doesn't exist
    #                     #     if seasonality_col not in st.session_state.modified_data.columns:
    #                     #         st.session_state.modified_data[seasonality_col] = None  # Initialize the column if it doesn't exist

    #                     #     # Iterate over all groups and calculate trend and seasonality
    #                     #     for group_name, group_data in grouped_data:
    #                     #         group_data = group_data.set_index(date_col)

    #                     #         # Perform STL decomposition for the current group
    #                     #         stl = sm.tsa.STL(group_data[analysis_var], period=12)
    #                     #         decomposition = stl.fit()

    #                     #         # Create a DataFrame for decomposition results
    #                     #         group_data_reset = group_data.reset_index()

    #                     #         # Ensure group_name is a tuple for comparison
    #                     #         if not isinstance(group_name, tuple):
    #                     #             group_name = (group_name,)

    #                     #         # Filter rows in modified_data that match the current group
    #                     #         group_mask = (st.session_state.modified_data[selected_columns].apply(tuple, axis=1) == group_name)

    #                     #         # Check if the lengths match
    #                     #         if group_mask.sum() != len(decomposition.trend):
    #                     #             # st.error(f"Length mismatch for group {group_name}. Expected {group_mask.sum()} rows, got {len(decomposition.trend)}.")
    #                     #             continue

    #                     #         # Assign trend and seasonality values to the main dataset (modified_data)
    #                     #         st.session_state.modified_data.loc[group_mask, trend_col] = decomposition.trend.values
    #                     #         st.session_state.modified_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

    #                     #     st.success(f"Trend and Seasonality components saved for all groups in '{analysis_var}'!")

    #                     # # Show Saved Data
    #                     # with st.expander('Show Saved Data'):
    #                     #     st.dataframe(st.session_state.modified_data)


    # #---------------------------------------------------------------------------------------------------- SEASONAL DECOMPOSE
    #                     # # Assuming d0_auto, selected_columns, date_col, and analysis_var are already defined

    #                     # if 'modified_data' not in st.session_state:
    #                     #     st.session_state.modified_data = d0_auto.copy()

    #                     # # Add option to append trend and seasonality as new features
    #                     # trend_col = f'{analysis_var}_Trend'
    #                     # seasonality_col = f'{analysis_var}_Seasonality'

    #                     # # Create a temporary DataFrame for preview
    #                     # preview_data = st.session_state.modified_data.copy()

    #                     # # Group data by selected columns
    #                     # grouped_data = st.session_state.modified_data.groupby(selected_columns)

    #                     # # Iterate over all groups and calculate trend and seasonality for preview
    #                     # for group_name, group_data in grouped_data:
    #                     #     group_data = group_data.set_index(date_col)

    #                     #     # Perform seasonal decomposition for the current group
    #                     #     decomposition = sm.tsa.seasonal_decompose(group_data[analysis_var], model='additive', period=12)

    #                     #     # Create a DataFrame for decomposition results
    #                     #     group_data_reset = group_data.reset_index()

    #                     #     # Ensure group_name is a tuple for comparison
    #                     #     if not isinstance(group_name, tuple):
    #                     #         group_name = (group_name,)

    #                     #     # Filter rows in preview_data that match the current group
    #                     #     group_mask = (preview_data[selected_columns].apply(tuple, axis=1) == group_name)

    #                     #     # Check if the lengths match
    #                     #     if group_mask.sum() != len(decomposition.trend):
    #                     #         # st.error(f"Length mismatch for group {group_name}. Expected {group_mask.sum()} rows, got {len(decomposition.trend)}.")
    #                     #         continue

    #                     #     # Assign trend and seasonality values to the preview dataset
    #                     #     preview_data.loc[group_mask, trend_col] = decomposition.trend.values
    #                     #     preview_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

    #                     # # Display the preview of the data with trend and seasonality columns
    #                     # st.info(f"Add Trend and Seasonality as features for every group based on '{analysis_var}':")
    #                     # st.dataframe(preview_data)

    #                     # # Add a SAVE button
    #                     # if st.button("SAVE"):
    #                     #     # Ensure the columns exist in modified_data
    #                     #     if trend_col not in st.session_state.modified_data.columns:
    #                     #         st.session_state.modified_data[trend_col] = None  # Initialize the column if it doesn't exist
    #                     #     if seasonality_col not in st.session_state.modified_data.columns:
    #                     #         st.session_state.modified_data[seasonality_col] = None  # Initialize the column if it doesn't exist

    #                     #     # Iterate over all groups and calculate trend and seasonality
    #                     #     for group_name, group_data in grouped_data:
    #                     #         group_data = group_data.set_index(date_col)

    #                     #         # Perform seasonal decomposition for the current group
    #                     #         decomposition = sm.tsa.seasonal_decompose(group_data[analysis_var], model='additive', period=12)

    #                     #         # Create a DataFrame for decomposition results
    #                     #         group_data_reset = group_data.reset_index()

    #                     #         # Ensure group_name is a tuple for comparison
    #                     #         if not isinstance(group_name, tuple):
    #                     #             group_name = (group_name,)

    #                     #         # Filter rows in modified_data that match the current group
    #                     #         group_mask = (st.session_state.modified_data[selected_columns].apply(tuple, axis=1) == group_name)

    #                     #         # Check if the lengths match
    #                     #         if group_mask.sum() != len(decomposition.trend):
    #                     #             # st.error(f"Length mismatch for group {group_name}. Expected {group_mask.sum()} rows, got {len(decomposition.trend)}.")
    #                     #             continue

    #                     #         # Assign trend and seasonality values to the main dataset (modified_data)
    #                     #         st.session_state.modified_data.loc[group_mask, trend_col] = decomposition.trend.values
    #                     #         st.session_state.modified_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

    #                     #     st.success(f"Trend and Seasonality components saved for all groups in '{analysis_var}'!")

    #                     # # Show Saved Data
    #                     # with st.expander('Show Saved Data'):
    #                     #     st.dataframe(st.session_state.modified_data)


    #                     # if 'modified_data' not in st.session_state:
    #                     st.session_state.modified_data = d0_auto.copy()

    #                     # Add option to append trend and seasonality as new features
    #                     trend_col = f'{analysis_var}_Trend'
    #                     seasonality_col = f'{analysis_var}_Seasonality'

    #                     # Create a temporary DataFrame for preview
    #                     preview_data = st.session_state.modified_data.copy()

    #                     # Group data by selected columns
    #                     grouped_data = st.session_state.modified_data.groupby(selected_columns)

    #                     # Iterate over all groups and calculate trend and seasonality for preview
    #                     for group_name, group_data in grouped_data:
    #                         group_data = group_data.set_index(date_col)

    #                         # Perform seasonal decomposition for seasonality
    #                         decomposition = sm.tsa.seasonal_decompose(group_data[analysis_var], model='additive', period=12)
                            
    #                         # Perform STL decomposition for trend
    #                         stl = STL(group_data[analysis_var], period=12)
    #                         stl_result = stl.fit()

    #                         # Create a DataFrame for decomposition results
    #                         group_data_reset = group_data.reset_index()

    #                         # Ensure group_name is a tuple for comparison
    #                         if not isinstance(group_name, tuple):
    #                             group_name = (group_name,)

    #                         # Filter rows in preview_data that match the current group
    #                         group_mask = (preview_data[selected_columns].apply(tuple, axis=1) == group_name)

    #                         # Check if the lengths match
    #                         if group_mask.sum() != len(stl_result.trend):  # Using STL trend length
    #                             # st.error(f"Length mismatch for group {group_name}. Expected {group_mask.sum()} rows, got {len(stl_result.trend)}.")
    #                             continue

    #                         # Assign trend (from STL) and seasonality (from seasonal decompose) values to the preview dataset
    #                         preview_data.loc[group_mask, trend_col] = stl_result.trend.values
    #                         preview_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

    #                     # Display the preview of the data with trend and seasonality columns
    #                     st.info(f"Add Trend and Seasonality as features for every group based on '{analysis_var}':")
    #                     st.dataframe(preview_data)

    #                     # Add a SAVE button
    #                     if st.button("SAVE"):
    #                         # Ensure the columns exist in modified_data
    #                         if trend_col not in st.session_state.modified_data.columns:
    #                             st.session_state.modified_data[trend_col] = None  # Initialize the column if it doesn't exist
    #                         if seasonality_col not in st.session_state.modified_data.columns:
    #                             st.session_state.modified_data[seasonality_col] = None  # Initialize the column if it doesn't exist

    #                         # Iterate over all groups and calculate trend and seasonality
    #                         for group_name, group_data in grouped_data:
    #                             group_data = group_data.set_index(date_col)

    #                             # Perform seasonal decomposition for seasonality
    #                             decomposition = sm.tsa.seasonal_decompose(group_data[analysis_var], model='additive', period=12)
                                
    #                             # Perform STL decomposition for trend
    #                             stl = STL(group_data[analysis_var], period=12)
    #                             stl_result = stl.fit()

    #                             # Create a DataFrame for decomposition results
    #                             group_data_reset = group_data.reset_index()

    #                             # Ensure group_name is a tuple for comparison
    #                             if not isinstance(group_name, tuple):
    #                                 group_name = (group_name,)

    #                             # Filter rows in modified_data that match the current group
    #                             group_mask = (st.session_state.modified_data[selected_columns].apply(tuple, axis=1) == group_name)

    #                             # Check if the lengths match
    #                             if group_mask.sum() != len(stl_result.trend):  # Using STL trend length
    #                                 # st.error(f"Length mismatch for group {group_name}. Expected {group_mask.sum()} rows, got {len(stl_result.trend)}.")
    #                                 continue

    #                             # Assign trend (from STL) and seasonality (from seasonal decompose) values to the main dataset (modified_data)
    #                             st.session_state.modified_data.loc[group_mask, trend_col] = stl_result.trend.values
    #                             st.session_state.modified_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

    #                         st.success(f"Trend and Seasonality components saved for all groups in '{analysis_var}'!")

    #                     # Show Saved Data
    #                     with st.expander('Show Saved Data'):
    #                         st.dataframe(st.session_state.modified_data)


    #                     d0_auto = st.session_state.modified_data.copy()

                        

    #---------------------------------------------------------------------------------------------------- SEASONAL DECOMPOSE


                # with tab5:


                    # # Initialize session state to store the original and modified data
                    # if 'original_d0' not in st.session_state:
                    #     st.session_state.original_d0 = d0_auto.copy()  # Save the original data

                    # if 'temp_data' not in st.session_state:
                    #     st.session_state.temp_data = {}  # Save the original data
                    
                    # # Use a temporary DataFrame for working changes
                    # if 'modified_data' in st.session_state:
                    #     st.session_state.temp_data = st.session_state.modified_data.copy()

                    # # Use the temporary data for all operations
                    # temp_data = st.session_state.temp_data

                    # all_features = [col for col in temp_data.columns if col not in [target_col, date_col]+basis_columns+['Fiscal Year']]

        
                    # # col22, col23 = st.columns(2)

                    # # with col22:
                    # st.subheader("Transformation Options")

                    # # Option to skip transformations
                    # apply_transformation = st.radio("Do you want to apply transformations?", ["Yes", "No"], index=1, horizontal=True)
                    # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    # if apply_transformation == "Yes":

                    #     # with st.expander("Apply Transformations:"):
                    #     # Select the feature to transform
                    #     col14, col15 = st.columns(2)

                    #     with col14:
                    #         transform_var = st.selectbox("Select the feature to transform", all_features, index=0)
                        
                    #     with col15:
                    #         # Transformation options
                    #         transform_option = st.selectbox(
                    #             "Select transformation",
                    #             ["Log", "Square Root", "Exponential","Power","Residual"],
                    #             index=0
                    #         )
                        
                    #     # Apply transformation to a temporary copy of the data
                    #     temp_data_transformed = temp_data.copy()
                    #     if transform_option == "Log":
                    #         temp_data_transformed[f'log_{transform_var}'] = np.log(temp_data_transformed[transform_var])
                    #     elif transform_option == "Square Root":
                    #         temp_data_transformed[f'sqrt_{transform_var}'] = np.sqrt(temp_data_transformed[transform_var])
                    #     elif transform_option == "Exponential":
                    #         temp_data_transformed[f'exp_{transform_var}'] = np.exp(temp_data_transformed[transform_var])
                    #     elif transform_option == "Power":
                    #         power = st.number_input(
                    #         "Enter the power exponent", 
                    #             value=2.0,  # Default to squared transformation
                    #             step=0.1,
                    #             format="%.2f"
                    #         )
                            
                    #         # Apply power transformation with error handling for negative values
                    #         try:
                    #             temp_data_transformed[f'power_{power}_{transform_var}'] = np.power(temp_data_transformed[transform_var], power)
                    #         except ValueError as e:
                    #             st.error(f"Error in power transformation: {str(e)}")
                    #             st.warning("Some values might be negative and raised to a non-integer power.")
                                
                    #             # Alternative approach using absolute values
                    #             if st.checkbox("Use absolute values before power transformation?"):
                    #                 temp_data_transformed[f'power_{power}_abs_{transform_var}'] = np.power(np.abs(temp_data_transformed[transform_var]), power)
                    #                 temp_data_transformed[f'power_{power}_sign_{transform_var}'] = np.sign(temp_data_transformed[transform_var])
                    #                 st.info("Applied power transformation to absolute values. Sign preserved in separate column.")






                    #     elif transform_option=="Residual":
                    #         # Residual transformation specific UI
                    #         # st.markdown("### Residual Transformation Setup")
                            
                    #         col12, col13 = st.columns(2)
                    #         with col12:
                    #             y_variables = st.multiselect("Select Y variables:", temp_data_transformed.columns, key="y_variables")
                    #         with col13:
                    #             y_x_mapping = {}
                    #             for y_var in y_variables:
                    #                 y_x_mapping[y_var] = st.multiselect(
                    #                     f"Select X variables for {y_var}:", 
                    #                     [col for col in temp_data_transformed.columns if col != y_var], 
                    #                     key=f"x_variables_{y_var}"
                    #                 )
                            
                    #         # Display the selected Y and X variables
                    #         if y_variables:
                    #             residual_inputs_data = [{
                    #                 "Y Variable": y, 
                    #                 "X Variables": ", ".join(x_vars)
                    #             } for y, x_vars in y_x_mapping.items()]
                                
                    #             residual_inputs_df = pd.DataFrame(residual_inputs_data)
                    #             st.write("Selected Y and X Variables for Residuals:")
                    #             st.dataframe(residual_inputs_df)
                                
                    #             # Function to calculate residuals
                    #             def calculate_residuals(df, y_x_mapping):
                    #                 for y_col, x_cols in y_x_mapping.items():
                    #                     if not x_cols:
                    #                         st.warning(f"No X variables selected for {y_col}. Skipping residual calculation.")
                    #                         continue
                                        
                    #                     if any(x_col not in df.columns for x_col in x_cols) or y_col not in df.columns:
                    #                         st.warning(f"One or more selected variables not found for residual calculation of {y_col}.")
                    #                         continue
                                        
                    #                     y = df[y_col]
                    #                     X = df[x_cols]
                                        
                    #                     if X.isnull().any().any() or y.isnull().any():
                    #                         st.warning(f"Skipping residual calculation for {y_col} due to NaN values.")
                    #                         continue
                                        
                    #                     if (X.std() == 0).any():
                    #                         st.warning(f"Skipping residual calculation for {y_col} due to zero standard deviation in one or more X variables.")
                    #                         continue
                                        
                    #                     X_standardized = (X - X.mean()) / X.std()
                    #                     X_standardized = sm.add_constant(X_standardized)
                                        
                    #                     # Calculate the mean of the Y variable
                    #                     y_mean = y.mean()
                                        
                    #                     model = sm.OLS(y, X_standardized).fit()
                    #                     df[f"Res_{y_col}"] = model.resid + y_mean
                                    
                    #                 return df
                                
                    #             temp_data_transformed = calculate_residuals(temp_data_transformed.copy(), y_x_mapping)

                    #     # Save button for transformations
                    #     if st.button("Save Transformation"):
                    #         st.session_state.modified_data = temp_data_transformed.copy()  # Save the modified data
                    #         st.session_state.temp_data = st.session_state.modified_data.copy()  # Reset temp data
                    #         st.success("Transformation saved! You can now apply another transformation or create a new feature.")
                    #         # st.rerun()

                    #     # Display transformed data
                    #     st.write("Transformed Data:")
                    #     st.dataframe(temp_data_transformed)
                        

                    #     transformed_cols = [col for col in temp_data_transformed.columns if col.startswith(transform_option.lower())]

                    #     # if transformed_cols:  # Check if the list is not empty
                    #     #     transformed_col = transformed_cols[0]
                        
                    #     # else:
                    #     #     st.warning("No transformed column found. Please check the transformation.")

                    #     st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    #     # Option to rename columns
                    #     col18, col19 = st.columns(2)
                    #     with col18:
                    #         column_to_rename = st.selectbox("Select column to rename", temp_data.columns)
                    #     with col19:
                    #         new_column_name = st.text_input("New column name")


                    #     if st.button("Rename Column"):
                    #         if new_column_name:
                    #             temp_data.rename(columns={column_to_rename: new_column_name}, inplace=True)
                    #             st.session_state.modified_data = temp_data.copy()
                    #             st.session_state.temp_data = st.session_state.modified_data.copy()
                    #             st.success(f"Renamed column '{column_to_rename}' to '{new_column_name}'")
                    #         else:
                    #             st.warning("Please enter a new column name.")

                    #     st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    #     # Add options to delete or rename columns
                    #     st.write("Any Column to Delete?")

                    #     # Option to delete columns
                    #     columns_to_delete = st.multiselect("Select columns to delete", temp_data.columns)
                    #     if st.button("Delete Selected Columns"):
                    #         temp_data.drop(columns=columns_to_delete, inplace=True)
                    #         st.session_state.modified_data = temp_data.copy()
                    #         st.session_state.temp_data = st.session_state.modified_data.copy()
                    #         st.success(f"Deleted columns: {columns_to_delete}")


                    #     st.markdown('<hr class="thin">', unsafe_allow_html=True)


                    # # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    # with st.expander("Show Final Saved Data"):
                    #     # Display the current state of the saved data
                    #     st.write("Saved Data:")
                    #     st.dataframe(st.session_state.modified_data)

                    # # # Reset button to revert to original data
                    # # if st.button("Reset Data to Original"):
                    # #     st.session_state.modified_data = st.session_state.original_d0.copy()
                    # #     st.session_state.temp_data = st.session_state.modified_data.copy()
                    # #     st.success("Data reset to original state!")


                    







            #         if 'working_data' not in st.session_state:
            #             st.session_state.working_data =None

            #         if 'staging_data' not in st.session_state:
            #             st.session_state.staging_data =None

            #         if 'last_modified_snapshot' not in st.session_state:
            #             st.session_state.last_modified_snapshot =None


            #         # # Function to generate a hash of a DataFrame
            #         # def hash_dataframe(df):
            #         #     return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()

            #         # # Check if d0_auto has changed
            #         # if 'd0_auto_hash' not in st.session_state:
            #         #     st.session_state.d0_auto_hash = hash_dataframe(d0)  # Initialize hash


            #         # current_hash = hash_dataframe(d0)
            #         # if current_hash != st.session_state.d0_auto_hash:
            #         #     st.session_state.d0_auto_hash = current_hash  # Update hash
            #         #     st.session_state.modified_data = d0.copy()  # Update modified_data with latest d0_auto




            #         # Initialize session state with unique names
            #         if 'data_pipeline_initialized' not in st.session_state:
            #             # Master original copy (never changes)
            #             st.session_state.master_original_data = st.session_state.modified_data.copy()
                        
            #             # Working copy that gets updated
            #             st.session_state.working_data = st.session_state.modified_data.copy()
                        
            #             # Temporary staging area for transformations
            #             st.session_state.staging_data = st.session_state.modified_data.copy()
                        
            #             st.session_state.data_pipeline_initialized = True

            #         # Check for changes in modified_data and update working_data if needed
            #         if not st.session_state.modified_data.equals(st.session_state.last_modified_snapshot):
            #             st.session_state.working_data = st.session_state.modified_data.copy()
            #             st.session_state.staging_data = st.session_state.modified_data.copy()
            #             st.session_state.last_modified_snapshot = st.session_state.modified_data.copy()



            #         # Get current data copies
            #         current_data = st.session_state.working_data.copy()
            #         staging_data = st.session_state.staging_data.copy()

            #         # Available features for transformation
            #         transformable_features = [col for col in staging_data.columns 
            #                                 if col not in [date_col] + basis_columns + ['Fiscal Year']+['Year','Month','Week']]

            #         # --- UI Section ---
            #         st.subheader("Transformation")

            #         # Transformation toggle
            #         apply_transforms = st.radio(
            #             "Apply transformations?",
            #             ["Yes", "No"],
            #             index=1,
            #             horizontal=True,
            #             key='transform_toggle'
            #         )

            #         st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #         if apply_transforms == "Yes":
            #             # Transformation selection
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 selected_feature = st.selectbox(
            #                     "Feature to transform",
            #                     transformable_features,
            #                     key='feature_select'
            #                 )
            #             with col2:
            #                 transform_type = st.selectbox(
            #                     "Transformation type",
            #                     ["Log", "Square Root", "Exponential", "Power", "Residual"],
            #                     key='transform_type'
            #                 )

            #             # Apply transformation to staging data
            #             transformed_data = staging_data.copy()

            #             if transform_type == "Log":
            #                 transformed_data[f'log_{selected_feature}'] = np.log1p(transformed_data[selected_feature])
            #             elif transform_type == "Square Root":
            #                 transformed_data[f'sqrt_{selected_feature}'] = np.sqrt(transformed_data[selected_feature])
            #             elif transform_type == "Exponential":
            #                 transformed_data[f'exp_{selected_feature}'] = np.exp(transformed_data[selected_feature])
            #             elif transform_type == "Power":
            #                 power_val = st.number_input(
            #                     "Power value",
            #                     value=2.0,
            #                     step=0.1,
            #                     key='power_val'
            #                 )
            #                 try:
            #                     transformed_data[f'power_{power_val}_{selected_feature}'] = np.power(
            #                         transformed_data[selected_feature],
            #                         power_val
            #                     )
            #                 except ValueError:
            #                     st.warning("Negative values with fractional powers not supported")
            #                     if st.checkbox("Use absolute values?", key='abs_power'):
            #                         transformed_data[f'power_{power_val}_abs_{selected_feature}'] = np.power(
            #                             np.abs(transformed_data[selected_feature]),
            #                             power_val
            #                         )
            #                         transformed_data[f'power_{power_val}_sign_{selected_feature}'] = np.sign(
            #                             transformed_data[selected_feature]
            #                         )
                        
            #             elif transform_type == "Residual":
            #                 st.markdown("**Residual Calculation**")
            #                 col3, col4 = st.columns(2)
            #                 with col3:
            #                     y_var = st.selectbox(
            #                         "Dependent (Y) variable",
            #                         staging_data.columns,
            #                         key='residual_y'
            #                     )
            #                 with col4:
            #                     x_vars = st.multiselect(
            #                         "Independent (X) variables",
            #                         [col for col in staging_data.columns if col != y_var],
            #                         key='residual_x'
            #                     )
                            
            #                 if x_vars:
            #                     try:
            #                         X = transformed_data[x_vars]
            #                         y = transformed_data[y_var]
                                    
            #                         # Standardize and add constant
            #                         X = (X - X.mean()) / X.std()
            #                         X = sm.add_constant(X)
                                    
            #                         # Fit model
            #                         model = sm.OLS(y, X).fit()
                                    
            #                         # Store residuals + mean
            #                         transformed_data[f'Res_{y_var}'] = model.resid + y.mean()
                                    
            #                         st.success(f"Residuals calculated (R² = {model.rsquared:.3f})")
            #                     except Exception as e:
            #                         st.error(f"Residual calculation failed: {str(e)}")

            #             # Preview transformed data
            #             st.write("**Transformation Preview:**")
            #             st.dataframe(transformed_data.head())

            #             # Save button - updates working_data only when clicked
            #             if st.button("💾 Save Transformations", key='save_transforms'):
            #                 st.session_state.working_data = transformed_data.copy()
            #                 st.session_state.staging_data = transformed_data.copy()
            #                 st.success("Transformations saved to working dataset!")
            #                 st.rerun()

            #             st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #             # Column operations on working data
            #             current_columns = st.session_state.working_data.columns.tolist()

            #             # Rename columns
            #             st.subheader("Column Renaming")
            #             col5, col6 = st.columns(2)
            #             with col5:
            #                 old_col = st.selectbox(
            #                     "Column to rename",
            #                     current_columns,
            #                     key='col_to_rename'
            #                 )
            #             with col6:
            #                 new_col = st.text_input(
            #                     "New name",
            #                     key='new_col_name'
            #                 )
                        
            #             if st.button("🔄 Rename Column", key='rename_col'):
            #                 if new_col:
            #                     st.session_state.working_data.rename(
            #                         columns={old_col: new_col},
            #                         inplace=True
            #                     )
            #                     st.session_state.staging_data = st.session_state.working_data.copy()
            #                     st.success(f"Renamed '{old_col}' to '{new_col}'")
            #                     st.rerun()
            #                 else:
            #                     st.warning("Please enter a new name")

            #             # Delete columns
            #             st.subheader("Column Deletion")
            #             cols_to_remove = st.multiselect(
            #                 "Columns to delete",
            #                 current_columns,
            #                 key='cols_to_delete'
            #             )
                        
            #             if st.button("🗑️ Delete Columns", key='delete_cols'):
            #                 st.session_state.working_data.drop(
            #                     columns=cols_to_remove,
            #                     inplace=True
            #                 )
            #                 st.session_state.staging_data = st.session_state.working_data.copy()
            #                 st.success(f"Deleted {len(cols_to_remove)} columns")
            #                 st.rerun()

            #         # --- Data Display ---
            #         with st.expander("📊 Current Working Data", expanded=False):
            #             st.write(f"Shape: {st.session_state.working_data.shape}")
            #             st.dataframe(st.session_state.working_data)




            #         # # Always show the current saved data
            #         # with st.expander("Show Final Saved Data"):
            #         #     st.write("Current Saved Data:")
            #         #     st.dataframe(st.session_state.modified_data)










            #         # st.session_state.modified_data=st.session_state.working_data.sort_values(by=date_col)
                    



            #         # import plotly.express as px

            #         # # Line Graph Visualization with Plotly
            #         # st.markdown('<hr class="thin">', unsafe_allow_html=True)
            #         # st.subheader("Interactive Data Visualization")

            #         # # Check if there's a date column in the data
            #         # date_columns = [col for col in st.session_state.modified_data.columns 
            #         #             if pd.api.types.is_datetime64_any_dtype(st.session_state.modified_data[col])]
                    
            #         # col40,col41,col42=st.columns(3)

            #         # if date_columns:

            #         #     with col40:
            #         #         # Let user select date column
            #         #         date_col = st.selectbox("Select Date Column", date_columns,key="date_col_for_transformation")

            #         #     column_to_filter=selected_columns[0]


            #         #     with col41:

                        
            #         #         # Check if SubCategory exists
            #         #         if column_to_filter in st.session_state.modified_data.columns:
            #         #             # Get unique subcategories
            #         #             subcategories = st.session_state.modified_data[column_to_filter].unique()
            #         #             selected_subcategories = st.multiselect(
            #         #                 f"Filter by {column_to_filter}", 
            #         #                 subcategories,
            #         #                 default=subcategories[0] if len(subcategories) > 0 else None,key="selected_subcategories_for_transformation"
            #         #             )
                                
            #         #             # Filter data by selected subcategories
            #         #             if selected_subcategories:
            #         #                 filtered_data = st.session_state.modified_data[
            #         #                     st.session_state.modified_data[column_to_filter].isin(selected_subcategories)
            #         #                 ]
            #         #             else:
            #         #                 filtered_data = st.session_state.modified_data
            #         #                 st.warning(f"No {column_to_filter} selected - showing all data")
            #         #         else:
            #         #             filtered_data = st.session_state.modified_data
            #         #             st.warning(f"No {column_to_filter} column found - showing all data")
                        
            #         #     # Get numeric columns (excluding the date column and SubCategory)
            #         #     numeric_cols = [col for col in filtered_data.columns 
            #         #                 if col not in [date_col, column_to_filter] and pd.api.types.is_numeric_dtype(filtered_data[col])]
                        
                        
                        
            #         #     if numeric_cols:
            #         #         with col42:
            #         #             # Multi-select for columns to plot
            #         #             selected_cols = st.multiselect("Select Columns to Plot", numeric_cols, default=numeric_cols[0],key="selected_cols_for_transformation")
                            
            #         #         if selected_cols:
            #         #             # Create the plot
            #         #             if column_to_filter in filtered_data.columns:
            #         #                 # If we have SubCategory, use it for color coding
            #         #                 fig = px.line(
            #         #                     filtered_data,
            #         #                     x=date_col,
            #         #                     y=selected_cols[0],  # Plotly Express needs one y column at a time
            #         #                     color=column_to_filter,
            #         #                     title=f'{selected_cols[0]} Trend by {column_to_filter}',
            #         #                     labels={selected_cols[0]: 'Value', date_col: 'Date'},
            #         #                     template='plotly_white'
            #         #                 )
                                    
            #         #                 # If multiple columns selected, add them as separate lines
            #         #                 if len(selected_cols) > 1:
            #         #                     for col in selected_cols[1:]:
            #         #                         fig.add_scatter(
            #         #                             x=filtered_data[date_col],
            #         #                             y=filtered_data[col],
            #         #                             mode='lines',
            #         #                             name=col,
            #         #                             visible='legendonly'  # Starts hidden but can be toggled
            #         #                         )
            #         #             else:
            #         #                 # No SubCategory - just plot selected columns
            #         #                 fig = px.line(
            #         #                     filtered_data,
            #         #                     x=date_col,
            #         #                     y=selected_cols,
            #         #                     title='Trend Over Time',
            #         #                     labels={'value': 'Value', date_col: 'Date'},
            #         #                     template='plotly_white'
            #         #                 )
                                
            #         #             # Improve layout
            #         #             fig.update_layout(
            #         #                 hovermode='x unified',
            #         #                 xaxis=dict(title=date_col, showgrid=True),
            #         #                 yaxis=dict(title='Value', showgrid=True),
            #         #                 legend=dict(title='Categories'),
            #         #                 height=600
            #         #             )
                                
            #         #             # Add range slider
            #         #             fig.update_xaxes(
            #         #                 rangeslider_visible=True,
            #         #                 rangeselector=dict(
            #         #                     buttons=list([
            #         #                         dict(count=1, label="1m", step="month", stepmode="backward"),
            #         #                         dict(count=6, label="6m", step="month", stepmode="backward"),
            #         #                         dict(count=1, label="YTD", step="year", stepmode="todate"),
            #         #                         dict(count=1, label="1y", step="year", stepmode="backward"),
            #         #                         dict(step="all")
            #         #                     ])
            #         #                 )
            #         #             )
                                
            #         #             st.plotly_chart(fig, use_container_width=True)
            #         #         else:
            #         #             st.warning("Please select at least one column to plot.")
            #         #     else:
            #         #         st.warning("No numeric columns found for plotting.")
            #         # else:
            #         #     st.warning("No date columns found in the data. Cannot create time series plot.")





            #     with tab6:

            #         # Update the working_data reference
            #         st.session_state.working_data = st.session_state.working_data.copy()

            #         # Sync staging_data with working_data
            #         if 'working_data' in st.session_state:
            #             st.session_state.staging_data = st.session_state.working_data.copy()

            #         # Use the staging data for all operations
            #         current_data = st.session_state.staging_data.copy()

            #         # Get available features
            #         all_features = [col for col in current_data.columns 
            #                     if col not in [ date_col] + basis_columns + ['Fiscal Year']+['Year','Month','Week']]

            #         # Feature creation section
            #         st.subheader("Create New Features")
            #         create_new_feature = st.radio(
            #             "Do you want to create a new feature?", 
            #             ["Yes", "No"], 
            #             index=1, 
            #             horizontal=True,
            #             key='new_feature_toggle'
            #         )
            #         st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #         if create_new_feature == "Yes":
            #             # Column selection for new feature
            #             col16, col17, col24 = st.columns(3)
            #             with col16:
            #                 col1 = st.selectbox(
            #                     "Select the first column", 
            #                     all_features, 
            #                     index=0,
            #                     key='new_feature_col1'
            #                 )
            #             with col17:
            #                 col2 = st.selectbox(
            #                     "Select the second column", 
            #                     all_features , 
            #                     index=1,
            #                     key='new_feature_col2'
            #                 )
            #             with col24:
            #                 operation = st.selectbox(
            #                     "Select operation", 
            #                     ["Add", "Subtract", "Multiply", "Divide"], 
            #                     index=0,
            #                     key='new_feature_op'
            #                 )
                        
            #             # Generate new column name
            #             new_col_name = f'{col1}_{operation.lower()}_{col2}'
                        
            #             # Create new feature in staging data
            #             staging_with_new_feature = current_data.copy()
            #             if operation == "Add":
            #                 staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] + staging_with_new_feature[col2]
            #             elif operation == "Subtract":
            #                 staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] - staging_with_new_feature[col2]
            #             elif operation == "Multiply":
            #                 staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] * staging_with_new_feature[col2]
            #             elif operation == "Divide":
            #                 staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] / staging_with_new_feature[col2]

            #             # Save button for new features
            #             if st.button("💾 Save New Feature", key='save_new_feature'):
            #                 st.session_state.working_data = staging_with_new_feature.copy()
            #                 st.session_state.staging_data = staging_with_new_feature.copy()
            #                 st.success("New feature saved to working dataset!")
            #                 st.rerun()

            #             # Display preview
            #             st.write("Preview with New Feature:")
            #             st.dataframe(staging_with_new_feature.head())
            #             st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #             # Column operations
            #             current_columns = st.session_state.working_data.columns.tolist()
                        
            #             # Rename columns
            #             col26, col27 = st.columns(2)
            #             with col26:
            #                 column_to_rename = st.selectbox(
            #                     "Select column to rename", 
            #                     current_columns,
            #                     key='rename_col_select'
            #                 )
            #             with col27:
            #                 new_column_name = st.text_input(
            #                     "New column name",
            #                     key='new_col_name_input'
            #                 )

            #             if st.button("🔄 Rename Column", key='rename_col_btn'):
            #                 if new_column_name:
            #                     st.session_state.working_data.rename(
            #                         columns={column_to_rename: new_column_name},
            #                         inplace=True
            #                     )
            #                     st.session_state.staging_data = st.session_state.working_data.copy()
            #                     st.success(f"Renamed '{column_to_rename}' to '{new_column_name}'")
            #                     st.rerun()
            #                 else:
            #                     st.warning("Please enter a new column name")

            #             st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #             # Delete columns
            #             st.write("Any Column to Delete?")
            #             columns_to_delete = st.multiselect(
            #                 "Select columns to delete",
            #                 current_columns,
            #                 key='cols_to_delete_select'
            #             )
                        
            #             if st.button("🗑️ Delete Columns", key='delete_cols_btn'):
            #                 st.session_state.working_data.drop(
            #                     columns=columns_to_delete,
            #                     inplace=True
            #                 )
            #                 st.session_state.staging_data = st.session_state.working_data.copy()
            #                 st.success(f"Deleted {len(columns_to_delete)} columns")
            #                 st.rerun()

            #             st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #         # Display current saved data
            #         with st.expander("📊 Current Working Data", expanded=False):
            #             st.write(f"Shape: {st.session_state.working_data.shape}")
            #             st.dataframe(st.session_state.working_data)



            #         # Convert DataFrame to Excel
            #         output = io.BytesIO()
            #         with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            #             st.session_state.working_data.to_excel(writer, index=False, sheet_name='Sheet1')
            #         excel_data = output.getvalue()

            #         # Download button
            #         st.download_button(
            #             label="📥 Download the final data as Excel",
            #             data=excel_data,
            #             file_name='final_saved_data.xlsx',
            #             mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            #         )








            #         # # Reset button to revert to original data
            #         # if st.button("Reset Data to Original"):
            #         #     st.session_state.modified_data = st.session_state.original_d0.copy()
            #         #     st.session_state.temp_data = st.session_state.modified_data.copy()
            #         #     st.success("Data reset to original state!")


            #         st.session_state.modified_data=st.session_state.working_data.sort_values(by=date_col)


            #         import plotly.express as px

            #         # Line Graph Visualization with Plotly
            #         st.markdown('<hr class="thick">', unsafe_allow_html=True)
            #         st.subheader("Interactive Data Visualization")

            #         # Check if there's a date column in the data
            #         date_columns = [col for col in st.session_state.modified_data.columns 
            #                     if pd.api.types.is_datetime64_any_dtype(st.session_state.modified_data[col])]
                    
            #         col40,col41,col42=st.columns(3)

            #         if date_columns:

            #             with col40:
            #                 # Let user select date column
            #                 date_col = st.selectbox("Select Date Column", date_columns)

            #             column_to_filter=selected_columns[0]


            #             with col41:

            #                 # Check if SubCategory exists
            #                 if column_to_filter in st.session_state.modified_data.columns:
            #                     # Get unique subcategories
            #                     subcategories = st.session_state.modified_data[column_to_filter].unique()
            #                     selected_subcategories = st.multiselect(
            #                         f"Filter by {column_to_filter}", 
            #                         subcategories,
            #                         default=subcategories[0] if len(subcategories) > 0 else None
            #                     )
                                
            #                     # Filter data by selected subcategories
            #                     if selected_subcategories:
            #                         filtered_data = st.session_state.modified_data[
            #                             st.session_state.modified_data[column_to_filter].isin(selected_subcategories)
            #                         ]
            #                     else:
            #                         filtered_data = st.session_state.modified_data
            #                         st.warning(f"No {column_to_filter} selected - showing all data")
            #                 else:
            #                     filtered_data = st.session_state.modified_data
            #                     st.warning(f"No {column_to_filter} column found - showing all data")
                        
            #             # Get numeric columns (excluding the date column and SubCategory)
            #             numeric_cols = [col for col in filtered_data.columns 
            #                         if col not in [date_col, column_to_filter] and pd.api.types.is_numeric_dtype(filtered_data[col])]
                        
                        
                        
            #             if numeric_cols:
            #                 with col42:
            #                     # Multi-select for columns to plot
            #                     selected_cols = st.multiselect("Select Columns to Plot", numeric_cols, default=numeric_cols[0])
                            
            #                 if selected_cols:
            #                     # Create the plot
            #                     if column_to_filter in filtered_data.columns:
            #                         # If we have SubCategory, use it for color coding
            #                         fig = px.line(
            #                             filtered_data,
            #                             x=date_col,
            #                             y=selected_cols[0],  # Plotly Express needs one y column at a time
            #                             color=column_to_filter,
            #                             title=f'{selected_cols[0]} Trend by {column_to_filter}',
            #                             labels={selected_cols[0]: 'Value', date_col: 'Date'},
            #                             template='plotly_white'
            #                         )
                                    
            #                         # If multiple columns selected, add them as separate lines
            #                         if len(selected_cols) > 1:
            #                             for col in selected_cols[1:]:
            #                                 fig.add_scatter(
            #                                     x=filtered_data[date_col],
            #                                     y=filtered_data[col],
            #                                     mode='lines',
            #                                     name=col,
            #                                     visible='legendonly'  # Starts hidden but can be toggled
            #                                 )
            #                     else:
            #                         # No SubCategory - just plot selected columns
            #                         fig = px.line(
            #                             filtered_data,
            #                             x=date_col,
            #                             y=selected_cols,
            #                             title='Trend Over Time',
            #                             labels={'value': 'Value', date_col: 'Date'},
            #                             template='plotly_white'
            #                         )
                                
            #                     # Improve layout
            #                     fig.update_layout(
            #                         hovermode='x unified',
            #                         xaxis=dict(title=date_col, showgrid=True),
            #                         yaxis=dict(title='Value', showgrid=True),
            #                         legend=dict(title='Categories'),
            #                         height=600
            #                     )
                                
            #                     # Add range slider
            #                     fig.update_xaxes(
            #                         rangeslider_visible=True,
            #                         rangeselector=dict(
            #                             buttons=list([
            #                                 dict(count=1, label="1m", step="month", stepmode="backward"),
            #                                 dict(count=6, label="6m", step="month", stepmode="backward"),
            #                                 dict(count=1, label="YTD", step="year", stepmode="todate"),
            #                                 dict(count=1, label="1y", step="year", stepmode="backward"),
            #                                 dict(step="all")
            #                             ])
            #                         )
            #                     )
                                
            #                     st.plotly_chart(fig, use_container_width=True)
            #                 else:
            #                     st.warning("Please select at least one column to plot.")
            #             else:
            #                 st.warning("No numeric columns found for plotting.")
            #         else:
            #             st.warning("No date columns found in the data. Cannot create time series plot.")




            #     with tab4:
            #         frequency=st.session_state.frequency


            #         # # Ensure we're working with the latest data
            #         # current_data = st.session_state.working_data.copy()

            #         # col28, col29 = st.columns(2)

            #         # with col28:
            #         #     col20, col21 = st.columns(2)

            #         #     with col20:
            #         #         if selected_columns:
            #         #             # Group data by selected columns
            #         #             grouped_data = current_data.groupby(selected_columns)

            #         #             # Get the list of groups
            #         #             group_names = list(grouped_data.groups.keys())

            #         #             selected_group = st.selectbox(
            #         #                 f"Select the group", 
            #         #                 group_names, 
            #         #                 key="group_selection_for_trend"
            #         #             )

            #         #             group_data = grouped_data.get_group(selected_group).set_index(date_col)

            #         #     with col21:
            #         #         # Select the variable to analyze
            #         #         analysis_var = st.selectbox(
            #         #             "Select the variable to analyze", 
            #         #             [target_col] + all_features, 
            #         #             index=0,
            #         #             key='analysis_var_select'
            #         #         )

            #         #     # Perform seasonal decomposition
            #         #     decomposition = sm.tsa.seasonal_decompose(
            #         #         group_data[analysis_var], 
            #         #         model='additive', 
            #         #         period=12
            #         #     )
                        
            #         #     # Perform STL decomposition for trend extraction
            #         #     stl = STL(group_data[analysis_var], period=12)
            #         #     stl_result = stl.fit()
                        
            #         #     # Create a DataFrame for decomposition results
            #         #     decomposition_df = pd.DataFrame({
            #         #         'Date': group_data.index,
            #         #         'Observed': decomposition.observed,
            #         #         'Trend': stl_result.trend,  # Using trend from STL
            #         #         'Seasonal': decomposition.seasonal,  # Using seasonality from seasonal_decompose
            #         #         'Residual': decomposition.resid  # Using residual from seasonal_decompose
            #         #     }).set_index('Date')

            #         #     # Plot Observed and Trend in one graph
            #         #     fig1 = go.Figure()
            #         #     fig1.add_trace(go.Scatter(
            #         #         x=decomposition_df.index,
            #         #         y=decomposition_df['Observed'],
            #         #         name='Observed',
            #         #         line=dict(color='blue')
            #         #     ))
            #         #     fig1.add_trace(go.Scatter(
            #         #         x=decomposition_df.index,
            #         #         y=decomposition_df['Trend'],
            #         #         name='Trend',
            #         #         line=dict(color='green')
            #         #     ))
            #         #     fig1.update_layout(
            #         #         title=f"Observed and Trend of {analysis_var}",
            #         #         xaxis=dict(title="Date"),
            #         #         yaxis=dict(title="Value"),
            #         #         legend=dict(x=0.1, y=1.1),
            #         #         hovermode="x unified"
            #         #     )
            #         #     st.plotly_chart(fig1, use_container_width=True)
            #         #     st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #         #     # Plot Seasonality separately
            #         #     fig2 = go.Figure()
            #         #     fig2.add_trace(go.Scatter(
            #         #         x=decomposition_df.index,
            #         #         y=decomposition_df['Seasonal'],
            #         #         name='Seasonality',
            #         #         line=dict(color='red')
            #         #     ))
            #         #     fig2.update_layout(
            #         #         title=f"Seasonality of {analysis_var}",
            #         #         xaxis=dict(title="Date"),
            #         #         yaxis=dict(title="Value"),
            #         #         hovermode="x unified"
            #         #     )
            #         #     st.plotly_chart(fig2, use_container_width=True)
            #         #     st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #         #     # Plot Residuals separately
            #         #     fig3 = go.Figure()
            #         #     fig3.add_trace(go.Scatter(
            #         #         x=decomposition_df.index,
            #         #         y=decomposition_df['Residual'],
            #         #         name='Residuals',
            #         #         line=dict(color='purple')
            #         #     ))
            #         #     fig3.update_layout(
            #         #         title=f"Residuals of {analysis_var}",
            #         #         xaxis=dict(title="Date"),
            #         #         yaxis=dict(title="Value"),
            #         #         hovermode="x unified"
            #         #     )
            #         #     st.plotly_chart(fig3, use_container_width=True)

            #         # with col29:
            #         #     # Create column names for new features
            #         #     trend_col = f'{analysis_var}_Trend'
            #         #     seasonality_col = f'{analysis_var}_Seasonality'

            #         #     # Create a preview DataFrame
            #         #     preview_data = current_data.copy()

            #         #     # Group data by selected columns
            #         #     grouped_data = current_data.groupby(selected_columns)

            #         #     # Calculate trend and seasonality for all groups (preview only)
            #         #     for group_name, group_data in grouped_data:
            #         #         group_data = group_data.set_index(date_col)

            #         #         # Handle group name formatting
            #         #         group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

            #         #         # Calculate decompositions
            #         #         decomposition = sm.tsa.seasonal_decompose(
            #         #             group_data[analysis_var], 
            #         #             model='additive', 
            #         #             period=12
            #         #         )
            #         #         stl_result = STL(group_data[analysis_var], period=12).fit()

            #         #         # Create mask for current group
            #         #         group_mask = (preview_data[selected_columns].apply(tuple, axis=1) == group_name)

            #         #         # Only proceed if lengths match
            #         #         if group_mask.sum() == len(stl_result.trend):
            #         #             preview_data.loc[group_mask, trend_col] = stl_result.trend.values
            #         #             preview_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

            #         #     # Display preview
            #         #     st.info(f"Preview of Trend and Seasonality features for '{analysis_var}':")
            #         #     st.dataframe(preview_data)

            #         #     # Save button
            #         #     if st.button("💾 Save Trend Components", key='save_trend_components'):
            #         #         # Initialize columns if they don't exist
            #         #         if trend_col not in st.session_state.working_data.columns:
            #         #             st.session_state.working_data[trend_col] = None
            #         #         if seasonality_col not in st.session_state.working_data.columns:
            #         #             st.session_state.working_data[seasonality_col] = None

            #         #         # Calculate and save for all groups
            #         #         for group_name, group_data in grouped_data:
            #         #             group_data = group_data.set_index(date_col)
            #         #             group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

            #         #             decomposition = sm.tsa.seasonal_decompose(
            #         #                 group_data[analysis_var], 
            #         #                 model='additive', 
            #         #                 period=12
            #         #             )
            #         #             stl_result = STL(group_data[analysis_var], period=12).fit()

            #         #             group_mask = (st.session_state.working_data[selected_columns].apply(tuple, axis=1) == group_name)

            #         #             if group_mask.sum() == len(stl_result.trend):
            #         #                 st.session_state.working_data.loc[group_mask, trend_col] = stl_result.trend.values
            #         #                 st.session_state.working_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

            #         #         # Update staging data
            #         #         st.session_state.staging_data = st.session_state.working_data.copy()
            #         #         st.success(f"Saved trend and seasonality components for {analysis_var}!")
            #         #         st.rerun()

            #         #     # Show current working data
            #         #     with st.expander('📊 Current Working Data'):
            #         #         st.dataframe(st.session_state.working_data)




            #         # Ensure we're working with the latest data
            #         current_data = st.session_state.working_data.copy()

            #         col28, col29 = st.columns(2)

            #         with col28:
            #             col20, col21 = st.columns(2)

            #             with col20:
            #                 if selected_columns:
            #                     # Group data by selected columns
            #                     grouped_data = current_data.groupby(selected_columns)

            #                     # Get the list of groups
            #                     group_names = list(grouped_data.groups.keys())

            #                     selected_group = st.selectbox(
            #                         f"Select the group", 
            #                         group_names, 
            #                         key="group_selection_for_trend"
            #                     )

            #                     group_data = grouped_data.get_group(selected_group).set_index(date_col)

            #             with col21:
            #                 # Select the variable to analyze
            #                 analysis_var = st.selectbox(
            #                     "Select the variable to analyze", 
            #                     all_features, 
            #                     index=0,
            #                     key='analysis_var_select'
            #                 )

            #             # Set seasonal period based on frequency
            #             frequency_periods = {
            #                 "Daily": 7,      # Weekly seasonality
            #                 "Weekly": 52,    # Yearly seasonality
            #                 "Monthly": 12,   # Yearly seasonality
            #                 "Quarterly": 4,  # Yearly seasonality
            #                 "Yearly": 1      # No seasonality
            #             }
                        
            #             period = frequency_periods.get(frequency, 12)  # Default to 12 if frequency not found
                        
            #             if frequency == "Yearly":
            #                 st.warning("Yearly data doesn't have sufficient seasonality for decomposition")
                        
            #             # Perform seasonal decomposition (only if not yearly)
            #             if frequency != "Yearly":
            #                 decomposition = sm.tsa.seasonal_decompose(
            #                     group_data[analysis_var], 
            #                     model='additive', 
            #                     period=period
            #                 )
                            
            #                 # Perform STL decomposition for trend extraction
            #                 stl = STL(group_data[analysis_var], period=period)
            #                 stl_result = stl.fit()
                            
            #                 # Create a DataFrame for decomposition results
            #                 decomposition_df = pd.DataFrame({
            #                     'Date': group_data.index,
            #                     'Observed': decomposition.observed,
            #                     'Trend': stl_result.trend,
            #                     'Seasonal': decomposition.seasonal,
            #                     'Residual': decomposition.resid
            #                 }).set_index('Date')

            #                 # Plot Observed and Trend in one graph
            #                 fig1 = go.Figure()
            #                 fig1.add_trace(go.Scatter(
            #                     x=decomposition_df.index,
            #                     y=decomposition_df['Observed'],
            #                     name='Observed',
            #                     line=dict(color='blue')
            #                 ))
            #                 fig1.add_trace(go.Scatter(
            #                     x=decomposition_df.index,
            #                     y=decomposition_df['Trend'],
            #                     name='Trend',
            #                     line=dict(color='green')
            #                 ))
            #                 fig1.update_layout(
            #                     title=f"Observed and Trend of {analysis_var}",
            #                     xaxis=dict(title="Date"),
            #                     yaxis=dict(title="Value"),
            #                     legend=dict(x=0.1, y=1.1),
            #                     hovermode="x unified"
            #                 )
            #                 st.plotly_chart(fig1, use_container_width=True)
            #                 st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #                 # Plot Seasonality separately
            #                 fig2 = go.Figure()
            #                 fig2.add_trace(go.Scatter(
            #                     x=decomposition_df.index,
            #                     y=decomposition_df['Seasonal'],
            #                     name='Seasonality',
            #                     line=dict(color='red')
            #                 ))
            #                 fig2.update_layout(
            #                     title=f"Seasonality of {analysis_var}",
            #                     xaxis=dict(title="Date"),
            #                     yaxis=dict(title="Value"),
            #                     hovermode="x unified"
            #                 )
            #                 st.plotly_chart(fig2, use_container_width=True)
            #                 st.markdown('<hr class="thin">', unsafe_allow_html=True)

            #                 # Plot Residuals separately
            #                 fig3 = go.Figure()
            #                 fig3.add_trace(go.Scatter(
            #                     x=decomposition_df.index,
            #                     y=decomposition_df['Residual'],
            #                     name='Residuals',
            #                     line=dict(color='purple')
            #                 ))
            #                 fig3.update_layout(
            #                     title=f"Residuals of {analysis_var}",
            #                     xaxis=dict(title="Date"),
            #                     yaxis=dict(title="Value"),
            #                     hovermode="x unified"
            #                 )
            #                 st.plotly_chart(fig3, use_container_width=True)
            #             else:
            #                 st.info("Trend analysis not available for yearly data")

            #         with col29:
            #             if frequency != "Yearly":
            #                 # Create column names for new features
            #                 trend_col = f'{analysis_var}_Trend'
            #                 seasonality_col = f'{analysis_var}_Seasonality'

            #                 # Create a preview DataFrame
            #                 preview_data = current_data.copy()

            #                 # Group data by selected columns
            #                 grouped_data = current_data.groupby(selected_columns)

            #                 # Calculate trend and seasonality for all groups (preview only)
            #                 for group_name, group_data in grouped_data:
            #                     group_data = group_data.set_index(date_col)

            #                     # Handle group name formatting
            #                     group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

            #                     # Calculate decompositions
            #                     decomposition = sm.tsa.seasonal_decompose(
            #                         group_data[analysis_var], 
            #                         model='additive', 
            #                         period=period
            #                     )
            #                     stl_result = STL(group_data[analysis_var], period=period).fit()

            #                     # Create mask for current group
            #                     group_mask = (preview_data[selected_columns].apply(tuple, axis=1) == group_name)

            #                     # Only proceed if lengths match
            #                     if group_mask.sum() == len(stl_result.trend):
            #                         preview_data.loc[group_mask, trend_col] = stl_result.trend.values
            #                         preview_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

            #                 # Display preview
            #                 st.info(f"Preview of Trend and Seasonality features for '{analysis_var}':")
            #                 st.dataframe(preview_data)

            #                 # Save button
            #                 if st.button("💾 Save Trend Components", key='save_trend_components'):
            #                     # Initialize columns if they don't exist
            #                     if trend_col not in st.session_state.working_data.columns:
            #                         st.session_state.working_data[trend_col] = None
            #                     if seasonality_col not in st.session_state.working_data.columns:
            #                         st.session_state.working_data[seasonality_col] = None

            #                     # Calculate and save for all groups
            #                     for group_name, group_data in grouped_data:
            #                         group_data = group_data.set_index(date_col)
            #                         group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

            #                         decomposition = sm.tsa.seasonal_decompose(
            #                             group_data[analysis_var], 
            #                             model='additive', 
            #                             period=period
            #                         )
            #                         stl_result = STL(group_data[analysis_var], period=period).fit()

            #                         group_mask = (st.session_state.working_data[selected_columns].apply(tuple, axis=1) == group_name)

            #                         if group_mask.sum() == len(stl_result.trend):
            #                             st.session_state.working_data.loc[group_mask, trend_col] = stl_result.trend.values
            #                             st.session_state.working_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

            #                     # Update staging data
            #                     st.session_state.staging_data = st.session_state.working_data.copy()
            #                     st.success(f"Saved trend and seasonality components for {analysis_var}!")
            #                     st.rerun()
            #             else:
            #                 st.info("Cannot save trend components for yearly data")

            #         # Show current working data
            #         with st.expander('📊 Current Working Data'):
            #             st.dataframe(st.session_state.working_data)




                    








            # else:
            #     # If no file is uploaded, ensure session_state is cleared
            #     st.session_state.clear()







        # if selected=="MODELING":
        # with tab14:
        if selected=="MODEL":
            # render_workflow(2)
            # show_workflow("MODELING")

            if uploaded_file:

                if 'modified_data' in st.session_state:
                    d0_auto=st.session_state.modified_data
                # if uploaded_file.name.endswith(".csv"):
                #     d0 = pd.read_csv(uploaded_file)
                # else:
                #     d0 = pd.read_excel(uploaded_file)
                

                    # # with col1:
                    with st.expander("Show Data"):
                        st.dataframe(d0_auto)
                    
                    date_col = detect_date_column(d0_auto)
                    if not date_col:
                        date_col = st.selectbox("📅 Select the date column", d0_auto.columns, index=0)
                    
                    d0_auto[date_col] = pd.to_datetime(d0_auto[date_col])

                    d0_auto = d0_auto.sort_values(by=date_col)

                    basis_columns = ['Market', 'Channel', 'Region', 'Category','SubCategory', 'Brand', 'PPG', 'Variant', 'PackType', 'PackSize']


                    col1, col2 = st.columns(2)

                    with col1:
                        columns_with_multiple_unique_values = [
                                                                col for col in basis_columns 
                                                                if col in d0_auto.columns and d0_auto[col].nunique() > 0 and col != date_col
                                                            ]

                        # Allow user to select which columns to consider
                        # selected_columns = st.multiselect("COLUMNS CONSIDERED", columns_with_multiple_unique_values, default=columns_with_multiple_unique_values,key='selected_columns_for_forecasting')
                        selected_columns = [st.selectbox(
                        "COLUMN CONSIDERED", 
                        columns_with_multiple_unique_values, 
                        index=0 if columns_with_multiple_unique_values else None,key='selected_columns_for_forecasting'
                    )]

                    #     possible_prediction_basis = [col for col in d0.columns if col not in [date_col]]
                    #     prediction_basis = st.selectbox("SELECT THE CATEGORY", possible_prediction_basis)
                        

                    with col2:
                        # st.write(d0_auto.columns)
                        possible_target_col = [col for col in d0_auto.columns if col not in [date_col] + basis_columns+['Year','Month','Week','Fiscal Year']]
                        # st.write(possible_target_col)
                        # target_col = st.selectbox("SELECT WHAT TO FORECAST", possible_target_col,key='target_col_for_forecasting')



                        if 'target_col_for_forecasting' not in st.session_state:
                            st.session_state.target_col_for_forecasting = possible_target_col[0]

                        target_col = st.selectbox(
                            "SELECT WHAT TO FORECAST",
                            possible_target_col,
                            key='target_col_for_forecasting'  # Using the same key makes it persistent automatically
                        )






                    # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    all_features = [col for col in d0_auto.columns if col not in [target_col, date_col]+basis_columns+['Year','Month','Week']]

                    d0_df=d0_auto.copy()

                    d0_df = d0_df[[date_col] + selected_columns + [target_col]+all_features]
                    

                    d0_df=d0_df.groupby(selected_columns+[date_col], as_index=False)[[target_col]+all_features].sum()

                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                    # st.write(d0_auto)

                    df_auto=d0_auto.copy()


                    tab17,tab18 = st.tabs(['AutoRegressive','Feature-Based'])

                    with tab17:

                        # st.write('auto reg')

                        # perform_ar = st.checkbox("Perform Autoregressive Models", value=True)
                        perform_ar=st.radio("Perform Autoregressive Modeling", ["Yes","No"], index=1, horizontal=True)
                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        


                        if perform_ar=="Yes":




                            

                            col3, col4 = st.columns(2)

                            with col3:

                                # Get the range of available years
                                min_year = df_auto[date_col].dt.year.min()
                                max_year = df_auto[date_col].dt.year.max()

                                # Streamlit UI - Two-Way Slider for Year Range Selection
                                selected_year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

                                # Filter DataFrame based on selected year range
                                df_auto = df_auto[
                                    (df_auto[date_col].dt.year >= selected_year_range[0]) & 
                                    (df_auto[date_col].dt.year <= selected_year_range[1])
                                ]

                                if selected_columns:
                                    # Group data by selected columns
                                    grouped_data = df_auto.groupby(selected_columns)

                                    # Get the list of groups
                                    group_names = list(grouped_data.groups.keys())


                                    if 'frequency' in st.session_state:
                                        frequency=st.session_state.frequency


                                    col5,col6,col11,col30=st.columns(4)

                                    with col5:
                                        selected_group = st.selectbox(f"Select the group", group_names)
                    

                                    with col6:
                                        forecast_horizon = st.number_input(f"Forecast Horizon ({frequency})", min_value=1, max_value=120, value=12)

                

                                    with col11:


                                        fiscal_start_month = st.selectbox("Select Fiscal Year Start Month", range(1, 13), index=0)  # Default to July (7)

                                    with col30:
                                        frequency_options = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}

                                        # Ensure the frequency matches the keys in frequency_options
                                        if frequency in frequency_options:
                                            default_frequency_key = frequency
                                        else:
                                            # st.write("No exact match found! Defaulting to 'Monthly'.")
                                            default_frequency_key = "Monthly"

                                        # st.write(f"Default Frequency Key: {default_frequency_key}")  # Debugging print

                                        selected_frequency = st.selectbox("Select Data Frequency", list(frequency_options.keys()), 
                                                                        index=list(frequency_options.keys()).index(default_frequency_key),disabled=True )

                                        frequency = frequency_options[selected_frequency]  # Store the corresponding value
                                    
                                    group_data = grouped_data.get_group(selected_group).set_index(date_col)

                                    # Fit models and get forecast
                                    df_forecast, accuracy = fit_models(group_data, target_col, date_col, forecast_horizon,frequency)




                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)


                                    
                                    # Plot the results
                                    fig = make_subplots(rows=1, cols=1)
                                    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Actual"], mode="lines", name="Actual", line=dict(color="blue", width=2)))
                                    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["SMA"], mode="lines", name="SMA", line=dict(dash='dot', color="seagreen", width=2)))
                                    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Holt-Winters"], mode="lines", name="Holt-Winters", line=dict(dash='dot', color="magenta", width=2)))
                                    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["ETS"], mode="lines", name="ETS", line=dict(dash='dot', color="green", width=2)))
                                    # fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["ARIMA"], mode="lines", name="ARIMA", line=dict(dash='dot', color="red", width=2)))
                                    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["SARIMA"], mode="lines", name="SARIMA", line=dict(dash='dot', color="purple", width=2)))
                                    fig.add_trace(go.Scatter(x=df_forecast["Date"], y=df_forecast["Prophet"], mode="lines", name="Prophet", line=dict(dash='dot', color="orange", width=2)))

                                    fig.update_layout(title=f"FORECAST FOR {selected_columns} : {selected_group}", xaxis_title="Date", yaxis_title=target_col, template="plotly_dark")
                                    st.plotly_chart(fig, use_container_width=True)




                                    # Add download button for Excel export
                                    st.markdown("---")
                                    # st.write("Download Forecast Data")

                                    # Create a BytesIO buffer for the Excel file
                                    from io import BytesIO
                                    output = BytesIO()

                                    # Create a Pandas Excel writer using BytesIO as its "file"
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        df_forecast.to_excel(writer, index=False, sheet_name='Forecast')

                                    # Set up the download button
                                    st.download_button(
                                        label="Download Forecast as Excel",
                                        data=output.getvalue(),
                                        file_name=f"forecast_{selected_columns}_{selected_group}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )

                                    # Display accuracy for each model
                                    # st.subheader("MODEL ACCURACY (1 - MAPE)")
                                    

                                    # accuracy_df = pd.DataFrame.from_dict(accuracy, orient="index", columns=["Accuracy"])
                                    # st.dataframe(accuracy_df)

                                    # # Plot accuracy as a bar chart
                                    # fig_accuracy = go.Figure()
                                    # fig_accuracy.add_trace(go.Bar(
                                    #     x=accuracy_df.index,
                                    #     y=accuracy_df["Accuracy"],
                                    #     text=accuracy_df["Accuracy"].round(3),
                                    #     textposition="auto",
                                    #     marker_color="skyblue"
                                    # ))
                                    # fig_accuracy.update_layout(
                                    #     title="Model Accuracy",
                                    #     xaxis_title="Model",
                                    #     yaxis_title="Accuracy",
                                    #     template="plotly_dark"
                                    # )
                                    # st.plotly_chart(fig_accuracy, use_container_width=True)

                                    # Define color mapping for models

                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    model_colors = {
                                        "Actual": "blue",
                                        "SMA": "seagreen",
                                        "Holt-Winters": "magenta",
                                        "ETS": "green",
                                        "ARIMA": "red",
                                        "SARIMA": "purple",
                                        "Prophet": "orange"
                                    }

                                    # Accuracy Bar Chart
                                    fig_accuracy = go.Figure()
                                    for model, color in model_colors.items():
                                        if model in accuracy:
                                            fig_accuracy.add_trace(go.Bar(
                                                x=[model],  # Ensure each bar has its corresponding model name
                                                y=[accuracy[model]],  # Accuracy value
                                                text=[round(accuracy[model], 3)] if accuracy[model] else None,  # Display rounded accuracy
                                                textposition="auto",
                                                marker_color=color  # Matching the line color
                                            ))

                                    fig_accuracy.update_layout(
                                        title="Model Accuracy",
                                        xaxis_title="Model",
                                        showlegend=False,
                                        yaxis_title="Accuracy",
                                        template="plotly_dark"
                                    )

                                    st.plotly_chart(fig_accuracy, use_container_width=True)

                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        # else:
                        #     st.warning("SELECT A CATEGORY!")

                                # else:
                                #     feature_cols = st.multiselect("Select features", [col for col in d0.columns if col not in [target_col, date_col, prediction_basis]])
                                #     st.write("📝 Selected features: ", ", ".join(feature_cols) if feature_cols else "None")




                            with col4:

                                # if forecasting_type == "Autoregressive":

                                # User input for fiscal year start month
                                # fiscal_start_month = st.selectbox("Select Fiscal Year Start Month", range(1, 13), index=6)  # Default to July (7)

                                # Example usage with Streamlit
                                # Example usage with Streamlit
                                # with st.expander("Annual Growth Rates", expanded=True):
                                    # # Calculate annual growth rates
                                    # forecast_results = {selected_group: df_forecast}
                                    # growth_results = calculate_annual_growth_single(forecast_results, forecast_horizon, fiscal_start_month=fiscal_start_month)

                                    # # Plot growth rates
                                    # fig_growth = go.Figure()
                                    # for model, data in growth_results[selected_group].items():
                                    #     fig_growth.add_trace(go.Scatter(
                                    #         x=data['fiscal_year'],  # Use 'fiscal_year' column for x-axis
                                    #         y=data[f"growth_rate_{model}"],
                                    #         mode="lines+markers",
                                    #         name=f"{model}",
                                    #         line=dict(color=model_colors.get(model, "black"), width=2)
                                    #     ))
                                    # fig_growth.update_layout(
                                    #     title=f"Annual Growth Rate for {selected_group}",
                                    #     xaxis_title="Fiscal Year",
                                    #     yaxis_title="Growth Rate",
                                    #     template="plotly_dark"
                                    # )
                                    # st.plotly_chart(fig_growth, use_container_width=True)

                                    # st.markdown('<hr class="thin">', unsafe_allow_html=True)
                                
                                    # # Annual Growth Rates Table
                                    # growth_dfs = []
                                    # year_extracted = False  # Ensure "Year" column is only added once

                                    
                                    # for model, data in growth_results[selected_group].items():
                                    #     df = data.reset_index()

                                    #     # Dynamically find the date column (handling missing cases)
                                    #     date = next((col for col in df.columns if "fiscal_year" in col.lower()), None)


                                    #     # Extract "Year" only once from the first available date column
                                    #     if date and not year_extracted:
                                    #         # df["Year"] = pd.to_datetime(df[date]).dt.year
                                    #         df["Year"] = df[date]
                                    #         year_extracted = True  # Prevent extracting "Year" multiple times


                                    #     # Rename volume & growth rate columns for each model
                                    #     df = df.rename(columns={"volume": f"Volume_{model}", f"growth_rate_{model}": f"Growth Rate_{model}"})

                                    #     # Multiply growth rate by 100 and format with '%'
                                    #     if f"Growth Rate_{model}" in df.columns:
                                    #         df[f"Growth Rate_{model}"] = (df[f"Growth Rate_{model}"] * 100).round(2).astype(str) + " %"

                                    #     # Keep only relevant columns
                                    #     selected_columns = ["Year"] if "Year" in df.columns else []
                                    #     selected_columns += [f"Growth Rate_{model}", f"Volume_{model}"]

                                    #     # Append to the list
                                    #     growth_dfs.append(df[selected_columns])

                                    # # Concatenate along columns
                                    # growth_table = pd.concat(growth_dfs, axis=1)

                                    # # Set 'Year' as index (only once)
                                    # if "Year" in growth_table.columns:
                                    #     growth_table = growth_table.set_index("Year")

                                    # # Display the cleaned table
                                    # st.dataframe(growth_table.iloc[2:])

                                # Initialize session state to store growth results for all groups
                            
                                # Initialize session state to store selected models and growth results
                
                                import plotly.colors as pc

                                # Function to reset session state when a new file is uploaded
                                def reset_session_state():
                                    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
                                        st.session_state.selected_models = {}
                                        st.session_state.growth_results_all = {}
                                        st.session_state.uploaded_file = uploaded_file

                                # # File uploader
                                # uploaded_file = st.file_uploader("Upload a new file", type=["csv", "xlsx"])
                                # if uploaded_file is not None:
                                #     reset_session_state()  # Reset session state if a new file is uploaded

                                # Initialize session state to store selected models and growth results
                                if "selected_models" not in st.session_state:
                                    st.session_state.selected_models = {}

                                if "growth_results_all" not in st.session_state:
                                    st.session_state.growth_results_all = {}

                                # Function to calculate growth results for a group
                                def calculate_growth_for_group(group, df_forecast, forecast_horizon, fiscal_start_month):
                                    forecast_results = {group: df_forecast}
                                    return calculate_annual_growth_single(forecast_results, forecast_horizon,start_year=min_year, fiscal_start_month=fiscal_start_month,frequency=frequency)

                                # Inside your existing code
                                with st.expander("Annual Growth Rates:",expanded=True):
                                    # Add a selection widget for models
                                    models_list = list(calculate_growth_for_group(selected_group, df_forecast, forecast_horizon, fiscal_start_month)[selected_group].keys())
                                    selected_models = st.multiselect(
                                        f"Select models for {selected_group}",
                                        options=models_list,
                                        default=models_list  # Default to all models
                                    )

                                    # Save the selected models for this group
                                    st.session_state.selected_models[selected_group] = selected_models

                                    # Calculate growth results for all models in this group (not filtered by selection)
                                    growth_results = calculate_growth_for_group(selected_group, df_forecast, forecast_horizon, fiscal_start_month)
                                    all_growth_results = {
                                        model: growth_results[selected_group][model]
                                        for model in models_list
                                    }

                                    # Store the filtered growth results in session state (for combined view)
                                    filtered_growth_results = {
                                        model: growth_results[selected_group][model]
                                        for model in selected_models
                                    }
                                    st.session_state.growth_results_all[selected_group] = filtered_growth_results

                                    # Plot growth rates for all models (not filtered by selection)
                                    fig_growth = go.Figure()
                                    for model, data in all_growth_results.items():
                                        fig_growth.add_trace(go.Scatter(
                                            x=data['fiscal_year'],  # Use 'fiscal_year' column for x-axis
                                            y=data[f"growth_rate_{model}"],
                                            mode="lines+markers",
                                            name=f"{model}",
                                            line=dict(color=model_colors.get(model, "black"), width=2)
                                        ))
                                    fig_growth.update_layout(
                                        title=f"Annual Growth Rate for {selected_group}",
                                        xaxis_title="Fiscal Year",
                                        yaxis_title="Growth Rate",
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig_growth, use_container_width=True)

                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    # Annual Growth Rates Table (not filtered by selection)
                                    growth_dfs = []
                                    year_extracted = False  # Ensure "Year" column is only added once

                                    for model, data in all_growth_results.items():
                                        df = data.reset_index()

                                        # Dynamically find the date column (handling missing cases)
                                        date = next((col for col in df.columns if "fiscal_year" in col.lower()), None)

                                        # Extract "Year" only once from the first available date column
                                        if date and not year_extracted:
                                            df["Year"] = df[date]
                                            year_extracted = True  # Prevent extracting "Year" multiple times

                                        # Rename volume & growth rate columns for each model
                                        df = df.rename(columns={"volume": f"Volume_{model}", f"growth_rate_{model}": f"Growth Rate_{model}"})

                                        # Multiply growth rate by 100 and format with '%'
                                        if f"Growth Rate_{model}" in df.columns:
                                            df[f"Growth Rate_{model}"] = (df[f"Growth Rate_{model}"] * 100).round(2).astype(str) + " %"

                                        # Keep only relevant columns
                                        req_columns_yearly = ["Year"] if "Year" in df.columns else []
                                        req_columns_yearly += [f"Growth Rate_{model}", f"Volume_{model}"]

                                        # Append to the list
                                        growth_dfs.append(df[req_columns_yearly])

                                    # Concatenate along columns
                                    growth_table = pd.concat(growth_dfs, axis=1)

                                    # Set 'Year' as index (only once)
                                    if "Year" in growth_table.columns:
                                        growth_table = growth_table.set_index("Year")

                                    # Display the cleaned table
                                    st.dataframe(growth_table)

                                    # Excel download
                                    output = BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        growth_table.to_excel(writer, index=False, sheet_name='Sheet1')
                                    excel_data = output.getvalue()
                                    st.download_button(
                                        label="Download as Excel",
                                        data=excel_data,
                                        file_name='growth_data.xlsx',
                                        mime='application/vnd.ms-excel',
                                    )

                

                                # st.markdown('<hr class="thin">', unsafe_allow_html=True)


                                # Calculate half-yearly growth rates
                                with st.expander("Half-Yearly Growth Rates"):
                                    forecast_results = {selected_group: df_forecast}
                                    growth_results = calculate_halfyearly_growth_single(forecast_results, forecast_horizon,start_year=min_year,fiscal_start_month=fiscal_start_month,frequency=frequency)

                                    # Plot growth rates
                                    fig_growth = go.Figure()
                                    for model, data in growth_results[selected_group].items():
                                        # Create a combined 'Year-Half' column for x-axis
                                        data['Year-Half'] = data['fiscal_year'].astype(str) + " " + data['fiscal_half']
                                        fig_growth.add_trace(go.Scatter(
                                            x=data['Year-Half'],
                                            y=data[f"growth_rate_{model}"],
                                            mode="lines+markers",
                                            name=f"{model}",
                                            line=dict(color=model_colors.get(model, "black"), width=2)
                                        ))
                                    fig_growth.update_layout(
                                        title=f"Half-Yearly Growth Rate for {selected_group}",
                                        xaxis_title="Year-Half",
                                        yaxis_title="Growth Rate",
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig_growth, use_container_width=True)
                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)
                                    # Initialize list for storing DataFrames
                                    growth_dfs = []
                                    for model, data in growth_results[selected_group].items():
                                        df = data.copy()
                                        # Create a combined 'Year-Half' column
                                        df['Year-Half'] = df['fiscal_year'].astype(str) + " " + df['fiscal_half']
                                        # Rename volume & growth rate columns for each model
                                        df = df.rename(columns={"volume": f"Volume_{model}", f"growth_rate_{model}": f"Growth Rate_{model}"})
                                        # Multiply growth rate by 100 and format with '%'
                                        if f"Growth Rate_{model}" in df.columns:
                                            df[f"Growth Rate_{model}"] = (df[f"Growth Rate_{model}"] * 100).round(2).astype(str) + " %"
                                        # Keep only relevant columns
                                        req_columns_halfyear = ["Year-Half", f"Growth Rate_{model}", f"Volume_{model}"]
                                        # Append to the list
                                        growth_dfs.append(df[req_columns_halfyear])

                                    # Merge DataFrames on 'Year-Half' column
                                    growth_table = growth_dfs[0]  # Start with the first DataFrame
                                    for df in growth_dfs[1:]:
                                        growth_table = growth_table.merge(df, on="Year-Half", how="outer")

                                    # Display the cleaned table
                                    st.dataframe(growth_table)



                                # Example usage with Streamlit
                                with st.expander("Quarterly Growth Rates"):
                                    # Calculate quarterly growth rates
                                    forecast_results = {selected_group: df_forecast}
                                    growth_results = calculate_quarterly_growth_single(forecast_results, forecast_horizon, start_year=min_year,fiscal_start_month=fiscal_start_month,frequency=frequency)

                                    # Plot growth rates
                                    fig_growth = go.Figure()
                                    for model, data in growth_results[selected_group].items():
                                        # Create a combined 'Year-Quarter' column for x-axis
                                        data['Year-Quarter'] = data['fiscal_year'].astype(str) + " Q" + data['fiscal_quarter'].astype(str)
                                        fig_growth.add_trace(go.Scatter(
                                            x=data['Year-Quarter'],
                                            y=data[f"growth_rate_{model}"],
                                            mode="lines+markers",
                                            name=f"{model}",
                                            line=dict(color=model_colors.get(model, "black"), width=2)
                                        ))
                                    fig_growth.update_layout(
                                        title=f"Quarterly Growth Rate for {selected_group}",
                                        xaxis_title="Year-Quarter",
                                        yaxis_title="Growth Rate",
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig_growth, use_container_width=True)
                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)


                                    # Quarterly Growth Rates Table
                                    quarterly_growth_dfs = []
                                    year_quarter_extracted = False  # Ensure "Year-Quarter" column is only added once

                                    for model, data in growth_results[selected_group].items():
                                        df = data.reset_index()

                                        # Combine fiscal_year and fiscal_quarter into "Year-Quarter"
                                        if not year_quarter_extracted:
                                            df["Year-Quarter"] = df["fiscal_year"].astype(str) + " Q" + df["fiscal_quarter"].astype(str)
                                            year_quarter_extracted = True  # Prevent extracting "Year-Quarter" multiple times

                                        # Rename volume & growth rate columns for each model
                                        df = df.rename(columns={"volume": f"Volume_{model}", f"growth_rate_{model}": f"Growth Rate_{model}"})

                                        # Multiply growth rate by 100 and format with '%'
                                        if f"Growth Rate_{model}" in df.columns:
                                            df[f"Growth Rate_{model}"] = (df[f"Growth Rate_{model}"] * 100).round(2).astype(str) + " %"

                                        # Keep only relevant columns
                                        req_columns_quatyear = ["Year-Quarter"] if "Year-Quarter" in df.columns else []
                                        req_columns_quatyear += [f"Growth Rate_{model}", f"Volume_{model}"]

                                        # Append to the list
                                        quarterly_growth_dfs.append(df[req_columns_quatyear])

                                    # Concatenate along columns
                                    quarterly_growth_table = pd.concat(quarterly_growth_dfs, axis=1)

                                    # Drop duplicate "Year-Quarter" columns (if any)
                                    quarterly_growth_table = quarterly_growth_table.loc[:, ~quarterly_growth_table.columns.duplicated()]

                                    # Set 'Year-Quarter' as index (only once)
                                    if "Year-Quarter" in quarterly_growth_table.columns:
                                        quarterly_growth_table = quarterly_growth_table.set_index("Year-Quarter")

                                    # Display the cleaned table
                                    st.dataframe(quarterly_growth_table)

                                st.markdown('<hr class="thin">', unsafe_allow_html=True)


                                # Initialize session state variables if they don't exist
                                if "post_data" not in st.session_state:
                                    st.session_state.post_data = {}

                                if "post_data_saved" not in st.session_state:
                                    st.session_state.post_data_saved = False

                                with st.expander("Comparison of annual growth rate between Groups"):
                                    # Check if any selections have been saved and data is not saved
                                    if st.session_state.selected_models and not st.session_state.post_data_saved:
                                        # Initialize a combined figure for all selected groups
                                        fig_combined = go.Figure()

                                        # Initialize a list to store combined tables for all selected groups
                                        combined_dfs = []

                                        # Generate a dynamic color palette for groups
                                        group_colors = pc.qualitative.Plotly  # Use Plotly's qualitative color palette
                                        group_color_map = {
                                            group: group_colors[i % len(group_colors)]
                                            for i, group in enumerate(st.session_state.selected_models.keys())
                                        }

                                        # Loop through each group and its selected models
                                        for group, models in st.session_state.selected_models.items():
                                            # Filter growth results for the selected models in this group
                                            filtered_growth_results = st.session_state.growth_results_all[group]

                                            # If no models are explicitly selected, default to the average of all models
                                            if not models:
                                                models = list(filtered_growth_results.keys())  # Use all models by default

                                            # If only one model is selected, use its growth rate directly
                                            if len(models) == 1:
                                                for model, data in filtered_growth_results.items():
                                                    # Rename the growth rate column to "growth_rate"
                                                    data = data.rename(columns={f"growth_rate_{model}": "growth_rate"})

                                                    # Add trace for the single model's growth rate
                                                    fig_combined.add_trace(go.Scatter(
                                                        x=data['fiscal_year'],
                                                        y=data["growth_rate"],
                                                        mode="lines+markers",
                                                        name=f"{group} - {model}",
                                                        line=dict(color=group_color_map[group], width=2)
                                                    ))

                                                    # Append data to the combined table
                                                    combined_dfs.append(data.assign(group=group))

                                            # If multiple models are selected, calculate the average growth rate
                                            elif len(models) > 1:
                                                # Combine growth rates for all selected models
                                                combined_growth = pd.DataFrame()
                                                for model, data in filtered_growth_results.items():
                                                    if combined_growth.empty:
                                                        combined_growth = data[["fiscal_year", f"growth_rate_{model}"]].copy()
                                                        combined_growth.rename(columns={f"growth_rate_{model}": "growth_rate"}, inplace=True)
                                                    else:
                                                        combined_growth["growth_rate"] += data[f"growth_rate_{model}"]
                                                combined_growth["growth_rate"] /= len(models)  # Calculate average
                                                combined_growth["group"] = [group] * len(combined_growth)  # Add group name for identification
                                                combined_dfs.append(combined_growth)

                                                # Add trace for the average growth rate
                                                fig_combined.add_trace(go.Scatter(
                                                    x=combined_growth['fiscal_year'],
                                                    y=combined_growth["growth_rate"],
                                                    mode="lines+markers",
                                                    name=f"{group} (Average)",
                                                    line=dict(color=group_color_map[group], width=2)
                                                ))

                                        # Combine all tables into one
                                        combined_table = pd.concat(combined_dfs, axis=0)

                                        # Pivot the table to have years as rows and groups as columns
                                        combined_table_pivot = combined_table.pivot(index="fiscal_year", columns="group", values="growth_rate")

                                        # Display the combined graph
                                        fig_combined.update_layout(
                                            title="Combined Growth Rates for Selected Groups",
                                            xaxis_title="Fiscal Year",
                                            yaxis_title="Growth Rate",
                                            template="plotly_dark"
                                        )
                                        st.plotly_chart(fig_combined, use_container_width=True)

                                        # Display the combined table
                                        st.dataframe(combined_table_pivot)


                                        # Save the combined table to session state
                                        st.session_state.post_data = combined_table_pivot

                                    else:
                                        
                                        st.error("Data has been Saved! No changes can be made until Reset.")
                                        if "post_data" in st.session_state:
                                            combined_table_pivot = st.session_state['post_data']

                                            st.dataframe(combined_table_pivot)
                                        

                                    col12, col13=st.columns(2)


                                    with col12:

                                        # Save button to lock the data
                                        if st.button("Save"):
                                            st.session_state.post_data_saved = True
                                            st.success("Data saved!\n\n Changes will not affect the saved data until Reset.")

                                            

                                    with col13:

                                        # Reset session state when clicking the reset button
                                        if st.button("Reset"):
                                            st.success("Data has been reset!\n\n Changes can be made.")
                                            # for key in ["selected_models", "growth_results_all", "post_data_saved"]:
                                            #     if key in st.session_state:
                                            #         del st.session_state[key]
                                            st.session_state.post_data_saved = False
                                            st.rerun()


                                    # Reset session state when clicking the reset button
                                        if st.button("Reset Session"):
                                            st.success("Session has been Reset!")
                                            for key in ["selected_models", "growth_results_all", "post_data_saved"]:
                                                if key in st.session_state:
                                                    del st.session_state[key]
                                            # st.session_state.post_data_saved = False
                                            st.rerun()


                        # else:
                        #     st.warning("Please select a prediction basis.")

                


        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################
        ##########################################################################################feature-based################################################################################



                    with tab18:
                        # st.write('dtcfvgbhnj')

                        with st.expander("ENGINEER:"):

                            



                            if uploaded_file:
                                if 'd0'in st.session_state:
                                    d0=st.session_state.d0

                                if 'date_col'in st.session_state:
                                    date_col=st.session_state.date_col


                                st.session_state.modified_data = d0_auto.copy()
                        
                    


                                import plotly.express as px

                                

                                # Create tabs for different functionalities
                                tab4, tab5, tab6,tab7 = st.tabs(["Trends", "Transformation","Create","View"])


                                with tab5:



                                    if 'working_data' not in st.session_state:
                                        st.session_state.working_data =None

                                    if 'staging_data' not in st.session_state:
                                        st.session_state.staging_data =None

                                    if 'last_modified_snapshot' not in st.session_state:
                                        st.session_state.last_modified_snapshot =None


                    


                                    # Initialize session state with unique names
                                    if 'data_pipeline_initialized' not in st.session_state:
                                        # Master original copy (never changes)
                                        st.session_state.master_original_data = st.session_state.modified_data.copy()
                                        
                                        # Working copy that gets updated
                                        st.session_state.working_data = st.session_state.modified_data.copy()
                                        
                                        # Temporary staging area for transformations
                                        st.session_state.staging_data = st.session_state.modified_data.copy()
                                        
                                        st.session_state.data_pipeline_initialized = True

                                    # Check for changes in modified_data and update working_data if needed
                                    if not st.session_state.modified_data.equals(st.session_state.last_modified_snapshot):
                                        st.session_state.working_data = st.session_state.modified_data.copy()
                                        st.session_state.staging_data = st.session_state.modified_data.copy()
                                        st.session_state.last_modified_snapshot = st.session_state.modified_data.copy()



                                    # Get current data copies
                                    current_data = st.session_state.working_data.copy()
                                    staging_data = st.session_state.staging_data.copy()

                                    # Available features for transformation
                                    transformable_features = [col for col in staging_data.columns 
                                                            if col not in [date_col] + basis_columns + ['Fiscal Year']+['Year','Month','Week']]

                                    # --- UI Section ---
                                    st.subheader("Transformation")

                                    # Transformation toggle
                                    apply_transforms = st.radio(
                                        "Apply transformations?",
                                        ["Yes", "No"],
                                        index=1,
                                        horizontal=True,
                                        key='transform_toggle'
                                    )

                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    if apply_transforms == "Yes":
                                        # Transformation selection
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            selected_feature = st.selectbox(
                                                "Feature to transform",
                                                transformable_features,
                                                key='feature_select'
                                            )
                                        with col2:
                                            transform_type = st.selectbox(
                                                "Transformation type",
                                                ["Log", "Square Root", "Exponential", "Power", "Residual"],
                                                key='transform_type'
                                            )

                                        # Apply transformation to staging data
                                        transformed_data = staging_data.copy()

                                        if transform_type == "Log":
                                            transformed_data[f'log_{selected_feature}'] = np.log1p(transformed_data[selected_feature])
                                        elif transform_type == "Square Root":
                                            transformed_data[f'sqrt_{selected_feature}'] = np.sqrt(transformed_data[selected_feature])
                                        elif transform_type == "Exponential":
                                            transformed_data[f'exp_{selected_feature}'] = np.exp(transformed_data[selected_feature])
                                        elif transform_type == "Power":
                                            power_val = st.number_input(
                                                "Power value",
                                                value=2.0,
                                                step=0.1,
                                                key='power_val'
                                            )
                                            try:
                                                transformed_data[f'power_{power_val}_{selected_feature}'] = np.power(
                                                    transformed_data[selected_feature],
                                                    power_val
                                                )
                                            except ValueError:
                                                st.warning("Negative values with fractional powers not supported")
                                                if st.checkbox("Use absolute values?", key='abs_power'):
                                                    transformed_data[f'power_{power_val}_abs_{selected_feature}'] = np.power(
                                                        np.abs(transformed_data[selected_feature]),
                                                        power_val
                                                    )
                                                    transformed_data[f'power_{power_val}_sign_{selected_feature}'] = np.sign(
                                                        transformed_data[selected_feature]
                                                    )
                                        
                                        elif transform_type == "Residual":
                                            st.markdown("**Residual Calculation**")
                                            col3, col4 = st.columns(2)
                                            with col3:
                                                y_var = st.selectbox(
                                                    "Dependent (Y) variable",
                                                    staging_data.columns,
                                                    key='residual_y'
                                                )
                                            with col4:
                                                x_vars = st.multiselect(
                                                    "Independent (X) variables",
                                                    [col for col in staging_data.columns if col != y_var],
                                                    key='residual_x'
                                                )
                                            
                                            if x_vars:
                                                try:
                                                    X = transformed_data[x_vars]
                                                    y = transformed_data[y_var]
                                                    
                                                    # Standardize and add constant
                                                    X = (X - X.mean()) / X.std()
                                                    X = sm.add_constant(X)
                                                    
                                                    # Fit model
                                                    model = sm.OLS(y, X).fit()
                                                    
                                                    # Store residuals + mean
                                                    transformed_data[f'Res_{y_var}'] = model.resid + y.mean()
                                                    
                                                    st.success(f"Residuals calculated (R² = {model.rsquared:.3f})")
                                                except Exception as e:
                                                    st.error(f"Residual calculation failed: {str(e)}")

                                        # Preview transformed data
                                        st.write("**Transformation Preview:**")
                                        st.dataframe(transformed_data.head())

                                        # Save button - updates working_data only when clicked
                                        if st.button("💾 Save Transformations", key='save_transforms'):
                                            st.session_state.working_data = transformed_data.copy()
                                            st.session_state.staging_data = transformed_data.copy()
                                            st.success("Transformations saved to working dataset!")
                                            st.rerun()

                                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                        # Column operations on working data
                                        current_columns = st.session_state.working_data.columns.tolist()

                                        # Rename columns
                                        st.subheader("Column Renaming")
                                        col5, col6 = st.columns(2)
                                        with col5:
                                            old_col = st.selectbox(
                                                "Column to rename",
                                                current_columns,
                                                key='col_to_rename'
                                            )
                                        with col6:
                                            new_col = st.text_input(
                                                "New name",
                                                key='new_col_name'
                                            )
                                        
                                        if st.button("🔄 Rename Column", key='rename_col'):
                                            if new_col:
                                                st.session_state.working_data.rename(
                                                    columns={old_col: new_col},
                                                    inplace=True
                                                )
                                                st.session_state.staging_data = st.session_state.working_data.copy()
                                                st.success(f"Renamed '{old_col}' to '{new_col}'")
                                                st.rerun()
                                            else:
                                                st.warning("Please enter a new name")

                                        # Delete columns
                                        st.subheader("Column Deletion")
                                        cols_to_remove = st.multiselect(
                                            "Columns to delete",
                                            current_columns,
                                            key='cols_to_delete'
                                        )
                                        
                                        if st.button("🗑️ Delete Columns", key='delete_cols'):
                                            st.session_state.working_data.drop(
                                                columns=cols_to_remove,
                                                inplace=True
                                            )
                                            st.session_state.staging_data = st.session_state.working_data.copy()
                                            st.success(f"Deleted {len(cols_to_remove)} columns")
                                            st.rerun()

                                





                                with tab6:

                                    # Update the working_data reference
                                    st.session_state.working_data = st.session_state.working_data.copy()

                                    # Sync staging_data with working_data
                                    if 'working_data' in st.session_state:
                                        st.session_state.staging_data = st.session_state.working_data.copy()

                                    # Use the staging data for all operations
                                    current_data = st.session_state.staging_data.copy()

                                    # Get available features
                                    all_features = [col for col in current_data.columns 
                                                if col not in [ date_col] + basis_columns + ['Fiscal Year']+['Year','Month','Week']]

                                    # Feature creation section
                                    st.subheader("Create New Features")
                                    create_new_feature = st.radio(
                                        "Do you want to create a new feature?", 
                                        ["Yes", "No"], 
                                        index=1, 
                                        horizontal=True,
                                        key='new_feature_toggle'
                                    )
                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    if create_new_feature == "Yes":
                                        # Column selection for new feature
                                        col16, col17, col24 = st.columns(3)
                                        with col16:
                                            col1 = st.selectbox(
                                                "Select the first column", 
                                                all_features, 
                                                index=0,
                                                key='new_feature_col1'
                                            )
                                        with col17:
                                            col2 = st.selectbox(
                                                "Select the second column", 
                                                all_features , 
                                                index=1,
                                                key='new_feature_col2'
                                            )
                                        with col24:
                                            operation = st.selectbox(
                                                "Select operation", 
                                                ["Add", "Subtract", "Multiply", "Divide"], 
                                                index=0,
                                                key='new_feature_op'
                                            )
                                        
                                        # Generate new column name
                                        new_col_name = f'{col1}_{operation.lower()}_{col2}'
                                        
                                        # Create new feature in staging data
                                        staging_with_new_feature = current_data.copy()
                                        if operation == "Add":
                                            staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] + staging_with_new_feature[col2]
                                        elif operation == "Subtract":
                                            staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] - staging_with_new_feature[col2]
                                        elif operation == "Multiply":
                                            staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] * staging_with_new_feature[col2]
                                        elif operation == "Divide":
                                            staging_with_new_feature[new_col_name] = staging_with_new_feature[col1] / staging_with_new_feature[col2]

                                        # Save button for new features
                                        if st.button("💾 Save New Feature", key='save_new_feature'):
                                            st.session_state.working_data = staging_with_new_feature.copy()
                                            st.session_state.staging_data = staging_with_new_feature.copy()
                                            st.success("New feature saved to working dataset!")
                                            st.rerun()

                                        # Display preview
                                        st.write("Preview with New Feature:")
                                        st.dataframe(staging_with_new_feature.head())
                                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                        # Column operations
                                        current_columns = st.session_state.working_data.columns.tolist()
                                        
                                        # Rename columns
                                        col26, col27 = st.columns(2)
                                        with col26:
                                            column_to_rename = st.selectbox(
                                                "Select column to rename", 
                                                current_columns,
                                                key='rename_col_select'
                                            )
                                        with col27:
                                            new_column_name = st.text_input(
                                                "New column name",
                                                key='new_col_name_input'
                                            )

                                        if st.button("🔄 Rename Column", key='rename_col_btn'):
                                            if new_column_name:
                                                st.session_state.working_data.rename(
                                                    columns={column_to_rename: new_column_name},
                                                    inplace=True
                                                )
                                                st.session_state.staging_data = st.session_state.working_data.copy()
                                                st.success(f"Renamed '{column_to_rename}' to '{new_column_name}'")
                                                st.rerun()
                                            else:
                                                st.warning("Please enter a new column name")

                                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                        # Delete columns
                                        st.write("Any Column to Delete?")
                                        columns_to_delete = st.multiselect(
                                            "Select columns to delete",
                                            current_columns,
                                            key='cols_to_delete_select'
                                        )
                                        
                                        if st.button("🗑️ Delete Columns", key='delete_cols_btn'):
                                            st.session_state.working_data.drop(
                                                columns=columns_to_delete,
                                                inplace=True
                                            )
                                            st.session_state.staging_data = st.session_state.working_data.copy()
                                            st.success(f"Deleted {len(columns_to_delete)} columns")
                                            st.rerun()

                                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    # Display current saved data
                                    # with st.expander("📊 Current Working Data", expanded=False):
                                    #     st.write(f"Shape: {st.session_state.working_data.shape}")
                                    #     st.dataframe(st.session_state.working_data)



                                    # Convert DataFrame to Excel
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        st.session_state.working_data.to_excel(writer, index=False, sheet_name='Sheet1')
                                    excel_data = output.getvalue()

                                    # Download button
                                    st.download_button(
                                        label="📥 Download the final data as Excel",
                                        data=excel_data,
                                        file_name='final_saved_data.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                    )








                                    # # Reset button to revert to original data
                                    # if st.button("Reset Data to Original"):
                                    #     st.session_state.modified_data = st.session_state.original_d0.copy()
                                    #     st.session_state.temp_data = st.session_state.modified_data.copy()
                                    #     st.success("Data reset to original state!")


                                # with tab7:


                                #     st.session_state.modified_data=st.session_state.working_data.sort_values(by=date_col)


                                #     import plotly.express as px

                                #     # Line Graph Visualization with Plotly
                                #     # st.markdown('<hr class="thick">', unsafe_allow_html=True)
                                #     st.subheader("Interactive Data Visualization")

                                #     # Check if there's a date column in the data
                                #     date_columns = [col for col in st.session_state.modified_data.columns 
                                #                 if pd.api.types.is_datetime64_any_dtype(st.session_state.modified_data[col])]
                                    
                                #     col40,col41,col42=st.columns(3)

                                #     if date_columns:

                                #         with col40:
                                #             # Let user select date column
                                #             date_col = st.selectbox("Select Date Column", date_columns)

                                #         column_to_filter=selected_columns[0]


                                #         with col41:

                                #             # Check if SubCategory exists
                                #             if column_to_filter in st.session_state.modified_data.columns:
                                #                 # Get unique subcategories
                                #                 subcategories = st.session_state.modified_data[column_to_filter].unique()
                                #                 selected_subcategories = st.multiselect(
                                #                     f"Filter by {column_to_filter}", 
                                #                     subcategories,
                                #                     default=subcategories[0] if len(subcategories) > 0 else None
                                #                 )
                                                
                                #                 # Filter data by selected subcategories
                                #                 if selected_subcategories:
                                #                     filtered_data = st.session_state.modified_data[
                                #                         st.session_state.modified_data[column_to_filter].isin(selected_subcategories)
                                #                     ]
                                #                 else:
                                #                     filtered_data = st.session_state.modified_data
                                #                     st.warning(f"No {column_to_filter} selected - showing all data")
                                #             else:
                                #                 filtered_data = st.session_state.modified_data
                                #                 st.warning(f"No {column_to_filter} column found - showing all data")
                                        
                                #         # Get numeric columns (excluding the date column and SubCategory)
                                #         numeric_cols = [col for col in filtered_data.columns 
                                #                     if col not in [date_col, column_to_filter] and pd.api.types.is_numeric_dtype(filtered_data[col])]
                                        
                                        
                                        
                                #         if numeric_cols:
                                #             with col42:
                                #                 # Multi-select for columns to plot
                                #                 selected_cols = st.multiselect("Select Columns to Plot", numeric_cols, default=numeric_cols[0])
                                            
                                #             if selected_cols:
                                #                 # Create the plot
                                #                 if column_to_filter in filtered_data.columns:
                                #                     # If we have SubCategory, use it for color coding
                                #                     fig = px.line(
                                #                         filtered_data,
                                #                         x=date_col,
                                #                         y=selected_cols[0],  # Plotly Express needs one y column at a time
                                #                         color=column_to_filter,
                                #                         title=f'{selected_cols[0]} Trend by {column_to_filter}',
                                #                         labels={selected_cols[0]: 'Value', date_col: 'Date'},
                                #                         template='plotly_white'
                                #                     )
                                                    
                                #                     # If multiple columns selected, add them as separate lines
                                #                     if len(selected_cols) > 1:
                                #                         for col in selected_cols[1:]:
                                #                             fig.add_scatter(
                                #                                 x=filtered_data[date_col],
                                #                                 y=filtered_data[col],
                                #                                 mode='lines',
                                #                                 name=col,
                                #                                 visible='legendonly'  # Starts hidden but can be toggled
                                #                             )
                                #                 else:
                                #                     # No SubCategory - just plot selected columns
                                #                     fig = px.line(
                                #                         filtered_data,
                                #                         x=date_col,
                                #                         y=selected_cols,
                                #                         title='Trend Over Time',
                                #                         labels={'value': 'Value', date_col: 'Date'},
                                #                         template='plotly_white'
                                #                     )
                                                
                                #                 # Improve layout
                                #                 fig.update_layout(
                                #                     hovermode='x unified',
                                #                     xaxis=dict(title=date_col, showgrid=True),
                                #                     yaxis=dict(title='Value', showgrid=True),
                                #                     legend=dict(title='Categories'),
                                #                     height=600
                                #                 )
                                                
                                #                 # Add range slider
                                #                 fig.update_xaxes(
                                #                     rangeslider_visible=True,
                                #                     rangeselector=dict(
                                #                         buttons=list([
                                #                             dict(count=1, label="1m", step="month", stepmode="backward"),
                                #                             dict(count=6, label="6m", step="month", stepmode="backward"),
                                #                             dict(count=1, label="YTD", step="year", stepmode="todate"),
                                #                             dict(count=1, label="1y", step="year", stepmode="backward"),
                                #                             dict(step="all")
                                #                         ])
                                #                     )
                                #                 )
                                                
                                #                 st.plotly_chart(fig, use_container_width=True)
                                #             else:
                                #                 st.warning("Please select at least one column to plot.")
                                #         else:
                                #             st.warning("No numeric columns found for plotting.")
                                #     else:
                                #         st.warning("No date columns found in the data. Cannot create time series plot.")




                                with tab4:
                                    frequency=st.session_state.frequency



                                    # Ensure we're working with the latest data
                                    current_data = st.session_state.working_data.copy()

                                    col28, col29 = st.columns(2)

                                    with col28:
                                        col20, col21 = st.columns(2)

                                        with col20:
                                            if selected_columns:
                                                # Group data by selected columns
                                                grouped_data = current_data.groupby(selected_columns)

                                                # Get the list of groups
                                                group_names = list(grouped_data.groups.keys())

                                                selected_group = st.selectbox(
                                                    f"Select the group", 
                                                    group_names, 
                                                    key="group_selection_for_trend"
                                                )

                                                group_data = grouped_data.get_group(selected_group).set_index(date_col)

                                        with col21:
                                            # Select the variable to analyze
                                            analysis_var = st.selectbox(
                                                "Select the variable to analyze", 
                                                all_features, 
                                                index=0,
                                                key='analysis_var_select'
                                            )

                                        # Set seasonal period based on frequency
                                        frequency_periods = {
                                            "Daily": 7,      # Weekly seasonality
                                            "Weekly": 52,    # Yearly seasonality
                                            "Monthly": 12,   # Yearly seasonality
                                            "Quarterly": 4,  # Yearly seasonality
                                            "Yearly": 1      # No seasonality
                                        }
                                        
                                        period = frequency_periods.get(frequency, 12)  # Default to 12 if frequency not found
                                        
                                        if frequency == "Yearly":
                                            st.warning("Yearly data doesn't have sufficient seasonality for decomposition")
                                        
                                        # Perform seasonal decomposition (only if not yearly)
                                        if frequency != "Yearly":
                                            decomposition = sm.tsa.seasonal_decompose(
                                                group_data[analysis_var], 
                                                model='additive', 
                                                period=period
                                            )
                                            
                                            # Perform STL decomposition for trend extraction
                                            stl = STL(group_data[analysis_var], period=period)
                                            stl_result = stl.fit()
                                            
                                            # Create a DataFrame for decomposition results
                                            decomposition_df = pd.DataFrame({
                                                'Date': group_data.index,
                                                'Observed': decomposition.observed,
                                                'Trend': stl_result.trend,
                                                'Seasonal': decomposition.seasonal,
                                                'Residual': decomposition.resid
                                            }).set_index('Date')

                                            # Plot Observed and Trend in one graph
                                            fig1 = go.Figure()
                                            fig1.add_trace(go.Scatter(
                                                x=decomposition_df.index,
                                                y=decomposition_df['Observed'],
                                                name='Observed',
                                                line=dict(color='blue')
                                            ))
                                            fig1.add_trace(go.Scatter(
                                                x=decomposition_df.index,
                                                y=decomposition_df['Trend'],
                                                name='Trend',
                                                line=dict(color='green')
                                            ))
                                            fig1.update_layout(
                                                title=f"Observed and Trend of {analysis_var}",
                                                xaxis=dict(title="Date"),
                                                yaxis=dict(title="Value"),
                                                legend=dict(x=0.1, y=1.1),
                                                hovermode="x unified"
                                            )
                                            st.plotly_chart(fig1, use_container_width=True)
                                            st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                            # Plot Seasonality separately
                                            fig2 = go.Figure()
                                            fig2.add_trace(go.Scatter(
                                                x=decomposition_df.index,
                                                y=decomposition_df['Seasonal'],
                                                name='Seasonality',
                                                line=dict(color='red')
                                            ))
                                            fig2.update_layout(
                                                title=f"Seasonality of {analysis_var}",
                                                xaxis=dict(title="Date"),
                                                yaxis=dict(title="Value"),
                                                hovermode="x unified"
                                            )
                                            st.plotly_chart(fig2, use_container_width=True)
                                            st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                            # Plot Residuals separately
                                            fig3 = go.Figure()
                                            fig3.add_trace(go.Scatter(
                                                x=decomposition_df.index,
                                                y=decomposition_df['Residual'],
                                                name='Residuals',
                                                line=dict(color='purple')
                                            ))
                                            fig3.update_layout(
                                                title=f"Residuals of {analysis_var}",
                                                xaxis=dict(title="Date"),
                                                yaxis=dict(title="Value"),
                                                hovermode="x unified"
                                            )
                                            st.plotly_chart(fig3, use_container_width=True)
                                        else:
                                            st.info("Trend analysis not available for yearly data")

                                    with col29:
                                        if frequency != "Yearly":
                                            # Create column names for new features
                                            trend_col = f'{analysis_var}_Trend'
                                            seasonality_col = f'{analysis_var}_Seasonality'

                                            # Create a preview DataFrame
                                            preview_data = current_data.copy()

                                            # Group data by selected columns
                                            grouped_data = current_data.groupby(selected_columns)

                                            # Calculate trend and seasonality for all groups (preview only)
                                            for group_name, group_data in grouped_data:
                                                group_data = group_data.set_index(date_col)

                                                # Handle group name formatting
                                                group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

                                                # Calculate decompositions
                                                decomposition = sm.tsa.seasonal_decompose(
                                                    group_data[analysis_var], 
                                                    model='additive', 
                                                    period=period
                                                )
                                                stl_result = STL(group_data[analysis_var], period=period).fit()

                                                # Create mask for current group
                                                group_mask = (preview_data[selected_columns].apply(tuple, axis=1) == group_name)

                                                # Only proceed if lengths match
                                                if group_mask.sum() == len(stl_result.trend):
                                                    preview_data.loc[group_mask, trend_col] = stl_result.trend.values
                                                    preview_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

                                            # Display preview
                                            st.info(f"Preview of Trend and Seasonality features for '{analysis_var}':")
                                            st.dataframe(preview_data)

                                            # # Save button
                                            # if st.button("💾 Save Trend Components", key='save_trend_components'):
                                            #     # Initialize columns if they don't exist
                                            #     if trend_col not in st.session_state.working_data.columns:
                                            #         st.session_state.working_data[trend_col] = None
                                            #     if seasonality_col not in st.session_state.working_data.columns:
                                            #         st.session_state.working_data[seasonality_col] = None

                                            #     # Calculate and save for all groups
                                            #     for group_name, group_data in grouped_data:
                                            #         group_data = group_data.set_index(date_col)
                                            #         group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

                                            #         decomposition = sm.tsa.seasonal_decompose(
                                            #             group_data[analysis_var], 
                                            #             model='additive', 
                                            #             period=period
                                            #         )
                                            #         stl_result = STL(group_data[analysis_var], period=period).fit()

                                            #         group_mask = (st.session_state.working_data[selected_columns].apply(tuple, axis=1) == group_name)

                                            #         if group_mask.sum() == len(stl_result.trend):
                                            #             st.session_state.working_data.loc[group_mask, trend_col] = stl_result.trend.values
                                            #             st.session_state.working_data.loc[group_mask, seasonality_col] = decomposition.seasonal.values

                                            #     # Update staging data
                                            #     st.session_state.staging_data = st.session_state.working_data.copy()
                                            #     st.success(f"Saved trend and seasonality components for {analysis_var}!")
                                            #     st.rerun()

                                            # Inside the "Save Trend Components" button section, modify the column initialization and assignment:

                                            # Save button
                                            if st.button("💾 Save Trend Components", key='save_trend_components'):
                                                # Initialize columns if they don't exist - ensure they're numeric
                                                if trend_col not in st.session_state.working_data.columns:
                                                    st.session_state.working_data[trend_col] = pd.Series(dtype='float64')
                                                if seasonality_col not in st.session_state.working_data.columns:
                                                    st.session_state.working_data[seasonality_col] = pd.Series(dtype='float64')

                                                # Calculate and save for all groups
                                                for group_name, group_data in grouped_data:
                                                    group_data = group_data.set_index(date_col)
                                                    group_name = (group_name,) if not isinstance(group_name, tuple) else group_name

                                                    decomposition = sm.tsa.seasonal_decompose(
                                                        group_data[analysis_var], 
                                                        model='additive', 
                                                        period=period
                                                    )
                                                    stl_result = STL(group_data[analysis_var], period=period).fit()

                                                    group_mask = (st.session_state.working_data[selected_columns].apply(tuple, axis=1) == group_name)

                                                    if group_mask.sum() == len(stl_result.trend):
                                                        # Ensure numeric conversion
                                                        st.session_state.working_data.loc[group_mask, trend_col] = pd.to_numeric(stl_result.trend.values, errors='coerce')
                                                        st.session_state.working_data.loc[group_mask, seasonality_col] = pd.to_numeric(decomposition.seasonal.values, errors='coerce')

                                                # Update staging data
                                                st.session_state.staging_data = st.session_state.working_data.copy()
                                                st.success(f"Saved trend and seasonality components for {analysis_var}!")
                                                st.rerun()
                                        else:
                                            st.info("Cannot save trend components for yearly data")


                                        # # In the preview section before displaying the dataframe:
                                        # preview_data[trend_col] = pd.to_numeric(preview_data[trend_col], errors='coerce')
                                        # preview_data[seasonality_col] = pd.to_numeric(preview_data[seasonality_col], errors='coerce')
                                        # st.dataframe(preview_data)

                                    # Show current working data
                                    # with st.expander('📊 Current Working Data'):
                                    # st.dataframe(st.session_state.working_data)



                                with tab7:


                                    # st.session_state.modified_data=st.session_state.working_data.sort_values(by=date_col)
                                    # st.session_state.working_data=st.session_state.modified_data.sort_values(by=date_col)


                                    # import plotly.express as px

                                    # # Line Graph Visualization with Plotly
                                    # # st.markdown('<hr class="thick">', unsafe_allow_html=True)
                                    # st.subheader("Interactive Data Visualization")

                                    # # Check if there's a date column in the data
                                    # date_columns = [col for col in st.session_state.modified_data.columns 
                                    #             if pd.api.types.is_datetime64_any_dtype(st.session_state.modified_data[col])]
                                    
                                    # col40,col41,col42=st.columns(3)

                                    # if date_columns:

                                    #     with col40:
                                    #         # Let user select date column
                                    #         date_col = st.selectbox("Select Date Column", date_columns)

                                    #     column_to_filter=selected_columns[0]


                                    #     with col41:

                                    #         # Check if SubCategory exists
                                    #         if column_to_filter in st.session_state.modified_data.columns:
                                    #             # Get unique subcategories
                                    #             subcategories = st.session_state.modified_data[column_to_filter].unique()
                                    #             selected_subcategories = st.multiselect(
                                    #                 f"Filter by {column_to_filter}", 
                                    #                 subcategories,
                                    #                 default=subcategories[0] if len(subcategories) > 0 else None
                                    #             )
                                                
                                    #             # Filter data by selected subcategories
                                    #             if selected_subcategories:
                                    #                 filtered_data = st.session_state.modified_data[
                                    #                     st.session_state.modified_data[column_to_filter].isin(selected_subcategories)
                                    #                 ]
                                    #             else:
                                    #                 filtered_data = st.session_state.modified_data
                                    #                 st.warning(f"No {column_to_filter} selected - showing all data")
                                    #         else:
                                    #             filtered_data = st.session_state.modified_data
                                    #             st.warning(f"No {column_to_filter} column found - showing all data")
                                        
                                    #     # Get numeric columns (excluding the date column and SubCategory)
                                    #     numeric_cols = [col for col in filtered_data.columns 
                                    #                 if col not in [date_col, column_to_filter] and pd.api.types.is_numeric_dtype(filtered_data[col])]
                                        
                                        
                                        
                                    #     if numeric_cols:
                                    #         with col42:
                                    #             # Multi-select for columns to plot
                                    #             selected_cols = st.multiselect("Select Columns to Plot", numeric_cols, default=numeric_cols[0])
                                            
                                    #         if selected_cols:
                                    #             # Create the plot
                                    #             if column_to_filter in filtered_data.columns:
                                    #                 # If we have SubCategory, use it for color coding
                                    #                 fig = px.line(
                                    #                     filtered_data,
                                    #                     x=date_col,
                                    #                     y=selected_cols[0],  # Plotly Express needs one y column at a time
                                    #                     color=column_to_filter,
                                    #                     title=f'{selected_cols[0]} Trend by {column_to_filter}',
                                    #                     labels={selected_cols[0]: 'Value', date_col: 'Date'},
                                    #                     template='plotly_white'
                                    #                 )
                                                    
                                    #                 # If multiple columns selected, add them as separate lines
                                    #                 if len(selected_cols) > 1:
                                    #                     for col in selected_cols[1:]:
                                    #                         fig.add_scatter(
                                    #                             x=filtered_data[date_col],
                                    #                             y=filtered_data[col],
                                    #                             mode='lines',
                                    #                             name=col,
                                    #                             visible='legendonly'  # Starts hidden but can be toggled
                                    #                         )
                                    #             else:
                                    #                 # No SubCategory - just plot selected columns
                                    #                 fig = px.line(
                                    #                     filtered_data,
                                    #                     x=date_col,
                                    #                     y=selected_cols,
                                    #                     title='Trend Over Time',
                                    #                     labels={'value': 'Value', date_col: 'Date'},
                                    #                     template='plotly_white'
                                    #                 )
                                                
                                    #             # Improve layout
                                    #             fig.update_layout(
                                    #                 hovermode='x unified',
                                    #                 xaxis=dict(title=date_col, showgrid=True),
                                    #                 yaxis=dict(title='Value', showgrid=True),
                                    #                 legend=dict(title='Categories'),
                                    #                 height=600
                                    #             )
                                                
                                    #             # Add range slider
                                    #             fig.update_xaxes(
                                    #                 rangeslider_visible=True,
                                    #                 rangeselector=dict(
                                    #                     buttons=list([
                                    #                         dict(count=1, label="1m", step="month", stepmode="backward"),
                                    #                         dict(count=6, label="6m", step="month", stepmode="backward"),
                                    #                         dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    #                         dict(count=1, label="1y", step="year", stepmode="backward"),
                                    #                         dict(step="all")
                                    #                     ])
                                    #                 )
                                    #             )
                                                
                                    #             st.plotly_chart(fig, use_container_width=True)
                                    #         else:
                                    #             st.warning("Please select at least one column to plot.")
                                    #     else:
                                    #         st.warning("No numeric columns found for plotting.")
                                    # else:
                                    #     st.warning("No date columns found in the data. Cannot create time series plot.")


                                    st.session_state.modified_data = st.session_state.working_data.sort_values(by=date_col)

                                    # import plotly.express as px
                                    # import plotly.graph_objects as go

                                    # Visualization selection
                                    # st.subheader("Interactive Data Visualization")
                                    visualization_type = st.radio(
                                        "Select Visualization Type",
                                        ("Time Series Plot", "Correlation Heatmap"),
                                        horizontal=True
                                    )

                                    if visualization_type == "Time Series Plot":
                                        # Line Graph Visualization with Plotly
                                        # Check if there's a date column in the data
                                        date_columns = [col for col in st.session_state.modified_data.columns 
                                                    if pd.api.types.is_datetime64_any_dtype(st.session_state.modified_data[col])]
                                        
                                        col40, col41, col42 = st.columns(3)

                                        if date_columns:
                                            with col40:
                                                # Let user select date column
                                                date_col = st.selectbox("Select Date Column", date_columns)

                                            column_to_filter = selected_columns[0]

                                            with col41:
                                                # Check if SubCategory exists
                                                if column_to_filter in st.session_state.modified_data.columns:
                                                    # Get unique subcategories
                                                    subcategories = st.session_state.modified_data[column_to_filter].unique()
                                                    selected_subcategories = st.multiselect(
                                                        f"Filter by {column_to_filter}", 
                                                        subcategories,
                                                        default=subcategories[0] if len(subcategories) > 0 else None
                                                    )
                                                    
                                                    # Filter data by selected subcategories
                                                    if selected_subcategories:
                                                        filtered_data = st.session_state.modified_data[
                                                            st.session_state.modified_data[column_to_filter].isin(selected_subcategories)
                                                        ]
                                                    else:
                                                        filtered_data = st.session_state.modified_data
                                                        st.warning(f"No {column_to_filter} selected - showing all data")
                                                else:
                                                    filtered_data = st.session_state.modified_data
                                                    st.warning(f"No {column_to_filter} column found - showing all data")
                                            
                                            # Get numeric columns (excluding the date column and SubCategory)
                                            numeric_cols = [col for col in filtered_data.columns 
                                                        if col not in [date_col, column_to_filter] and pd.api.types.is_numeric_dtype(filtered_data[col])]
                                            
                                            if numeric_cols:
                                                with col42:
                                                    # Multi-select for columns to plot
                                                    selected_cols = st.multiselect("Select Columns to Plot", numeric_cols, default=numeric_cols[0])
                                                
                                                if selected_cols:
                                                    # Create the plot
                                                    if column_to_filter in filtered_data.columns:
                                                        # If we have SubCategory, use it for color coding
                                                        fig = px.line(
                                                            filtered_data,
                                                            x=date_col,
                                                            y=selected_cols[0],  # Plotly Express needs one y column at a time
                                                            color=column_to_filter,
                                                            title=f'{selected_cols[0]} Trend by {column_to_filter}',
                                                            labels={selected_cols[0]: 'Value', date_col: 'Date'},
                                                            template='plotly_white'
                                                        )
                                                        
                                                        # If multiple columns selected, add them as separate lines
                                                        if len(selected_cols) > 1:
                                                            for col in selected_cols[1:]:
                                                                fig.add_scatter(
                                                                    x=filtered_data[date_col],
                                                                    y=filtered_data[col],
                                                                    mode='lines',
                                                                    name=col,
                                                                    visible='legendonly'  # Starts hidden but can be toggled
                                                                )
                                                    else:
                                                        # No SubCategory - just plot selected columns
                                                        fig = px.line(
                                                            filtered_data,
                                                            x=date_col,
                                                            y=selected_cols,
                                                            title='Trend Over Time',
                                                            labels={'value': 'Value', date_col: 'Date'},
                                                            template='plotly_white'
                                                        )
                                                    
                                                    # Improve layout
                                                    fig.update_layout(
                                                        hovermode='x unified',
                                                        xaxis=dict(title=date_col, showgrid=True),
                                                        yaxis=dict(title='Value', showgrid=True),
                                                        legend=dict(title='Categories'),
                                                        height=600
                                                    )
                                                    
                                                    # Add range slider
                                                    fig.update_xaxes(
                                                        rangeslider_visible=True,
                                                        rangeselector=dict(
                                                            buttons=list([
                                                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                                                dict(step="all")
                                                            ])
                                                        )
                                                    )
                                                    
                                                    st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.warning("Please select at least one column to plot.")
                                            else:
                                                st.warning("No numeric columns found for plotting.")
                                        else:
                                            st.warning("No date columns found in the data. Cannot create time series plot.")

                                    else:  # Correlation Heatmap
                                        # st.subheader("Correlation Heatmap")
                                        
                                        # Get numeric columns
                                        numeric_cols = [col for col in st.session_state.modified_data.columns 
                                                    if pd.api.types.is_numeric_dtype(st.session_state.modified_data[col])]
                                        
                                        if len(numeric_cols) >= 2:
                                            # Let user select columns for correlation
                                            selected_cols = st.multiselect(
                                                "Select columns for correlation analysis",
                                                numeric_cols,
                                                default=numeric_cols[:min(10, len(numeric_cols))]  # Default to first 10 or all if less
                                            )
                                            
                                            if len(selected_cols) >= 2:
                                                # Calculate correlation matrix
                                                corr_matrix = st.session_state.modified_data[selected_cols].corr()
                                                
                                                # Create heatmap
                                                fig = go.Figure(data=go.Heatmap(
                                                    z=corr_matrix,
                                                    x=corr_matrix.columns,
                                                    y=corr_matrix.columns,
                                                    colorscale='RdBu',
                                                    zmin=-1,
                                                    zmax=1,
                                                    hoverongaps=False,
                                                    text=np.round(corr_matrix.values, 2),
                                                    texttemplate="%{text}"
                                                ))
                                                
                                                fig.update_layout(
                                                    title='Correlation Matrix',
                                                    xaxis_title="Columns",
                                                    yaxis_title="Columns",
                                                    height=600,
                                                    width=600
                                                )
                                                
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Add interpretation note
                                                st.markdown("""
                                                **Interpretation Guide:**
                                                - Values close to **1** indicate strong positive correlation
                                                - Values close to **-1** indicate strong negative correlation
                                                - Values close to **0** indicate no correlation
                                                """)
                                            else:
                                                st.warning("Please select at least 2 numeric columns for correlation analysis.")
                                        else:
                                            st.warning("Not enough numeric columns (need at least 2) for correlation analysis.")



                                




                                    








                                # else:
                                #     # If no file is uploaded, ensure session_state is cleared
                                #     st.session_state.clear()

















                        if 'comparison_data' not in st.session_state:
                            st.session_state.comparison_data = {}


                        # def save_to_comparison(model_name, growth_data, volume_data, column_to_filter):
                        #     if model_name not in st.session_state.comparison_data:
                        #         st.session_state.comparison_data[model_name] = {}
                            
                        #     # Store the data with the selected group/region as key
                        #     st.session_state.comparison_data[model_name][column_to_filter] = {
                        #         'growth_rates': growth_data,
                        #         'volumes': volume_data
                        #     }
                        #     st.success(f"Data for {model_name} saved to comparison table!")


                        # def save_to_comparison(model_name, growth_data, volume_data, column_to_filter, custom_name, features_used=None):
                        #     # Create a unique key for this run (combination of model_name and custom_name)
                        #     run_key = f"{model_name}_{custom_name}"
                            
                        #     # Store all data including the custom name
                        #     st.session_state.comparison_data[run_key] = {
                        #         'model_name': model_name,  # Store the original model name separately
                        #         'growth_rates': growth_data,
                        #         'volumes': volume_data,
                        #         'group': column_to_filter,
                        #         'custom_name': custom_name,
                        #         'features_used': features_used if features_used else []
                        #     }
                        #     st.success(f"Saved as: {custom_name}")


                        def save_forecast_results(model_name, segment_name, forecast_results, metrics, feature_elasticities, features_used=None, custom_name=None):
                            # """
                            # Save forecast results, metrics, and feature elasticities for a model and segment.
                            
                            # Args:
                            #     model_name (str): Name of the model (e.g., "Prophet", "Ridge")
                            #     segment_name (str): Name of the segment/group being forecasted
                            #     forecast_results (dict): Dictionary containing forecast data
                            #     metrics (dict): Dictionary containing performance metrics (MAPE, R-squared, etc.)
                            #     feature_elasticities (pd.DataFrame): DataFrame containing feature elasticities
                            #     features_used (list): List of features used in the model
                            #     custom_name (str): Optional custom name for this run
                            # """
                            # Create a unique key for this run (combination of model_name and segment_name)
                            run_key = f"{model_name}_{segment_name}"
                            
                            if custom_name is None:
                                custom_name = run_key
                            
                            # Convert feature elasticities DataFrame to dict if it exists
                            feature_elasticities_dict = None
                            if feature_elasticities is not None and not feature_elasticities.empty:
                                feature_elasticities_dict = feature_elasticities.to_dict('records')
                            
                            # Store all data
                            st.session_state.comparison_data[run_key] = {
                                'model_name': model_name,
                                'segment_name': segment_name,
                                'custom_name': custom_name,
                                'forecast_results': forecast_results,
                                'metrics': metrics,
                                'feature_elasticities': feature_elasticities_dict,
                                'features_used': features_used if features_used else []
                            }
                            
                            st.success(f"Saved forecast results for {model_name} - {segment_name} as: {custom_name}")



                        



                        # Assuming d0_df, target_col, date_col, basis_columns, and selected_columns are predefined

                        # At the beginning of your script
                        # Set default values for session state variables if they are not already set
                        if 'models' not in st.session_state:
                            st.session_state.models = []  # or your default model list

                        if 'features' not in st.session_state:
                            st.session_state.features = []  # or default list of features

                        if 'selected_group' not in st.session_state:
                            st.session_state.selected_group = None  # or default group

                        if 'forecast_horizon' not in st.session_state:
                            st.session_state.forecast_horizon = 12  # or your default forecast horizon

                        # if 'fiscal_start_month' not in st.session_state:
                        #     st.session_state.fiscal_start_month = 1  # e.g., January

                        # if 'frequency' not in st.session_state:
                        #     st.session_state.frequency = 'M'  # 'D' for daily, 'H' for hourly, 'M' for monthly, etc....



                        df_fea = d0_auto.copy()

                        # model_types = st.selectbox(
                        #             "Select Model Types", 
                        #             ["None","Time-Series","ML"]   #, "Lasso", "Elastic Net"
                        #             # default=["None"]
                        #         )





                        frequency=st.session_state.frequency


                        # with st.expander("Select Saved Model and its Configurations:"):


                        #     # with col50:
                        #     saved_configs = [f.replace(".pkl", "") for f in os.listdir("saved_configs") if f.endswith(".pkl")]
                        #     # saved_configs = ["None"]+ [f.replace(".pkl", "") for f in os.listdir("saved_configs") if f.endswith(".pkl")]
                        #     if saved_configs:
                        #         selected = st.selectbox("Saved Models", saved_configs)
                        #         if st.checkbox("Load Selected"):
                        #             try:
                        #                 with open(f"saved_configs/{selected}.pkl", "rb") as f:
                        #                     loaded_config = pickle.load(f)
                                        
                        #                 # Store the loaded config in session state
                        #                 st.session_state['loaded_config'] = loaded_config
                                        
                        #                 # Display the loaded configuration
                        #                 st.write(f"Loaded Model: {selected}")
                        #                 st.json(loaded_config)  # More compact display than multiple st.write()
                                        
                        #                 if st.checkbox("Use This Configuration"):
                        #                     # Update session state with loaded values
                        #                     st.session_state.models = loaded_config.get('model', 'None')
                        #                     models = st.session_state.models

                        #                     st.session_state.valid_features = loaded_config.get('features', [])
                        #                     feature_cols = st.session_state.valid_features
                                            
                        #                     st.session_state.selected_group = loaded_config.get('group')
                        #                     selected_group=st.session_state.selected_group
                                            
                        #                     st.session_state.forecast_horizon = loaded_config.get('forecast_period')
                        #                     forecast_horizon=st.session_state.forecast_horizon 


                        #                     st.session_state.fiscal_start_month = loaded_config.get('fiscal_month')
                        #                     fiscal_start_month=st.session_state.fiscal_start_month


                        #                     # st.session_state.frequency = loaded_config.get('frequency')
                                            
                        #                     # st.success("Configuration applied successfully! Refresh the page to see changes.")
                        #                     st.success("Configuration applied successfully!")
                        #                     # st.rerun()

                        #             except Exception as e:
                        #                 st.error(f"Error loading configuration: {str(e)}")

                        #         # else:
                        #         #     st.info("No saved configurations available")


                        # frequency_options = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}

                        # # Ensure the frequency matches the keys in frequency_options
                        # if frequency in frequency_options:
                        #     default_frequency_key = frequency
                        # else:
                        #     # st.write("No exact match found! Defaulting to 'Monthly'.")
                        #     default_frequency_key = "Monthly"

                        # # st.write(f"Default Frequency Key: {default_frequency_key}")  # Debugging print

                        # selected_frequency = st.selectbox("Select Data Frequency", list(frequency_options.keys()), 
                        #                                 index=list(frequency_options.keys()).index(default_frequency_key))

                        # frequency = frequency_options[selected_frequency]  # Store the corresponding value







        


                        # # Model types selection
                        # models = st.selectbox(
                        #     "SELECT MODEL", 
                        #     ["None","Prophet","Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"],   #, "Lasso", "Elastic Net"
                        #     # default=["None"]
                        # )

                        # Model selection with session state default
                        # models = st.selectbox(
                        #     "SELECT MODEL", 
                        #     options=["None", "Prophet", "Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"],
                        #     index=["None", "Prophet", "Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"].index(
                        #         st.session_state.get('models', 'None')  # Default to 'None' if not in session state
                        #     ) if st.session_state.get('models', 'None') in ["None", "Prophet", "Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"] else 0
                        # )



    #-------------------------------------------multiselect option


                        # Model selection with session state default (multiselect)
                        available_models = [

                            "Prophet", 
                            "Generalized Constrained Ridge", 
                            "Generalized Constrained Lasso", 
                            "Ridge", 
                            "Linear Regression"
                        ]

                        # Get previously selected models from session state, default to ['None'] if not set or invalid
                        default_models = st.session_state.get('models', ['None'])
                        if not isinstance(default_models, list):
                            default_models = [default_models]

                        # Filter only valid models for default selection
                        default_models = [model for model in default_models if model in available_models]

                        # models = st.multiselect(
                        #     "SELECT MODEL", 
                        #     options=available_models,
                        #     default=default_models
                        # )


                        
                        models = st.multiselect(
                            label="**SELECT MODEL**",
                            options=sorted(available_models),  # Optional: sort for better UX
                            default=default_models,
                            help="You can select multiple models to compare performance."
                        )






                        # Optional: update session state if needed
                        st.session_state.models = models
    # #-------------------------------------------multiselect option






                        # if models in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]:
                        if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression","Prophet"]):

                        # valid_models = ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]

                        # # Ensure `models` is always treated as a list
                        # models_list = models if isinstance(models, list) else [models]

                        # if any(m in valid_models for m in models_list):

                            import re

                            column_to_filter = selected_columns[0]

                            


                            df=d0_auto.copy()

                            df_fea = d0_auto.copy()

                            min_year = df_fea[date_col].dt.year.min()
                            max_year = df_fea[date_col].dt.year.max()


                            # col38,col39=st.columns(2)

                            # with col38:


                            class GeneralizedConstraintRidgeRegression(BaseEstimator, RegressorMixin):
                                def __init__(self, l2_penalty=0.01, learning_rate=0.00001, iterations=100):
                                    self.learning_rate = learning_rate
                                    self.iterations = iterations
                                    self.l2_penalty = l2_penalty

                                def fit(self, X, Y, feature_names, constraints=None):
                                    """
                                    Fit the model with constraints on specific features.

                                    Parameters:
                                    - X: Feature matrix (numpy array)
                                    - Y: Target variable (numpy array)
                                    - feature_names: List of feature names (used for constraints)
                                    - constraints: Dictionary with feature constraints. Example:
                                        {
                                            'negative': ['scaled_Price', 'Market_X_interaction_scaled_Price'],
                                            'positive': ['scaled_Stores', 'scaled_Total A&P Spend']
                                        }
                                    """
                                    self.m, self.n = X.shape
                                    self.W = np.zeros(self.n)
                                    self.b = 0
                                    self.X = X
                                    self.Y = Y
                                    self.feature_names = feature_names
                                    self.constraints = constraints or {}
                                    # st.write("NaN values in X:", X)  # Check how many NaNs are in X


                                    # Prepare indices for constraints
                                    self.negative_indices = [
                                        feature_names.index(var) for var in self.constraints.get('negative', []) if var in feature_names
                                    ]
                                    self.positive_indices = [
                                        feature_names.index(var) for var in self.constraints.get('positive', []) if var in feature_names
                                    ]

                                    for _ in range(self.iterations):
                                        self.update_weights()
                                    return self

                                def update_weights(self):
                                    """Gradient Descent with Constraint Application."""
                                    Y_pred = self.predict(self.X)
                                    # st.write("NaN values in X:", np.isnan(Y_pred).sum())  # Check how many NaNs are in X
                                    dW = (-(2 * (self.X.T).dot(self.Y - Y_pred)) + (2 * self.l2_penalty * self.W)) / self.m
                                    db = -2 * np.sum(self.Y - Y_pred) / self.m

                                    self.W -= self.learning_rate * dW
                                    self.b -= self.learning_rate * db

                                    # Apply constraints
                                    self.apply_constraints()
                                    return self

                                def apply_constraints(self):
                                    """Apply constraints on weights."""

                                    # 1. Ensure specified weights are negative or zero
                                    for index in self.negative_indices:
                                        self.W[index] = min(self.W[index], 0)

                                    # 2. Ensure specified weights are positive or zero
                                    for index in self.positive_indices:
                                        self.W[index] = max(self.W[index], 0)

                                    # 3. Ensure sum of weights in 'sum_negative' groups is negative or zero
                                    if 'sum_negative' in self.constraints:
                                        for group in self.constraints['sum_negative']:
                                            if len(group) == 2:  # Handle exactly two variables (base and interaction)
                                                var, interaction = group
                                                if var in self.feature_names and interaction in self.feature_names:
                                                    var_index = self.feature_names.index(var)
                                                    interaction_index = self.feature_names.index(interaction)
                                                    total_beta = self.W[var_index] + self.W[interaction_index]
                                                    if total_beta > 0:  # Violation of constraint
                                                        deficit = -total_beta
                                                        # Adjust equally between the base variable and interaction term
                                                        self.W[var_index] += deficit / 2
                                                        self.W[interaction_index] += deficit / 2

                                    # 4. Ensure sum of weights in 'sum_positive' groups is positive or zero
                                    if 'sum_positive' in self.constraints:
                                        for group in self.constraints['sum_positive']:
                                            if len(group) == 2:  # Handle exactly two variables (base and interaction)
                                                var, interaction = group
                                                if var in self.feature_names and interaction in self.feature_names:
                                                    var_index = self.feature_names.index(var)
                                                    interaction_index = self.feature_names.index(interaction)
                                                    total_beta = self.W[var_index] + self.W[interaction_index]
                                                    if total_beta < 0:  # Violation of constraint
                                                        deficit = -total_beta
                                                        # Adjust equally between the base variable and interaction term
                                                        self.W[var_index] += deficit / 2
                                                        self.W[interaction_index] += deficit / 2



                                def predict(self, X):
                                    """Predict the target variable."""
                                    return X.dot(self.W) + self.b

                            class GeneralizedConstraintLassoRegression(BaseEstimator, RegressorMixin):
                                def __init__(self, l1_penalty=0.01, learning_rate=0.0001, iterations=100):
                                    self.learning_rate = learning_rate
                                    self.iterations = iterations
                                    self.l1_penalty = l1_penalty

                                def fit(self, X, Y, feature_names, constraints=None):
                                    """
                                    Fit the model with L1 regularization and constraints.

                                    Parameters:
                                    - X: Feature matrix (numpy array)
                                    - Y: Target variable (numpy array)
                                    - feature_names: List of feature names (used for constraints)
                                    - constraints: Dictionary with feature constraints. Example:
                                        {
                                            'negative': ['scaled_Price', 'Market_X_interaction_scaled_Price'],
                                            'positive': ['scaled_Stores', 'scaled_Total A&P Spend']
                                        }
                                    """
                                    self.m, self.n = X.shape
                                    self.W = np.zeros(self.n)
                                    self.b = 0
                                    self.X = X
                                    self.Y = Y
                                    self.feature_names = feature_names
                                    self.constraints = constraints or {}

                                    # Prepare indices for constraints
                                    self.negative_indices = [
                                        feature_names.index(var) for var in self.constraints.get('negative', []) if var in feature_names
                                    ]
                                    self.positive_indices = [
                                        feature_names.index(var) for var in self.constraints.get('positive', []) if var in feature_names
                                    ]

                                    for _ in range(self.iterations):
                                        self.update_weights()
                                    return self

                                def update_weights(self):
                                    """Gradient Descent with L1 Regularization and Constraint Application."""
                                    Y_pred = self.predict(self.X)

                                    # Compute gradients
                                    dW = (-2 * (self.X.T).dot(self.Y - Y_pred)) / self.m

                                    # Apply L1 regularization (soft thresholding)
                                    dW += self.l1_penalty * np.sign(self.W) / self.m

                                    db = -2 * np.sum(self.Y - Y_pred) / self.m

                                    # Gradient step
                                    self.W -= self.learning_rate * dW
                                    self.b -= self.learning_rate * db

                                    # Apply constraints
                                    self.apply_constraints()
                                    return self

                                # def apply_constraints(self):
                                #     """Apply constraints on weights."""

                                #     # Ensure specified weights are negative or zero
                                #     for index in self.negative_indices:
                                #         self.W[index] = min(self.W[index], 0)

                                #     # Ensure specified weights are positive or zero
                                #     for index in self.positive_indices:
                                #         self.W[index] = max(self.W[index], 0)

                                def apply_constraints(self):
                                    """Apply constraints on weights."""

                                    # 1. Ensure specified weights are negative or zero
                                    for index in self.negative_indices:
                                        self.W[index] = min(self.W[index], 0)

                                    # 2. Ensure specified weights are positive or zero
                                    for index in self.positive_indices:
                                        self.W[index] = max(self.W[index], 0)

                                    # 3. Ensure sum of weights in 'sum_negative' groups is negative or zero
                                    if 'sum_negative' in self.constraints:
                                        for group in self.constraints['sum_negative']:
                                            if len(group) == 2:  # Handle exactly two variables (base and interaction)
                                                var, interaction = group
                                                if var in self.feature_names and interaction in self.feature_names:
                                                    var_index = self.feature_names.index(var)
                                                    interaction_index = self.feature_names.index(interaction)
                                                    total_beta = self.W[var_index] + self.W[interaction_index]
                                                    if total_beta > 0:  # Violation of constraint
                                                        deficit = -total_beta
                                                        # Adjust equally between the base variable and interaction term
                                                        self.W[var_index] += deficit / 2
                                                        self.W[interaction_index] += deficit / 2

                                    # 4. Ensure sum of weights in 'sum_positive' groups is positive or zero
                                    if 'sum_positive' in self.constraints:
                                        for group in self.constraints['sum_positive']:
                                            if len(group) == 2:  # Handle exactly two variables (base and interaction)
                                                var, interaction = group
                                                if var in self.feature_names and interaction in self.feature_names:
                                                    var_index = self.feature_names.index(var)
                                                    interaction_index = self.feature_names.index(interaction)
                                                    total_beta = self.W[var_index] + self.W[interaction_index]
                                                    if total_beta < 0:  # Violation of constraint
                                                        deficit = -total_beta
                                                        # Adjust equally between the base variable and interaction term
                                                        self.W[var_index] += deficit / 2
                                                        self.W[interaction_index] += deficit / 2

                                def predict(self, X):
                                    """Predict the target variable."""
                                    return X.dot(self.W) + self.b



                            def generate_constraints_dynamic(feature_names,  PositiveConstraint_variables, NonPositiveConstraint_variables, NonConstraint_variables, interaction_suffix="_interaction_"):
                                """
                                Dynamically generate constraints based on provided media and other variables.

                                Args:
                                    feature_names (list): List of all feature names in the dataset.
                                    media_variables (list): List of media variables.
                                    PositiveConstraint_variables (list): List of other variables.
                                    interaction_suffix (str): Suffix used to identify interaction terms.

                                Returns:
                                    dict: Generated constraints for the regression model.
                                """
                                # Standardize feature names for matching
                                # standardized_media_vars = [f"{media}_transformed" for media in media_variables]
                                standardized_PositiveConstraint_variables = [f"scaled_{var}" for var in PositiveConstraint_variables]
                                standardized_NonPositiveConstraint_variables = [f"scaled_{var}" for var in NonPositiveConstraint_variables]
                                standardized_NonConstraint_variables = [f"scaled_{var}" for var in NonConstraint_variables]

                                # Identify media and other variables
                                # identified_media_vars = [var for var in feature_names if var in standardized_media_vars]
                                identified_PositiveConstraint_variables = [var for var in feature_names if var in standardized_PositiveConstraint_variables]
                                identified_NonPositiveConstraint_variables = [var for var in feature_names if var in standardized_NonPositiveConstraint_variables]
                                identified_NonConstraint_variables = [var for var in feature_names if var in standardized_NonConstraint_variables]

                                # Separate Price variable
                                # price_vars = [var for var in identified_other_vars if "Price" in var]
                                # non_price_other_vars = [var for var in identified_other_vars if var not in price_vars]

                                # Identify interaction terms
                                interaction_terms = [var for var in feature_names if interaction_suffix in var]

                                # Group interaction terms by base variables
                                interaction_map = {}
                                for interaction in interaction_terms:
                                    base_var = interaction.split(interaction_suffix)[1]  # Extract base variable from interaction term
                                    if base_var in interaction_map:
                                        interaction_map[base_var].append(interaction)
                                    else:
                                        interaction_map[base_var] = [interaction]

                                # Generate constraints
                                sum_positive_constraints = []
                                for var in  identified_PositiveConstraint_variables:
                                    if var in interaction_map:
                                        for interaction_var in interaction_map[var]:
                                            sum_positive_constraints.append([var, interaction_var])

                                sum_negative_constraints = []
                                for price_var in identified_NonPositiveConstraint_variables:
                                    if price_var in interaction_map:
                                        for interaction_var in interaction_map[price_var]:
                                            sum_negative_constraints.append([price_var, interaction_var])

                                # Final constraints dictionary
                                constraints = {
                                    'negative': identified_NonPositiveConstraint_variables,  # Ensure Price effects are non-positive
                                    'positive':  identified_PositiveConstraint_variables,  # Ensure media and non-Price variables have non-negative effects
                                    'sum_positive': sum_positive_constraints,  # Ensure media and other variables + their interactions are non-negative
                                    'sum_negative': sum_negative_constraints,  # Ensure Price + its interactions are non-positive
                                }

                                print(constraints)
                                return constraints


                            # Ridge Regression
                            def ridge_model(alpha=0.1):
                                return Ridge(alpha=alpha)

                            # Linear Regression
                            def linear_model():
                                return LinearRegression()

                            # Lasso Regression
                            def lasso_model(alpha=0.1):
                                return Lasso(alpha=alpha)

                            # Elastic Net
                            def elastic_net_model(alpha=0.1, l1_ratio=0.5):
                                return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)


                            def calculate_aic_bic(Y_true, Y_pred, n_features):
                                """Calculates AIC and BIC."""
                                residuals = Y_true - Y_pred
                                rss = np.sum(residuals**2)
                                n_samples = len(Y_true)
                                
                                # Avoid division by zero errors
                                if n_samples == 0 or rss == 0:
                                    return np.inf, np.inf  # Return large numbers to indicate error

                                # AIC and BIC formulae
                                aic = n_samples * np.log(rss / n_samples) + 2 * n_features
                                bic = n_samples * np.log(rss / n_samples) + np.log(n_samples) * n_features

                                return aic, bic

                            def apply_transformations_by_market(df, PositiveConstraint_variables, NonPositiveConstraint_variables, NonConstraint_variables, standardization_method):
                                """
                                Applies adstock, logistic transformations, and standardization to the given DataFrame
                                for each region separately, then appends the transformed data.

                                Parameters:
                                - df: The DataFrame containing the data.
                                - media_variables: List of media variable names to transform.
                                - PositiveConstraint_variables: List of other variable names to standardize.
                                - current_transformations: List of transformation parameters (growth_rate, carryover, midpoint).
                                - standardization_method: The method for standardization ('minmax', 'zscore', or 'none').
                                """
                                from sklearn.preprocessing import MinMaxScaler, StandardScaler

                                transformed_data_list = []  # To store transformed data for each region
                                unique_regions = df[column_to_filter].unique()  # Get unique regions

                                # Choose standardization method
                                if standardization_method == 'minmax':
                                    scaler_class = MinMaxScaler
                                    scaler_params = {'feature_range': (0, 1)}
                                elif standardization_method == 'zscore':
                                    scaler_class = StandardScaler
                                    scaler_params = {}
                                elif standardization_method == 'None':
                                    scaler_class = None  # No scaling
                                else:
                                    raise ValueError(f"Unsupported standardization method: {standardization_method}")

                                for region in unique_regions:
                                    # Filter data for the current region
                                    region_df = df[df[column_to_filter] == region].copy()

                                    # Standardize other variables
                                    for var in PositiveConstraint_variables:
                                        if scaler_class:
                                            scaler = scaler_class(**scaler_params)
                                            region_df[f"scaled_{var}"] = scaler.fit_transform(region_df[[var]])
                                        else:
                                            region_df[f"scaled_{var}"] = region_df[var]  # No scaling

                                    # # Transform media variables
                                    # for media_idx, media_var in enumerate(media_variables):
                                    #     gr, co, mp = current_transformations[media_idx]
                                    #     adstocked = adstock_function(region_df[media_var], co)
                                    #     standardized = (adstocked - np.mean(adstocked)) / np.std(adstocked)
                                    #     region_df[f"{media_var}_transformed"] = logistic_function(standardized, gr, mp)

                                    #     # Replace NaN values (if any) with 0
                                    #     region_df[f"{media_var}_transformed"] = np.nan_to_num(region_df[f"{media_var}_transformed"])

                                    #     if scaler_class:
                                    #         scaler = scaler_class(**scaler_params)
                                    #         region_df[f"{media_var}_transformed"] = scaler.fit_transform(
                                    #             region_df[[f"{media_var}_transformed"]]
                                    #         )

                                    # Keep non-scaled variables as is
                                    for var in NonPositiveConstraint_variables:
                                        if scaler_class:
                                            scaler = scaler_class(**scaler_params)
                                            region_df[f"scaled_{var}"] = scaler.fit_transform(region_df[[var]])
                                        else:
                                            region_df[f"scaled_{var}"] = region_df[var]  # No scaling
                                    # Keep non-scaled variables as is
                                    for var in NonConstraint_variables:
                                        if scaler_class:
                                            scaler = scaler_class(**scaler_params)
                                            region_df[f"scaled_{var}"] = scaler.fit_transform(region_df[[var]])
                                        else:
                                            region_df[f"scaled_{var}"] = region_df[var]  # No scaling

                                    # Append the transformed region data to the list
                                    transformed_data_list.append(region_df)
                                
                                # Concatenate all transformed data
                                transformed_df = pd.concat(transformed_data_list, axis=0).reset_index(drop=True)
                                return transformed_df


                            def calculate_region_specific_predictions_and_mape(
                                stacked_model, X, Y_actual, feature_names, regions, PositiveConstraint_variables, NonPositiveConstraint_variables,NonConstraint_variables
                            ):
                                """
                                Calculate region-wise predictions and MAPE for stacked models.
                                
                                Parameters:
                                - stacked_model: Fitted stacked regression model.
                                - X: DataFrame containing independent variables (features).
                                - Y_actual: Array or Series of actual dependent variable values.
                                - feature_names: List of feature names from the model.
                                - regions: List of unique regions.
                                - media_variables: List of media variables used in the model.
                                - PositiveConstraint_variables: List of other variables used in the model.
                                
                                Returns:
                                - region_mapes: Dictionary containing MAPE for each region.
                                - region_predictions: DataFrame of predictions for each region.
                                """
                                region_mapes = {}
                                region_predictions = {}

                                for region in regions:
                                    # Ensure the region dummy column exists
                                    region_column = f"Region_{region}"
                                    if region_column not in X.columns:
                                        raise ValueError(f"Expected region dummy column '{region_column}' not found in X.")

                                    # Filter the data for the specific region
                                    X_region = X[X[region_column] == 1]
                                    Y_region_actual = Y_actual[X_region.index]

                                    # Extract the intercept for the region
                                    base_intercept = stacked_model.intercept_ if hasattr(stacked_model, 'intercept_') else stacked_model.b
                                    region_intercept = 0  # Initialize to 0 by default

                                    # Check if the region-specific coefficient exists and add it
                                    if f"Region_{region}" in feature_names:
                                        region_index = feature_names.index(f"Region_{region}")
                                        region_intercept = stacked_model.coef_[region_index] if hasattr(stacked_model, 'coef_') else stacked_model.W[region_index]

                                    # Calculate the total intercept for the region
                                    intercept = base_intercept + region_intercept

                                    # Initialize adjusted weights for the region
                                    adjusted_weights = {}
                                    for var in [f"scaled_{var}" for var in PositiveConstraint_variables] +  [f"scaled_{var}" for var in NonPositiveConstraint_variables]+  [f"scaled_{var}" for var in NonConstraint_variables]:
                                        base_beta = 0  # Default to 0 if not found
                                        interaction_beta = 0  # Default to 0 if no interaction term

                                        # Check for base coefficient
                                        if var in feature_names:
                                            var_index = feature_names.index(var)
                                            base_beta = stacked_model.coef_[var_index] if hasattr(stacked_model, 'coef_') else stacked_model.W[var_index]

                                        # Check for interaction term
                                        interaction_term = f"{region}_interaction_{var}"
                                        if interaction_term in feature_names:
                                            interaction_index = feature_names.index(interaction_term)
                                            interaction_beta = (
                                                stacked_model.coef_[interaction_index] if hasattr(stacked_model, 'coef_') else stacked_model.W[interaction_index]
                                            )

                                        # Calculate the adjusted weight
                                        adjusted_weights[var] = base_beta + interaction_beta

                                    # # Debug output
                                    # print("Intercept:", intercept)
                                    # print("Adjusted Weights:", adjusted_weights)

                                    # print(intercept,adjusted_weights)        

                                    # Extract relevant columns and calculate predictions
                                    relevant_columns = list(adjusted_weights.keys())
                                    X_filtered = X_region[relevant_columns]
                                    adjusted_weights_array = np.array([adjusted_weights[col] for col in relevant_columns])
                                    Y_region_predicted = intercept + np.dot(X_filtered, adjusted_weights_array)

                                    # Calculate MAPE for the region
                                    mape = mean_absolute_percentage_error(Y_region_actual, Y_region_predicted)
                                    region_mapes[region] = mape
                                    region_predictions[region] = Y_region_predicted

                                return region_mapes, region_predictions



                            def recursive_modeling(
                                df, Region, Market, Brand, y_variable, PositiveConstraint_variables, NonPositiveConstraint_variables, NonConstraint_variables,
                                model_type, standardization_method, results=None, model_counter=1
                            ):
                                if results is None:
                                    results = []

                                is_stacked = len(Region) > 1  # Determine if it's a stacked model
                                model_type_label = f"Stacked_{model_type}" if is_stacked else model_type

                                # Build filter conditions
                                filter_conditions = []
                                
                                # Always filter by Region
                                if is_stacked:
                                    filter_conditions.append(df[column_to_filter].isin(Region))
                                else:
                                    filter_conditions.append(df[column_to_filter] == Region[0])
                                
                                # Only filter by Market if Market is provided and not empty
                                if Market:
                                    filter_conditions.append(df["Market"].isin(Market))
                                
                                # Only filter by Brand if Brand is provided and not empty
                                if Brand:
                                    filter_conditions.append(df["Brand"] == Brand)
                                
                                # Apply all filters
                                df_filtered = df.copy()
                                for condition in filter_conditions:
                                    df_filtered = df_filtered[condition]

                                if len(Region) == 2:
                                    # Simplify logic for two regions
                                    df_filtered = pd.concat([group for _, group in df_filtered.groupby(column_to_filter)], ignore_index=True)
                                else:
                                    # Default logic for other cases
                                    df_filtered = df_filtered.groupby(column_to_filter).apply(lambda x: x).reset_index(drop=True)

                                if df_filtered.empty:
                                    print(f"No data found for: {Region}, {Market if Market else 'All Markets'}, {Brand if Brand else 'All Brands'}.")
                                    return results

                                df_transformed = apply_transformations_by_market(
                                    df_filtered, PositiveConstraint_variables, NonPositiveConstraint_variables, NonConstraint_variables, standardization_method
                                )

                                # Group by 'Region' and calculate the mean for the specified columns
                                columns_to_average = [f"scaled_{var}" for var in PositiveConstraint_variables] + [f"scaled_{var}" for var in NonPositiveConstraint_variables] + [f"scaled_{var}" for var in NonConstraint_variables] 
                                df_transformed_Region_means = df_transformed.groupby(column_to_filter)[columns_to_average].mean()

                                # Create comma-separated mean strings for each variable
                                variable_means_dict = {
                                    f"{col}_mean": ', '.join([f"{region}:{round(df_transformed_Region_means.loc[region, col], 4)}"
                                                            for region in df_transformed_Region_means.index])
                                    for col in columns_to_average
                                }

                                # print("variable_means_dict :",variable_means_dict)

                                x_columns = [f"scaled_{var}" for var in PositiveConstraint_variables] + [f"scaled_{var}" for var in NonPositiveConstraint_variables] + [f"scaled_{var}" for var in NonConstraint_variables] 
                                unique_regions = df_transformed[column_to_filter].unique()

                                if is_stacked:
                                    for region in unique_regions:
                                        df_transformed[f"Region_{region}"] = (df_transformed[column_to_filter] == region).astype(int)
                                        for var in x_columns:
                                            if not var.startswith("Region_"):
                                                df_transformed[f"{region}_interaction_{var}"] = df_transformed[f"Region_{region}"] * df_transformed[var]

                                    x_columns += [f"Region_{region}" for region in unique_regions]
                                    x_columns += [f"{region}_interaction_{var}" for region in unique_regions for var in x_columns if not var.startswith("Region_")]

                                X = df_transformed[x_columns].fillna(0).reset_index(drop=True)
                                # print(X.columns)
                                Y = df_filtered[[y_variable]].reset_index(drop=True).loc[X.index]
                                # Y = pd.to_numeric(Y, errors='coerce')
                                # st.write(X.isna().sum())
                                # vif_df = calculate_vif(X)
                                feature_names = X.columns.tolist()
                                X_np, Y_np = X.values, Y.values.flatten()
                                # st.write(X.dtypes)
                                # st.write(Y.dtypes)

                                if model_type == "Ridge":
                                    model = ridge_model(alpha=0.1)
                                    model.fit(X_np, Y_np)
                                    Y_pred = model.predict(X_np)  # Define Y_pred before using it
                                    aic, bic = calculate_aic_bic(Y_np, Y_pred, X_np.shape[1])  # Correct
                                    p_values = None  # No p-values for Ridge regression


                                    # Calculate region-specific predictions and MAPE
                                    regions = [col.split('Region_')[1] for col in feature_names if col.startswith('Region_')]
                                    region_mapes, region_predictions = calculate_region_specific_predictions_and_mape(
                                        model, X, Y_np, feature_names, regions,  PositiveConstraint_variables,NonPositiveConstraint_variables,NonConstraint_variables
                                    )

                                elif model_type == "Linear Regression":
                                    model = linear_model()
                                    model.fit(X_np, Y_np)
                                    Y_pred = model.predict(X_np)  # Define Y_pred before using it

                                    # To get p-values for Linear Regression, use statsmodels OLS
                                    X_sm = add_constant(X_np)  # Add constant for intercept
                                    model_ols = sm.OLS(Y_np, X_sm)
                                    results_ols = model_ols.fit()
                                    p_values = results_ols.pvalues
                                    aic = results_ols.aic  # AIC from OLS
                                    bic = results_ols.bic  # BIC from OLS

                                    # Calculate region-specific predictions and MAPE
                                    regions = [col.split('Region_')[1] for col in feature_names if col.startswith('Region_')]
                                    region_mapes, region_predictions = calculate_region_specific_predictions_and_mape(
                                        model, X, Y_np, feature_names, regions, PositiveConstraint_variables,NonPositiveConstraint_variables,NonConstraint_variables
                                    )

                                elif model_type == "Generalized Constrained Ridge":
                                    constraints = generate_constraints_dynamic(
                                        feature_names=feature_names,
                                        # media_variables=media_variables,
                                        PositiveConstraint_variables=PositiveConstraint_variables,
                                        NonPositiveConstraint_variables=NonPositiveConstraint_variables,
                                        NonConstraint_variables=NonPositiveConstraint_variables,
                                        interaction_suffix="_interaction_"
                                    )
                                    model = GeneralizedConstraintRidgeRegression(l2_penalty=0.1, learning_rate=0.0001, iterations=1000000)
                                    model.fit(X_np, Y_np, feature_names, constraints)
                                    Y_pred = model.predict(X_np)  # Define Y_pred before using it
                                    aic, bic = calculate_aic_bic(Y_np, Y_pred, X_np.shape[1])  # Correct  # Calculate AIC and BIC for Generalized Constrained Ridge
                                    p_values = None  # No p-values for Generalized Constrained Ridge

                                    # Calculate region-specific predictions and MAPE
                                    regions = [col.split('Region_')[1] for col in feature_names if col.startswith('Region_')]
                                    region_mapes, region_predictions = calculate_region_specific_predictions_and_mape(
                                        model, X, Y_np, feature_names, regions, PositiveConstraint_variables,NonPositiveConstraint_variables,NonConstraint_variables
                                    )

                                elif model_type == "Generalized Constrained Lasso":
                                    constraints = generate_constraints_dynamic(
                                        feature_names=feature_names,
                                        # media_variables=media_variables,
                                        PositiveConstraint_variables=PositiveConstraint_variables,
                                        NonPositiveConstraint_variables=NonPositiveConstraint_variables,
                                        NonConstraint_variables=NonConstraint_variables,
                                        interaction_suffix="_interaction_"
                                    )
                                    model = GeneralizedConstraintLassoRegression(l1_penalty=0.01, learning_rate=0.0001, iterations=1000000)
                                    model.fit(X_np, Y_np, feature_names, constraints)
                                    Y_pred = model.predict(X_np)  # Define Y_pred before using it
                                    aic, bic = calculate_aic_bic(Y_np, Y_pred, X_np.shape[1])  # Correct  # Calculate AIC and BIC for Generalized Constrained Ridge
                                    p_values = None  # No p-values for Generalized Constrained Ridge

                                    # Calculate region-specific predictions and MAPE
                                    regions = [col.split('Region_')[1] for col in feature_names if col.startswith('Region_')]
                                    region_mapes, region_predictions = calculate_region_specific_predictions_and_mape(
                                        model, X, Y_np, feature_names, regions, PositiveConstraint_variables,NonPositiveConstraint_variables,NonConstraint_variables
                                    )

                                else:
                                    raise ValueError(f"Unsupported model type: {model_type}")

                                Y_pred = model.predict(X_np)
                                mape = mean_absolute_percentage_error(Y_np, Y_pred)
                                r2 = r2_score(Y_np, Y_pred)
                                adjusted_r2 = 1 - ((1 - r2) * (len(Y_np) - 1) / (len(Y_np) - len(feature_names) - 1)) if len(Y_np) - len(feature_names) - 1 != 0 else float('nan')
                                print(f"model_{model_counter}")

                                results_dict = {
                                    'Model_num': f"model_{model_counter}",
                                    'Model_type': model_type_label,
                                    'Brand': Brand,
                                    'Market': Market,
                                    'Region': Region,
                                    'Model_selected': 0,
                                    'MAPE': round(mape, 4),
                                    "Region_MAPEs": ','.join([f"{region}:{round(mape, 4)}" for region, mape in region_mapes.items()]),
                                    'R_squared': round(r2, 4),
                                    'Adjusted_R_squared': round(adjusted_r2, 4),
                                    'AIC': round(aic, 4),
                                    'Y': y_variable,
                                    'beta0': model.intercept_ if hasattr(model, 'intercept_') else model.b,
                                    **{f'beta_{feature_names[i]}': model.coef_[i] if hasattr(model, 'coef_') else model.W[i] for i in range(len(feature_names))},
                                    # **variable_means_dict,  # Add variable-specific mean strings
                                    'BIC': round(bic, 4),
                                    # 'Growth_rate': ','.join(map(str, [t[0] for t in current_transformations])),
                                    # 'Mid_point': ','.join(map(str, [t[2] for t in current_transformations])),
                                    # 'Carryover': ','.join(map(str, [t[1] for t in current_transformations])),
                                    "Standardization_method": standardization_method
                                }

                                # if p_values is not None:
                                #     for i, feature in enumerate(feature_names):
                                #         results_dict[f'p_value_{feature}'] = "Yes" if p_values[i + 1] <= 0.05 else "No"

                                if p_values is not None:
                                    for i, feature in enumerate(feature_names):
                                        results_dict[f'p_value_{feature}'] = p_values[i + 1] 

                                # for _, row in vif_df.iterrows():
                                #     results_dict[f'VIF_{row["Feature"]}'] = row["VIF"]

                                results.append(results_dict)
                                return results

                                # media_var = media_variables[current_media_idx]
                                # for gr in growth_rates:
                                #     for co in carryover_rates:
                                #         for mp in midpoints:
                                #             results = recursive_modeling(
                                #                 df, Region, Market, Brand, y_variable, media_variables, PositiveConstraint_variables,NonPositiveConstraint_variables,
                                #                 growth_rates, carryover_rates, midpoints,
                                #                 model_type, standardization_method,
                                #                 current_media_idx + 1, current_transformations + [(gr, co, mp)], results, model_counter
                                #             )
                                #             model_counter += 1
                                            

                                # return results

                            def generalized_modeling_recursive(
                                df, Region, Market, Brand, y_variables, PositiveConstraint_variables, 
                                NonPositiveConstraint_variables, NonConstraint_variables,
                                model_types, standardization_method
                            ):
                                """
                                Generalized function to run models for multiple brands step-by-step.
                                Includes both stacked models and individual models for each region.
                                
                                Parameters:
                                - df: DataFrame containing the data.
                                - Region: List of regions to include.
                                - Market: List of markets to include (optional - if empty, all markets are considered).
                                - Brand: List of brands to include (optional - if empty, all brands are considered).
                                - y_variables: List of dependent variables to model.
                                - PositiveConstraint_variables: List of variables with positive constraints.
                                - NonPositiveConstraint_variables: List of variables with non-positive constraints.
                                - NonConstraint_variables: List of variables with no constraints.
                                - model_types: List of model types to fit (e.g., Ridge, Linear Regression).
                                - standardization_method: Standardization method to use ('minmax', 'zscore', or 'none').
                                
                                Returns:
                                - A DataFrame containing results for all models (stacked and region-specific).
                                """
                                results = []  # To store results for all models
                                model_counter = 1

                                # Handle case when Brand is not provided or empty
                                brands_to_process = Brand if Brand else [None]
                                
                                for brand in brands_to_process:
                                    brand_label = brand if brand else "All Brands"
                                    print(f"Processing brand: {brand_label}")

                                    for model_type in model_types:
                                        print(f"  Running {model_type} models...")

                                        for y_variable in y_variables:
                                            print(f"    Modeling for dependent variable: {y_variable}")

                                            # Stacked Model (all regions together with dummies and interactions)
                                            results = recursive_modeling(
                                                df, Region, Market, brand, y_variable,  
                                                PositiveConstraint_variables, NonPositiveConstraint_variables, NonConstraint_variables,
                                                model_type, standardization_method, 
                                                results=results, model_counter=model_counter
                                            )
                                            model_counter += 1

                                            # Region-Specific Models (one per region)
                                            for region in Region:
                                                print(f"    Running region-specific model for Region: {region}")
                                                
                                                results = recursive_modeling(
                                                    df, [region], Market, brand, y_variable, 
                                                    PositiveConstraint_variables, NonPositiveConstraint_variables, NonConstraint_variables,
                                                    model_type, standardization_method, 
                                                    results=results, model_counter=model_counter
                                                )
                                                model_counter += 1
                                                
                                # Combine all results into a single DataFrame
                                return pd.DataFrame(results)
                                                        




                            # Page title
                            # st.subheader("Modeling Options")

                            import plotly.express as px



                            with st.expander("BUILD MODEL:"):


                                if selected_columns:

                                    df_fea = df_fea.sort_values(by=date_col)


                                    # Group data by selected columns
                                    grouped_data = df_fea.groupby(selected_columns)
                                    
                                    # Get the list of groups
                                    group_names = list(grouped_data.groups.keys())
                                    
                                    if 'frequency' in st.session_state:
                                        frequency = st.session_state.frequency





                                col1, col2, col3 = st.columns(3)
                                # Extract unique values for dynamic dropdowns
                                available_regions = df[column_to_filter].unique().tolist()
                                # available_markets = df["Market"].unique().tolist()
                                # available_brands = df["Brand"].unique().tolist()

                                available_markets = df["Market"].unique().tolist() if "Market" in df.columns else []
                                available_brands = df["Brand"].unique().tolist() if "Brand" in df.columns else []

                            
                            
                                with col1:
                                    
                                    # # selected_group = st.selectbox(f"Select the group", group_names, key='group_for_featurebased')
                                    # selected_group = st.selectbox(
                                    #     f"Select the group", 
                                    #     group_names, 
                                    #     index=group_names.index(st.session_state.get('selected_group', group_names[0])) if st.session_state.get('selected_group') in group_names else 0,
                                    #     key='group_for_featurebased'
                                    # )

                            
                                    # excluded_from_regions = {"Prophet"}

                                    # regions = [selected_group] if selected_group in available_regions and selected_group not in excluded_from_regions else []



                                    # Pre-select previously selected group(s) or default to the first one
                                    default_selection = (
                                        st.session_state.get('group_for_featurebased') 
                                        if isinstance(st.session_state.get('group_for_featurebased'), list) 
                                        else [st.session_state.get('group_for_featurebased')] 
                                        if st.session_state.get('group_for_featurebased') in group_names 
                                        else [group_names[0]]
                                    )

                                    # Multiselect for groups
                                    selected_group = st.multiselect(
                                        "Select the group(s)",
                                        group_names,
                                        default=default_selection,
                                        key='group_for_featurebased'
                                    )

                                    excluded_from_regions = {"Prophet"}

                                    # Only include groups that are in available_regions and not excluded
                                    regions = [group for group in selected_group if group in available_regions and group not in excluded_from_regions]
                                    


                                with col2:
                                    # Market selection
                                    markets = st.multiselect("Select Markets", available_markets, default=available_markets)

                                with col3:
                                    # Brand selection
                                    brands = st.multiselect("Select Brands", available_brands, default=available_brands)

                            

                                import streamlit as st
                                import pandas as pd

                                def clean_and_match(stored_list, df_columns):
                                    if isinstance(stored_list, str):
                                        items = [item.strip() for item in stored_list.split(",") if item.strip()]
                                    elif isinstance(stored_list, (list, np.ndarray, pd.Series)):
                                        items = [str(item).strip() for item in stored_list]
                                    else:
                                        items = []

                                    # Filter out invalid columns
                                    valid_items = [item for item in items if item in df_columns]
                                    return valid_items

                                # Sample DataFrame (This should be replaced with actual df)
                                # df = pd.DataFrame(columns=["D1", "Price", "A&P_Amount_Spent", "Region_Brand_seasonality"])

                                # # Initialize session state for storing selected variables
                                # if "stored_variables" not in st.session_state:
                                #     st.session_state.stored_variables = pd.DataFrame(columns=["Model_ID", "Positive Variables", "NonPositive Variables","Non Constraint Variables"])

                                col5, col6,col7 = st.columns(3)

                                # with col4:
                                #     # Dependent variable selection
                                #     # y_variables = [st.selectbox("Select Dependent Variable (y)", target_col)]  #, default=["Filtered Volume Index"]
                                y_variables=[target_col]
                                    


                                only_prophet_selected = len(models) == 1 and "Prophet" in models

                                if not only_prophet_selected and any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression", "Prophet"]):



                                    

                                
                                    with col5:
                                        
                                        PositiveConstraint_variables = st.multiselect(
                                            "Select Positive Constraint Variables",
                                            [col for col in df.columns if col not in ["Year", "Month","Week",'Fiscal Year',date_col]+y_variables+basis_columns],
                                            # default=valid_positive
                                        )

                                        
                                    with col6:
                                    
                                        
                                        NonPositiveConstraint_variables = st.multiselect(
                                            "Select NonPositive Constraint Variables",
                                            [col for col in df.columns if col not in ["Year", "Month","Week",'Fiscal Year',date_col]+y_variables+basis_columns],
                                            # default=valid_non_positive
                                        )

                                    with col7:
                                        # # Non-Constraint Variables
                                        # valid_non_constraint = [v for v in retrieved_non_constraint if v in df.columns]
                                        NonConstraint_variables = st.multiselect(
                                            "Select Non Constraint Variables",
                                            [col for col in df.columns if col not in ["Year", "Month","Week",'Fiscal Year',date_col]+y_variables+basis_columns],
                                            # default=valid_non_constraint
                                        )

                                    valid_features = PositiveConstraint_variables+NonPositiveConstraint_variables+NonConstraint_variables


                                    # st.session_state.features = valid_features

                                elif only_prophet_selected:

                                    st.markdown('<hr class="thin">', unsafe_allow_html=True)


                                    # Show the simplified feature selection for Prophet only
                                    all_features = [col for col in df_fea.columns if col not in [target_col, date_col] + basis_columns]
                                    
                                    exclude_cols = [target_col, date_col] + basis_columns + ['Year', 'Month', 'Fiscal Year']
                                    all_features = [col for col in df_fea.columns if col not in exclude_cols]
                                    
                                    # Checkbox to select/deselect all
                                    select_all = st.checkbox("Select All Features", value=True)
                                    
                                    # Multiselect with dynamic default based on the checkbox
                                    if select_all:
                                        feature_cols = st.multiselect(
                                            "Choose features to include:", 
                                            all_features, 
                                            default=st.session_state.get('valid_features', all_features)
                                        )
                                    else:
                                        feature_cols = st.multiselect(
                                            "Choose features to include:", 
                                            all_features, 
                                            default=st.session_state.get('valid_features', [])
                                        )
                                    
                                    valid_features = feature_cols

                                    
                                    
                                st.session_state.features = valid_features



                                col8,col9,col10,col11=st.columns(4)

                                with col8:

                                    # forecast_horizon = st.number_input(f"Forecast Horizon ({frequency})", min_value=1, max_value=120, value=12, key='horizen_for_featurebased')
                                    forecast_horizon = st.number_input(
                                        f"Forecast Horizon ({frequency})", 
                                        min_value=1, 
                                        max_value=120, 
                                        value=st.session_state.get('forecast_horizon', 12),  # Default to 12 if nothing loaded
                                        key='horizen_for_featurebased'
                                    )




                                with col9:
                                    fiscal_start_month = st.selectbox(
                                        "Select Fiscal Year Start Month", 
                                        range(1, 13), 
                                        index=st.session_state.get('fiscal_start_month', 1) - 1,  # Convert month number to index
                                        key='fiscal_for_featurebased'
                                        # ,disabled=True
                                    )
                                    

                                with col10:

                                    frequency_options = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
                                    if frequency in frequency_options:
                                        default_frequency_key = frequency
                                    else:
                                        default_frequency_key = "Monthly"
                                    selected_frequency = st.selectbox("Select Data Frequency", list(frequency_options.keys()), 
                                                                    index=list(frequency_options.keys()).index(default_frequency_key), key='frequecy_for_featurebased',disabled=True)
                                    
                                    frequency_options = frequency_options[selected_frequency]
                                
                                with col11:
                                    standardization_method = st.selectbox("Select Standardization Method", [ 'zscore','minmax','None'], index=0)


                                st.session_state.frequency_options=frequency_options

                                st.session_state.fiscal_start_month=fiscal_start_month











                                


                                model_type = [m for m in models if m != 'Prophet']



                                # group_data = grouped_data.get_group(selected_group).set_index(date_col)
                                
                                # Add a checkbox to choose whether to use PCA or not
                                df_fea = df_fea.set_index(date_col)
                                
                                
                                # st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                # # use_pca = st.checkbox("Use PCA", value=True)
                                # use_pca = st.checkbox("**PERFORM PCA**")

                                



                                import streamlit as st

                                # Function to parse user input
                                def parse_input(input_text):
                                    # Convert list to string if needed
                                    if isinstance(input_text, list):
                                        input_text = ",".join(map(str, input_text))  # Join list elements into a string
                                    try:
                                        return [float(x.strip()) for x in input_text.split(",") if x.strip()]
                                    except ValueError:
                                        return []
                                    
                                

                            

                                st.markdown(
                                    """ 
                                    <div style="height: 3px; background-color: black; margin: 20px 0;"></div>
                                    """, 
                                    unsafe_allow_html=True
                                    )  
                                
                                # Initialize session state
                                if "model_status" not in st.session_state:
                                    st.session_state.model_status = "idle"
                                if "model_results" not in st.session_state:
                                    st.session_state.model_results = None  # To store model results

                                # Run modeling button
                                if st.button("Run Model"):
                                    st.session_state.model_status = "running"
                                    st.rerun()  # Refresh the UI


                            if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]):

                                # with st.expander("Review Model:"):

                                # if valid_features:


                                
                                

                                    # Display message based on model status
                                    if st.session_state.model_status == "running":
                                        st.write("Running model with the selected options...")


                                        if PositiveConstraint_variables or NonPositiveConstraint_variables or NonConstraint_variables:
                                            
                                            # Call the modeling function with selected parameters
                                            results_df = generalized_modeling_recursive(
                                                df=df,
                                                Region=regions,
                                                Market=markets,
                                                Brand=brands,
                                                y_variables=y_variables,
                                                # media_variables=media_variables,
                                                PositiveConstraint_variables=PositiveConstraint_variables,
                                                NonPositiveConstraint_variables = NonPositiveConstraint_variables,
                                                NonConstraint_variables = NonConstraint_variables,
                                                # growth_rates=growth_rates,
                                                # carryover_rates=carryover_rates,
                                                # midpoints=midpoints,
                                                model_types=model_type,
                                                standardization_method=standardization_method,
                                                # apply_same_params=apply_same_params
                                            )

                                            # Store results in session state and update status
                                            st.session_state.model_results = results_df
                                            st.session_state.model_status = "completed"
                                            st.rerun()
                                            # st.session_state["results_df"] = results_df
                                            # results_df


                                        else:
                                            st.error("No Variables Selected.")

                                    elif st.session_state.model_status == "completed":
                                        # st.write("Modeling completed!")

                                    
                                        
                                        # st.write("Modeling completed!")
                                        # results_df = results_df.drop_duplicates()
                                        # st.dataframe(results_df)  # Display results
                                        # Display results if available
                                        if st.session_state.model_results is not None:
                                            results_df = st.session_state.model_results  # Retrieve stored results
                                            # st.dataframe(st.session_state.model_results)  # Show results in UI

                                        # Store results_df in session_state
                                        st.session_state["results_df"] = results_df

                                        
                                        # st.dataframe(results_df)  # Display results
                                        # else:
                                        #     st.warning("Please upload a CSV file to proceed.")
                            

                                        # if st.button("Final Result df"):
                                        if "results_df" not in st.session_state:
                                            st.warning("Please run the model first by clicking 'Run Model'.")
                                        else:

                                            column_to_filter = selected_columns[0]
                                            # st.write(column_to_filter)


                                            results_df = st.session_state["results_df"]  # Retrieve stored results
                                            
                                            # Identify columns that contain lists
                                            for col in results_df.columns:
                                                if results_df[col].apply(lambda x: isinstance(x, list)).any():
                                                    results_df[col] = results_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

                                            # Now safely drop duplicates
                                            subset_cols = [col for col in results_df.columns if col != 'Model_num']
                                            results_df = results_df.drop_duplicates(subset=subset_cols, keep='first')
                                            

                                            

                                            
                                            
                                            results_df['Region'] = results_df['Region'].astype(str).apply(
                                                lambda x: re.findall(r"\['(.*?)'\]", x)[0] if re.findall(r"\['(.*?)'\]", x) else x
                                            )

                                            results_df['Region'] = results_df['Region'].astype(str).str.replace(r"[(),'']", "", regex=True)

                                            results_df['Region'].unique()

                                            expanded_results = []

                                            # Loop through each model in the results dataframe
                                            for _, model_row in results_df.iterrows():
                                                # Extract model type and Region
                                                model_type = model_row['Model_type']
                                                original_region = model_row.get(column_to_filter, None)  # Get the original region from the results dataframe

                                                # Determine if the model is stacked
                                                is_stacked = model_type.startswith("Stacked")

                                                # Extract feature names and parameters dynamically
                                                feature_names = [
                                                    col.split('beta_')[1] for col in model_row.keys() if col.startswith('beta_')
                                                ]
                                                model = {
                                                    "b": model_row['beta0'],
                                                    "W": np.array([model_row[f'beta_{col}'] for col in feature_names]),
                                                }

                                                # Handle stacked models
                                                if is_stacked:
                                                    # Extract regions dynamically
                                                    Region = [col.split('Region_')[1] for col in feature_names if col.startswith('Region_')]

                                                    # Parse Region_MAPEs into a dictionary
                                                    region_mapes = {
                                                        region_mape.split(':')[0]: float(region_mape.split(':')[1])
                                                        for region_mape in model_row['Region_MAPEs'].split(',')
                                                    }

                                                    # Loop through each region
                                                    for region in Region:
                                                        # Extract base intercept and region-specific intercept
                                                        base_intercept = model["b"]
                                                        region_intercept = (
                                                            model["W"][feature_names.index(f"Region_{region}")] if f"Region_{region}" in feature_names else 0
                                                        )
                                                        adjusted_intercept = base_intercept + region_intercept

                                                        # Calculate adjusted betas
                                                        adjusted_betas = {}
                                                        for var in feature_names:
                                                            # Skip interaction terms (variables starting with {region}_interaction_)
                                                            if not var.startswith("Region_") and not any(var.startswith(f"{region}_interaction") for region in Region):
                                                                # Base coefficient for the variable
                                                                base_beta = model["W"][feature_names.index(var)]

                                                                # Interaction term adjustment (if exists)
                                                                interaction_term = f"{region}_interaction_{var}"
                                                                interaction_beta = (
                                                                    model["W"][feature_names.index(interaction_term)]
                                                                    if interaction_term in feature_names else 0
                                                                )

                                                                # Store adjusted beta
                                                                adjusted_betas[var] = base_beta + interaction_beta

                                                        # Prepare a dictionary for the region-specific row
                                                        region_row = {
                                                            'Model_num': model_row['Model_num'],
                                                            'Model_type': model_type,
                                                            'Market': model_row['Market'],
                                                            'Brand': model_row['Brand'],
                                                            column_to_filter: region,
                                                            'Model_selected': model_row['Model_selected'],
                                                            'MAPE': model_row['MAPE'],
                                                            'Region_MAPEs': region_mapes.get(region, None),  # Assign region-specific MAPE
                                                            'R_squared': model_row['R_squared'],
                                                            'Adjusted_R_squared': model_row['Adjusted_R_squared'],
                                                            'AIC': model_row['AIC'],
                                                            'BIC': model_row['BIC'],
                                                            'Y': model_row['Y'],
                                                            'beta0': adjusted_intercept,
                                                            **{
                                                                f'beta_{var}': adjusted_betas[var]
                                                                for var in adjusted_betas.keys()
                                                            },  # Add region-specific means for each variable
                                                            # 'Growth_rate': model_row['Growth_rate'],
                                                            # 'Mid_point': model_row['Mid_point'],
                                                            # 'Carryover': model_row['Carryover'],
                                                            'Standardization_method': model_row['Standardization_method'],
                                                        }

                                                        # Append the region-specific row to the expanded results
                                                        expanded_results.append(region_row)
                                                else:
                                                    # For non-stacked models, retain the original row
                                                    region_row = model_row.to_dict()  # Convert the row to a dictionary
                                                
                                                    
                                                    expanded_results.append(region_row)

                                            # Replace the original results with the expanded results
                                            # Add a unique identifier to each row
                                            expanded_results_df = pd.DataFrame(expanded_results)  # Convert to DataFrame for further use
                                            # Replace None values in 'Region_MAPEs' with the corresponding 'MAPE' values
                                            expanded_results_df['Region_MAPEs'] = expanded_results_df.apply(
                                                lambda row: row['MAPE'] if pd.isna(row['Region_MAPEs']) or row['Region_MAPEs'] in ["None", "nan", ""] else row['Region_MAPEs'],
                                                axis=1
                                            )
                                            import re

                                            # Extract the region name only if it's in list format, otherwise keep the original value
                                            expanded_results_df['Region'] = expanded_results_df['Region'].astype(str).apply(
                                                lambda x: re.findall(r"\['(.*?)'\]", x)[0] if re.findall(r"\['(.*?)'\]", x) else x
                                            )


                                            expanded_results_df['Unique_ID'] = range(1, len(expanded_results_df) + 1)  # Assign unique IDs starting from 1
                                            # expanded_results_df["Approach"] = "Bottom Up"
                                            # st.write(expanded_results_df.columns[expanded_results_df.columns.duplicated()])


                                            # if "Approach" in expanded_results_df.columns:
                                            #     expanded_results_df = expanded_results_df.drop(columns=["Approach"])

                                            # expanded_results_df["Approach"] = "Bottom Up"

                                            # Reorder columns to make Unique_ID the first column
                                            columns = ['Unique_ID'] +  [col for col in expanded_results_df.columns if col != 'Unique_ID']
                                            expanded_results_df = expanded_results_df[columns]

                                            # Drop columns with all NaN values
                                            expanded_results_df = expanded_results_df.dropna(axis=1, how='all')

                                            # Store the expanded results in session_state
                                            st.session_state["expanded_results_df"] = expanded_results_df

                                        # # st.write("Modeling completed!")
                                        # st.dataframe(expanded_results_df)  # Display results

                                        # st.write(expanded_results_df.columns[expanded_results_df.columns.duplicated()])

                                        # if st.button("Get Elasticity"):
                                        # if "expanded_results_df" not in st.session_state:
                                        #     st.warning("Please run the model first by clicking 'Run Model'.")
                                        # else:
                                            expanded_results_df = st.session_state["expanded_results_df"]  # Retrieve stored results
                                            # expanded_results_df

                                            # # st.write("Modeling completed!")


                                            # st.dataframe(expanded_results_df[["Model_type",'Region',"MAPE","R_squared","Adjusted_R_squared"]])  # Display results




                                            expanded_results_df['Region'].unique()

                                            if 'ml_review' not in st.session_state:

                                                st.session_state.ml_review = None




                                    # df_transformed = apply_transformations_by_market(
                                    #     df, PositiveConstraint_variables,NonPositiveConstraint_variables, standardization_method
                                    # )

                                            # Group by 'Region' and calculate the mean for the specified columns
                                            columns_to_average = [f"{var}" for var in PositiveConstraint_variables] + [f"{var}" for var in NonPositiveConstraint_variables] + [f"{var}" for var in NonConstraint_variables]
                                            # if not df.empty: 
                                            # # print("Variables :", columns_to_average)
                                            #     df_transformed_Region_means = df.groupby(column_to_filter)[columns_to_average].agg(['mean', 'std'])

                                            if not columns_to_average:
                                                st.error("No Variables Selected.")
                                            elif not df.empty:
                                                df_transformed_Region_means = df.groupby(column_to_filter)[columns_to_average].agg(['mean', 'std'])


                                    

                                            

                                                # Flatten MultiIndex columns (Mean & Std in separate columns)
                                                df_transformed_Region_means.columns = [f"{col[0]}_{col[1]}" for col in df_transformed_Region_means.columns]

                                                # Reset index to make 'Region' a column instead of index
                                                df_transformed_Region_means.reset_index(inplace=True)

                                                df_transformed_Region_means=df_transformed_Region_means.rename(columns={column_to_filter:'Region'})

                                                # df_transformed_Region_means = df_transformed_Region_means[df_transformed_Region_means["Region"]!="Category"]

                                                # # df_transformed_Region_means
                                                # st.markdown(
                                                # """ 
                                                # <div style="height: 1px; background-color: black; margin: 10px 0;"></div>
                                                # """, 
                                                # unsafe_allow_html=True
                                                # )  

                                                # st.write("##### Final Model Result")

                                                # # if not expanded_results_df.empty: 
                                                model_type = expanded_results_df["Model_type"].unique()
                                                # expanded_results_df
                                                # expanded_results_df = expanded_results_df[expanded_results_df["Model_type"].isin(model_type)]

                                                # st.write(expanded_results_df)
                                                # st.write(df_transformed_Region_means)
                                                


                                                final_result_df = pd.merge(expanded_results_df,df_transformed_Region_means,on='Region',how="left")
                                                # final_result_df

                                                # Ensure df_final is not empty before proceeding
                                                # if not final_result_df.empty:
                                                # st.write(final_result_df["Model_type"].unique())
                                                # if final_result_df["Model_type"].unique() == "Linear Regression":
                                                if "Linear Regression" in final_result_df["Model_type"].unique():

                                                    if not final_result_df.empty:
                                                    
                                                        beta_columns = [col for col in final_result_df.columns if col.startswith("beta_scaled_")]
                                                        mean_columns = [col for col in final_result_df.columns if col.endswith("_mean")]
                                                        std_columns = [col for col in final_result_df.columns if col.endswith("_std")]
                                                        p_value_columns = [col for col in final_result_df.columns if col.startswith("p_value_scaled_")]

                                                        # Extract beta0 separately
                                                        beta0_df = final_result_df[['Region', 'beta0',"Y","Model_type"]]

                                                        # Melt the DataFrame to get a long format
                                                        df_long = final_result_df.melt(id_vars=['Region'], 
                                                                                        value_vars=beta_columns + mean_columns + std_columns + p_value_columns,
                                                                                        var_name='Variable_Type', 
                                                                                        value_name='Value')

                                                        # Extract variable names from column names
                                                        df_long['Variable'] = df_long['Variable_Type'].str.replace('beta_scaled_|_mean|_std|p_value_scaled_', '', regex=True)

                                                        # Create a new column indicating whether it's a beta, mean, or std value
                                                        df_long['Metric'] = df_long['Variable_Type'].apply(lambda x: 'Beta' if 'beta_scaled_' in x 
                                                                                                        else ('Mean' if '_mean' in x 
                                                                                                        else 'p_value' if 'p_value_scaled_' in x
                                                                                                        else 'Std')
                                                                                                        )

                                                        # Pivot to get the desired format
                                                        df_final = df_long.pivot_table(index=['Region', 'Variable'], 
                                                                                    columns='Metric', 
                                                                                    values='Value', 
                                                                                    aggfunc='first').reset_index()

                                                        # Rename columns for clarity
                                                        df_final.columns.name = None  # Remove multi-index name
                                                        df_final.rename(columns={'Beta': 'Scaled_Beta', 'Mean': 'Mean_Value', 'Std': 'Std_Value'}, inplace=True)

                                                        # Merge beta0 as a separate column
                                                        df_final = df_final.merge(beta0_df, on='Region', how='left')

                                                        # Rename beta0 column for clarity
                                                        df_final.rename(columns={'beta0': 'Beta0_Scaled'}, inplace=True)

                                                        # print(df_final)

                                                        # Display the final transformed DataFrame
                                                        df_final["Non_Scaled_Beta"] = df_final["Scaled_Beta"]/df_final["Std_Value"]

                                                        vol_mean = df.groupby(column_to_filter)['Volume'].mean()
                                                        # vol_mean

                                                        # Map the corresponding region's mean volume to df_final and store it
                                                        df_final["Region_Vol_Mean"] = df_final['Region'].map(vol_mean)

                                                        

                                                        # Calculate Elasticity
                                                        df_final["Elasticity"] = df_final["Non_Scaled_Beta"] * (df_final["Mean_Value"] / df_final["Region_Vol_Mean"])
                                                        df_final=df_final.rename(columns={'Region':"Segment"})

                                                        df_final1 = df_final.rename(columns={'Region':"Segment"})

                                                        df_final1['Target_Column']=target_col

                                                        # df_final1['Target_Mean']=vol_mean



                                                        # Select relevant columns
                                                        # if "Linear Regression" in final_result_df["Model_type"].unique():
                                                        #     df_final1 = df_final1[["Model_type", "Segment", "Y", "Variable", "Elasticity", "P_Value"]]
                                                        # else:
                                                        #     df_final1 = df_final1[["Model_type", "Segment", "Y", "Variable", "Elasticity"]]

                                                        df_final1 = df_final1[["Model_type","Segment","Variable","Elasticity","p_value","Beta0_Scaled","Region_Vol_Mean","Target_Column"]]

                                                        # st.write(df_final1)
                                                        
                                                        st.session_state.ml_review=df_final1

                                                        # st.write(df_final1)
                                                else:

                                                    if not final_result_df.empty:
                                                                
                                                        beta_columns = [col for col in final_result_df.columns if col.startswith("beta_scaled_")]
                                                        mean_columns = [col for col in final_result_df.columns if col.endswith("_mean")]
                                                        std_columns = [col for col in final_result_df.columns if col.endswith("_std")]
                                                        # p_value_columns = [col for col in final_result_df.columns if col.startswith("p_value_scaled_")]

                                                        # Extract beta0 separately
                                                        beta0_df = final_result_df[['Region', 'beta0',"Y","Model_type"]]

                                                        # Melt the DataFrame to get a long format
                                                        df_long = final_result_df.melt(id_vars=['Region'], 
                                                                                        value_vars=beta_columns + mean_columns + std_columns,
                                                                                        var_name='Variable_Type', 
                                                                                        value_name='Value')

                                                        # Extract variable names from column names
                                                        df_long['Variable'] = df_long['Variable_Type'].str.replace('beta_scaled_|_mean|_std|', '', regex=True)

                                                        # Create a new column indicating whether it's a beta, mean, or std value
                                                        df_long['Metric'] = df_long['Variable_Type'].apply(lambda x: 'Beta' if 'beta_scaled_' in x 
                                                                                                        else ('Mean' if '_mean' in x 
                                                                                                        else 'Std')
                                                                                                        )

                                                        # Pivot to get the desired format
                                                        df_final = df_long.pivot_table(index=['Region', 'Variable'], 
                                                                                    columns='Metric', 
                                                                                    values='Value', 
                                                                                    aggfunc='first').reset_index()

                                                        # Rename columns for clarity
                                                        df_final.columns.name = None  # Remove multi-index name
                                                        df_final.rename(columns={'Beta': 'Scaled_Beta', 'Mean': 'Mean_Value', 'Std': 'Std_Value'}, inplace=True)

                                                        # Merge beta0 as a separate column
                                                        df_final = df_final.merge(beta0_df, on='Region', how='left')

                                                        # Rename beta0 column for clarity
                                                        df_final.rename(columns={'beta0': 'Beta0_Scaled'}, inplace=True)

                                                        # print(df_final)

                                                        # Display the final transformed DataFrame
                                                        df_final["Non_Scaled_Beta"] = df_final["Scaled_Beta"]/df_final["Std_Value"]

                                                        vol_mean = df.groupby(column_to_filter)["Volume"].mean()
                                                        # vol_mean

                                                        # Map the corresponding region's mean volume to df_final and store it
                                                        df_final["Region_Vol_Mean"] = df_final['Region'].map(vol_mean)

                                                        # Calculate Elasticity
                                                        df_final["Elasticity"] = df_final["Non_Scaled_Beta"] * (df_final["Mean_Value"] / df_final["Region_Vol_Mean"])
                                                        df_final=df_final.rename(columns={'Region':"Segment"})

                                                        # st.write(df_final)

                                                        df_final1 = df_final.rename(columns={'Region':"Segment"})

                                                        df_final1 = df_final1[["Model_type","Segment","Variable","Elasticity","Beta0_Scaled"]]

                                                        st.session_state.ml_review=df_final1


                                                


                                    ##################################################################################################################################

                                        
                                        def apply_transformations_with_contributions(df, region_weight_df):
                                            """
                                            Applies adstock, logistic transformations, and standardization to the original DataFrame,
                                            and calculates contributions of each variable by multiplying betas with scaled and transformed variables.

                                            Parameters:
                                            - df: The DataFrame containing the original data (media and other variables).
                                            - region_weight_df: DataFrame containing region weights and transformation parameters.

                                            Returns:
                                            - DataFrame: Transformed data with contribution columns.
                                            """
                                            transformed_data_list = []  # To store transformed data for each region
                                            unique_regions = df[column_to_filter].unique()

                                            # region_weight_df.columns

                                            # Extract media variables dynamically
                                            media_variables = [
                                                col.replace('_adjusted', '')
                                                for col in region_weight_df.columns
                                                if col.endswith('_adjusted') 
                                            ]

                                            # Extract other variables dynamically
                                            other_variables = [
                                                col.replace('beta_scaled_', '')
                                                for col in region_weight_df.columns
                                                if col.startswith('beta_scaled_')
                                            ]
                                            # other_variables

                                            # Include beta0 in the calculations
                                            if 'beta0' in region_weight_df.columns:
                                                include_beta0 = True
                                            else:
                                                include_beta0 = False

                                            # # Add additional media variables
                                            # additional_media_vars = ['TV_Total_Unique_Reach', 'Digital_Total_Unique_Reach']
                                            # media_variables += additional_media_vars

                                            # Filter data by Region and Brand
                                            filtered_data = {
                                            region: df[df[column_to_filter] == region].copy() for region in unique_regions
                                            }
                                            
                                            for region in unique_regions:
                                                brand = region_weight_df.loc[region_weight_df[column_to_filter] == region].iloc[0]
                                                region_df = filtered_data.get((region), pd.DataFrame())

                                                if region_df.empty:
                                                    print(f"Warning: No data found for Region={region}, Brand={brand}. Skipping.")
                                                    continue

                                                region_row = region_weight_df[region_weight_df[column_to_filter] == region].iloc[0]

                                                # Add beta0 contribution if available
                                                if include_beta0:
                                                    beta0_value = float(region_row['beta0'])
                                                    region_df['beta0'] = beta0_value


                                                # Ensure column exists before accessing
                                                if "Standardization_method" not in region_weight_df.columns:
                                                    raise KeyError("The column 'Standardization_method' is missing in region_weight_df.")

                                                # Access the standardization method for the current region
                                                standardization_method = region_row["Standardization_method"]

                                                if standardization_method == 'minmax':
                                                    scaler_class = MinMaxScaler
                                                    scaler_params = {'feature_range': (0, 1)}
                                                elif standardization_method == 'zscore':
                                                    scaler_class = StandardScaler
                                                    scaler_params = {}
                                                elif standardization_method == 'none':
                                                    scaler_class = None
                                                else:
                                                    raise ValueError(f"Unsupported standardization method: {standardization_method}")

                                                # Standardize other variables
                                                for var in other_variables:
                                                    if var in region_df.columns:
                                                        if scaler_class:
                                                            scaler = scaler_class(**scaler_params)
                                                            region_df[f"scaled_{var}"] = scaler.fit_transform(region_df[[var]])
                                                        else:
                                                            region_df[f"scaled_{var}"] = region_df[var]

                                                # Calculate contributions for other variables
                                                for var in other_variables:
                                                    beta_col = f"beta_scaled_{var}"
                                                    if beta_col in region_row and f"scaled_{var}" in region_df.columns:
                                                        beta_value = float(region_row[beta_col])
                                                        region_df[f"{var}_contribution"] = beta_value * region_df[f"scaled_{var}"]
                                                        

                                                transformed_data_list.append(region_df)

                                            # Concatenate all transformed data
                                            transformed_df = pd.concat(transformed_data_list, axis=0).reset_index(drop=True)
                                            return transformed_df
                                        



                            # segment_data = df_fea[df_fea[selected_columns[0]] == selected_group]

                            # if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression","Prophet"]):

                            #     with st.expander("Final Features and Constraints:"):

                            #         exclude_columns = ["Market", "Brand", column_to_filter, target_col, "date", "Year", "Month", "Fiscal Year"]

                            #         if valid_features:
        
                            #             # Non-PCA method
                            #             scaler = StandardScaler()
                            #             X_scaled = scaler.fit_transform(segment_data[valid_features])
                            #             y = segment_data[target_col]

                            #             # Train XGBoost model
                            #             xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                            #             xgb_model.fit(X_scaled, y)

                            #             # Get feature importance
                            #             feature_importances = pd.DataFrame({
                            #                 'Feature': valid_features,
                            #                 'Importance': xgb_model.feature_importances_
                            #             }).sort_values(by='Importance', ascending=False)


                            #             if not only_prophet_selected and any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression", "Prophet"]):
                            #                 final_features = valid_features

                            #             elif only_prophet_selected:

                            #                 if len(valid_features) == 1:
                            #                     final_features = valid_features  # No need to drop any features

                            #                 else:

                            #                     # Hierarchical clustering to remove correlated features
                            #                     corr_matrix = segment_data[valid_features].corr().abs()
                            #                     corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Ensure symmetry

                            #                     # Check for NaN or invalid values in the correlation matrix
                            #                     if corr_matrix.isna().any().any():
                            #                         st.error(f"Correlation matrix contains NaN values for segment {selected_group}. Please check the data.")
                            #                         return

                            #                     distance_matrix = 1 - corr_matrix
                            #                     distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure symmetry

                            #                     # Check for NaN or invalid values in the distance matrix
                            #                     if distance_matrix.isna().any().any():
                            #                         st.error(f"Distance matrix contains NaN values for segment {selected_group}. Please check the data.")
                            #                         return

                            #                     # Perform hierarchical clustering
                            #                     Z = linkage(distance_matrix, method='complete')
                            #                     clusters = fcluster(Z, t=0.1, criterion='distance')  # Adjust `t` to control the correlation threshold

                            #                     # Create a dictionary to map clusters to features
                            #                     cluster_to_features = {}
                            #                     for cluster, feature in zip(clusters, valid_features):
                            #                         if cluster not in cluster_to_features:
                            #                             cluster_to_features[cluster] = []
                            #                         cluster_to_features[cluster].append(feature)

                            #                     # Drop less important features in each cluster
                            #                     to_drop = set()
                            #                     for cluster, features in cluster_to_features.items():
                            #                         if len(features) > 1:  # Only process clusters with more than one feature
                            #                             most_important_feature = None
                            #                             highest_importance = -1
                            #                             for feature in features:
                            #                                 importance = feature_importances.loc[feature_importances['Feature'] == feature, 'Importance'].values[0]
                            #                                 if importance > highest_importance:
                            #                                     highest_importance = importance
                            #                                     most_important_feature = feature
                            #                             for feature in features:
                            #                                 if feature != most_important_feature:
                            #                                     to_drop.add(feature)

                            #                     final_features = [f for f in valid_features if f not in to_drop]

                                        

                            #                 scaler = StandardScaler()
                            #                 X_scaled = scaler.fit_transform(segment_data[final_features])
                            #                 y = segment_data[target_col]

                            #                 # Train XGBoost model
                            #                 xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                            #                 xgb_model.fit(X_scaled, y)

                            #                 # Get feature importance
                            #                 feature_importances = pd.DataFrame({
                            #                     'Feature': final_features,
                            #                     'Importance': xgb_model.feature_importances_
                            #                 }).sort_values(by='Importance', ascending=False)

                            #                 # Filter features with importance > 0.01
                            #                 final_features = feature_importances[
                            #                     (feature_importances['Feature'].isin(final_features)) & 
                            #                     (feature_importances['Importance'] > 0.02)
                            #                 ]['Feature'].tolist()

                                            
                            #                 st.write("Features retained after correlation filtering and importance threshold:")
                            #                 final_feature_importance = feature_importances[
                            #                     (feature_importances['Feature'].isin(final_features))
                            #                 ]
                            #                 st.write(final_feature_importance)
                            #                 st.markdown('<hr class="thin">', unsafe_allow_html=True)



                            #             if 'growth_rate_adjustments' not in st.session_state:
                            #                 st.session_state.growth_rate_adjustments = {}

                            #             # Feature forecasting using Prophet - WITHOUT SCALING
                            #             feature_forecasts = {}
                            #             for feature in final_features:
                            #                 # Prepare Prophet data - using raw values
                            #                 prophet_df = segment_data[[feature]].reset_index().copy()
                            #                 prophet_df['y'] = prophet_df[feature]  # Use raw values
                            #                 prophet_df = prophet_df.rename(columns={date_col: 'ds'})[['ds', 'y']]
                                            
                            #                 # Train model
                            #                 model_prophet = Prophet()
                            #                 model_prophet.fit(prophet_df)

                            #                 # Generate future forecasts only
                            #                 future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq=frequency_options, include_history=False)
                            #                 forecast = model_prophet.predict(future)

                            #                 historical_fit = model_prophet.predict(prophet_df.rename(columns={'ds': 'ds', 'y': 'y'}))
                                            
                            #                 # Store results - all in original scale
                            #                 feature_forecasts[feature] = {
                            #                     'actual_values': prophet_df['y'].tolist(),
                            #                     'actual_dates': prophet_df['ds'].tolist(),
                            #                     'past_forecast': historical_fit['yhat'].tolist(),
                            #                     'future_forecast': forecast['yhat'].tolist(),
                            #                     'future_dates': future['ds'].tolist()
                            #                 }

                            #             # Calculate growth rates based on fiscal years
                            #             growth_dfs = []
                            #             for feature in final_features:
                            #                 # Combine actual and forecast data
                            #                 all_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'] + feature_forecasts[feature]['future_dates'])
                            #                 all_values = feature_forecasts[feature]['actual_values'] + feature_forecasts[feature]['future_forecast']
                                            
                            #                 # Create DataFrame (all values already in original scale)
                            #                 df_g = pd.DataFrame({
                            #                     'date': all_dates,
                            #                     'value': all_values,
                            #                     'type': ['actual']*len(feature_forecasts[feature]['actual_dates']) + ['forecast']*len(feature_forecasts[feature]['future_dates'])
                            #                 })
                                            
                            #                 # Calculate fiscal years
                            #                 df_g['fiscal_year'] = df_g['date'] - pd.offsets.DateOffset(months=fiscal_start_month-1)
                            #                 df_g['fiscal_year'] = df_g['fiscal_year'].dt.year
                            #                 # if fiscal_start_month != 1:
                            #                 #     df_g['fiscal_year'] = df_g['fiscal_year'] + 1
                                            
                            #                 # Group by fiscal year and calculate mean
                            #                 annual_df = df_g.groupby('fiscal_year')['value'].mean().reset_index()

                            #                 # Calculate PROPER growth rates
                            #                 annual_df['prev_year_mean'] = annual_df['value'].shift(1)  # Get previous year's mean
                            #                 annual_df['Growth Rate (%)'] = ((annual_df['value'] - annual_df['prev_year_mean']) / 
                            #                                             annual_df['prev_year_mean']) * 100
                            #                 annual_df['Growth Rate (%)'] = annual_df['Growth Rate (%)'].fillna(0).round(2)
                            #                 annual_df['Feature'] = feature
                                            
                            #                 growth_dfs.append(annual_df[['Feature', 'fiscal_year', 'Growth Rate (%)']])

                            #             # Create and display editable growth rate table
                            #             combined_growth = pd.concat(growth_dfs).reset_index(drop=True)
                            #             combined_growth_pivoted = combined_growth.pivot(
                            #                 index='Feature',
                            #                 columns='fiscal_year',
                            #                 values='Growth Rate (%)'
                            #             ).reset_index()

                            #             year_columns = sorted([col for col in combined_growth_pivoted.columns if isinstance(col, int)])
                            #             combined_growth_pivoted = combined_growth_pivoted[['Feature'] + year_columns]

                            #             edited_growth_pivoted = st.data_editor(
                            #                 combined_growth_pivoted,
                            #                 key="growth_rate_editor",
                            #                 num_rows="fixed"
                            #             )

                            #             # Create a single combined plot for all features
                            #             def plot_combined_forecasts(features):
                            #                 fig = go.Figure()
                                            
                            #                 # Define a color palette (expand as needed for more features)
                            #                 colors = px.colors.qualitative.Plotly
                                            
                            #                 for i, feature in enumerate(features):
                            #                     color = colors[i % len(colors)]
                                                
                            #                     # Plot actual data
                            #                     actual_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'])
                            #                     actual_values = feature_forecasts[feature]['actual_values']
                                                
                            #                     fig.add_trace(go.Scatter(
                            #                         x=actual_dates,
                            #                         y=actual_values,
                            #                         name=f'{feature} (Actual)',
                            #                         line=dict(color=color),
                            #                         legendgroup=feature,
                            #                         showlegend=True
                            #                     ))
                                                
                            #                     # Plot forecast data
                            #                     forecast_dates = pd.to_datetime(feature_forecasts[feature]['future_dates'])
                            #                     forecast_values = feature_forecasts[feature]['future_forecast']
                                                
                            #                     fig.add_trace(go.Scatter(
                            #                         x=forecast_dates,
                            #                         y=forecast_values,
                            #                         name=f'{feature} (Forecast)',
                            #                         line=dict(color=color, dash='dash'),
                            #                         legendgroup=feature,
                            #                         showlegend=True
                            #                     ))
                                            
                            #                 # Add vertical line at forecast start point (using the first feature's forecast start)
                            #                 if len(features) > 0 and len(feature_forecasts[features[0]]['future_dates']) > 0:
                            #                     first_forecast_date = pd.to_datetime(feature_forecasts[features[0]]['future_dates'][0])
                            #                     fig.add_vline(
                            #                         x=first_forecast_date.timestamp() * 1000,
                            #                         line_dash="dot",
                            #                         line_color="gray",
                            #                         annotation_text="Forecast Start",
                            #                         annotation_position="top left"
                            #                     )
                                            
                            #                 fig.update_layout(
                            #                     title='Combined Feature Forecasts',
                            #                     xaxis_title='Date',
                            #                     yaxis_title='Value',
                            #                     hovermode='x unified',
                            #                     legend=dict(
                            #                         orientation="h",
                            #                         yanchor="bottom",
                            #                         y=1.02,
                            #                         xanchor="right",
                            #                         x=1
                            #                     )
                            #                 )
                                            
                            #                 st.plotly_chart(fig, use_container_width=True)


                                        # if 'adjusted_forecasts' not in st.session_state:
                                        #     st.session_state.adjusted_forecasts = {}

                                        # # Apply adjustments
                                        # if st.button("Apply Adjustments"):
                                        #     # Get the original unedited growth rates for comparison
                                        #     original_growth = combined_growth.copy()
                                            
                                        #     # Melt the edited table
                                        #     edited_growth = edited_growth_pivoted.melt(
                                        #         id_vars=['Feature'],
                                        #         var_name='fiscal_year',
                                        #         value_name='Growth Rate (%)'
                                        #     )
                                            
                                        #     # Convert fiscal_year to int for merging
                                        #     edited_growth['fiscal_year'] = edited_growth['fiscal_year'].astype(int)
                                        #     original_growth['fiscal_year'] = original_growth['fiscal_year'].astype(int)
                                            
                                        #     # Process each feature's adjustments
                                        #     for feature in final_features:
                                        #         # Get original and new rates for this feature
                                        #         feature_rates_new = edited_growth[edited_growth['Feature'] == feature].sort_values('fiscal_year')
                                        #         feature_rates_old = original_growth[original_growth['Feature'] == feature].sort_values('fiscal_year')
                                                
                                        #         # Find which years were actually changed
                                        #         merged = pd.merge(
                                        #             feature_rates_new, 
                                        #             feature_rates_old,
                                        #             on=['Feature', 'fiscal_year'],
                                        #             suffixes=('_new', '_old')
                                        #         )
                                        #         changed_years = merged[merged['Growth Rate (%)_new'] != merged['Growth Rate (%)_old']]
                                                
                                        #         if len(changed_years) == 0:
                                        #             continue  # Skip if no changes for this feature
                                                    
                                        #         # Create dictionary of new rates {year: rate}
                                        #         new_rates = feature_rates_new.set_index('fiscal_year')['Growth Rate (%)'] / 100
                                                
                                        #         # Get the data (already in original scale)
                                        #         all_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'] + feature_forecasts[feature]['future_dates'])
                                        #         all_values = np.array(feature_forecasts[feature]['past_forecast'] + feature_forecasts[feature]['future_forecast'])
                                                
                                        #         # Create DataFrame
                                        #         df_g = pd.DataFrame({'date': all_dates, 'value': all_values})
                                                
                                        #         # Calculate fiscal years
                                        #         df_g['fiscal_year'] = df_g['date'] - pd.offsets.DateOffset(months=fiscal_start_month-1)
                                        #         df_g['fiscal_year'] = df_g['fiscal_year'].dt.year
                                        #         # if not (1 <= fiscal_start_month <= 5):
                                        #         #     df_g['fiscal_year'] = df_g['fiscal_year'] + 1
                                                
                                        #         # Sort and group
                                        #         df_g = df_g.sort_values('fiscal_year')
                                        #         yearly_groups = df_g.groupby('fiscal_year', sort=True)
                                                
                                        #         # MODIFIED APPROACH: Only adjust the specific changed years
                                        #         for year in changed_years['fiscal_year']:
                                        #             # Get the group for this year
                                        #             year_mask = df_g['fiscal_year'] == year
                                        #             year_group = df_g[year_mask]
                                                    
                                        #             if len(year_group) == 0:
                                        #                 continue
                                                        
                                        #             # Calculate current average for this year
                                        #             current_avg = year_group['value'].mean()
                                                    
                                        #             # Calculate target based on edited growth rate
                                        #             prev_year = year - 1
                                        #             prev_year_group = df_g[df_g['fiscal_year'] == prev_year]
                                        #             if len(prev_year_group) == 0:
                                        #                 continue  # Can't calculate growth if no previous year
                                                        
                                        #             prev_avg = prev_year_group['value'].mean()
                                        #             target_avg = prev_avg * (1 + new_rates[year])
                                                    
                                        #             # Calculate and apply scaling only to this year
                                        #             if current_avg != 0:
                                        #                 scaling_factor = target_avg / current_avg
                                        #                 df_g.loc[year_mask, 'value'] *= scaling_factor
                                                
                                        #         # Update forecasts (still in original scale)
                                        #         hist_length = len(feature_forecasts[feature]['past_forecast'])
                                        #         feature_forecasts[feature]['past_forecast'] = df_g['value'].values[:hist_length].tolist()
                                        #         feature_forecasts[feature]['future_forecast'] = df_g['value'].values[hist_length:].tolist()

                                        #     # Store the adjusted forecasts in session state
                                        #     for feature in final_features:
                                        #         st.session_state.adjusted_forecasts[feature] = {
                                        #             'past_forecast': feature_forecasts[feature]['past_forecast'],
                                        #             'future_forecast': feature_forecasts[feature]['future_forecast']
                                        #         }
                                            
                                        #     st.success("Growth rate adjustments applied!")
                                        #     st.rerun()  # Refresh to show updated forecasts

                                        # # When loading forecasts, check for existing adjustments
                                        # for feature in final_features:
                                        #     if feature in st.session_state.adjusted_forecasts:
                                        #         # Apply previously made adjustments
                                        #         feature_forecasts[feature]['past_forecast'] = st.session_state.adjusted_forecasts[feature]['past_forecast']
                                        #         feature_forecasts[feature]['future_forecast'] = st.session_state.adjusted_forecasts[feature]['future_forecast']

                                        


                                        # # Display the combined plot
                                        # if len(final_features) > 0:
                                        #     plot_combined_forecasts(final_features)


                                        #     # Create a scaled version of the forecasts at the end
                                        # scaled_feature_forecasts = {}
                                        # for feature in final_features:
                                        #     # Create and fit scaler
                                        #     feature_scaler = StandardScaler()
                                        #     feature_scaler.fit(segment_data[[feature]])
                                            
                                        #     # Scale all values
                                        #     scaled_feature_forecasts[feature] = {
                                        #         'actual_values': feature_scaler.transform(np.array(feature_forecasts[feature]['actual_values']).reshape(-1, 1)).flatten().tolist(),
                                        #         'actual_dates': feature_forecasts[feature]['actual_dates'],
                                        #         'past_forecast': feature_scaler.transform(np.array(feature_forecasts[feature]['past_forecast']).reshape(-1, 1)).flatten().tolist(),
                                        #         'future_forecast': feature_scaler.transform(np.array(feature_forecasts[feature]['future_forecast']).reshape(-1, 1)).flatten().tolist(),
                                        #         'future_dates': feature_forecasts[feature]['future_dates'],
                                        #         '_scaler': feature_scaler
                                        #     }
                                            
                                        
                                        # # Forecast Volume using SARIMAX with selected features as exogenous variables
                                        # scaler_prophet = StandardScaler()
                                        # segment_data[final_features] = scaler_prophet.fit_transform(segment_data[final_features])

                                        # exog_train = segment_data[final_features]
                                        # future_exog = pd.DataFrame({feature: feature_forecasts[feature]['future_forecast'] for feature in final_features})
        
                                        # # Train SARIMAX model for Volume
                                        # auto_sarimax_model = pm.auto_arima(segment_data[target_col], exogenous=exog_train, seasonal=True, m=12, stepwise=True, trace=False)
                                        # best_order = auto_sarimax_model.order
                                        # best_seasonal_order = auto_sarimax_model.seasonal_order
                                        
                                        # model_sarimax = SARIMAX(segment_data[target_col], order=best_order, seasonal_order=best_seasonal_order, exog=exog_train).fit()
                                        
                                        # # # Past forecast (using historical feature values)
                                        # # past_forecast_sarimax = model_sarimax.predict(start=segment_data.index[0], end=segment_data.index[-1], exog=exog_train)
                                        
                                        # # # Future forecast (using forecasted feature values)
                                        # # future_forecast_sarimax = model_sarimax.forecast(steps=forecast_horizon, exog=future_exog)
                                        
                                        # # Prepare data for Prophet
                                        # prophet_df = segment_data[[target_col] + final_features].reset_index().rename(columns={date_col: 'ds', target_col: 'y'})
                                        
                                        # # Initialize Prophet model

                                        # model_prophet = Prophet()              
                                        # # Add regressors (features) to the Prophet model
                                        # for feature in final_features:
                                        #     model_prophet.add_regressor(feature)
                                        
                                        # # Fit the model
                                        # model_prophet.fit(prophet_df)
                                        
                                        # # Past forecast (using historical feature values)
                                        # past_forecast_prophet = model_prophet.predict(prophet_df[['ds'] + final_features])
                                        
                                        # # Future forecast (using forecasted feature values)
                                        # if frequency_options == 'Y':
                                        #     offset = pd.DateOffset(years=1)
                                        # elif frequency_options == 'M':
                                        #     offset = pd.DateOffset(months=1)
                                        # elif frequency_options == 'W':
                                        #     offset = pd.DateOffset(weeks=1)
                                        # elif frequency_options == 'D':
                                        #     offset = pd.DateOffset(days=1)
                                        # elif frequency_options == 'T':  # 'T' is the alias for minutely frequency
                                        #     offset = pd.DateOffset(minutes=1)
                                        # else:
                                        #     raise ValueError("Unsupported frequency. Use 'Y', 'M', 'W', 'D', or 'T'.")

                                        # future_dates = pd.date_range(start=segment_data.index.max() + offset, periods=forecast_horizon, freq=frequency_options)

                                        # # future_dates = pd.date_range(start=segment_data.index.max() + pd.DateOffset(months=1), periods=forecast_horizon, freq=frequency)

                                        # future_df = pd.DataFrame({'ds': future_dates})
                                        # for feature in final_features:
                                        #     future_df[feature] = scaled_feature_forecasts[feature]['future_forecast']
                                        
                                        # future_forecast_prophet = model_prophet.predict(future_df)


    
                                        
                                        # # Store results
                                        # forecast_results = {
                                        #     selected_group: {
                                        #         'Model_type':'Prophet',
                                        #         'actual_dates': segment_data.index,
                                        #         'actual_volume': segment_data[target_col],
                                        #         # 'past_sarimax_forecast': past_forecast_sarimax,
                                        #         # 'sarimax_future_forecast': future_forecast_sarimax,
                                        #         'past_prophet_forecast': past_forecast_prophet['yhat'],
                                        #         'prophet_future_forecast': future_forecast_prophet['yhat'],
                                        #         'feature_forecasts': feature_forecasts
                                        #         # 'feature_forecasts': forecast['yhat'].iloc[-forecast_horizon:].tolist(),
                                        #     }
                                        # }



                                        # # Calculate MAPE for Prophet model
                                        # actual = segment_data[target_col]
                                        # prophet_predictions = past_forecast_prophet['yhat'].values
                                        # mape = np.mean(np.abs((actual - prophet_predictions) / actual)) 

                                        # # Create results DataFrame
                                        # prophet_results_df = pd.DataFrame({
                                        #     'Model_type': ['Prophet'],
                                        #     'Segment': [selected_group],
                                        #     'MAPE': [mape]
                                        # })

                                        



                                        # # --- Get Standardized Coefficients from Prophet ---
                                        # num_regressors = len(final_features)
                                        # regressor_coefficients = model_prophet.params['beta'][:, :num_regressors].mean(axis=0)  # Average across chains

                                        # standardized_coefficients = dict(zip(final_features, regressor_coefficients))

                                        # # --- Destandardize Coefficients ---
                                        # feature_means = scaler_prophet.mean_
                                        # feature_stds = np.sqrt(scaler_prophet.var_)
                                        # mean_target = segment_data[target_col].mean()

                                        # destandardized_coefficients = {
                                        #     feature: (standardized_coefficients[feature]* mean_target)/ feature_stds[i]
                                            
                                        #     for i, feature in enumerate(final_features)
                                        # }

                                        # # --- Calculate Elasticity ---
                                        
                                        # elasticity = {
                                        #     feature: destandardized_coefficients[feature] * (feature_means[i] / mean_target)
                                        #     for i, feature in enumerate(final_features)
                                        # }

                                        # # --- Display Results ---
                                        # feature_metrics = pd.DataFrame({
                                        #     'Model_type': 'Prophet',
                                        #     'Segment':selected_group,
                                        #     'Variable': final_features,
                                            
                                        #     # 'Standardized_Coefficient': regressor_coefficients,
                                        #     # 'Destandardized_Coefficient': [destandardized_coefficients[feature] for feature in final_features],
                                        #     # 'Original_feature_Mean': feature_means,
                                        #     # 'Original_feature_std': feature_stds,
                                        #     # 'Target Mean' : mean_target,
                                        #     'Elasticity': [elasticity[feature] for feature in final_features],
                                        #     'p_value':'None'
                                        # })

                                        # # st.write("\nFeature Importance Metrics:")
                                        # # st.write(feature_metrics.round(4))



                                        # # Create an empty list to store rows
                                        # rows = []

                                        # for feature_name, forecast_data in feature_forecasts.items():
                                        #     values = forecast_data['future_forecast']
                                        #     dates = forecast_data['future_dates']
                                            
                                        #     # Ensure values and dates are aligned
                                        #     for date, value in zip(dates, values):
                                        #         rows.append({
                                        #             'Feature': feature_name,
                                        #             'Date': pd.to_datetime(date),
                                        #             'Value': value
                                        #         })

                                        # # Create a DataFrame
                                        # df_forecast = pd.DataFrame(rows)





                            segment_data_list = []
                            for group in selected_group:
                                segment_data = df_fea[df_fea[selected_columns[0]] == group]
                                segment_data_list.append((group, segment_data))

                            if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression","Prophet"]):
                                with st.expander("Final Features and Constraints:"):
                                    exclude_columns = ["Market", "Brand", column_to_filter, target_col, "date", "Year", "Month", "Fiscal Year"]
                                    
                                    # Initialize containers for group-specific results
                                    all_group_features = {}
                                    all_feature_importances = {}
                                    all_feature_forecasts = {}
                                    
                                    # Define the plotting function
                                    def plot_combined_forecasts(features, feature_forecasts, group_name=""):
                                        fig = go.Figure()
                                        
                                        colors = px.colors.qualitative.Plotly
                                        
                                        for i, feature in enumerate(features):
                                            color = colors[i % len(colors)]
                                            


                                            
                                            # Plot actual data
                                            actual_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'])
                                            actual_values = feature_forecasts[feature]['actual_values']
                                            
                                            fig.add_trace(go.Scatter(
                                                x=actual_dates,
                                                y=actual_values,
                                                name=f'{feature} (Actual)',
                                                line=dict(color=color),
                                                legendgroup=feature,
                                                showlegend=True
                                            ))
                                            
                                            # Plot forecast data
                                            forecast_dates = pd.to_datetime(feature_forecasts[feature]['future_dates'])
                                            forecast_values = feature_forecasts[feature]['future_forecast']
                                            
                                            fig.add_trace(go.Scatter(
                                                x=forecast_dates,
                                                y=forecast_values,
                                                name=f'{feature} (Forecast)',
                                                line=dict(color=color, dash='dash'),
                                                legendgroup=feature,
                                                showlegend=True
                                            ))
                                        
                                        # Add vertical line at forecast start point
                                        if len(features) > 0 and len(feature_forecasts[features[0]]['future_dates']) > 0:
                                            first_forecast_date = pd.to_datetime(feature_forecasts[features[0]]['future_dates'][0])
                                            fig.add_vline(
                                                x=first_forecast_date.timestamp() * 1000,
                                                line_dash="dot",
                                                line_color="gray",
                                                annotation_text="Forecast Start",
                                                annotation_position="top left"
                                            )
                                        
                                        title = 'Feature Forecasts'
                                        if group_name and only_prophet_selected:
                                            title = f'{title} - {group_name}'
                                            
                                        fig.update_layout(
                                            title=title,
                                            xaxis_title='Date',
                                            yaxis_title='Value',
                                            hovermode='x unified',
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)



                                    # # Define the plotting function (modified to show only selected feature)
                                    # def plot_combined_forecasts(features, feature_forecasts, group_name=""):
                                    #     # Only plot the selected feature (first item in features list)
                                    #     if not features:
                                    #         return
                                        
                                    #     fig = go.Figure()
                                    #     color = px.colors.qualitative.Plotly[0]  # Use first color
                                        
                                    #     feature = features[0]  # Only the selected feature
                                        
                                    #     # Plot actual data
                                    #     actual_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'])
                                    #     actual_values = feature_forecasts[feature]['actual_values']
                                        
                                    #     fig.add_trace(go.Scatter(
                                    #         x=actual_dates,
                                    #         y=actual_values,
                                    #         name=f'{feature} (Actual)',
                                    #         line=dict(color=color),
                                    #         showlegend=True
                                    #     ))
                                        
                                    #     # Plot forecast data
                                    #     forecast_dates = pd.to_datetime(feature_forecasts[feature]['future_dates'])
                                    #     forecast_values = feature_forecasts[feature]['future_forecast']
                                        
                                    #     fig.add_trace(go.Scatter(
                                    #         x=forecast_dates,
                                    #         y=forecast_values,
                                    #         name=f'{feature} (Forecast)',
                                    #         line=dict(color=color, dash='dash'),
                                    #         showlegend=True
                                    #     ))
                                        
                                    #     # Add vertical line at forecast start point
                                    #     if len(feature_forecasts[feature]['future_dates']) > 0:
                                    #         first_forecast_date = pd.to_datetime(feature_forecasts[feature]['future_dates'][0])
                                    #         fig.add_vline(
                                    #             x=first_forecast_date.timestamp() * 1000,
                                    #             line_dash="dot",
                                    #             line_color="gray",
                                    #             annotation_text="Forecast Start",
                                    #             annotation_position="top left"
                                    #         )
                                        
                                    #     title = f'{feature} Forecast'
                                    #     if group_name and only_prophet_selected:
                                    #         title = f'{title} - {group_name}'
                                            
                                    #     fig.update_layout(
                                    #         title=title,
                                    #         xaxis_title='Date',
                                    #         yaxis_title='Value',
                                    #         hovermode='x unified',
                                    #         legend=dict(
                                    #             orientation="h",
                                    #             yanchor="bottom",
                                    #             y=1.02,
                                    #             xanchor="right",
                                    #             x=1
                                    #         )
                                    #     )
                                        
                                    #     st.plotly_chart(fig, use_container_width=True)







                                    combined_growth_dict = {}
                                    combined_growth_pivoted_dict = {}

                                    # Process first group to get features (same for all groups when multiple models selected)
                                    first_group, first_segment_data = segment_data_list[0]
                                    
                                    if valid_features:
                                        # Non-PCA method
                                        scaler = StandardScaler()
                                        X_scaled = scaler.fit_transform(first_segment_data[valid_features])
                                        y = first_segment_data[target_col]

                                        # Train XGBoost model
                                        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                                        xgb_model.fit(X_scaled, y)

                                        # Get feature importance
                                        feature_importances = pd.DataFrame({
                                            'Feature': valid_features,
                                            'Importance': xgb_model.feature_importances_
                                        }).sort_values(by='Importance', ascending=False)

                                        if not only_prophet_selected and any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression", "Prophet"]):
                                            final_features = valid_features
                                        elif only_prophet_selected:
                                            if len(valid_features) == 1:
                                                final_features = valid_features
                                            else:
                                                # Hierarchical clustering to remove correlated features
                                                corr_matrix = first_segment_data[valid_features].corr().abs()
                                                corr_matrix = (corr_matrix + corr_matrix.T) / 2

                                                if corr_matrix.isna().any().any():
                                                    st.error(f"Correlation matrix contains NaN values. Please check the data.")
                                                    return

                                                distance_matrix = 1 - corr_matrix
                                                distance_matrix = (distance_matrix + distance_matrix.T) / 2

                                                if distance_matrix.isna().any().any():
                                                    st.error(f"Distance matrix contains NaN values. Please check the data.")
                                                    return

                                                Z = linkage(distance_matrix, method='complete')
                                                clusters = fcluster(Z, t=0.1, criterion='distance')

                                                cluster_to_features = {}
                                                for cluster, feature in zip(clusters, valid_features):
                                                    if cluster not in cluster_to_features:
                                                        cluster_to_features[cluster] = []
                                                    cluster_to_features[cluster].append(feature)

                                                to_drop = set()
                                                for cluster, features in cluster_to_features.items():
                                                    if len(features) > 1:
                                                        most_important_feature = None
                                                        highest_importance = -1
                                                        for feature in features:
                                                            importance = feature_importances.loc[feature_importances['Feature'] == feature, 'Importance'].values[0]
                                                            if importance > highest_importance:
                                                                highest_importance = importance
                                                                most_important_feature = feature
                                                        for feature in features:
                                                            if feature != most_important_feature:
                                                                to_drop.add(feature)

                                                final_features = [f for f in valid_features if f not in to_drop]

                                            scaler = StandardScaler()
                                            X_scaled = scaler.fit_transform(first_segment_data[final_features])
                                            y = first_segment_data[target_col]

                                            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                                            xgb_model.fit(X_scaled, y)

                                            feature_importances = pd.DataFrame({
                                                'Feature': final_features,
                                                'Importance': xgb_model.feature_importances_
                                            }).sort_values(by='Importance', ascending=False)

                                            final_features = feature_importances[
                                                (feature_importances['Feature'].isin(final_features)) & 
                                                (feature_importances['Importance'] > 0.02)
                                            ]['Feature'].tolist()

                                        # Show features only once if multiple models selected
                                        if not only_prophet_selected:
                                            # st.write("Features retained after correlation filtering and importance threshold (same for all groups):")
                                            final_feature_importance = feature_importances[
                                                (feature_importances['Feature'].isin(final_features))
                                            ]
                                            st.write(final_feature_importance)
                                            st.markdown('<hr class="thin">', unsafe_allow_html=True)

                            

                                    



                                    # for group, segment_data in segment_data_list:
                                    #     if valid_features:
                                    #         # Store group-specific results (using same features for all groups when multiple models)
                                    #         all_group_features[group] = final_features
                                    #         all_feature_importances[group] = feature_importances

                                    #         # Show group-specific features only if only Prophet is selected
                                    #         if only_prophet_selected:
                                    #             st.subheader(f"Group: {group}")
                                    #             st.write("Features retained after correlation filtering and importance threshold:")
                                    #             final_feature_importance = feature_importances[
                                    #                 (feature_importances['Feature'].isin(final_features))
                                    #             ]
                                    #             st.write(final_feature_importance)
                                    #             st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    #         # Feature forecasting using Prophet
                                    #         feature_forecasts = {}
                                    #         for feature in final_features:
                                    #             prophet_df = segment_data[[feature]+['Fiscal Year']].reset_index().copy()
                                                

                                    #             prophet_df['y'] = prophet_df[feature]
                                                
                                    #             prophet_df = prophet_df.rename(columns={date_col: 'ds'})[['ds', 'y','Fiscal Year']]

                                    #             # prophet_df

                                    #             # Align dates based on frequency
                                    #             if frequency_options == 'D':
                                    #                 pass  # no alignment needed for daily
                                    #             elif frequency_options == 'W':
                                    #                 prophet_df['ds'] = prophet_df['ds'] + pd.offsets.Week(weekday=6)  # align to Sunday
                                    #             elif frequency_options == 'M':
                                    #                 prophet_df['ds'] = prophet_df['ds'] + pd.offsets.MonthEnd(0)
                                    #             elif frequency_options == 'Q':
                                    #                 prophet_df['ds'] = prophet_df['ds'] + pd.offsets.QuarterEnd(startingMonth=3)
                                    #             elif frequency_options == 'Y':
                                    #                 prophet_df['ds'] = prophet_df['ds'] + pd.offsets.YearEnd(0)
                                                
                                    #             model_prophet = Prophet()
                                    #             model_prophet.fit(prophet_df)

                                    #             future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq=frequency_options, include_history=False)
                                    #             forecast = model_prophet.predict(future)

                                    #             historical_fit = model_prophet.predict(prophet_df.rename(columns={'ds': 'ds', 'y': 'y'}))


                                    #             # prophet_df
                                    #             # Calculate statistics
                                    #             overall_mean = prophet_df['y'].mean()
                                    #             overall_std = prophet_df['y'].std()
                                                
                                    #             # # Calculate previous year mean
                                    #             # prophet_df['year'] = prophet_df['ds'].dt.year
                                    #             # previous_year = prophet_df['year'].max() - 1
                                    #             # previous_year_mean = prophet_df[prophet_df['year'] == previous_year]['y'].mean()

                                    #             # Calculate previous fiscal year mean
                                    #             max_fiscal_year = prophet_df['Fiscal Year'].max()  # Get the latest fiscal year (e.g., 'FY25')

                                    #             # Extract the year number and create previous fiscal year string
                                    #             current_fy_number = int(max_fiscal_year[2:])  # Extracts '25' from 'FY25'
                                    #             previous_fy = f'FY{current_fy_number - 1}'    # Creates 'FY24'

                                    #             # Calculate mean for previous fiscal year
                                    #             previous_year_mean = prophet_df[prophet_df['Fiscal Year'] == previous_fy]['y'].mean()
                                                
                                    #             feature_forecasts[feature] = {
                                    #                 'actual_values': prophet_df['y'].tolist(),
                                    #                 'actual_dates': prophet_df['ds'].tolist(),
                                    #                 'past_forecast': historical_fit['yhat'].tolist(),
                                    #                 'future_forecast': forecast['yhat'].tolist(),
                                    #                 'future_dates': future['ds'].tolist(),
                                    #                 'overall_mean': overall_mean,
                                    #                 'overall_std': overall_std,
                                    #                 'previous_year_mean': previous_year_mean
                                    #             }

                                    #         all_feature_forecasts[group] = feature_forecasts


                                    

                                    




                                    # Add model selection for feature forecasting (before the loop)
                                    # # st.subheader("Feature Forecasting Models")
                                    # feature_forecast_model = st.selectbox(
                                    #     "Select models for feature forecasting:",
                                    #     options=["Prophet", "SARIMA", "Holt-Winters", "ETS"],
                                    #     index=0
                                    # )

                                    # # Inside the group loop, after feature selection:
                                    # if valid_features:
                                    #     # Store group-specific results (same as before)
                                    #     all_group_features[group] = final_features
                                    #     all_feature_importances[group] = feature_importances

                                    #     # Show group-specific features (same as before)
                                    #     if only_prophet_selected:
                                    #         st.subheader(f"Group: {group}")
                                    #         st.write("Features retained after correlation filtering and importance threshold:")
                                    #         final_feature_importance = feature_importances[
                                    #             (feature_importances['Feature'].isin(final_features))
                                    #         ]
                                    #         st.write(final_feature_importance)
                                    #         st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                    #     # Feature forecasting using selected model
                                    #     feature_forecasts = {}
                                    #     for feature in final_features:
                                    #         # Prepare data (same as before)
                                    #         prophet_df = segment_data[[feature]+['Fiscal Year']].reset_index().copy()
                                    #         prophet_df['y'] = prophet_df[feature]
                                    #         prophet_df = prophet_df.rename(columns={date_col: 'ds'})[['ds', 'y','Fiscal Year']]
                                            
                                    #         # Align dates based on frequency (same as before)
                                    #         if frequency_options == 'D':
                                    #             pass
                                    #         elif frequency_options == 'W':
                                    #             prophet_df['ds'] = prophet_df['ds'] + pd.offsets.Week(weekday=6)
                                    #         elif frequency_options == 'M':
                                    #             prophet_df['ds'] = prophet_df['ds'] + pd.offsets.MonthEnd(0)
                                    #         elif frequency_options == 'Q':
                                    #             prophet_df['ds'] = prophet_df['ds'] + pd.offsets.QuarterEnd(startingMonth=3)
                                    #         elif frequency_options == 'Y':
                                    #             prophet_df['ds'] = prophet_df['ds'] + pd.offsets.YearEnd(0)
                                            
                                    #         # Calculate statistics (same as before)
                                    #         overall_mean = prophet_df['y'].mean()
                                    #         overall_std = prophet_df['y'].std()
                                    #         max_fiscal_year = prophet_df['Fiscal Year'].max()
                                    #         current_fy_number = int(max_fiscal_year[2:])
                                    #         previous_fy = f'FY{current_fy_number - 1}'
                                    #         previous_year_mean = prophet_df[prophet_df['Fiscal Year'] == previous_fy]['y'].mean()

                                    #         # Initialize forecast storage
                                    #         forecast_data = {
                                    #             'actual_values': prophet_df['y'].tolist(),
                                    #             'actual_dates': prophet_df['ds'].tolist(),
                                    #             'overall_mean': overall_mean,
                                    #             'overall_std': overall_std,
                                    #             'previous_year_mean': previous_year_mean
                                    #         }

                                    #         # Prophet forecasting
                                    #         if feature_forecast_model == "Prophet":
                                    #             model_prophet = Prophet()
                                    #             model_prophet.fit(prophet_df)
                                    #             future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq=frequency_options, include_history=False)
                                    #             forecast = model_prophet.predict(future)
                                    #             historical_fit = model_prophet.predict(prophet_df.rename(columns={'ds': 'ds', 'y': 'y'}))
                                                
                                    #             forecast_data.update({
                                    #                 'past_forecast': historical_fit['yhat'].tolist(),
                                    #                 'future_forecast': forecast['yhat'].tolist(),
                                    #                 'future_dates': future['ds'].tolist()
                                    #             })

                                    #         # SARIMA forecasting
                                    #         elif feature_forecast_model == "SARIMA":
                                    #             try:
                                    #                 ts_data = prophet_df.set_index('ds')['y']
                                                    
                                    #                 # Determine seasonal order based on frequency
                                    #                 if frequency_options == 'D':
                                    #                     seasonal_order = (1, 1, 1, 7)  # weekly seasonality
                                    #                 elif frequency_options == 'W':
                                    #                     seasonal_order = (1, 1, 1, 52)  # yearly seasonality
                                    #                 elif frequency_options == 'M':
                                    #                     seasonal_order = (1, 1, 1, 12)  # yearly seasonality
                                    #                 elif frequency_options == 'Q':
                                    #                     seasonal_order = (1, 1, 1, 4)   # yearly seasonality
                                    #                 else:  # yearly
                                    #                     seasonal_order = (0, 0, 0, 0)    # no seasonality
                                                    
                                    #                 model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=seasonal_order)
                                    #                 model_fit = model.fit(disp=False)
                                                    
                                    #                 # Generate forecasts
                                    #                 forecast_result = model_fit.get_forecast(steps=forecast_horizon)
                                    #                 future_dates = pd.date_range(
                                    #                     start=ts_data.index[-1] + pd.Timedelta(days=1),
                                    #                     periods=forecast_horizon,
                                    #                     freq=frequency_options
                                    #                 )
                                                    
                                    #                 forecast_data.update({
                                    #                     'past_forecast': model_fit.predict(start=0, end=len(ts_data)-1).tolist(),
                                    #                     'future_forecast': forecast_result.predicted_mean.tolist(),
                                    #                     'future_dates': future_dates.tolist()
                                    #                 })
                                    #             except Exception as e:
                                    #                 st.warning(f"SARIMA failed for feature {feature}: {str(e)}")
                                    #                 continue  # Skip this feature if model fails

                                    #         # Holt-Winters forecasting
                                    #         elif feature_forecast_model == "Holt-Winters":
                                    #             try:
                                    #                 ts_data = prophet_df.set_index('ds')['y']
                                    #                 seasonal_periods = None
                                                    
                                    #                 if frequency_options == 'M':
                                    #                     seasonal_periods = 12
                                    #                 elif frequency_options == 'Q':
                                    #                     seasonal_periods = 4
                                    #                 elif frequency_options == 'W':
                                    #                     seasonal_periods = 52
                                                    
                                    #                 model = ExponentialSmoothing(
                                    #                     ts_data,
                                    #                     trend='add',
                                    #                     seasonal='add' if seasonal_periods else None,
                                    #                     seasonal_periods=seasonal_periods
                                    #                 ).fit()
                                                    
                                    #                 future_dates = pd.date_range(
                                    #                     start=ts_data.index[-1] + pd.Timedelta(days=1),
                                    #                     periods=forecast_horizon,
                                    #                     freq=frequency_options
                                    #                 )
                                                    
                                    #                 forecast_data.update({
                                    #                     'past_forecast': model.fittedvalues.tolist(),
                                    #                     'future_forecast': model.forecast(forecast_horizon).tolist(),
                                    #                     'future_dates': future_dates.tolist()
                                    #                 })
                                    #             except Exception as e:
                                    #                 st.warning(f"Holt-Winters failed for feature {feature}: {str(e)}")
                                    #                 continue

                                    #         # ETS forecasting using ExponentialSmoothing (Holt-Winters)
                                    #         elif feature_forecast_model == "ETS":
                                    #             try:
                                    #                 ts_data = prophet_df.set_index('ds')['y']
                                                    
                                    #                 # Determine seasonal periods based on frequency
                                    #                 if frequency_options == 'M':
                                    #                     seasonal_periods = 12  # monthly data with yearly seasonality
                                    #                 elif frequency_options == 'Q':
                                    #                     seasonal_periods = 4   # quarterly data with yearly seasonality
                                    #                 elif frequency_options == 'W':
                                    #                     seasonal_periods = 52   # weekly data with yearly seasonality
                                    #                 else:
                                    #                     seasonal_periods = None # no seasonality for other frequencies
                                                    
                                    #                 # Fit ETS model (using ExponentialSmoothing implementation)
                                    #                 ets_model = ExponentialSmoothing(
                                    #                     ts_data,
                                    #                     trend='add',
                                    #                     seasonal='add' if seasonal_periods else None,
                                    #                     seasonal_periods=seasonal_periods
                                    #                 ).fit()
                                                    
                                    #                 # Generate future dates
                                    #                 future_dates = pd.date_range(
                                    #                     start=ts_data.index[-1] + pd.Timedelta(days=1),
                                    #                     periods=forecast_horizon,
                                    #                     freq=frequency_options
                                    #                 )
                                                    
                                    #                 forecast_data.update({
                                    #                     'past_forecast': ets_model.fittedvalues.tolist(),
                                    #                     'future_forecast': ets_model.forecast(forecast_horizon).tolist(),
                                    #                     'future_dates': future_dates.tolist()
                                    #                 })
                                                    
                                    #             except Exception as e:
                                    #                 st.warning(f"ETS forecasting failed for feature {feature}: {str(e)}")
                                    #                 continue

                                    #         feature_forecasts[feature] = forecast_data

                                    #     all_feature_forecasts[group] = feature_forecasts




                                    # st.subheader("Feature Forecasting Models")
                                    # Remove the initial model selection since we'll test all models first

                                    # Inside the group loop, after feature selection:
                                    if valid_features:
                                        # Store group-specific results (same as before)
                                        all_group_features[group] = final_features
                                        all_feature_importances[group] = feature_importances

                                        # Show group-specific features (same as before)
                                        if only_prophet_selected:
                                            st.subheader(f"Group: {group}")
                                            st.write("Features retained after correlation filtering and importance threshold:")
                                            final_feature_importance = feature_importances[
                                                (feature_importances['Feature'].isin(final_features))
                                            ]
                                            st.write(final_feature_importance)
                                            st.markdown('<hr class="thin">', unsafe_allow_html=True)

                                        # Test all forecasting models first
                                        all_models = ["Prophet", "SARIMA", "Holt-Winters", "ETS"]
                                        model_results = {}
                                        growth_rate_comparison = {}
                                        
                                        for model_name in all_models:
                                            feature_forecasts = {}
                                            model_growth_rates = {}
                                            
                                            for feature in final_features:
                                                # Prepare data (same as before)
                                                prophet_df = segment_data[[feature]+['Fiscal Year']].reset_index().copy()
                                                prophet_df['y'] = prophet_df[feature]
                                                prophet_df = prophet_df.rename(columns={date_col: 'ds'})[['ds', 'y','Fiscal Year']]
                                                
                                                # Align dates based on frequency (same as before)
                                                if frequency_options == 'D':
                                                    pass
                                                elif frequency_options == 'W':
                                                    prophet_df['ds'] = prophet_df['ds'] + pd.offsets.Week(weekday=6)
                                                elif frequency_options == 'M':
                                                    prophet_df['ds'] = prophet_df['ds'] + pd.offsets.MonthEnd(0)
                                                elif frequency_options == 'Q':
                                                    prophet_df['ds'] = prophet_df['ds'] + pd.offsets.QuarterEnd(startingMonth=3)
                                                elif frequency_options == 'Y':
                                                    prophet_df['ds'] = prophet_df['ds'] + pd.offsets.YearEnd(0)
                                                
                                                # Calculate statistics (same as before)
                                                overall_mean = prophet_df['y'].mean()
                                                overall_std = prophet_df['y'].std()
                                                max_fiscal_year = prophet_df['Fiscal Year'].max()
                                                current_fy_number = int(max_fiscal_year[2:])
                                                previous_fy = f'FY{current_fy_number - 1}'
                                                previous_year_mean = prophet_df[prophet_df['Fiscal Year'] == previous_fy]['y'].mean()

                                                # Initialize forecast storage
                                                forecast_data = {
                                                    'actual_values': prophet_df['y'].tolist(),
                                                    'actual_dates': prophet_df['ds'].tolist(),
                                                    'overall_mean': overall_mean,
                                                    'overall_std': overall_std,
                                                    'previous_year_mean': previous_year_mean
                                                }

                                                try:
                                                    # Prophet forecasting
                                                    if model_name == "Prophet":
                                                        model_prophet = Prophet()
                                                        model_prophet.fit(prophet_df)
                                                        future = model_prophet.make_future_dataframe(periods=forecast_horizon, freq=frequency_options, include_history=False)
                                                        forecast = model_prophet.predict(future)
                                                        historical_fit = model_prophet.predict(prophet_df.rename(columns={'ds': 'ds', 'y': 'y'}))
                                                        
                                                        forecast_data.update({
                                                            'Model': 'Prophet',
                                                            'past_forecast': historical_fit['yhat'].tolist(),
                                                            'future_forecast': forecast['yhat'].tolist(),
                                                            'future_dates': future['ds'].tolist()
                                                        })

                                                    # SARIMA forecasting
                                                    elif model_name == "SARIMA":
                                                        ts_data = prophet_df.set_index('ds')['y']
                                                        
                                                        # Determine seasonal order based on frequency
                                                        if frequency_options == 'D':
                                                            seasonal_order = (1, 1, 1, 7)  # weekly seasonality
                                                        elif frequency_options == 'W':
                                                            seasonal_order = (1, 1, 1, 52)  # yearly seasonality
                                                        elif frequency_options == 'M':
                                                            seasonal_order = (1, 1, 1, 12)  # yearly seasonality
                                                        elif frequency_options == 'Q':
                                                            seasonal_order = (1, 1, 1, 4)   # yearly seasonality
                                                        else:  # yearly
                                                            seasonal_order = (0, 0, 0, 0)    # no seasonality
                                                        
                                                        model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=seasonal_order)
                                                        model_fit = model.fit(disp=False)
                                                        
                                                        # Generate forecasts
                                                        forecast_result = model_fit.get_forecast(steps=forecast_horizon)
                                                        future_dates = pd.date_range(
                                                            start=ts_data.index[-1] + pd.Timedelta(days=1),
                                                            periods=forecast_horizon,
                                                            freq=frequency_options
                                                        )
                                                        
                                                        forecast_data.update({
                                                            'Model': 'SARIMA',
                                                            'past_forecast': model_fit.predict(start=0, end=len(ts_data)-1).tolist(),
                                                            'future_forecast': forecast_result.predicted_mean.tolist(),
                                                            'future_dates': future_dates.tolist()
                                                        })

                                                    # Holt-Winters forecasting
                                                    elif model_name == "Holt-Winters":
                                                        ts_data = prophet_df.set_index('ds')['y']
                                                        seasonal_periods = None
                                                        
                                                        if frequency_options == 'M':
                                                            seasonal_periods = 12
                                                        elif frequency_options == 'Q':
                                                            seasonal_periods = 4
                                                        elif frequency_options == 'W':
                                                            seasonal_periods = 52
                                                        
                                                        model = ExponentialSmoothing(
                                                            ts_data,
                                                            trend='add',
                                                            seasonal='add' if seasonal_periods else None,
                                                            seasonal_periods=seasonal_periods
                                                        ).fit()
                                                        
                                                        future_dates = pd.date_range(
                                                            start=ts_data.index[-1] + pd.Timedelta(days=1),
                                                            periods=forecast_horizon,
                                                            freq=frequency_options
                                                        )
                                                        
                                                        forecast_data.update({
                                                            'Model': 'Holt-Winters',
                                                            'past_forecast': model.fittedvalues.tolist(),
                                                            'future_forecast': model.forecast(forecast_horizon).tolist(),
                                                            'future_dates': future_dates.tolist()
                                                        })

                                                    # ETS forecasting
                                                    elif model_name == "ETS":
                                                        ts_data = prophet_df.set_index('ds')['y']
                                                        
                                                        if frequency_options == 'M':
                                                            seasonal_periods = 12
                                                        elif frequency_options == 'Q':
                                                            seasonal_periods = 4
                                                        elif frequency_options == 'W':
                                                            seasonal_periods = 52
                                                        else:
                                                            seasonal_periods = None
                                                        
                                                        ets_model = ExponentialSmoothing(
                                                            ts_data,
                                                            trend='add',
                                                            seasonal='add' if seasonal_periods else None,
                                                            seasonal_periods=seasonal_periods
                                                        ).fit()
                                                        
                                                        future_dates = pd.date_range(
                                                            start=ts_data.index[-1] + pd.Timedelta(days=1),
                                                            periods=forecast_horizon,
                                                            freq=frequency_options
                                                        )
                                                        
                                                        forecast_data.update({
                                                            'Model': 'ETS',
                                                            'past_forecast': ets_model.fittedvalues.tolist(),
                                                            'future_forecast': ets_model.forecast(forecast_horizon).tolist(),
                                                            'future_dates': future_dates.tolist()
                                                        })
                                                    
                                                    # Calculate annual growth rates for this feature
                                                    actual_values = forecast_data['actual_values']
                                                    forecast_values = forecast_data['future_forecast']
                                                    fiscal_years = prophet_df['Fiscal Year'].unique()
                                                    
                                                    # Get last actual year and forecast year values
                                                    last_actual = np.mean(actual_values[-12:]) if len(actual_values) >= 12 else actual_values[-1]
                                                    first_forecast = np.mean(forecast_values[:12]) if len(forecast_values) >= 12 else forecast_values[0]
                                                    
                                                    growth_rate = ((first_forecast - last_actual) / last_actual) * 100
                                                    model_growth_rates[feature] = growth_rate
                                                    
                                                    feature_forecasts[feature] = forecast_data
                                                    
                                                except Exception as e:
                                                    st.warning(f"{model_name} failed for feature {feature}: {str(e)}")
                                                    model_growth_rates[feature] = None
                                                    continue
                                            
                                            model_results[model_name] = feature_forecasts
                                            growth_rate_comparison[model_name] = model_growth_rates

                                        col1, col2 = st.columns(2)

                                        with col1:
                                            # Function to determine fiscal year based on date and start month
                                            def get_fiscal_year(date, start_month):
                                                if date.month >= start_month:
                                                    return f"FY{(date.year + 1) % 100:02d}"
                                                else:
                                                    return f"FY{date.year % 100:02d}"

                                            # Function to calculate annual growth rates with fiscal year consideration
                                            def calculate_annual_growth(df, forecast_dates, forecast_values, fiscal_start_month):
                                                """Calculate annual growth rates considering fiscal year start month"""
                                                # Create full DataFrame with actual and forecasted values
                                                full_df = pd.DataFrame({
                                                    'date': pd.to_datetime(df['ds']).tolist() + pd.to_datetime(forecast_dates).tolist(),
                                                    'value': df['y'].tolist() + forecast_values,
                                                    'type': ['actual'] * len(df) + ['forecast'] * len(forecast_dates)
                                                })
                                                
                                                # Add fiscal year column based on start month
                                                full_df['fiscal_year'] = full_df['date'].apply(
                                                    lambda x: get_fiscal_year(x, fiscal_start_month)
                                                )
                                                
                                                # Calculate average by fiscal year
                                                yearly_avg = full_df.groupby(['fiscal_year', 'type'])['value'].mean().unstack()
                                                yearly_avg = yearly_avg.sort_index()  # Sort fiscal years
                                                
                                                # Calculate growth rates
                                                growth_rates = {}
                                                fiscal_years = yearly_avg.index.tolist()
                                                
                                                for i in range(1, len(fiscal_years)):
                                                    current_fy = fiscal_years[i]
                                                    prev_fy = fiscal_years[i-1]
                                                    
                                                    # Use forecast if available, otherwise actual
                                                    current_value = yearly_avg.loc[current_fy, 'forecast'] if 'forecast' in yearly_avg.columns and not pd.isna(yearly_avg.loc[current_fy, 'forecast']) else yearly_avg.loc[current_fy, 'actual']
                                                    prev_value = yearly_avg.loc[prev_fy, 'forecast'] if 'forecast' in yearly_avg.columns and not pd.isna(yearly_avg.loc[prev_fy, 'forecast']) else yearly_avg.loc[prev_fy, 'actual']
                                                    
                                                    if prev_value != 0:
                                                        growth_rate = (((current_value - prev_value) / prev_value) * 100).round(2)
                                                    else:
                                                        growth_rate = 0
                                                    
                                                    growth_rates[current_fy] = growth_rate
                                                
                                                return growth_rates

                                            # Create a dictionary to hold growth rates for all models
                                            all_model_growth = {}

                                            for model_name in all_models:
                                                model_growth = {}
                                                
                                                for feature in final_features:
                                                    if feature in model_results[model_name]:
                                                        data = model_results[model_name][feature]
                                                        prophet_df = pd.DataFrame({
                                                            'ds': data['actual_dates'],
                                                            'y': data['actual_values']
                                                        })
                                                        
                                                        growth_rates = calculate_annual_growth(
                                                            prophet_df,
                                                            data['future_dates'],
                                                            data['future_forecast'],
                                                            fiscal_start_month
                                                        )
                                                        
                                                        model_growth[feature] = growth_rates
                                                    else:
                                                        model_growth[feature] = {}
                                                
                                                all_model_growth[model_name] = model_growth

                                            # Feature selection dropdown
                                            selected_feature = st.selectbox(
                                                "Select feature to view growth rates:",
                                                options=final_features,
                                                index=0,
                                                key=f"feature_selector_{group}"
                                            )

                                            # Create comparison DataFrame for selected feature
                                            st.write(f"Growth Rate Comparison: {selected_feature}")

                                            # Collect all fiscal years for the selected feature
                                            all_fys = set()
                                            for model_name in all_models:
                                                if selected_feature in all_model_growth[model_name]:
                                                    all_fys.update(all_model_growth[model_name][selected_feature].keys())
                                            all_fys = sorted(all_fys)

                                            # Create comparison DataFrame
                                            comparison_data = []
                                            for fy in all_fys:
                                                row = {'Fiscal Year': fy}
                                                for model_name in all_models:
                                                    if selected_feature in all_model_growth[model_name] and fy in all_model_growth[model_name][selected_feature]:
                                                        row[model_name] = all_model_growth[model_name][selected_feature][fy]
                                                    else:
                                                        row[model_name] = None
                                                comparison_data.append(row)

                                            comparison_df = pd.DataFrame(comparison_data).set_index('Fiscal Year')

                                            # Display styled DataFrame
                                            st.dataframe(
                                                comparison_df, use_container_width=True
                                            )

                                        with col2:
                                            # # Let user select which model to proceed with
                                            # selected_model = st.selectbox(
                                            #     "Select model to use for further analysis:",
                                            #     options=all_models,
                                            #     index=0,  # Default to Prophet
                                            #     key=f"model_select_{group}"
                                            # )


                                            # Let user select which model to proceed with using radio buttons
                                            selected_model = st.radio(
                                                "Select model to use for further analysis:",
                                                options=all_models,
                                                index=0,  # Default to Prophet (first item in all_models)
                                                key=f"model_select_{group}",
                                                horizontal=True  # This displays the options horizontally
                                            )





                                            # Get the selected model's forecasts
                                            selected_model_forecasts = model_results[selected_model]
                                            
                                            # Store the selected model's forecasts
                                            all_feature_forecasts[group] = selected_model_forecasts
                                            
                                            # Plot the selected model's forecasts
                                            if valid_features:
                                                # Show plots (only once if multiple models, per group if only Prophet)
                                                if only_prophet_selected or group == first_group:
                                                    plot_title = group if only_prophet_selected else "All Groups"
                                                    plot_combined_forecasts(final_features, selected_model_forecasts, plot_title)



















                                            # # Calculate growth rates (show only for first group if multiple models)
                                            # if only_prophet_selected or group == first_group:
                                            #     growth_dfs = []
                                            #     for feature in final_features:
                                            #         all_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'] + feature_forecasts[feature]['future_dates'])
                                            #         all_values = feature_forecasts[feature]['actual_values'] + feature_forecasts[feature]['future_forecast']
                                                    
                                            #         df_g = pd.DataFrame({
                                            #             'date': all_dates,
                                            #             'value': all_values,
                                            #             'type': ['actual']*len(feature_forecasts[feature]['actual_dates']) + ['forecast']*len(feature_forecasts[feature]['future_dates'])
                                            #         })
                                                    
                                            #         df_g['fiscal_year'] = df_g['date'] - pd.offsets.DateOffset(months=fiscal_start_month-1)
                                            #         df_g['fiscal_year'] = df_g['fiscal_year'].dt.year
                                                    
                                            #         annual_df = df_g.groupby('fiscal_year')['value'].mean().reset_index()

                                            #         annual_df['prev_year_mean'] = annual_df['value'].shift(1)
                                            #         annual_df['Growth Rate (%)'] = ((annual_df['value'] - annual_df['prev_year_mean']) / 
                                            #                                     annual_df['prev_year_mean']) * 100
                                            #         annual_df['Growth Rate (%)'] = annual_df['Growth Rate (%)'].fillna(0).round(2)
                                            #         annual_df['Feature'] = feature
                                                    
                                            #         growth_dfs.append(annual_df[['Feature', 'fiscal_year', 'Growth Rate (%)']])

                                            #     # Create and display editable growth rate table
                                            #     combined_growth = pd.concat(growth_dfs).reset_index(drop=True)
                                            #     combined_growth_dict[group] = combined_growth
                                            #     combined_growth_pivoted = combined_growth.pivot(
                                            #         index='Feature',
                                            #         columns='fiscal_year',
                                            #         values='Growth Rate (%)'
                                            #     ).reset_index()

                                            #     year_columns = sorted([col for col in combined_growth_pivoted.columns if isinstance(col, int)])
                                            #     combined_growth_pivoted = combined_growth_pivoted[['Feature'] + year_columns]
                                            #     combined_growth_pivoted_dict[group] = combined_growth_pivoted

                                            #     if only_prophet_selected:
                                            #         st.write(f"Growth Rates - {group}")
                                            #     else:
                                            #         st.write("Growth Rates")
                                                    
                                            #     edited_growth_pivoted = st.data_editor(
                                            #         combined_growth_pivoted,
                                            #         key=f"growth_rate_editor_{group if only_prophet_selected else 'all'}",
                                            #         num_rows="fixed"
                                            #     )

                                            

                                    # # Adjustment button (single button for all groups)
                                    # if 'adjusted_forecasts' not in st.session_state:
                                    #     st.session_state.adjusted_forecasts = {}

                                    # if st.button("Apply Adjustments"):
                                    #     for group, segment_data in segment_data_list:
                                    #         final_features = all_group_features.get(group, [])
                                    #         feature_forecasts = all_feature_forecasts.get(group, {})
                                            
                                    #         if not final_features:
                                    #             continue
                                                
                                    #         try:
                                    #             # Get the edited growth rates from the data editor
                                    #             editor_key = f"growth_rate_editor_{group}" if only_prophet_selected else "growth_rate_editor_all"
                                    #             edited_data = st.session_state[editor_key]
                                                
                                    #             # Convert to proper DataFrame format
                                    #             edited_growth = pd.DataFrame(edited_data['edited_rows'].values())
                                                
                                    #             # Get original growth rates
                                    #             original_growth = combined_growth_dict[group if only_prophet_selected else first_group].copy()
                                    #             original_growth['fiscal_year'] = original_growth['fiscal_year'].astype(int)
                                                
                                    #             # Process each feature's adjustments
                                    #             for feature in final_features:
                                    #                 # Get the edited rates for this feature
                                    #                 feature_rates_edited = edited_growth[edited_growth['Feature'] == feature]
                                                    
                                    #                 if feature_rates_edited.empty:
                                    #                     continue
                                                        
                                    #                 # Get original rates for this feature
                                    #                 feature_rates_original = original_growth[original_growth['Feature'] == feature]
                                                    
                                    #                 # Create mapping of year to new growth rate
                                    #                 growth_rate_changes = feature_rates_edited.set_index('fiscal_year')['Growth Rate (%)'] / 100
                                                    
                                    #                 # Get all forecast dates and values
                                    #                 all_dates = pd.to_datetime(feature_forecasts[feature]['actual_dates'] + feature_forecasts[feature]['future_dates'])
                                    #                 all_values = np.array(feature_forecasts[feature]['past_forecast'] + feature_forecasts[feature]['future_forecast'])
                                                    
                                    #                 # Create working DataFrame
                                    #                 df_g = pd.DataFrame({
                                    #                     'date': all_dates,
                                    #                     'value': all_values
                                    #                 })
                                                    
                                    #                 # Calculate fiscal years
                                    #                 df_g['fiscal_year'] = (df_g['date'] - pd.offsets.DateOffset(months=fiscal_start_month-1)).dt.year
                                                    
                                    #                 # Apply adjustments year by year
                                    #                 for year, new_rate in growth_rate_changes.items():
                                    #                     # Get data for this year
                                    #                     year_mask = df_g['fiscal_year'] == year
                                    #                     year_data = df_g[year_mask]
                                                        
                                    #                     if year_data.empty:
                                    #                         continue
                                                            
                                    #                     # Get previous year's average
                                    #                     prev_year = year - 1
                                    #                     prev_year_data = df_g[df_g['fiscal_year'] == prev_year]
                                                        
                                    #                     if prev_year_data.empty:
                                    #                         continue
                                                            
                                    #                     # Calculate scaling factor
                                    #                     current_avg = year_data['value'].mean()
                                    #                     prev_avg = prev_year_data['value'].mean()
                                    #                     target_avg = prev_avg * (1 + new_rate)
                                                        
                                    #                     if current_avg != 0:
                                    #                         scaling_factor = target_avg / current_avg
                                    #                         df_g.loc[year_mask, 'value'] *= scaling_factor
                                                    
                                    #                 # Update forecasts
                                    #                 hist_length = len(feature_forecasts[feature]['past_forecast'])
                                    #                 feature_forecasts[feature]['past_forecast'] = df_g['value'].values[:hist_length].tolist()
                                    #                 feature_forecasts[feature]['future_forecast'] = df_g['value'].values[hist_length:].tolist()
                                                    
                                    #                 # Store in session state
                                    #                 if group not in st.session_state.adjusted_forecasts:
                                    #                     st.session_state.adjusted_forecasts[group] = {}
                                                    
                                    #                 st.session_state.adjusted_forecasts[group][feature] = {
                                    #                     'past_forecast': feature_forecasts[feature]['past_forecast'],
                                    #                     'future_forecast': feature_forecasts[feature]['future_forecast']
                                    #                 }
                                                    
                                    #         except Exception as e:
                                    #             st.error(f"Failed to adjust forecasts for group {group}: {str(e)}")
                                    #             continue
                                        
                                    #     st.success("Growth rate adjustments applied successfully!")
                                    #     st.rerun()

                                    
            




                                    # # When loading forecasts, check for existing adjustments (inside group loop)
                                    # for group, segment_data in segment_data_list:
                                    #     final_features = all_group_features.get(group, [])
                                    #     feature_forecasts = all_feature_forecasts.get(group, {})
                                        
                                    #     if group in st.session_state.adjusted_forecasts:
                                    #         for feature in final_features:
                                    #             if feature in st.session_state.adjusted_forecasts[group]:
                                    #                 # Apply previously made adjustments
                                    #                 feature_forecasts[feature]['past_forecast'] = st.session_state.adjusted_forecasts[group][feature]['past_forecast']
                                    #                 feature_forecasts[feature]['future_forecast'] = st.session_state.adjusted_forecasts[group][feature]['future_forecast']

                                    # Create scaled version of forecasts for each group
                                    scaled_feature_forecasts_dict = {}
                                    for group, segment_data in segment_data_list:
                                        final_features = all_group_features.get(group, [])
                                        feature_forecasts = all_feature_forecasts.get(group, {})
                                        
                                        scaled_feature_forecasts = {}
                                        for feature in final_features:
                                            # Create and fit scaler
                                            feature_scaler = StandardScaler()
                                            feature_scaler.fit(segment_data[[feature]])
                                            
                                            # Scale all values
                                            scaled_feature_forecasts[feature] = {
                                                'actual_values': feature_scaler.transform(np.array(feature_forecasts[feature]['actual_values']).reshape(-1, 1)).flatten().tolist(),
                                                'actual_dates': feature_forecasts[feature]['actual_dates'],
                                                'past_forecast': feature_scaler.transform(np.array(feature_forecasts[feature]['past_forecast']).reshape(-1, 1)).flatten().tolist(),
                                                'future_forecast': feature_scaler.transform(np.array(feature_forecasts[feature]['future_forecast']).reshape(-1, 1)).flatten().tolist(),
                                                'future_dates': feature_forecasts[feature]['future_dates'],
                                                '_scaler': feature_scaler
                                            }
                                        scaled_feature_forecasts_dict[group] = scaled_feature_forecasts



                                    # Initialize dictionaries to store forecast results
                                    forecast_results = {}
                                    prophet_results_df_list = []
                                    feature_metrics_list = []
                                    all_forecast_dfs = []

                                    for group, segment_data in segment_data_list:
                                        final_features = all_group_features.get(group, [])
                                        feature_forecasts = all_feature_forecasts.get(group, {})
                                        scaled_feature_forecasts = scaled_feature_forecasts_dict.get(group, {})
                                        
                                        if not final_features:
                                            continue
                                            
                                        # Scale features for current group
                                        scaler_prophet = StandardScaler()
                                        segment_data[final_features] = scaler_prophet.fit_transform(segment_data[final_features])
                                        exog_train = segment_data[final_features]
                                        
                                        # Prepare future exogenous variables
                                        future_exog = pd.DataFrame({
                                            feature: scaled_feature_forecasts[feature]['future_forecast'] 
                                            for feature in final_features
                                        })
                                        
                                        # SARIMAX Model (optional - uncomment if needed)
                                        
                                        # auto_sarimax_model = pm.auto_arima(
                                        #     segment_data[target_col], 
                                        #     exogenous=exog_train, 
                                        #     seasonal=True, 
                                        #     m=12, 
                                        #     stepwise=True, 
                                        #     trace=False
                                        # )
                                        # best_order = auto_sarimax_model.order
                                        # best_seasonal_order = auto_sarimax_model.seasonal_order
                                        
                                        # model_sarimax = SARIMAX(
                                        #     segment_data[target_col], 
                                        #     order=best_order, 
                                        #     seasonal_order=best_seasonal_order, 
                                        #     exog=exog_train
                                        # ).fit()
                                        
                                        
                                        # Prophet Model
                                        prophet_df = segment_data[[target_col] + final_features+['Fiscal Year']].reset_index().rename(
                                            columns={date_col: 'ds', target_col: 'y'}
                                        )
                                        
                                        model_prophet = Prophet()              
                                        for feature in final_features:
                                            model_prophet.add_regressor(feature)
                                        
                                        model_prophet.fit(prophet_df)
                                        
                                        # Past forecast
                                        past_forecast_prophet = model_prophet.predict(prophet_df[['ds'] + final_features])
                                        
                                        # Future forecast
                                        if frequency_options == 'Y':
                                            offset = pd.DateOffset(years=1)
                                        elif frequency_options == 'M':
                                            offset = pd.DateOffset(months=1)
                                        elif frequency_options == 'W':
                                            offset = pd.DateOffset(weeks=1)
                                        elif frequency_options == 'D':
                                            offset = pd.DateOffset(days=1)
                                        elif frequency_options == 'T':
                                            offset = pd.DateOffset(minutes=1)
                                        else:
                                            raise ValueError("Unsupported frequency. Use 'Y', 'M', 'W', 'D', or 'T'.")

                                        future_dates = pd.date_range(
                                            start=segment_data.index.max() + offset, 
                                            periods=forecast_horizon, 
                                            freq=frequency_options
                                        )
                                        
                                        future_df = pd.DataFrame({'ds': future_dates})
                                        for feature in final_features:
                                            future_df[feature] = scaled_feature_forecasts[feature]['future_forecast']
                                        
                                        future_forecast_prophet = model_prophet.predict(future_df)

                                        historical_mean = prophet_df['y'].mean()

                                        # # Get previous year mean from actual values (not forecasts)
                                        # last_date = prophet_df['ds'].max()
                                        # previous_year = last_date.year - 1
                                        # prev_year_mask = prophet_df['ds'].dt.year == previous_year
                                        # prev_year_mean = prophet_df.loc[prev_year_mask, 'y'].mean()


                                        # st.write(prophet_df)


                                        # Get previous fiscal year mean from actual values
                                        last_date = prophet_df['ds'].max()
                                        max_fiscal_year = prophet_df.loc[prophet_df['ds'] == last_date, 'Fiscal Year'].values[0]

                                        # Extract the year number (assuming format like 'FY24')
                                        current_fy_number = int(max_fiscal_year[2:])
                                        previous_fy = f'FY{current_fy_number - 1}'

                                        # Calculate mean for previous fiscal year
                                        prev_year_mask = prophet_df['Fiscal Year'] == previous_fy
                                        prev_year_mean = prophet_df.loc[prev_year_mask, 'y'].mean()





                                        # Store results for this group
                                        forecast_results[group] = {
                                            'Model_type': 'Prophet',
                                            'actual_dates': segment_data.index,
                                            'actual_volume': segment_data[target_col],
                                            'past_prophet_forecast': past_forecast_prophet['yhat'],
                                            'prophet_future_forecast': future_forecast_prophet['yhat'],
                                            'feature_forecasts': feature_forecasts,
                                            'historical_mean': historical_mean,  # Added this line
                                            'previous_year_mean': prev_year_mean 
                                        }
                                        
                                        # st.write(feature_forecasts)

                                        # Calculate MAPE for this group
                                        actual = segment_data[target_col]
                                        prophet_predictions = past_forecast_prophet['yhat'].values
                                        mape = np.mean(np.abs((actual - prophet_predictions) / actual)) # Convert to percentage
                                        
                                        prophet_results_df_list.append(pd.DataFrame({
                                            'Model_type': ['Prophet'],
                                            'Segment': [group],
                                            'MAPE': [mape],
                                            'R_squared':['None'],
                                            'Adjusted_R_squared':['None']
                                        }))

                                    #     # Feature importance metrics for this group
                                    #     num_regressors = len(final_features)
                                    #     regressor_coefficients = model_prophet.params['beta'][:, :num_regressors].mean(axis=0)
                                        
                                    #     standardized_coefficients = dict(zip(final_features, regressor_coefficients))
                                    #     feature_means = scaler_prophet.mean_
                                    #     feature_stds = np.sqrt(scaler_prophet.var_)
                                    #     mean_target = segment_data[target_col].mean()
                                        
                                    #     destandardized_coefficients = {
                                    #         feature: (standardized_coefficients[feature] * mean_target) / feature_stds[i]
                                    #         for i, feature in enumerate(final_features)
                                    #     }
                                        
                                    #     elasticity = {
                                    #         feature: destandardized_coefficients[feature] * (feature_means[i] / mean_target)
                                    #         for i, feature in enumerate(final_features)
                                    #     }
                                        
                                    #     feature_metrics_list.append(pd.DataFrame({
                                    #         'Model_type': 'Prophet',
                                    #         'Segment': group,
                                    #         'Variable': final_features,
                                    #         'Elasticity': [elasticity[feature] for feature in final_features],
                                    #         'p_value': 'None'
                                    #     }))
                                        
                                    #     # Create forecast DataFrame for this group's features
                                    #     rows = []
                                    #     for feature_name, forecast_data in feature_forecasts.items():
                                    #         values = forecast_data['future_forecast']
                                    #         dates = forecast_data['future_dates']
                                            
                                    #         for date, value in zip(dates, values):
                                    #             rows.append({
                                    #                 'Feature': feature_name,
                                    #                 'Date': pd.to_datetime(date),
                                    #                 'Value': value,
                                    #                 'Group': group
                                    #             })
                                        
                                    #     all_forecast_dfs.append(pd.DataFrame(rows))

                                    # # Combine all group results
                                    # if prophet_results_df_list:
                                    #     prophet_results_df = pd.concat(prophet_results_df_list, ignore_index=True)
                                        
                                    # if feature_metrics_list:
                                    #     feature_metrics = pd.concat(feature_metrics_list, ignore_index=True)
                                        
                                    # if all_forecast_dfs:
                                    #     df_forecast = pd.concat(all_forecast_dfs, ignore_index=True)

                                    # if final_features:



                                        # Feature importance metrics (historical data only)
                                        num_regressors = len(final_features)
                                        regressor_coefficients = model_prophet.params['beta'][:, :num_regressors].mean(axis=0)
                                        # intercept = model_prophet.params['beta'][:, num_regressors].mean()

                                        # Get the trend at all historical time points
                                        trend_component = model_prophet.predict(prophet_df)['trend']  

                                        # The intercept is the mean of the trend (baseline when regressors=0)
                                        intercept = trend_component.mean()  

                                        # Get time period information
                                        last_date = segment_data.index.max()
                                        current_year = last_date.year
                                        previous_year = current_year - 1

                                        # Calculate previous year means (historical data only)
                                        prev_year_mask = segment_data.index.year == previous_year
                                        prev_year_data = segment_data.loc[prev_year_mask]

                                        # Destandardized means (full period and previous year)
                                        feature_means = scaler_prophet.mean_  # Original means (already destandardized)
                                        feature_stds = np.sqrt(scaler_prophet.var_)  # Original stds (already destandardized)
                                        mean_target = segment_data[target_col].mean()  # Full period target mean

                                        # Previous year destandardized means
                                        # prev_year_target_mean = prev_year_data[target_col].mean()

                                        prev_year_target_mean = forecast_results[group]['previous_year_mean']



                                        # Get previous year means from feature_forecasts
                                        prev_year_feature_means_from_forecasts = {
                                            feature: feature_forecasts[feature]['previous_year_mean']
                                            for feature in final_features
                                        }

                                        # Standardized coefficients conversion
                                        standardized_coefficients = dict(zip(final_features, regressor_coefficients))
                                        destandardized_coefficients = {
                                            feature: (standardized_coefficients[feature] * mean_target) / feature_stds[i]
                                            for i, feature in enumerate(final_features)
                                        }

                                        # Destandardized intercept calculation
                                        destandardized_intercept = intercept * mean_target - sum(
                                            standardized_coefficients[feature] * feature_means[i] * mean_target / feature_stds[i]
                                            for i, feature in enumerate(final_features)
                                        )

                                        # since scaler_prophet.mean_ and scaler_prophet.var_ were calculated from original data
                                        elasticity = {
                                            feature: destandardized_coefficients[feature] * (feature_means[i] / mean_target)
                                            for i, feature in enumerate(final_features)
                                        }

                                        # Create feature metrics DataFrame
                                        feature_metrics_df = pd.DataFrame({
                                            'Model_type': 'Prophet',
                                            'Segment': group,
                                            'Target_Column': target_col,
                                            'Target_Mean': mean_target,
                                            'Prev_Year_Target_Mean': prev_year_target_mean,  # Previous year target mean
                                            'Variable': final_features,
                                            'Feature_Mean': feature_means,  # Full period original means
                                            'Feature_Std': feature_stds,    # Full period original stds
                                            'Prev_Year_Feature_Mean': [prev_year_feature_means_from_forecasts[feature] for feature in final_features],
                                            'Elasticity': [elasticity[feature] for feature in final_features],
                                            # 'Destandardized_Beta': [destandardized_coefficients[feature] for feature in final_features],
                                            'intercept': intercept,
                                            'Destandardized_Intercept': destandardized_intercept,
                                            'p_value': 'None'
                                        })

                                        feature_metrics_list.append(feature_metrics_df)

                                    # Combine all group results
                                    if prophet_results_df_list:
                                        prophet_results_df = pd.concat(prophet_results_df_list, ignore_index=True)

                                    if feature_metrics_list:
                                        feature_metrics = pd.concat(feature_metrics_list, ignore_index=True)


                                    # st.write(feature_metrics)

                                    # st.write(feature_metrics)


                    
                                        
                                    if all_forecast_dfs:
                                        df_forecast = pd.concat(all_forecast_dfs, ignore_index=True)





                                    



























                                        









                                        if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]):



                                            # st.write(columns_to_forecast)
                                            frequency = st.session_state.frequency
                                            # st.write(frequency)

                                            # st.write(frequency)

                                            if frequency == 'Yearly':

                                


                                                def calculate_all_beta_adjusted(df_forecast, df_final_sp):
                                                    # Initialize empty DataFrames to store results
                                                    all_detailed_output = pd.DataFrame()
                                                    all_predicted_output = pd.DataFrame()
                                                    
                                                    # Get unique groups from the forecast data
                                                    unique_groups = df_forecast['Group'].unique() if 'Group' in df_forecast.columns else [None]
                                                    
                                                    for group in unique_groups:
                                                        # Filter data for current group
                                                        if group is not None:
                                                            group_forecast = df_forecast[df_forecast['Group'] == group]
                                                            # Ensure we're using the right coefficients for this group
                                                            group_final_sp = df_final_sp[df_final_sp['Segment'] == group]
                                                        else:
                                                            group_forecast = df_forecast
                                                            group_final_sp = df_final_sp
                                                        
                                                        if len(group_final_sp) == 0:
                                                            continue  # Skip if no coefficients for this group
                                                            
                                                        # Merge with df_final_sp to get mean, std, beta
                                                        df_merged = group_forecast.merge(
                                                            group_final_sp, 
                                                            left_on='Feature', 
                                                            right_on='Variable', 
                                                            how='left'
                                                        )
                                                        
                                                        # Standardize values
                                                        df_merged['Standardized'] = (df_merged['Value'] - df_merged['Mean_Value']) / df_merged['Std_Value']
                                                        
                                                        # Multiply with beta
                                                        df_merged['Beta_adjusted_value'] = df_merged['Standardized'] * df_merged['Scaled_Beta']
                                                        
                                                        # Detailed output for this group
                                                        detailed_output = df_merged[[
                                                            'Variable', 'Date', 'Value', 'Standardized', 
                                                            'Beta_adjusted_value', 'Model_type'
                                                        ]]
                                                        
                                                        # Add group information if available
                                                        if group is not None:
                                                            detailed_output['Group'] = group
                                                        
                                                        # Sum beta-adjusted values per date
                                                        prediction_per_date = detailed_output.groupby(['Date', 'Model_type'])['Beta_adjusted_value'].sum().reset_index()
                                                        prediction_per_date.rename(columns={'Beta_adjusted_value': 'Predicted_Y_no_intercept'}, inplace=True)
                                                        
                                                        # Add intercept (Beta0_Scaled)
                                                        beta0 = group_final_sp['Beta0_Scaled'].iloc[0]
                                                        prediction_per_date['Predicted_Y'] = prediction_per_date['Predicted_Y_no_intercept'] + beta0
                                                        
                                                        # Add group information if available
                                                        if group is not None:
                                                            prediction_per_date['Group'] = group
                                                        
                                                        # Combine with overall results
                                                        all_detailed_output = pd.concat([all_detailed_output, detailed_output], ignore_index=True)
                                                        all_predicted_output = pd.concat([all_predicted_output, prediction_per_date], ignore_index=True)
                                                    
                                                    return all_detailed_output, all_predicted_output

                                                # Usage example
                                                detailed_df, predicted_df = calculate_all_beta_adjusted(df_forecast, df_final)

                                                # Rename columns and select relevant ones
                                                detailed_df = detailed_df.rename(columns={'Variable': 'Feature'})
                                                predicted_df = predicted_df.rename(columns={'Predicted_Y': "Volume"})

                                                # Include group in output if available
                                                if 'Group' in predicted_df.columns:
                                                    predicted_df = predicted_df[['Model_type', 'Date', 'Volume', 'Group']]
                                                else:
                                                    predicted_df = predicted_df[['Model_type', 'Date', 'Volume']]










            ######################################################################################## Monthly Forecasting ###########################################################################################
            ######################################################################################################################################################################################
            #######################################################################################################################################################################################

                                            if frequency == 'Monthly':
                                                    
                                                if 'df_final' in locals() and not df_final.empty:


                

                                                    def calculate_all_beta_adjusted(df_forecast, df_final_sp):
                                                        # Initialize empty DataFrames to store results
                                                        all_detailed_output = pd.DataFrame()
                                                        all_predicted_output = pd.DataFrame()
                                                        
                                                        # Get unique groups from the forecast data
                                                        unique_groups = df_forecast['Group'].unique() if 'Group' in df_forecast.columns else [None]
                                                        
                                                        for group in unique_groups:
                                                            # Filter data for current group
                                                            if group is not None:
                                                                group_forecast = df_forecast[df_forecast['Group'] == group]
                                                                # Ensure we're using the right coefficients for this group
                                                                group_final_sp = df_final_sp[df_final_sp['Segment'] == group]
                                                            else:
                                                                group_forecast = df_forecast
                                                                group_final_sp = df_final_sp
                                                            
                                                            if len(group_final_sp) == 0:
                                                                continue  # Skip if no coefficients for this group
                                                                
                                                            # Merge with df_final_sp to get mean, std, beta
                                                            df_merged = group_forecast.merge(
                                                                group_final_sp, 
                                                                left_on='Feature', 
                                                                right_on='Variable', 
                                                                how='left'
                                                            )
                                                            # st.write(df_merged)
                                                            
                                                            # Standardize values
                                                            df_merged['Standardized'] = (df_merged['Value'] - df_merged['Mean_Value']) / df_merged['Std_Value']
                                                            
                                                            # Multiply with beta
                                                            df_merged['Beta_adjusted_value'] = df_merged['Standardized'] * df_merged['Scaled_Beta']
                                                            
                                                            # Detailed output for this group
                                                            detailed_output = df_merged[[
                                                                'Variable', 'Date', 'Value', 'Standardized', 
                                                                'Beta_adjusted_value', 'Model_type'
                                                            ]]
                                                            
                                                            # Add group information if available
                                                            if group is not None:
                                                                detailed_output['Group'] = group

                                                            # detailed_output
                                                            
                                                            # Sum beta-adjusted values per date
                                                            prediction_per_date = detailed_output.groupby(['Date', 'Model_type','Group'])['Beta_adjusted_value'].sum().reset_index()
                                                            prediction_per_date.rename(columns={'Beta_adjusted_value': 'Predicted_Y_no_intercept'}, inplace=True)
                                                            # group_final_sp
                                                            # # df = group_final_sp.merge(prediction_per_date, how = 'left')
                                                            # # df
                                                            # # Add intercept (Beta0_Scaled)
                                                            # beta0 = group_final_sp['Beta0_Scaled'].iloc[0]
                                                            # # st.write(beta0)
                                                            # prediction_per_date['Predicted_Y'] = prediction_per_date['Predicted_Y_no_intercept'] + beta0
                                                            # # prediction_per_date

                                                            # For each model type in prediction_per_date, add its corresponding Beta0_Scaled
                                                            # First create a mapping of Model_type to Beta0_Scaled from group_final_sp
                                                            beta0_mapping = group_final_sp.set_index('Model_type')['Beta0_Scaled'].to_dict()
                                                            # beta0_mapping

                                                            # Add intercept (Beta0_Scaled) based on model type
                                                            prediction_per_date['Predicted_Y'] = prediction_per_date.apply(
                                                                lambda row: row['Predicted_Y_no_intercept'] + beta0_mapping.get(row['Model_type'], 0),
                                                                axis=1
                                                            )
                                                            # prediction_per_date
                                                            # Add group information if available
                                                            if group is not None:
                                                                prediction_per_date['Group'] = group


                                                            
                                                            
                                                            # Combine with overall results
                                                            all_detailed_output = pd.concat([all_detailed_output, detailed_output], ignore_index=True)
                                                            all_predicted_output = pd.concat([all_predicted_output, prediction_per_date], ignore_index=True)
                                                        
                                                        return all_detailed_output, all_predicted_output

                                                    # Usage example
                                                    detailed_df, predicted_df = calculate_all_beta_adjusted(df_forecast, df_final)

                                                    # Rename columns and select relevant ones
                                                    detailed_df = detailed_df.rename(columns={'Variable': 'Feature'})
                                                    predicted_df = predicted_df.rename(columns={'Predicted_Y': "Volume"})

                                                    predicted_df=predicted_df.rename(columns={'Group': "Segment"})

                                                    # Include group in output if available
                                                    if 'Segment' in predicted_df.columns:
                                                        predicted_df = predicted_df[['Model_type', 'Date', 'Volume', 'Segment']]
                                                    else:
                                                        predicted_df = predicted_df[['Model_type', 'Date', 'Volume']]






                                            if frequency == "Quarterly":

                                            

                                                    def calculate_all_beta_adjusted(df_forecast, df_final_sp):
                                                        # Merge with df_final_sp to get mean, std, beta
                                                        df_merged = df_forecast.merge(df_final_sp, left_on='Feature', right_on='Variable', how='left')

                                                        # Standardize values
                                                        df_merged['Standardized'] = (df_merged['Value'] - df_merged['Mean_Value']) / df_merged['Std_Value']

                                                        # Multiply with beta
                                                        df_merged['Beta_adjusted_value'] = df_merged['Standardized'] * df_merged['Scaled_Beta']

                                                        # Detailed output
                                                        detailed_output = df_merged[['Variable', 'Date', 'Value', 'Standardized', 'Beta_adjusted_value','Model_type']]

                                                        # Sum beta-adjusted values per date
                                                        prediction_per_date = detailed_output.groupby(['Date','Model_type'])['Beta_adjusted_value'].sum().reset_index()
                                                        prediction_per_date.rename(columns={'Beta_adjusted_value': 'Predicted_Y_no_intercept'}, inplace=True)
                                                        # prediction_per_date['Predicted_Y_no_intercept']

                                                        # Add intercept (Beta0_Scaled)
                                                        beta0 = df_final_sp['Beta0_Scaled'].iloc[0]
                                                        prediction_per_date['Predicted_Y'] = prediction_per_date['Predicted_Y_no_intercept'] + beta0

                                                        return detailed_output, prediction_per_date
                                                    
                                                    detailed_df, predicted_df = calculate_all_beta_adjusted(df_forecast, df_final)
                                                    detailed_df = detailed_df.rename(columns={'Variable': 'Feature'})
                                                    # detailed_df
                                                    predicted_df = predicted_df.rename(columns={'Predicted_Y': "Volume"})
                                                    predicted_df = predicted_df[['Model_type','Date', 'Volume']]
                                                    # predicted_df








                                            if frequency == "Weekly":

                                                

                                                    def calculate_all_beta_adjusted(df_forecast, df_final_sp):
                                                        # Merge with df_final_sp to get mean, std, beta
                                                        df_merged = df_forecast.merge(df_final_sp, left_on='Feature', right_on='Variable', how='left')

                                                        # Standardize values
                                                        df_merged['Standardized'] = (df_merged['Value'] - df_merged['Mean_Value']) / df_merged['Std_Value']

                                                        # Multiply with beta
                                                        df_merged['Beta_adjusted_value'] = df_merged['Standardized'] * df_merged['Scaled_Beta']

                                                        # Detailed output
                                                        detailed_output = df_merged[['Variable', 'Date', 'Value', 'Standardized', 'Beta_adjusted_value','Model_type']]

                                                        # Sum beta-adjusted values per date
                                                        prediction_per_date = detailed_output.groupby(['Date','Model_type'])['Beta_adjusted_value'].sum().reset_index()
                                                        prediction_per_date.rename(columns={'Beta_adjusted_value': 'Predicted_Y_no_intercept'}, inplace=True)
                                                        # prediction_per_date['Predicted_Y_no_intercept']

                                                        # Add intercept (Beta0_Scaled)
                                                        beta0 = df_final_sp['Beta0_Scaled'].iloc[0]
                                                        prediction_per_date['Predicted_Y'] = prediction_per_date['Predicted_Y_no_intercept'] + beta0

                                                        return detailed_output, prediction_per_date
                                                    
                                                    detailed_df, predicted_df = calculate_all_beta_adjusted(df_forecast, df_final)
                                                    detailed_df = detailed_df.rename(columns={'Variable': 'Feature'})
                                                    # detailed_df
                                                    predicted_df = predicted_df.rename(columns={'Predicted_Y': "Volume"})
                                                    predicted_df = predicted_df[['Model_type','Date', 'Volume']]
                                                    # predicted_df

                                            if frequency == "Daily":

                                                    
                            

                                                    def calculate_all_beta_adjusted(df_forecast, df_final_sp):
                                                        # Merge with df_final_sp to get mean, std, beta
                                                        df_merged = df_forecast.merge(df_final_sp, left_on='Feature', right_on='Variable', how='left')

                                                        # Standardize values
                                                        df_merged['Standardized'] = (df_merged['Value'] - df_merged['Mean_Value']) / df_merged['Std_Value']

                                                        # Multiply with beta
                                                        df_merged['Beta_adjusted_value'] = df_merged['Standardized'] * df_merged['Scaled_Beta']

                                                        # Detailed output
                                                        detailed_output = df_merged[['Variable', 'Date', 'Value', 'Standardized', 'Beta_adjusted_value','Model_type']]

                                                        # Sum beta-adjusted values per date
                                                        prediction_per_date = detailed_output.groupby(['Date','Model_type'])['Beta_adjusted_value'].sum().reset_index()
                                                        prediction_per_date.rename(columns={'Beta_adjusted_value': 'Predicted_Y_no_intercept'}, inplace=True)
                                                        # prediction_per_date['Predicted_Y_no_intercept']

                                                        # Add intercept (Beta0_Scaled)
                                                        beta0 = df_final_sp['Beta0_Scaled'].iloc[0]
                                                        prediction_per_date['Predicted_Y'] = prediction_per_date['Predicted_Y_no_intercept'] + beta0

                                                        return detailed_output, prediction_per_date
                                                    
                                                    detailed_df, predicted_df = calculate_all_beta_adjusted(df_forecast, df_final)
                                                    detailed_df = detailed_df.rename(columns={'Variable': 'Feature'})
                                                    # detailed_df
                                                    predicted_df = predicted_df.rename(columns={'Predicted_Y': "Volume"})
                                                    predicted_df = predicted_df[['Model_type','Date', 'Volume']]
                                                    # predicted_df


                                with st.expander('Review Model:'):

                                    if valid_features:




                                        # # First rename the column in expanded_results_df if it exists
                                        # if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                        #     expanded_results_df = expanded_results_df.rename(columns={'Region': 'Segment'})

                                        # # Prepare prophet_results_df with all columns from expanded_results_df
                                        # if 'prophet_results_df' in locals() or 'prophet_results_df' in globals():
                                        #     # Get all columns from expanded_results_df (if it exists)
                                        #     all_columns = expanded_results_df.columns if 'expanded_results_df' in locals() or 'expanded_results_df' in globals() else prophet_results_df.columns
                                            
                                        #     # Reindex prophet_results_df to include all columns
                                        #     prophet_results_df_full = prophet_results_df.reindex(columns=all_columns)
                                            
                                        #     # Concatenate the DataFrames
                                        #     if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                        #         combined_results = pd.concat([expanded_results_df, prophet_results_df_full], axis=0)
                                        #     else:
                                        #         combined_results = prophet_results_df_full
                                            
                                        #     # Filter for only selected models
                                        #     if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in models:
                                        #         combined_results = combined_results[combined_results['Model_type'].isin(models)]
                                            
                                        #     # Display the results
                                        #     st.dataframe(combined_results[["Segment", "Model_type", "MAPE", "R_squared", "Adjusted_R_squared"]], use_container_width=True)
                                        # else:
                                        #     # Just display expanded_results_df if prophet_results_df doesn't exist
                                        #     if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                        #         st.dataframe(expanded_results_df[["Segment", "Model_type","MAPE", "R_squared", "Adjusted_R_squared"]], use_container_width=True)




                                        # if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in models:
                                        #     st.write("\nFeature Metrics:")
                                            
                                        #     if 'ml_review' in st.session_state:
                                        #         ml_review = st.session_state.ml_review.sort_values(by=['Model_type'])

                                        #         # st.write(ml_review)
                                        #         ml_review=ml_review.round(6)
                                        #         feature_metrics_prep = feature_metrics.round(4)
                                                
                                        #         # # Standardize column names if needed (assuming index is Model_type in feature_metrics)
                                        #         # feature_metrics_prep = feature_metrics_prep.rename(columns={'index': 'Model_type'})
                                                
                                        #         # Get union of all columns from both DataFrames
                                        #         all_columns = list(set(ml_review.columns).union(set(feature_metrics_prep.columns)))
                                                
                                        #         # Reindex both DataFrames to have all columns
                                        #         ml_review_full = ml_review.reindex(columns=all_columns)
                                        #         feature_metrics_full = feature_metrics_prep.reindex(columns=all_columns)
                                                
                                        #         # Concatenate vertically
                                        #         combined_feat = pd.concat([ml_review_full, feature_metrics_full], axis=0)
                                        #         # st.write(combined)
                                                
                                        #         # Filter for only selected models
                                        #         combined_feat = combined_feat[combined_feat['Model_type'].isin(models)]
                                                
                                        #         # Fill NA values
                                        #         combined_feat = combined_feat.fillna('None')
                                                
                                        #         # Sort by Model_type for clean display
                                        #         combined_feat = combined_feat.sort_values('Model_type')
                                                
                                        #         st.dataframe(combined_feat[['Segment','Model_type','Variable','Elasticity','p_value']], use_container_width=True)



                                        # Get unique model types from both dataframes
                                        all_models = set()
                                        if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                            all_models.update(expanded_results_df['Model_type'].unique())
                                        if 'prophet_results_df' in locals() or 'prophet_results_df' in globals():
                                            all_models.update(prophet_results_df['Model_type'].unique())

                                        # Convert to sorted list
                                        all_models = sorted(list(models))

                                        # # Create multiselect filter at the top
                                        # selected_models = st.multiselect(
                                        #     "Filter by Model Type:",
                                        #     options=all_models,
                                        #     default=all_models  # Show all by default
                                        # )


                                        # Get unique segments if available
                                        all_segments = []
                                        if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                            all_segments = expanded_results_df['Region'].unique()
                                        elif 'prophet_results_df' in locals() or 'prophet_results_df' in globals():
                                            all_segments = prophet_results_df['Segment'].unique()

                                        # Convert to sorted list
                                        all_segments = sorted(list(all_segments))

                                        # Create filters in columns layout
                                        col1, col2 = st.columns(2)

                                        # Model type filter (always shown)
                                        with col1:
                                            selected_models = st.multiselect(
                                                "Filter by Model Type:",
                                                options=all_models,
                                                default=all_models  # Show all by default
                                            )

                                        # Segment filter (only shown if multiple segments exist)
                                        with col2:
                                            if len(all_segments) > 1:
                                                selected_segments = st.multiselect(
                                                    "Filter by Segment:",
                                                    options=all_segments,
                                                    default=all_segments  # Show all by default
                                                )
                                            else:
                                                selected_segments = all_segments  # Use all (which is just one)











                                        # # Get unique model types from both dataframes
                                        # all_models = set()
                                        # stacked_models = set()

                                        # if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                        #     expanded_models = expanded_results_df['Model_type'].unique()
                                        #     all_models.update(expanded_models)
                                        #     # Find models starting with "Stacked"
                                        #     stacked_models.update([model for model in expanded_models if str(model).startswith('Stacked')])
                                            
                                        # if 'prophet_results_df' in locals() or 'prophet_results_df' in globals():
                                        #     all_models.update(prophet_results_df['Model_type'].unique())

                                        # # Convert to sorted list
                                        # all_models = sorted(list(all_models))

                                        # # Set default selection - all models plus any stacked models
                                        # default_selection = all_models.copy()
                                        # if stacked_models:
                                        #     # Ensure stacked models are selected (in case they're not already in all_models)
                                        #     default_selection.extend([model for model in stacked_models if model not in default_selection])
                                        #     default_selection = sorted(list(set(default_selection)))  # Remove duplicates and sort

                                        # # Create multiselect filter at the top
                                        # selected_models = st.multiselect(
                                        #     "Filter by Model Type:",
                                        #     options=all_models,
                                        #     default=default_selection  # Show all by default, with stacked models included
                                        # )






                                        def color_mape(val):
                                            """
                                            Takes a scalar and returns a string with CSS styling
                                            if the MAPE value is greater than 30%
                                            """
                                            try:
                                                # Remove '%' and convert to float
                                                num = float(val.strip('%'))
                                                if num > 30:
                                                    return 'background-color: #ffcccc'  # Light red
                                                

                                                # if num < 5:
                                                #     return 'background-color: lightgreen'  # Light red
                                            except:
                                                pass
                                            return ''


                                        # First rename the column in expanded_results_df if it exists
                                        if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                            expanded_results_df = expanded_results_df.rename(columns={'Region': 'Segment'})

                                        # Prepare prophet_results_df with all columns from expanded_results_df
                                        if 'prophet_results_df' in locals() or 'prophet_results_df' in globals():
                                            # Get all columns from expanded_results_df (if it exists)
                                            all_columns = expanded_results_df.columns if 'expanded_results_df' in locals() or 'expanded_results_df' in globals() else prophet_results_df.columns
                                            
                                            # Reindex prophet_results_df to include all columns
                                            prophet_results_df_full = prophet_results_df.reindex(columns=all_columns)
                                            
                                            # Concatenate the DataFrames
                                            if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                                combined_results = pd.concat([expanded_results_df, prophet_results_df_full], axis=0)
                                            else:
                                                combined_results = prophet_results_df_full

                                
                                            
                                            # Filter for only selected models (from the multiselect)
                                            combined_results = combined_results[combined_results['Model_type'].isin(selected_models) &
                                                (combined_results['Segment'].isin(selected_segments))]
                                            
                                            # Display the results
                                            # st.dataframe(combined_results[["Segment", "Model_type", "MAPE", "R_squared", "Adjusted_R_squared"]], use_container_width=True)


                                            display_df = combined_results[["Segment", "Model_type", "MAPE", "R_squared", "Adjusted_R_squared"]].copy()
        
                                            # Convert columns to numeric (coerce errors to NaN)
                                            for col in ["MAPE", "R_squared", "Adjusted_R_squared"]:
                                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                                            
                                            # Format the percentage columns
                                            display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
                                            display_df["R_squared"] = display_df["R_squared"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
                                            display_df["Adjusted_R_squared"] = display_df["Adjusted_R_squared"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
                                            display_df = display_df.reset_index(drop=True)


                                            styled_df = display_df.style.applymap(color_mape, subset=['MAPE'])
                                            
                                            st.dataframe(styled_df, use_container_width=True)




                                        else:
                                            # Just display expanded_results_df if prophet_results_df doesn't exist
                                            if 'expanded_results_df' in locals() or 'expanded_results_df' in globals():
                                                # Filter for selected models
                                                filtered_expanded_results = expanded_results_df[expanded_results_df['Model_type'].isin(selected_models)&
                                                (combined_results['Segment'].isin(selected_segments))]


                                                # st.dataframe(filtered_expanded_results[["Segment", "Model_type","MAPE", "R_squared", "Adjusted_R_squared"]], use_container_width=True)

                                                # Create a copy to avoid modifying the original dataframe
                                                display_df = filtered_expanded_results[["Segment", "Model_type", "MAPE", "R_squared", "Adjusted_R_squared"]].copy()
                                                
                                                # Convert columns to numeric (coerce errors to NaN)
                                                for col in ["MAPE", "R_squared", "Adjusted_R_squared"]:
                                                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                                                
                                                # Format the percentage columns
                                                display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
                                                display_df["R_squared"] = display_df["R_squared"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
                                                display_df["Adjusted_R_squared"] = display_df["Adjusted_R_squared"].apply(lambda x: f"{float(x)*100:.1f}%" if pd.notnull(x) else "N/A")
                                                display_df = display_df.reset_index(drop=True)


                                                styled_df = display_df.style.applymap(color_mape, subset=['MAPE'])
                                                
                                                st.dataframe(styled_df, use_container_width=True)






                                        if any(m in selected_models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in selected_models:
                                            st.write("\nFeature Metrics:")
                                            
                                            if 'ml_review' in st.session_state:
                                                ml_review = st.session_state.ml_review.sort_values(by=['Model_type'])
                                                ml_review = ml_review.round(6)

                                                

                                                ml_review=ml_review.dropna()


                                                # if ml_review['Segment'].isna().any():
                                                #     st.warning("Warning: There are NaN values in the Segment column. These rows will be filled with NaN for Prev_Year_Target_Mean.")

                                                # # Safely map with handling for NaN/missing segments
                                                # ml_review['Prev_Year_Target_Mean'] = ml_review['Segment'].map(
                                                #     lambda group: forecast_results.get(group, {}).get('previous_year_mean', float('nan')))

                                                # st.write(ml_review)

                                                ml_review=ml_review.rename(columns={'Beta0_Scaled':"intercept"})

                                                ml_review=ml_review.rename(columns={'Region_Vol_Mean':"Target_Mean"})

                                                # Create a DataFrame from feature_forecasts
                                                feature_stats = pd.DataFrame({
                                                    'Variable': feature_forecasts.keys(),
                                                    'Feature_Mean': [v['overall_mean'] for v in feature_forecasts.values()],
                                                    'Feature_Std': [v['overall_std'] for v in feature_forecasts.values()],
                                                    'Prev_Year_Feature_Mean': [v['previous_year_mean'] for v in feature_forecasts.values()],
                                                })


                                                ml_review = ml_review.merge(
                                                    feature_stats,
                                                    on='Variable',  # Ensure this column matches in both DataFrames
                                                    how='left'      # Keep all rows from ml_review
                                                )
                                                
                                                st.write(ml_review)

                                            



                                                ml_review['Prev_Year_Target_Mean'] = ml_review['Segment'].map(
                                                    lambda group: forecast_results.get(group, {}).get('previous_year_mean', np.nan)
                                                )



                                                # st.write(ml_review)


                                                # st.write(ml_review)
                                                feature_metrics_prep = feature_metrics.round(4)

                                                # st.write(feature_metrics_prep)
                                                
                                                # Get union of all columns from both DataFrames
                                                all_columns = list(set(ml_review.columns).union(set(feature_metrics_prep.columns)))
                                                
                                                # Reindex both DataFrames to have all columns
                                                ml_review_full = ml_review.reindex(columns=all_columns)
                                                feature_metrics_full = feature_metrics_prep.reindex(columns=all_columns)
                                                
                                                # Concatenate vertically
                                                combined_feat = pd.concat([ml_review_full, feature_metrics_full], axis=0)

                                                # combined_feat
                                                
                                                # Filter for only selected models (from the multiselect)
                                                combined_feat = combined_feat[combined_feat['Model_type'].isin(selected_models) &
                                                (combined_feat['Segment'].isin(selected_segments))]
                                                
                                                # Fill NA values
                                                combined_feat = combined_feat.fillna('None')
                                                
                                                # Sort by Model_type for clean display
                                                combined_feat = combined_feat.sort_values('Model_type')

                                                overall_data=combined_feat.copy()
                                                
                                                # Pivot the data to have Variables as columns
                                                try:
                                                    # Create separate dataframes for elasticity and p-values
                                                    elasticity_df = combined_feat.pivot_table(
                                                        index=['Segment', 'Model_type'],
                                                        columns='Variable',
                                                        values='Elasticity',
                                                        aggfunc='first'  # Takes the first value if there are duplicates
                                                    ).reset_index()
                                                    
                                                    pvalue_df = combined_feat.pivot_table(
                                                        index=['Segment', 'Model_type'],
                                                        columns='Variable',
                                                        values='p_value',
                                                        aggfunc='first'
                                                    ).reset_index()
                                                    
                                                    # Add suffix to distinguish metrics
                                                    elasticity_df.columns = [f"{col}_elasticity" if col not in ['Segment', 'Model_type'] else col for col in elasticity_df.columns]
                                                    pvalue_df.columns = [f"{col}_pvalue" if col not in ['Segment', 'Model_type'] else col for col in pvalue_df.columns]
                                                    
                                                    # Merge the two dataframes
                                                    pivoted_df = pd.merge(elasticity_df, pvalue_df, on=['Segment', 'Model_type'])
                                                    
                                                    # Display the pivoted dataframe
                                                    st.dataframe(pivoted_df, use_container_width=True)
                                                    
                                                except Exception as e:
                                                    st.warning(f"Could not pivot the data. Showing original format. Error: {e}")
                                                    st.dataframe(combined_feat[['Segment','Model_type','Variable','Elasticity','p_value']], use_container_width=True)

                                                
                                                
                                                
                                                # st.dataframe(overall_data[['Segment','Model_type','Variable','Elasticity','p_value','Feature_Mean','Feature_Std','Target_Column','Target_Mean','intercept','Prev_Year_Target_Mean','Prev_Year_Feature_Mean']], use_container_width=True)

                                                # overall_data.columns











                                st.markdown('<hr class="thick">', unsafe_allow_html=True)


                                # # Initialize session state for elasticity data if it doesn't exist
                                # if 'elasticity_data' not in st.session_state:
                                #     st.session_state.elasticity_data = {}

                                # # Initialize storage if it doesn't exist
                                # if 'saved_models' not in st.session_state:
                                #     st.session_state.saved_models = {}

                                # # After your model calculations
                                # with st.container():
                                #     st.markdown("#### Save Current Results:")
                                    
                                #     custom_name = st.text_input(
                                #         "Enter a name for this model set:",
                                #         key="current_model_set_name",
                                #         placeholder="My Model Results"
                                #     )
                                    
                                #     if st.button("💾 SAVE CURRENT RESULTS", key="save_current_results"):
                                #         if not custom_name.strip():
                                #             st.warning("Please enter a name before saving")
                                #         else:
                                #             # Get all unique segments from all data sources
                                #             segments = set()
                                #             if 'combined_results' in locals():
                                #                 segments.update(combined_results['Segment'].unique())
                                #             if 'combined_feat' in locals():
                                #                 segments.update(combined_feat['Segment'].unique())
                                #             if 'forecast_results' in locals():
                                #                 segments.update(forecast_results.keys())
                                            
                                #             for segment in segments:
                                #                 # Get all models for this segment from all sources
                                #                 models_in_segment = set()
                                #                 if 'combined_results' in locals():
                                #                     models_in_segment.update(
                                #                         combined_results[combined_results['Segment'] == segment]['Model_type'].unique()
                                #                     )
                                #                 if 'combined_feat' in locals():
                                #                     models_in_segment.update(
                                #                         combined_feat[combined_feat['Segment'] == segment]['Model_type'].unique()
                                #                     )
                                #                 if 'forecast_results' in locals() and segment in forecast_results:
                                #                     models_in_segment.add(forecast_results[segment]['Model_type'])
                                                
                                #                 for model in models_in_segment:
                                #                     model_key = f"{custom_name}_{model}_{segment}"
                                #                     model_data = {
                                #                         'custom_name': custom_name,
                                #                         'model_type': model,
                                #                         'segment': segment,
                                #                     }
                                                    
                                #                     # 1. Add metrics from combined_results (for all models)
                                #                     if 'combined_results' in locals():
                                #                         model_metrics = combined_results[
                                #                             (combined_results['Segment'] == segment) & 
                                #                             (combined_results['Model_type'] == model)
                                #                         ]
                                #                         if not model_metrics.empty:
                                #                             model_data.update({
                                #                                 'MAPE': model_metrics['MAPE'].values[0] if 'MAPE' in model_metrics else None,
                                #                                 'R_squared': model_metrics['R_squared'].values[0] if 'R_squared' in model_metrics else None,
                                #                                 'Adjusted_R_squared': model_metrics['Adjusted_R_squared'].values[0] if 'Adjusted_R_squared' in model_metrics else None
                                #                             })
                                                    
                                #                     # 2. Add feature information (handled differently for Prophet vs other models)
                                #                     features_used = []
                                #                     feature_elasticities = []
                                                    
                                #                     # For Prophet models - get from feature_forecasts
                                #                     if model == 'Prophet' and 'forecast_results' in locals() and segment in forecast_results:
                                #                         if 'feature_forecasts' in forecast_results[segment]:
                                #                             prophet_features = forecast_results[segment]['feature_forecasts']
                                #                             features_used.extend(prophet_features.keys())
                                #                             # Store the actual forecast values for each feature
                                #                             model_data['feature_forecasts'] = {
                                #                                 feat: values.tolist() if hasattr(values, 'tolist') else values
                                #                                 for feat, values in prophet_features.items()
                                #                             }
                                                    
                                #                     # For all models - get elasticities from combined results
                                #                     if 'combined_results' in locals():
                                #                         combined_features = combined_results[
                                #                             (combined_results['Segment'] == segment) &
                                #                             (combined_results['Model_type'] == model)
                                #                         ]
                                #                         if not combined_features.empty and 'Variable' in combined_features:
                                #                             feature_elasticities.extend(
                                #                                 combined_features[['Variable', 'Elasticity', 'p_value']]
                                #                                 .to_dict('records')
                                #                             )
                                #                             features_used.extend(combined_features['Variable'].unique().tolist())
                                                    
                                #                     # For non-Prophet models - get from ml_review if available
                                #                     if 'combined_feat' in locals():
                                #                         combined_featelas = combined_feat[
                                #                             (combined_feat['Segment'] == segment) &
                                #                             (combined_feat['Model_type'] == model)
                                #                         ]
                                #                         if not combined_featelas.empty:
                                #                             feature_elasticities.extend(
                                #                                 combined_featelas[['Variable', 'Elasticity', 'p_value']]
                                #                                 .to_dict('records')
                                #                             )
                                #                             features_used.extend(combined_featelas['Variable'].unique().tolist())
                                                    
                                #                     # Store the combined feature information
                                #                     model_data['features_used'] = list(set(features_used))
                                #                     if feature_elasticities:
                                #                         model_data['feature_elasticities'] = feature_elasticities
                                                    
                                #                     # 3. Add forecast/prediction data
                                #                     if model == 'Prophet' and 'forecast_results' in locals() and segment in forecast_results:
                                #                         model_data['forecast_data'] = {
                                #                             'actual': forecast_results[segment]['actual_volume'].tolist(),
                                #                             'dates': forecast_results[segment]['actual_dates'].astype(str).tolist(),
                                #                             'future_forecast': forecast_results[segment].get('prophet_future_forecast', []).tolist()
                                #                         }
                                #                     elif 'predicted_df' in locals():
                                #                         preds = predicted_df[
                                #                             (predicted_df['Segment'] == segment) &
                                #                             (predicted_df['Model_type'] == model)
                                #                         ]
                                #                         if not preds.empty:
                                #                             model_data['predictions'] = preds.to_dict('records')
                                                    
                                #                     # Store the model data
                                #                     st.session_state.saved_models[model_key] = model_data
                                                    
                                #                     # Update elasticity data ONLY for this model
                                #                     if 'combined_feat' in locals():
                                #                         model_features = combined_feat[
                                #                             (combined_feat['Segment'] == segment) & 
                                #                             (combined_feat['Model_type'] == model)
                                #                         ]
                                                        
                                #                         if not model_features.empty:
                                #                             st.session_state.elasticity_data[model_key] = []
                                #                             for _, row in model_features.iterrows():
                                #                                 st.session_state.elasticity_data[model_key].append({
                                #                                     'Feature': row['Variable'],
                                #                                     'Elasticity': row['Elasticity'],
                                #                                     'p-value': row.get('p_value', None)
                                #                                 })
                                            
                                #             st.success(f"✅ Saved results as: {custom_name}")   



                                # Initialize session state
                                if 'saved_models' not in st.session_state:
                                    st.session_state.saved_models = {}
                                if 'elasticity_data' not in st.session_state:
                                    st.session_state.elasticity_data = {}

                                # Save Current Results Section
                                with st.container():
                                    st.markdown("#### Save Current Results:")
                                    
                                    custom_name = st.text_input(
                                        "Enter a base name for these results:",
                                        key="current_model_set_name",
                                        placeholder="My Analysis"
                                    )
                                    
                                    # if st.button("💾 SAVE CURRENT RESULTS", key="save_current_results"):
                                    #     if not custom_name.strip():
                                    #         st.warning("Please enter a name before saving")
                                    #     else:
                                    #         saved_count = 0
                                            
                                    #         # Process each segment and model in selected_models
                                    #         for segment in set(combined_results['Segment'].unique()):
                                    #             for model in selected_models:
                                    #                 # Create unique model-specific name
                                    #                 segment_safe_name = segment.replace(' ', '_').replace('/', '_')  # Make segment name filesystem-safe
                                    #                 model_specific_name = f"{custom_name}_{segment_safe_name}_{model.replace(' ', '_')}"
                                    #                 model_key = f"{model_specific_name}"
                                                    
                                    #                 # Prepare model data with individual naming
                                    #                 model_data = {
                                    #                     'base_name': custom_name,  # Original name user entered
                                    #                     'custom_name': model_specific_name,  # Model-specific name
                                    #                     'model_type': model,
                                    #                     'segment': segment,
                                    #                     # 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    #                 }
                                                    
                                    #                 # 1. Add metrics from combined_results (for all models)
                                    #                 if 'combined_results' in locals():
                                    #                     model_metrics = combined_results[
                                    #                         (combined_results['Segment'] == segment) & 
                                    #                         (combined_results['Model_type'] == model)
                                    #                     ]
                                    #                     if not model_metrics.empty:
                                    #                         model_data.update({
                                    #                             'MAPE': model_metrics['MAPE'].values[0] if 'MAPE' in model_metrics else None,
                                    #                             'R_squared': model_metrics['R_squared'].values[0] if 'R_squared' in model_metrics else None,
                                    #                             'Adjusted_R_squared': model_metrics['Adjusted_R_squared'].values[0] if 'Adjusted_R_squared' in model_metrics else None
                                    #                         })



                                    #                 # 2. Add the new metrics from overall_data
                                    #                 if 'overall_data' in locals():
                                    #                     model_overall = overall_data[
                                    #                         (overall_data['Segment'] == segment) &
                                    #                         (overall_data['Model_type'] == model)
                                    #                     ]
                                                        
                                    #                     if not model_overall.empty:
                                    #                         # Store all the new metrics
                                    #                         model_data.update({
                                    #                             'feature_metrics': model_overall[[
                                    #                                 'Variable', 'Elasticity', 'p_value', 'Feature_Mean', 
                                    #                                 'Feature_Std', 'Target_Column', 'Target_Mean',
                                    #                                 'intercept', 'Prev_Year_Target_Mean', 'Prev_Year_Feature_Mean'
                                    #                             ]].to_dict('records')
                                    #                         })
                                                    
                                    #                 # 2. Add feature information - now using pivoted format
                                    #                 features_used = []
                                    #                 feature_elasticities = {}
                                                    
                                    #                 # For Prophet models - get from feature_forecasts
                                    #                 if model == 'Prophet' and 'forecast_results' in locals() and segment in forecast_results:
                                    #                     if 'feature_forecasts' in forecast_results[segment]:
                                    #                         prophet_features = forecast_results[segment]['feature_forecasts']
                                    #                         features_used.extend(prophet_features.keys())
                                    #                         # Store the actual forecast values for each feature
                                    #                         model_data['feature_forecasts'] = {
                                    #                             feat: values.tolist() if hasattr(values, 'tolist') else values
                                    #                             for feat, values in prophet_features.items()
                                    #                         }
                                                    
                                    #                 # For all models - get elasticities in pivoted format
                                    #                 if 'pivoted_df' in locals():
                                    #                     model_pivoted = pivoted_df[
                                    #                         (pivoted_df['Segment'] == segment) &
                                    #                         (pivoted_df['Model_type'] == model)
                                    #                     ]
                                                        
                                    #                     if not model_pivoted.empty:
                                    #                         # Extract all feature columns (ending with _elasticity or _pvalue)
                                    #                         feature_cols = [col for col in model_pivoted.columns 
                                    #                                     if col.endswith('_elasticity') or col.endswith('_pvalue')]
                                                            
                                    #                         # Group by feature base name (without suffix)
                                    #                         features = set(col.rsplit('_', 1)[0] for col in feature_cols)
                                                            
                                    #                         for feature in features:
                                    #                             elasticity_col = f"{feature}_elasticity"
                                    #                             pvalue_col = f"{feature}_pvalue"
                                                                
                                    #                             if elasticity_col in model_pivoted.columns:
                                    #                                 feature_elasticities[feature] = {
                                    #                                     'Elasticity': model_pivoted[elasticity_col].values[0],
                                    #                                     'p_value': model_pivoted[pvalue_col].values[0] if pvalue_col in model_pivoted.columns else None
                                    #                                 }
                                    #                                 features_used.append(feature)
                                                    
                                    #                 # Store the combined feature information
                                    #                 model_data['features_used'] = list(set(features_used))
                                    #                 if feature_elasticities:
                                    #                     model_data['feature_elasticities'] = feature_elasticities
                                                    
                                    #                 # 3. Add forecast/prediction data
                                    #                 if model == 'Prophet' and 'forecast_results' in locals() and segment in forecast_results:
                                    #                     model_data['forecast_data'] = {
                                    #                         'actual': forecast_results[segment]['actual_volume'].tolist(),
                                    #                         'dates': forecast_results[segment]['actual_dates'].astype(str).tolist(),
                                    #                         'future_forecast': forecast_results[segment].get('prophet_future_forecast', []).tolist()
                                    #                     }
                                    #                 elif 'predicted_df' in locals():
                                    #                     preds = predicted_df[
                                    #                         (predicted_df['Segment'] == segment) &
                                    #                         (predicted_df['Model_type'] == model)
                                    #                     ]
                                    #                     if not preds.empty:
                                    #                         model_data['predictions'] = preds.to_dict('records')
                                                    
                                    #                 # Store the model data (overwrites if same key exists)
                                    #                 st.session_state.saved_models[model_key] = model_data
                                                    
                                    #                 # Update elasticity data
                                    #                 if 'pivoted_df' in locals():
                                    #                     model_pivoted = pivoted_df[
                                    #                         (pivoted_df['Segment'] == segment) & 
                                    #                         (pivoted_df['Model_type'] == model)
                                    #                     ]
                                                        
                                    #                     if not model_pivoted.empty:
                                    #                         st.session_state.elasticity_data[model_key] = {}
                                    #                         for col in model_pivoted.columns:
                                    #                             if col.endswith('_elasticity'):
                                    #                                 feature = col.replace('_elasticity', '')
                                    #                                 pvalue_col = f"{feature}_pvalue"
                                    #                                 st.session_state.elasticity_data[model_key][feature] = {
                                    #                                     'Elasticity': model_pivoted[col].values[0],
                                    #                                     'p-value': model_pivoted[pvalue_col].values[0] if pvalue_col in model_pivoted.columns else None
                                    #                                 }
                                                    
                                    #                 saved_count += 1
                                        
                                    #     if saved_count > 0:
                                    #         st.success(f"✅ Saved {saved_count} model results as: {custom_name}")
                                    #         st.session_state.last_saved_set = custom_name  # Store last saved set name
                                    #     else:
                                    #         st.warning("No models matched the current selection to save")



                                    if st.button("💾 SAVE CURRENT RESULTS", key="save_current_results"):
                                        if not custom_name.strip():
                                            st.warning("Please enter a name before saving")
                                        else:
                                            saved_count = 0
                                            
                                            # Process each segment and model in selected_models
                                            for segment in set(combined_results['Segment'].unique()):
                                                for model in selected_models:
                                                    # Create unique model-specific name
                                                    segment_safe_name = segment.replace(' ', '_').replace('/', '_')
                                                    model_specific_name = f"{custom_name}_{segment_safe_name}_{model.replace(' ', '_')}"
                                                    model_key = f"{model_specific_name}"
                                                    
                                                    # Prepare model data with individual naming
                                                    model_data = {
                                                        'base_name': custom_name,
                                                        'custom_name': model_specific_name,
                                                        'model_type': model,
                                                        'segment': segment,
                                                    }
                                                    
                                                    # 1. Add metrics from combined_results
                                                    if 'combined_results' in locals():
                                                        model_metrics = combined_results[
                                                            (combined_results['Segment'] == segment) & 
                                                            (combined_results['Model_type'] == model)
                                                        ]
                                                        if not model_metrics.empty:
                                                            model_data.update({
                                                                'MAPE': model_metrics['MAPE'].values[0] if 'MAPE' in model_metrics else None,
                                                                'R_squared': model_metrics['R_squared'].values[0] if 'R_squared' in model_metrics else None,
                                                                'Adjusted_R_squared': model_metrics['Adjusted_R_squared'].values[0] if 'Adjusted_R_squared' in model_metrics else None
                                                            })
                                                    
                                                    # 2. Add the new metrics from overall_data
                                                    if 'overall_data' in locals():
                                                        model_overall = overall_data[
                                                            (overall_data['Segment'] == segment) &
                                                            (overall_data['Model_type'] == model)
                                                        ]
                                                        
                                                        if not model_overall.empty:
                                                            # Store all the new metrics
                                                            model_data.update({
                                                                'feature_metrics': model_overall[[
                                                                    'Variable', 'Elasticity', 'p_value', 'Feature_Mean', 
                                                                    'Feature_Std', 'Target_Column', 'Target_Mean',
                                                                    'intercept', 'Prev_Year_Target_Mean', 'Prev_Year_Feature_Mean'
                                                                ]].to_dict('records')
                                                            })
                                                    
                                                    # 3. Add feature information
                                                    features_used = []
                                                    feature_elasticities = {}
                                                    
                                                    # For Prophet models
                                                    if model == 'Prophet' and 'forecast_results' in locals() and segment in forecast_results:
                                                        if 'feature_forecasts' in forecast_results[segment]:
                                                            prophet_features = forecast_results[segment]['feature_forecasts']
                                                            features_used.extend(prophet_features.keys())
                                                            model_data['feature_forecasts'] = {
                                                                feat: values.tolist() if hasattr(values, 'tolist') else values
                                                                for feat, values in prophet_features.items()
                                                            }
                                                    
                                                    # For all models - get elasticities
                                                    if 'pivoted_df' in locals():
                                                        model_pivoted = pivoted_df[
                                                            (pivoted_df['Segment'] == segment) &
                                                            (pivoted_df['Model_type'] == model)
                                                        ]
                                                        
                                                        if not model_pivoted.empty:
                                                            feature_cols = [col for col in model_pivoted.columns 
                                                                        if col.endswith('_elasticity') or col.endswith('_pvalue')]
                                                            
                                                            features = set(col.rsplit('_', 1)[0] for col in feature_cols)
                                                            
                                                            for feature in features:
                                                                elasticity_col = f"{feature}_elasticity"
                                                                pvalue_col = f"{feature}_pvalue"
                                                                
                                                                if elasticity_col in model_pivoted.columns:
                                                                    feature_elasticities[feature] = {
                                                                        'Elasticity': model_pivoted[elasticity_col].values[0],
                                                                        'p_value': model_pivoted[pvalue_col].values[0] if pvalue_col in model_pivoted.columns else None
                                                                    }
                                                                    features_used.append(feature)
                                                    
                                                    model_data['features_used'] = list(set(features_used))
                                                    if feature_elasticities:
                                                        model_data['feature_elasticities'] = feature_elasticities
                                                    
                                                    # 4. Add forecast/prediction data
                                                    if model == 'Prophet' and 'forecast_results' in locals() and segment in forecast_results:
                                                        model_data['forecast_data'] = {
                                                            'actual': forecast_results[segment]['actual_volume'].tolist(),
                                                            'dates': forecast_results[segment]['actual_dates'].astype(str).tolist(),
                                                            'future_forecast': forecast_results[segment].get('prophet_future_forecast', []).tolist()
                                                        }
                                                    elif 'predicted_df' in locals():
                                                        preds = predicted_df[
                                                            (predicted_df['Segment'] == segment) &
                                                            (predicted_df['Model_type'] == model)
                                                        ]
                                                        if not preds.empty:
                                                            model_data['predictions'] = preds.to_dict('records')
                                                    
                                                    # Store the complete model data
                                                    st.session_state.saved_models[model_key] = model_data
                                                    
                                                    # Update elasticity data
                                                    if 'pivoted_df' in locals():
                                                        model_pivoted = pivoted_df[
                                                            (pivoted_df['Segment'] == segment) & 
                                                            (pivoted_df['Model_type'] == model)
                                                        ]
                                                        
                                                        if not model_pivoted.empty:
                                                            st.session_state.elasticity_data[model_key] = {}
                                                            for col in model_pivoted.columns:
                                                                if col.endswith('_elasticity'):
                                                                    feature = col.replace('_elasticity', '')
                                                                    pvalue_col = f"{feature}_pvalue"
                                                                    st.session_state.elasticity_data[model_key][feature] = {
                                                                        'Elasticity': model_pivoted[col].values[0],
                                                                        'p-value': model_pivoted[pvalue_col].values[0] if pvalue_col in model_pivoted.columns else None
                                                                    }
                                                    
                                                    saved_count += 1

                                            if saved_count > 0:
                                                st.success(f"✅ Saved {saved_count} model results as: {custom_name}")
                                                st.session_state.last_saved_set = custom_name
                                            else:
                                                st.warning("No models matched the current selection to save")












                                with st.expander("SAVED MODELS:"):

                            

                                    # Display Saved Results - Showing Individual Models
                                    if st.session_state.saved_models:
                                        st.markdown("#### Saved Model Results")
                                        
                                        # Create display dataframe with model-specific names
                                        display_data = []
                                        for model_key, data in st.session_state.saved_models.items():
                                            display_data.append({
                                                'Base Name': data['base_name'],
                                                'Model Name': data['custom_name'],  # Showing model-specific name
                                                'Type': data['model_type'],
                                                'Segment': data['segment'],
                                                # 'Saved At': data['timestamp'],
                                                'MAPE': data.get('MAPE', 'N/A'),
                                                'R²': data.get('R_squared', 'N/A'),
                                                'AdjR²':data.get('Adjusted_R_squared', 'N/A'),
                                                'features_used':data['features_used']
                                            })
                                        
                                        display_df = pd.DataFrame(display_data)
                                        
                                        # # Add filters
                                        # col1, col2 = st.columns(2)
                                        # with col1:
                                        #     name_filter = st.multiselect(
                                        #         "Filter by base name:",
                                        #         options=display_df['Base Name'].unique(),
                                        #         default=display_df['Base Name'].unique()
                                        #     )
                                        # with col2:
                                        #     type_filter = st.multiselect(
                                        #         "Filter by model type:",
                                        #         options=display_df['Type'].unique(),
                                        #         default=display_df['Type'].unique()
                                        #     )
                                        
                                        # # Apply filters
                                        # filtered_df = display_df[
                                        #     (display_df['Base Name'].isin(name_filter)) & 
                                        #     (display_df['Type'].isin(type_filter))
                                        # ]
                                        
                                        # Display with model-specific names
                                        st.dataframe(
                                            display_df.sort_values(['Base Name', 'Model Name']),
                                            column_config={
                                                "Model Name": "Saved Model",
                                                "Type": "Model Type"
                                            },
                                            use_container_width=True
                                        )
                                            
                                        # # Display elasticities from session state - in pivoted format
                                        # st.write("#### Feature Elasticities Summary")
                                        
                                        # if st.session_state.elasticity_data:
                                        #     # Combine all elasticity data
                                        #     all_elasticity_data = []
                                        #     for model_key, features in st.session_state.elasticity_data.items():
                                        #         data = st.session_state.saved_models[model_key]
                                        #         # Only include if in selected_models (if selected_models exists)
                                        #         if 'selected_models' not in locals() or data['model_type'] in selected_models:
                                        #             for feature_name, feature_data in features.items():
                                        #                 all_elasticity_data.append({
                                        #                     # 'Base Name':data['base_name'],
                                        #                     'Custom Name': data['custom_name'],
                                        #                     'Model Type': data['model_type'],
                                        #                     'Segment': data['segment'],
                                        #                     'Feature': feature_name,
                                        #                     'Elasticity': feature_data.get('Elasticity'),
                                        #                     'p-value': feature_data.get('p-value')
                                        #                 })
                                            
                                        #     if all_elasticity_data:
                                        #         elasticity_df = pd.DataFrame(all_elasticity_data)
                                                
                                        #         # Convert to numeric safely
                                        #         elasticity_df['Elasticity'] = pd.to_numeric(elasticity_df['Elasticity'], errors='coerce')
                                        #         elasticity_df['p-value'] = pd.to_numeric(elasticity_df['p-value'], errors='coerce')
                                                
                                        #         # Safe significance calculation
                                        #         def get_significance(p_value):
                                        #             try:
                                        #                 p = float(p_value)
                                        #                 if p < 0.001:
                                        #                     return '***'
                                        #                 elif p < 0.01:
                                        #                     return '**'
                                        #                 elif p < 0.05:
                                        #                     return '*'
                                        #                 return ''
                                        #             except (TypeError, ValueError):
                                        #                 return ''
                                                
                                        #         elasticity_df['Significance'] = elasticity_df['p-value'].apply(get_significance)
                                                
                                        #         # Pivot the data to match our display format
                                        #         try:
                                        #             pivoted_elasticity = elasticity_df.pivot_table(
                                        #                 index=['Custom Name', 'Model Type', 'Segment'],
                                        #                 columns='Feature',
                                        #                 values=['Elasticity', 'p-value'],
                                        #                 aggfunc='first'
                                        #             )
                                                    
                                        #             # Flatten multi-index columns
                                        #             pivoted_elasticity.columns = [f"{col[1]}_{col[0]}" for col in pivoted_elasticity.columns]
                                        #             pivoted_elasticity = pivoted_elasticity.reset_index()
                                                    
                                                    
                                        #             st.dataframe(
                                        #                 pivoted_elasticity,
                                        #                 use_container_width=True,
                                        #                 column_config={
                                        #                     col: st.column_config.NumberColumn(format="%.4f") 
                                        #                     for col in pivoted_elasticity.columns 
                                        #                     if 'Elasticity' in col or 'p-value' in col
                                        #                 }
                                        #             )
                                        #         except Exception as e:
                                        #             st.warning(f"Could not pivot elasticity data. Showing raw format. Error: {e}")
                                        #             st.dataframe(
                                        #                 elasticity_df,
                                        #                 use_container_width=True,
                                        #                 column_config={
                                        #                     "Elasticity": st.column_config.NumberColumn(format="%.4f"),
                                        #                     "p-value": st.column_config.NumberColumn(format="%.4f")
                                        #                 }
                                        #             )
                                        #     else:
                                        #         st.warning("No feature elasticities data available for selected models")
                                        # else:
                                        #     st.warning("No feature elasticities data available")



                                        # Display elasticities from session state - in pivoted format
                                        st.write("#### Feature Elasticities Summary")

                                        if st.session_state.elasticity_data:
                                            # Combine all elasticity data
                                            all_elasticity_data = []
                                            for model_key, features in st.session_state.elasticity_data.items():
                                                data = st.session_state.saved_models[model_key]
                                                # Only filter for display purposes, but keep all data in session_state
                                                if 'selected_models' not in st.session_state or data['model_type'] in st.session_state.selected_models:
                                                    for feature_name, feature_data in features.items():
                                                        all_elasticity_data.append({
                                                            'Custom Name': data['custom_name'],
                                                            'Model Type': data['model_type'],
                                                            'Segment': data['segment'],
                                                            'Feature': feature_name,
                                                            'Elasticity': feature_data.get('Elasticity'),
                                                            'p-value': feature_data.get('p-value')
                                                        })
                                            
                                            if all_elasticity_data:
                                                elasticity_df = pd.DataFrame(all_elasticity_data)
                                                
                                                # Convert to numeric safely
                                                elasticity_df['Elasticity'] = pd.to_numeric(elasticity_df['Elasticity'], errors='coerce')
                                                elasticity_df['p-value'] = pd.to_numeric(elasticity_df['p-value'], errors='coerce')
                                                
                                                # Safe significance calculation
                                                def get_significance(p_value):
                                                    try:
                                                        p = float(p_value)
                                                        if p < 0.001:
                                                            return '***'
                                                        elif p < 0.01:
                                                            return '**'
                                                        elif p < 0.05:
                                                            return '*'
                                                        return ''
                                                    except (TypeError, ValueError):
                                                        return ''
                                                
                                                elasticity_df['Significance'] = elasticity_df['p-value'].apply(get_significance)
                                                
                                                # Pivot the data to match our display format
                                                try:
                                                    pivoted_elasticity = elasticity_df.pivot_table(
                                                        index=['Custom Name', 'Model Type', 'Segment'],
                                                        columns='Feature',
                                                        values=['Elasticity', 'p-value'],
                                                        aggfunc='first'
                                                    )
                                                    
                                                    # Flatten multi-index columns
                                                    pivoted_elasticity.columns = [f"{col[1]}_{col[0]}" for col in pivoted_elasticity.columns]
                                                    pivoted_elasticity = pivoted_elasticity.reset_index()
                                                    
                                                    st.dataframe(
                                                        pivoted_elasticity,
                                                        use_container_width=True,
                                                        column_config={
                                                            col: st.column_config.NumberColumn(format="%.4f") 
                                                            for col in pivoted_elasticity.columns 
                                                            if 'Elasticity' in col or 'p-value' in col
                                                        }
                                                    )
                                                except Exception as e:
                                                    st.warning(f"Could not pivot elasticity data. Showing raw format. Error: {e}")
                                                    st.dataframe(
                                                        elasticity_df,
                                                        use_container_width=True,
                                                        column_config={
                                                            "Elasticity": st.column_config.NumberColumn(format="%.4f"),
                                                            "p-value": st.column_config.NumberColumn(format="%.4f")
                                                        }
                                                    )
                                            else:
                                                st.warning("No feature elasticities data available for selected models")
                                        else:
                                            st.warning("No feature elasticities data available")






                                    # else:
                                    #     st.warning("No saved models match the current selection")
                                    else:
                                        st.warning("No saved models available")



                                    













                    



                                            










                                                

                                    
                                # with st.expander("Model Results:"):

                                #     if valid_features:


                                #         if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in models:
        

                                #             # Create a figure for each group
                                #             for group in selected_group:
                                #                 # Check if we have forecast results for this group
                                #                 if group in forecast_results:
                                #                     # Create the figure for this group
                                #                     fig = go.Figure()
                                #                     data = forecast_results[group]
                                                    
                                #                     # Actual Volume Data (always shown)
                                #                     fig.add_trace(go.Scatter(
                                #                         x=data['actual_dates'], 
                                #                         y=data['actual_volume'],
                                #                         mode='lines', 
                                #                         line=dict(color='blue', width=2.5), 
                                #                         name='Actual Volume',
                                #                         legendgroup="actual"
                                #                     ))

                                #                     # Handle non-Prophet models if they exist
                                #                     try:
                                #                         if not df_final.empty and 'predicted_df' in locals():
                                #                             # Filter predictions for this group
                                #                             group_predictions = predicted_df[predicted_df['Segment'] == group]
                                #                             for model_name, model_group in group_predictions.groupby('Model_type'):
                                #                                 fig.add_trace(go.Scatter(
                                #                                     x=model_group['Date'],
                                #                                     y=model_group['Volume'],
                                #                                     mode='lines',
                                #                                     name=f'{model_name} Forecast',
                                #                                     line=dict(dash='dot', width=3)
                                #                                 ))
                                #                     except NameError:
                                #                         st.warning("`df_final_sp` is not defined.")
                                #                     except Exception as e:
                                #                         st.warning(f"An error occurred while displaying predictions: {e}")

                                #                     # Handle Prophet forecasts if selected
                                #                     if "Prophet" in models:
                                #                         # Prophet Future Forecasts
                                #                         fig.add_trace(go.Scatter(
                                #                             x=future_dates, 
                                #                             y=data['prophet_future_forecast'],
                                #                             mode='lines', 
                                #                             name='Prophet Forecast', 
                                #                             line=dict(dash='dot', color='peru', width=3),
                                #                             legendgroup="prophet"
                                #                         ))

                                #                     # Update layout for this group's figure
                                #                     fig.update_layout(
                                #                         title=f"Forecast for {group}", 
                                #                         xaxis_title="Date", 
                                #                         yaxis_title=target_col, 
                                #                         template="plotly_dark",
                                #                         legend=dict(
                                #                             traceorder="normal",
                                #                             itemsizing="constant"
                                #                         )
                                #                     )

                                #                     # Display the figure for this group
                                #                     st.plotly_chart(fig, use_container_width=True)
                                #                     st.markdown('<hr class="thin">', unsafe_allow_html=True)


                                            









                                # with st.expander("Growth Rates:"):

                                #     if valid_features:


                                #         def calculate_and_display_growth_analysis(forecast_results, predicted_df, models, 
                                #                                                 forecast_horizon=12, start_year=2020, 
                                #                                                 fiscal_start_month=1, frequency="M"):
                                        
                                #             # Calculate and display growth rates for yearly, half-yearly, and quarterly periods
                                #             # with all groups plotted together for each model.
                                        
                                #             # Add frequency selector
                                #             analysis_freq = st.radio("Growth Rate Frequency:", 
                                #                                     ["Yearly", "Half-Yearly", "Quarterly"],
                                #                                     horizontal=True)
                                            
                                #             # Calculate growth rates for all available models
                                #             growth_results = calculate_growth_rates_with_forecasts(
                                #                 forecast_results=forecast_results,
                                #                 predicted_df=predicted_df,
                                #                 forecast_horizon=forecast_horizon,
                                #                 start_year=start_year,
                                #                 fiscal_start_month=fiscal_start_month,
                                #                 frequency=frequency,
                                #                 analysis_freq=analysis_freq.lower()
                                #             )
                                            
                                #             # Prepare data for table and chart - keeping same DataFrame structure
                                #             table_data = []
                                #             chart_data = []
                                            
                                #             for segment, model_data in growth_results.items():
                                #                 for model_name, result_df in model_data.items():
                                #                     if model_name in models or (model_name == "Prophet" and "Prophet" in models):
                                #                         table_entry = result_df.copy()
                                #                         table_entry['Segment'] = segment
                                #                         table_entry['Model'] = model_name
                                #                         table_data.append(table_entry)
                                                        
                                #                         for _, row in result_df.iterrows():
                                #                             chart_data.append({
                                #                                 'Segment': segment,
                                #                                 'Model': model_name,
                                #                                 'period': row['period'],
                                #                                 'growth_rate': row['growth_rate'],
                                #                                 'volume': row['volume']
                                #                             })
                                            
                                #             if not table_data:
                                #                 st.warning("No growth rates available for selected models.")
                                #                 return
                                            
                                #             # Create combined DataFrame (same structure as original)
                                #             combined_df = pd.concat(table_data, ignore_index=True)
                                #             chart_df = pd.DataFrame(chart_data)
                                            
                                #             # Display in columns
                                #             col1, col2 = st.columns(2)
                                            
                                #             with col1:
                                #                 # Create one chart with all groups for each model
                                #                 fig = px.line(
                                #                     chart_df,
                                #                     x='period',
                                #                     y='growth_rate',
                                #                     color='Model',
                                #                     line_dash='Segment',  # Different dash patterns for segments
                                #                     labels={'growth_rate': 'Growth Rate (%)', 'period': 'Period'},
                                #                     title=f'{analysis_freq} Growth Rate Comparison by Model',
                                #                     markers=True,
                                #                     text='growth_rate'
                                #                 )
                                                
                                #                 fig.update_yaxes(tickformat=".1%")
                                #                 fig.update_traces(
                                #                     hovertemplate="<b>%{fullData.name}</b><br>" +
                                #                                 "Segment: %{customdata[0]}<br>" +
                                #                                 "Period: %{x}<br>" +
                                #                                 "Growth Rate: %{y:.2%}<extra></extra>",
                                #                     texttemplate='%{y:.1%}',
                                #                     textposition='top center',
                                #                     customdata=chart_df[['Segment']]
                                #                 )
                                                
                                #                 fig.update_layout(
                                #                     hovermode='x unified',
                                #                     legend_title_text='Model/Segment',
                                #                     template='plotly_white',
                                #                     height=500,
                                #                     uniformtext_minsize=8,
                                #                     uniformtext_mode='hide'
                                #                 )
                                                
                                #                 st.plotly_chart(fig, use_container_width=True)
                                            
                                #             with col2:
                                #                 # Table remains the same
                                #                 combined_df = combined_df.dropna()
                                #                 st.dataframe(combined_df.style.format({
                                #                     'growth_rate': '{:.4f}',
                                #                     'volume': '{:,.2f}'
                                #                 }), use_container_width=True)


                                #         def calculate_growth_rates_with_forecasts(forecast_results, predicted_df=None, forecast_horizon=12, 
                                #                                     start_year=2020, fiscal_start_month=1, frequency="M", 
                                #                                     analysis_freq="yearly"):
                                        
                                #             # Calculate growth rates for different time periods with proper frequency handling
                                    
                                #             growth_results = {}
                                            
                                #             for segment, data in forecast_results.items():
                                #                 actual_dates = pd.Series(pd.to_datetime(data['actual_dates']))
                                #                 actual_volume = pd.Series(data['actual_volume'])
                                                
                                #                 segment_growth = {}
                                                
                                #                 # Process Prophet model if available
                                #                 if "prophet_future_forecast" in data:
                                #                     # Calculate proper date offset based on frequency
                                #                     if frequency == "D":
                                #                         offset = pd.DateOffset(days=1)
                                #                     elif frequency == "W":
                                #                         offset = pd.DateOffset(weeks=1)
                                #                     elif frequency == "M":
                                #                         offset = pd.DateOffset(months=1)
                                #                     elif frequency == "Q":
                                #                         offset = pd.DateOffset(months=3)
                                #                     elif frequency == "Y":
                                #                         offset = pd.DateOffset(years=1)
                                #                     else:
                                #                         offset = pd.DateOffset(months=1)  # Default to monthly
                                                    
                                #                     future_dates = pd.Series(pd.date_range(
                                #                         start=actual_dates.max() + offset,  # Use the calculated offset
                                #                         periods=forecast_horizon,
                                #                         freq=frequency
                                #                     ))
                                                    
                                #                     prophet_df = pd.DataFrame({
                                #                         'date': pd.concat([actual_dates, future_dates], ignore_index=True),
                                #                         'volume': pd.concat([actual_volume, 
                                #                                         pd.Series(data['prophet_future_forecast'])], 
                                #                                         ignore_index=True),
                                #                         'Model_type': 'Prophet',
                                #                         'Segment': segment  # Add segment information
                                #                     })
                                                    
                                #                     prophet_result = _calculate_model_growth_by_freq(
                                #                         prophet_df, start_year, fiscal_start_month, frequency, analysis_freq
                                #                     )
                                #                     segment_growth['Prophet'] = prophet_result
                                                
                                #                 # Process other models
                                #                 if predicted_df is not None and not predicted_df.empty:
                                #                     predicted_df['date'] = pd.to_datetime(predicted_df['Date'])
                                                    
                                #                     # Filter predicted_df for current segment if Segment column exists
                                #                     if 'Segment' in predicted_df.columns:
                                #                         segment_predicted_df = predicted_df[predicted_df['Segment'] == segment]
                                #                     else:
                                #                         segment_predicted_df = predicted_df
                                                    
                                #                     for model_name, model_df in segment_predicted_df.groupby('Model_type'):
                                #                         model_combined = pd.DataFrame({
                                #                             'date': pd.concat([actual_dates, 
                                #                                             pd.Series(model_df['date'])], 
                                #                                             ignore_index=True),
                                #                             'volume': pd.concat([actual_volume, 
                                #                                             pd.Series(model_df['Volume'])], 
                                #                                             ignore_index=True),
                                #                             'Model_type': model_name,
                                #                             'Segment': segment  # Ensure segment is preserved
                                #                         })
                                                        
                                #                         model_result = _calculate_model_growth_by_freq(
                                #                             model_combined, start_year, fiscal_start_month, frequency, analysis_freq
                                #                         )
                                #                         segment_growth[model_name] = model_result
                                                
                                #                 growth_results[segment] = segment_growth
                                            
                                #             return growth_results

                                #         def _calculate_model_growth_by_freq(df, start_year, fiscal_start_month, frequency_options, analysis_freq):
                                #             """Calculate growth rates with proper period handling for all frequencies"""
                                #             df = df.copy()
                                #             df['date'] = pd.to_datetime(df['date'])
                                #             df = df.sort_values('date')
                                #             df = df[df['date'].dt.year >= start_year - 1]
                                #             df.set_index('date', inplace=True)
                                            
                                #             # Determine period grouping based on analysis frequency
                                #             if analysis_freq == "yearly":
                                #                 # Handle fiscal year offset
                                #                 if frequency_options in ["D", "W", "M", "Q"]:
                                #                     df['period'] = (df.index - pd.offsets.DateOffset(months=fiscal_start_month-1)).year
                                #                 else:
                                #                     df['period'] = df.index.year
                                                
                                #                 # if not (1 <= fiscal_start_month <= 5):
                                #                 #     df['period'] = df['period'] + 1
                                #                 period_name = 'Year'
                                                
                                #             elif analysis_freq == "half-yearly":
                                #                 # Create proper half-year periods considering fiscal year
                                #                 if fiscal_start_month == 1:  # Calendar year
                                #                     df['half'] = np.where(df.index.month <= 6, 'H1', 'H2')
                                #                     df['period'] = df.index.year.astype(str) + '-' + df['half']
                                #                 else:
                                #                     # Adjust for fiscal year - convert to numpy array for modification
                                #                     adjusted_month = (df.index.month - (fiscal_start_month - 1)).to_numpy()
                                #                     adjusted_month[adjusted_month <= 0] += 12
                                #                     df['half'] = np.where(adjusted_month <= 6, 'H1', 'H2')
                                #                     year = np.where(adjusted_month <= 12, df.index.year, df.index.year + 1)
                                #                     df['period'] = year.astype(str) + '-' + df['half']
                                                
                                #                 period_name = 'Half-Year'
                                                
                                #             elif analysis_freq == "quarterly":
                                #                 # Create proper quarterly periods considering fiscal year
                                #                 if fiscal_start_month == 1:  # Calendar year
                                #                     df['period'] = df.index.to_period('Q').astype(str)
                                #                 else:
                                #                     # Adjust for fiscal year - convert to numpy array for modification
                                #                     adjusted_month = (df.index.month - fiscal_start_month).to_numpy()
                                #                     adjusted_month[adjusted_month < 0] += 12
                                #                     df['quarter'] = (adjusted_month // 3) + 1
                                #                     year = np.where(df.index.month >= fiscal_start_month, df.index.year, df.index.year - 1)
                                #                     df['period'] = year.astype(str) + 'Q' + df['quarter'].astype(str)
                                                
                                #                 period_name = 'Quarter'
                                            
                                #             # Group by period and calculate mean volume
                                #             period_df = df.groupby('period')['volume'].mean().reset_index()
                                            
                                #             # Calculate growth rates
                                #             model_name = df['Model_type'].iloc[0]
                                #             period_df['growth_rate'] = period_df['volume'].pct_change()
                                #             period_df['Model'] = model_name
                                #             period_df['period_type'] = period_name
                                            
                                #             # Sort periods chronologically
                                #             if analysis_freq == "half-yearly":
                                #                 period_df['sort_year'] = period_df['period'].str[:4].astype(int)
                                #                 period_df['sort_half'] = np.where(period_df['period'].str.contains('H1'), 1, 2)
                                #                 period_df = period_df.sort_values(['sort_year', 'sort_half']).drop(['sort_year', 'sort_half'], axis=1)
                                #             elif analysis_freq == "quarterly":
                                #                 period_df['sort_year'] = period_df['period'].str.extract(r'(\d+)Q')[0].astype(int)
                                #                 period_df['sort_qtr'] = period_df['period'].str.extract(r'Q(\d+)')[0].astype(int)
                                #                 period_df = period_df.sort_values(['sort_year', 'sort_qtr']).drop(['sort_year', 'sort_qtr'], axis=1)
                                            
                                #             return period_df


                                            
                                #         if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in models:
                                #             calculate_and_display_growth_analysis(
                                #                 forecast_results=forecast_results,
                                #                 predicted_df=predicted_df if 'predicted_df' in locals() else None,
                                #                 models=models,
                                #                 forecast_horizon=forecast_horizon,
                                #                 start_year=min_year,
                                #                 fiscal_start_month=fiscal_start_month
                                #             )








                                    



    #--------------------------------------------------------------------------------------MODEL SAVING-------------------------------------------------------------

                        # if st.session_state.comparison_data:

                        #     st.markdown('<hr class="thick">', unsafe_allow_html=True)


                        #     with st.expander("Model Comparison:"):
                        #         # Create filter controls at the top
                        #         col53, col54, col55 = st.columns(3)
                                
                        #         # Get unique values for filters from the comparison data
                        #         all_models = list(set(data['model_name'] for data in st.session_state.comparison_data.values()))
                        #         all_groups = list(set(data['group'] for data in st.session_state.comparison_data.values()))
                        #         all_years = list(set(year for data in st.session_state.comparison_data.values() for year in data['growth_rates'].index))
                                
                        #         with col53:
                        #             selected_models = st.multiselect("Filter by Model", all_models, default=all_models)
                        #         with col54:
                        #             selected_groups = st.multiselect("Filter by Group", all_groups, default=all_groups)
                        #         with col55:
                        #             selected_years = st.multiselect("Filter by Year", sorted(all_years), default=sorted(all_years))

                        #         st.markdown('<hr class="thin">', unsafe_allow_html=True)
                                
                        #         # Filter the data based on selections
                        #         filtered_data = {}
                        #         for run_key, data in st.session_state.comparison_data.items():
                        #             if (data['model_name'] in selected_models and 
                        #                 data['group'] in selected_groups and 
                        #                 any(year in selected_years for year in data['growth_rates'].index)):
                                        
                        #                 # Filter the years within each dataset
                        #                 year_mask = [year in selected_years for year in data['growth_rates'].index]
                        #                 filtered_data[run_key] = {
                        #                     'model_name': data['model_name'],
                        #                     'custom_name': data['custom_name'],
                        #                     'group': data['group'],
                        #                     'features_used': data['features_used'],
                        #                     'growth_rates': data['growth_rates'][year_mask],
                        #                     'volumes': data['volumes'][year_mask]
                        #                 }
                                
                        #         # Create tabs for different views
                        #         tab9, tab10 = st.tabs(["Growth Rates Comparison", "Volumes Comparison"])
                                
                        #         with tab9:
                        #             fig_compare_growth = go.Figure()
                                    
                        #             for run_key, data in filtered_data.items():
                        #                 fig_compare_growth.add_trace(go.Scatter(
                        #                     x=data['growth_rates'].index,
                        #                     y=data['growth_rates'],
                        #                     mode="lines+markers+text",
                        #                     name=f"{data['model_name']} - {data['group']} ({data['custom_name']})",
                        #                     line=dict(width=2),
                        #                     text=[f"{y:.1f}%" for y in data['growth_rates']],
                        #                     textposition="top center",
                        #                     textfont=dict(size=12)
                        #                 ))
                                    
                        #             fig_compare_growth.update_layout(
                        #                 title="Growth Rate Comparison (Filtered)",
                        #                 xaxis_title="Year",
                        #                 yaxis_title="Growth Rate (%)",
                        #                 template="plotly_dark",
                        #                 hovermode="x unified",
                        #                 margin=dict(t=40, b=40)
                        #             )
                                    
                        #             fig_compare_growth.add_hline(y=0, line_dash="dot", line_color="white")
                        #             st.plotly_chart(fig_compare_growth, use_container_width=True)

                        #         with tab10:
                        #             fig_compare_volume = go.Figure()
                                    
                        #             for run_key, data in filtered_data.items():
                        #                 fig_compare_volume.add_trace(go.Scatter(
                        #                     x=data['volumes'].index,
                        #                     y=data['volumes'],
                        #                     mode="lines+markers",
                        #                     name=f"{data['model_name']} - {data['group']} ({data['custom_name']})",
                        #                     line=dict(width=2)
                        #                 ))
                                    
                        #             fig_compare_volume.update_layout(
                        #                 title="Volume Comparison (Filtered)",
                        #                 xaxis_title="Year",
                        #                 yaxis_title="Volume",
                        #                 template="plotly_dark",
                        #                 hovermode="x unified"
                        #             )
                        #             st.plotly_chart(fig_compare_volume, use_container_width=True)

                        #         st.markdown('<hr class="thin">', unsafe_allow_html=True)
                                
                        #         # Display filtered comparison table
                        #         st.write("Comparison Data (Filtered)")

                        #         comparison_dfs = []
                        #         for run_key, data in filtered_data.items():
                        #             temp_df = pd.DataFrame({
                        #                 'Model': data['model_name'],
                        #                 'Model Name': data['custom_name'],
                        #                 'Group': data['group'],
                        #                 'Year': data['growth_rates'].index,
                        #                 'Growth Rate': data['growth_rates'].astype(str) + '%',
                        #                 'Volume': data['volumes'],
                        #                 'Features Used': ", ".join(data['features_used'])
                        #             })
                        #             comparison_dfs.append(temp_df)

                        #         if comparison_dfs:
                        #             combined_df = pd.concat(comparison_dfs)
                        #             st.dataframe(
                        #                 combined_df.sort_values(['Model','Model Name'], ascending=True),
                        #                 use_container_width=True
                        #             )
                                    
                        #             csv = combined_df.to_csv(index=False)
                        #             st.download_button(
                        #                 label="Download Filtered Comparison Data",
                        #                 data=csv,
                        #                 file_name='filtered_model_comparison.csv',
                        #                 mime='text/csv'
                        #             )
                                
                        #         if st.button("Clear Comparison Data"):
                        #             st.session_state.comparison_data = {}
                        #             st.rerun()


                            # with st.expander("Model Comparison:"):
                                
                            #     # Create tabs for different views
                            #     tab9, tab10 = st.tabs(["Growth Rates Comparison", "Volumes Comparison"])
                                
                            
                            #     with tab9:
                            #         fig_compare_growth = go.Figure()
                                    
                            #         # Updated to work with the new data structure
                            #         for run_key, data in st.session_state.comparison_data.items():
                            #             fig_compare_growth.add_trace(go.Scatter(
                            #                 x=data['growth_rates'].index,
                            #                 y=data['growth_rates'],
                            #                 mode="lines+markers+text",
                            #                 name=f"{data['model_name']} - {data['group']} ({data['custom_name']})",  # Include custom name in legend
                            #                 line=dict(width=2),
                            #                 text=[f"{y:.1f}%" for y in data['growth_rates']],
                            #                 textposition="top center",
                            #                 textfont=dict(size=12)
                            #             ))
                                    
                            #         fig_compare_growth.update_layout(
                            #             title="Growth Rate Comparison",
                            #             xaxis_title="Year",
                            #             yaxis_title="Growth Rate (%)",
                            #             template="plotly_dark",
                            #             hovermode="x unified",
                            #             margin=dict(t=40, b=40)
                            #         )
                                    
                            #         # Add horizontal line at 0% for reference
                            #         fig_compare_growth.add_hline(y=0, line_dash="dot", line_color="white")
                            #         st.plotly_chart(fig_compare_growth, use_container_width=True)

                            #     with tab10:
                            #         # Plot volume comparison
                            #         fig_compare_volume = go.Figure()
                                    
                            #         for run_key, data in st.session_state.comparison_data.items():
                            #             fig_compare_volume.add_trace(go.Scatter(
                            #                 x=data['volumes'].index,
                            #                 y=data['volumes'],
                            #                 mode="lines+markers",
                            #                 name=f"{data['model_name']} - {data['group']} ({data['custom_name']})",  # Include custom name in legend
                            #                 line=dict(width=2)
                            #             ))
                                    
                            #         fig_compare_volume.update_layout(
                            #             title="Volume Comparison",
                            #             xaxis_title="Year",
                            #             yaxis_title="Volume",
                            #             template="plotly_dark",
                            #             hovermode="x unified"
                            #         )
                            #         st.plotly_chart(fig_compare_volume, use_container_width=True)

                            #     st.markdown('<hr class="thin">', unsafe_allow_html=True)
                                
                            #     # Display comparison table
                            #     st.write("Comparison Data")

                            #     # Enhanced comparison table                        
                            #     comparison_dfs = []
                            #     for run_key, data in st.session_state.comparison_data.items():
                            #         temp_df = pd.DataFrame({
                            #             'Model': data['model_name'],  # Use the stored model name
                            #             'Model Name': data['custom_name'],
                            #             'Group': data['group'],
                            #             'Year': data['growth_rates'].index,
                            #             'Growth Rate': data['growth_rates'].astype(str) + '%',
                            #             'Volume': data['volumes'],
                            #             'Features Used': ", ".join(data['features_used'])
                            #         })
                            #         comparison_dfs.append(temp_df)


                            #     if comparison_dfs:
                            #         combined_df = pd.concat(comparison_dfs)
                            #         st.dataframe(
                            #             combined_df.sort_values(['Model','Model Name'], ascending=True),
                            #             use_container_width=True
                            #         )
                                    
                            #         # Download button
                            #         csv = combined_df.to_csv(index=False)
                            #         st.download_button(
                            #             label="Download Comparison Data",
                            #             data=csv,
                            #             file_name='model_comparison.csv',
                            #             mime='text/csv'
                            #         )
                                
                            #     if st.button("Clear Comparison Data"):
                            #         st.session_state.comparison_data = {}
                            #         st.rerun()











                    
                    # st.markdown('<hr class="thick">', unsafe_allow_html=True)

                    # os.makedirs("saved_configs", exist_ok=True)



                    # # Save/Load Configuration
                    # # with st.expander("Configuration Management"):
                    # config_name = st.text_input("Model Name")
                    
                    # col49, col50 = st.columns(2)
                    # with col49:
                    #     if st.button("Save Current Model"):
                    #         if not config_name:
                    #             st.error("Please enter a name")
                    #         else:
                    #             config = {
                    #                 'model': st.session_state.get('models', models),
                    #                 'features': st.session_state.get('valid_features', valid_features),
                    #                 'group': st.session_state.get('selected_group', selected_group),
                    #                 'forecast_period': st.session_state.get('forecast_horizon', forecast_horizon),
                    #                 'fiscal_month': st.session_state.get('fiscal_start_month', fiscal_start_month),
                    #                 'frequency': st.session_state.get('frequency', frequency)
                    #             }
                                
                    #             with open(f"saved_configs/{config_name}.pkl", "wb") as f:
                    #                 pickle.dump(config, f)
                    #             st.success(f"Saved '{config_name}'")

                    #     # with col50:
                    #     #     saved_configs = [f.replace(".pkl", "") for f in os.listdir("saved_configs") if f.endswith(".pkl")]
                    #     #     if saved_configs:
                    #     #         selected = st.selectbox("Saved Configs", saved_configs)
                    #     #         if st.checkbox("Load Selected"):
                    #     #             try:
                    #     #                 with open(f"saved_configs/{selected}.pkl", "rb") as f:
                    #     #                     loaded_config = pickle.load(f)
                                        
                    #     #                 # Store the loaded config in session state
                    #     #                 st.session_state['loaded_config'] = loaded_config
                                        
                    #     #                 # Display the loaded configuration
                    #     #                 st.subheader(f"Loaded Configuration: {selected}")
                    #     #                 st.json(loaded_config)  # More compact display than multiple st.write()
                                        
                    #     #                 if st.checkbox("Use This Configuration"):
                    #     #                     # Update session state with loaded values
                    #     #                     st.session_state.models = loaded_config.get('model', 'None')
                    #     #                     models = st.session_state.models

                    #     #                     st.session_state.valid_features = loaded_config.get('features', [])
                    #     #                     feature_cols = st.session_state.valid_features
                                            
                    #     #                     st.session_state.selected_group = loaded_config.get('group')
                    #     #                     selected_group=st.session_state.selected_group
                                            
                    #     #                     st.session_state.forecast_horizon = loaded_config.get('forecast_period')
                    #     #                     forecast_horizon=st.session_state.forecast_horizon 


                    #     #                     st.session_state.fiscal_start_month = loaded_config.get('fiscal_month')
                    #     #                     fiscal_start_month=st.session_state.fiscal_start_month


                    #     #                     # st.session_state.frequency = loaded_config.get('frequency')
                                            
                    #     #                     st.success("Configuration applied successfully! Refresh the page to see changes.")
                    #     #                     # st.rerun()

                    #     #             except Exception as e:
                    #     #                 st.error(f"Error loading configuration: {str(e)}")
                    #         # else:
                    #         #     st.info("No saved configurations available")

                            





            # import plotly.express as px

            

            # if selected == "POST-MODELING":
            # with tab15:
        if selected=="EVALUATE":
                
            import plotly.express as px
            # render_workflow(3)
            # show_workflow("POST-MODELING")

            # st.write('ghjkl')

            tab25,tab26 = st.tabs(['AutoRegressive','Feature-Based'])

            with tab25:
                if "post_data" in st.session_state:
                    if "original_post_data" not in st.session_state:
                        st.session_state['original_post_data'] = st.session_state['post_data'].copy()
                    combined_table_pivot = st.session_state['post_data']
                    combined_table_pivot = combined_table_pivot.dropna()

                    combined_table_pivot.columns = [
                        str(tuple(col)) if isinstance(col, list) else str(col)
                        for col in combined_table_pivot.columns
                    ]

                    col24, col25 = st.columns(2)

                    with col24:
                        # Create a copy of the original dataframe for display in the expander
                        original_combined_table_pivot = st.session_state['original_post_data']
                        original_combined_table_pivot = original_combined_table_pivot.dropna()
                        original_combined_table_pivot.columns = [
                        str(tuple(col)) if isinstance(col, list) else str(col)
                        for col in original_combined_table_pivot.columns
                    ]

                        # Allow users to edit the growth rate values
                        with st.expander('Original Growth Rate Values'):
                            

                            # Create the Plotly line chart using the original data
                            fig = px.line(original_combined_table_pivot, x=original_combined_table_pivot.index, y=original_combined_table_pivot.columns, title="", markers=True)
                            st.plotly_chart(fig)
                            st.markdown('<hr class="thin">', unsafe_allow_html=True)
                            st.write("Growth Rate Values:")
                            st.dataframe(original_combined_table_pivot)

                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        # Extract group names from the columns of the combined_table_pivot
                        groups = combined_table_pivot.columns.tolist()

                        # Separate groups into single-name, two-name, three-name, etc.
                        def categorize_groups(groups):
                            categorized = {}
                            for group in groups:
                                # Check if the group is a tuple string (e.g., "('assam', 'Deluxe')")
                                if group.startswith("(") and group.endswith(")"):
                                    # Count the number of commas to determine the category
                                    num_commas = group.count(",")
                                    if num_commas == 1:
                                        category = "2nd Level"
                                    elif num_commas == 2:
                                        category = "3rd Level"
                                    elif num_commas == 3:
                                        category = "4th Level"
                                    elif num_commas == 4:
                                        category = "5th Level"
                                    elif num_commas == 5:
                                        category = "6th Level"
                                    elif num_commas == 6:
                                        category = "7th Level"
                                    elif num_commas == 7:
                                        category = "8th Level"
                                    else:
                                        category = f"{num_commas + 1}th Level"  # General case for n-name groups
                                else:
                                    # Single-name groups (e.g., 'Deluxe', 'Premium')
                                    category = "All India"

                                # Add the group to the appropriate category
                                if category not in categorized:
                                    categorized[category] = []
                                categorized[category].append(group)
                            return categorized

                        # Categorize the groups
                        categorized_groups = categorize_groups(groups)

                        # Define the hierarchical order of categories
                        hierarchical_order = ["All India", "2nd Level", "3rd Level", "4th Level", "5th Level", "6th Level", "7th Level", "8th Level", "9th Level", "10th Level"]

                        # Use the original data's index for the year options
                        selected_years = st.multiselect(
                            "Select Year",
                            original_combined_table_pivot.index.tolist(),  # Use original data's index
                            default=original_combined_table_pivot.index.tolist()  # Default to all years
                        )

                        # Apply year filtering to the editable dataframe and weighted averages
                        if selected_years:
                            filtered_combined_table_pivot = combined_table_pivot.loc[selected_years]
                        else:
                            filtered_combined_table_pivot = combined_table_pivot  # Use full data if no years are selected

                        # Show the combined_table_pivot again but make it editable
                        st.write("Edit the growth rates below. Changes will be used to calculate averages.")
                        edited_combined_table = st.data_editor(filtered_combined_table_pivot, key="combined_data_editor")

                        # Save the edited combined_table_pivot back to st.session_state
                        # Only save edits to the original data, not the filtered data
                        if selected_years:
                            combined_table_pivot.loc[selected_years] = edited_combined_table
                        else:
                            combined_table_pivot = edited_combined_table
                        st.session_state['post_data'] = combined_table_pivot

                        # Add a reset button to revert to the original data
                        if st.button("Reset to Original Values"):
                            st.session_state['post_data'] = st.session_state['original_post_data'].copy()
                            st.rerun()

                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        # Calculate and display Weighted Average growth rates for 2nd Level groups
                        st.write("The weighted average growth rates are calculated based on the corresponding All India names.")

                        # Create a DataFrame to store weighted averages
                        weighted_avg_growth_rates = pd.DataFrame(index=combined_table_pivot.index)



                        if "All India" in categorized_groups and "2nd Level" in categorized_groups:
                            all_india_groups = categorized_groups["All India"]
                            second_level_groups = categorized_groups["2nd Level"]

                            for all_india_group in all_india_groups:
                                # Extract the All India name (e.g., 'Deluxe' from 'All India - Deluxe')
                                all_india_name = all_india_group

                                # Find matching 2nd Level groups based on the All India name
                                matching_second_level_groups = [
                                    group for group in second_level_groups if all_india_name in group
                                ]

                                if matching_second_level_groups:
                                    # Calculate the average growth rate for matching 2nd Level groups for each year
                                    avg_growth_rates = combined_table_pivot[matching_second_level_groups].mean(axis=1)
                                    weighted_avg_growth_rates[f"Avg ({all_india_name})"] = avg_growth_rates
                                

                        # Display the weighted average growth rates table
                        st.dataframe(weighted_avg_growth_rates)

                        # # Add a comparison between All India and Weighted Average growth rates
                        # st.markdown("### Comparison: All India vs. Weighted Average Growth Rates")
                        # comparison_table = pd.DataFrame(index=combined_table_pivot.index)

                        # if "All India" in categorized_groups:
                        #     for all_india_group in all_india_groups:
                        #         all_india_name = all_india_group
                        #         comparison_table[f"All India - {all_india_name}"] = combined_table_pivot[all_india_group]
                        #         if f"Avg ({all_india_name})" in weighted_avg_growth_rates.columns:
                        #             comparison_table[f"Avg ({all_india_name})"] = weighted_avg_growth_rates[f"Avg ({all_india_name})"]

                        # # Display the comparison table
                        # st.dataframe(comparison_table)

                    with col25:

                        # # Add a multi-select year selector in the sidebar
                        # selected_years = st.multiselect(
                        #     "Select Year",
                        #     combined_table_pivot.index.tolist(),
                        #     default=combined_table_pivot.index.tolist()  # Default to all years
                        # )


                        # Add a group category selector in the sidebar
                        st.write("Select Group Categories to Display:")
                        selected_categories = []
                        for category in hierarchical_order:
                            if category in categorized_groups:  # Only show categories that exist
                                if st.checkbox(f"Show {category}", value=True, key=category):
                                    selected_categories.append(category)

                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

                        # Add a toggle for grouping by first name or second name for 2nd Level groups
                        if "2nd Level" in selected_categories:
                            grouping_option = st.radio(
                                "Group 2nd Level by:",
                                options=["First Name", "Second Name"],
                                key="grouping_option",
                                horizontal=True
                            )
                            st.markdown('<hr class="thin">', unsafe_allow_html=True)
                        else:
                            grouping_option = "First Name"  # Default to first name if 2nd Level is not selected

                        

                        # Function to create a bar graph for a list of groups
                        def create_bar_graph(groups, title, category):
                            if groups:
                                if category == "2nd Level":
                                    # Group the groups based on the selected option (first name or second name)
                                    grouped_data = {}
                                    for group in groups:
                                        if grouping_option == "First Name":
                                            # Group by the first name before the comma
                                            key = group.split(",")[0].strip("('\"")
                                        else:
                                            # Group by the second name after the comma
                                            key = group.split(",")[1].strip(" '\")")
                                        
                                        if key not in grouped_data:
                                            grouped_data[key] = []
                                        grouped_data[key].append(group)

                                    # Create a separate bar graph for each group
                                    for key, group_list in grouped_data.items():
                                        fig = go.Figure()
                                        colors = px.colors.qualitative.Set3
                                        if len(selected_years) > 1:
                                            # Grouped column bar for multiple years (color based on years)
                                            for i, year in enumerate(selected_years):
                                                y_values = [combined_table_pivot.loc[year, group] if group in combined_table_pivot.columns else None
                                                        for group in group_list]
                                                fig.add_trace(go.Bar(
                                                    x=group_list,
                                                    y=y_values,
                                                    name=str(year),
                                                    marker_color=colors[i % len(colors)],
                                                    text=[f"{val:.2f}" if val is not None else "N/A" for val in y_values],
                                                    textposition="auto"
                                                ))
                                        else:
                                            # Regular bar graph for a single year (color based on groups)
                                            for i, group in enumerate(group_list):
                                                if group in combined_table_pivot.columns:
                                                    fig.add_trace(go.Bar(
                                                        x=[group],
                                                        y=[combined_table_pivot.loc[selected_years[0], group]],
                                                        name=group,
                                                        marker_color=colors[i % len(colors)],
                                                        text=[f"{combined_table_pivot.loc[selected_years[0], group]:.2f}"],
                                                        textposition="auto"
                                                    ))
                                                else:
                                                    st.warning(f"Group '{group}' not found in data.")
                                        fig.update_layout(
                                            title=f"{title} - {key}",
                                            xaxis_title="Group",
                                            yaxis_title="Growth Rate",
                                            template="plotly_white",
                                            barmode="group",  # Side-by-side bars
                                            showlegend=True,
                                            xaxis=dict(
                                                title=None,
                                                tickfont=dict(size=12, family="Arial", color="black", weight="bold")
                                            ),
                                            yaxis=dict(
                                                title=dict(text="Growth Rate", font=dict(size=14, family="Arial", color="black", weight="bold")),
                                                tickfont=dict(size=12, family="Arial", color="black", weight="bold")
                                            )
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    # For other categories, create a single bar graph
                                    fig = go.Figure()
                                    colors = px.colors.qualitative.Set3
                                    if len(selected_years) > 1:
                                        # Grouped column bar for multiple years (color based on years)
                                        for i, year in enumerate(selected_years):
                                            y_values = [combined_table_pivot.loc[year, group] if group in combined_table_pivot.columns else None
                                                    for group in groups]
                                            fig.add_trace(go.Bar(
                                                x=groups,
                                                y=y_values,
                                                name=str(year),
                                                marker_color=colors[i % len(colors)],
                                                text=[f"{val:.2f}" if val is not None else "N/A" for val in y_values],
                                                textposition="auto"
                                            ))
                                    else:
                                        # Regular bar graph for a single year (color based on groups)
                                        for i, group in enumerate(groups):
                                            if group in combined_table_pivot.columns:
                                                fig.add_trace(go.Bar(
                                                    x=[group],
                                                    y=[combined_table_pivot.loc[selected_years[0], group]],
                                                    name=group,
                                                    marker_color=colors[i % len(colors)],
                                                    text=[f"{combined_table_pivot.loc[selected_years[0], group]:.2f}"],
                                                    textposition="auto"
                                                ))
                                            else:
                                                st.warning(f"Group '{group}' not found in data.")
                                    fig.update_layout(
                                        title=title,
                                        xaxis_title="Group",
                                        yaxis_title="Growth Rate",
                                        template="plotly_white",
                                        barmode="group",  # Side-by-side bars
                                        showlegend=True,
                                        xaxis=dict(
                                            title=None,
                                            tickfont=dict(size=12, family="Arial", color="black", weight="bold")
                                        ),
                                        yaxis=dict(
                                            title=dict(text="Growth Rate", font=dict(size=14, family="Arial", color="black", weight="bold")),
                                            tickfont=dict(size=12, family="Arial", color="black", weight="bold")
                                        )
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write(f"No {title.lower()} groups available.")



                        # Create bar graphs for each selected category (in hierarchical order)
                        for category in hierarchical_order:
                            if category in selected_categories:  # Only display selected categories
                                groups_in_category = categorized_groups[category]
                                create_bar_graph(groups_in_category, f"Growth Rates for {category} Group", category)
                                st.markdown('<hr class="thin">', unsafe_allow_html=True)

                else:
                    st.warning("No post-modeling data available. Please run the modeling step first.")








            with tab26:
                # Evaluation Page
    
                # st.subheader("📈 Model Evaluation")
        
                # # Get unique custom names for filtering
                # custom_names = sorted(list({
                #     data['custom_name'] 
                #     for data in st.session_state.saved_models.values()
                # }))
                
                # # Custom name filter
                # selected_custom_name = st.selectbox(
                #     "Filter by Custom Name:",
                #     options=custom_names,
                #     index=0
                # )
                
                # # Filter models by selected custom name
                # filtered_models = {
                #     key: data for key, data in st.session_state.saved_models.items() 
                #     if data['custom_name'] == selected_custom_name
                # }

                # if filtered_models:

                #     with st.expander("Performance Summary and Feature Elasticites",expanded=True):
                
                
                #         # 1. Show Performance Summary
                #         st.write(f"Performance Summary: {selected_custom_name}")
                        
                #         performance_data = []
                #         for data in filtered_models.values():
                #             performance_data.append({
                #                 'Custom Name':data['custom_name'],
                #                 'Model Type': data['model_type'],
                #                 'Segment': data['segment'],
                #                 'MAPE': data.get('MAPE'),
                #                 'R-squared': data.get('R_squared'),
                #                 'Adj R-squared': data.get('Adjusted_R_squared'),
                #                 'Features Used': ', '.join(data.get('features_used', []))
                #             })
                        
                #         performance_df = pd.DataFrame(performance_data)
                #         st.dataframe(
                #             performance_df.sort_values(['Model Type', 'Segment']),
                #             use_container_width=True
                #         )
                        
                #         # 2. Show Elasticities from session state
                #         st.write(f"Feature Elasticities: {selected_custom_name}")

                #         elasticity_data = []
                #         for model_key in filtered_models:
                #             if model_key in st.session_state.elasticity_data:
                #                 data = st.session_state.saved_models[model_key]
                #                 # The structure is now model_key: {feature_name: {Elasticity: x, p-value: y}}
                #                 for feature_name, feature_data in st.session_state.elasticity_data[model_key].items():
                #                     p_value = feature_data.get('p-value')
                #                     significance = ('***' if isinstance(p_value, (int, float)) and p_value < 0.001 
                #                                 else '**' if isinstance(p_value, (int, float)) and p_value < 0.01 
                #                                 else '*' if isinstance(p_value, (int, float)) and p_value < 0.05 
                #                                 else '')
                                    
                #                     elasticity_data.append({
                #                         'Custom Name':data['custom_name'],
                #                         'Model Type': data['model_type'],
                #                         'Segment': data['segment'],
                #                         'Feature': feature_name,
                #                         'Elasticity': feature_data.get('Elasticity'),
                #                         'p-value': p_value,
                #                         'Significance': significance
                #                     })

                #         if elasticity_data:
                #             elasticity_df = pd.DataFrame(elasticity_data)
                            
                #             # Convert to numeric safely
                #             for col in ['Elasticity', 'p-value']:
                #                 if col in elasticity_df:
                #                     elasticity_df[col] = pd.to_numeric(elasticity_df[col], errors='coerce')
                            
                #             # Sort by absolute elasticity
                #             elasticity_df['abs_elasticity'] = elasticity_df['Elasticity'].abs()
                #             elasticity_df = elasticity_df.sort_values(
                #                 ['Model Type', 'Segment', 'abs_elasticity'],
                #                 ascending=[True, True, False]
                #             ).drop('abs_elasticity', axis=1)
                            
                #             st.dataframe(
                #                 elasticity_df,
                #                 use_container_width=True,
                #                 column_config={
                #                     "Elasticity": st.column_config.NumberColumn(format="%.4f"),
                #                     "p-value": st.column_config.NumberColumn(format="%.4f")
                #                 }
                #             )
                #         else:
                #             st.warning("No feature elasticities data available")




                #         # # 2. Show Elasticities from session state in pivoted format
                #         # st.write(f"Feature Elasticities: {selected_custom_name}")

                #         # elasticity_data = []
                #         # for model_key in filtered_models:
                #         #     if model_key in st.session_state.elasticity_data:
                #         #         data = st.session_state.saved_models[model_key]
                #         #         for feature_name, feature_data in st.session_state.elasticity_data[model_key].items():
                #         #             p_value = feature_data.get('p-value')
                #         #             significance = ('***' if isinstance(p_value, (int, float)) and p_value < 0.001 
                #         #                         else '**' if isinstance(p_value, (int, float)) and p_value < 0.01 
                #         #                         else '*' if isinstance(p_value, (int, float)) and p_value < 0.05 
                #         #                         else '')
                                    
                #         #             elasticity_data.append({
                #         #                 'Model Type': data['model_type'],
                #         #                 'Segment': data['segment'],
                #         #                 'Feature': feature_name,
                #         #                 'Elasticity': feature_data.get('Elasticity'),
                #         #                 'p-value': p_value,
                #         #                 'Significance': significance
                #         #             })

                #         # if elasticity_data:
                #         #     elasticity_df = pd.DataFrame(elasticity_data)
                            
                #         #     # Convert to numeric safely
                #         #     for col in ['Elasticity', 'p-value']:
                #         #         if col in elasticity_df:
                #         #             elasticity_df[col] = pd.to_numeric(elasticity_df[col], errors='coerce')
                            
                #         #     # Create pivoted tables for Elasticity and p-values
                #         #     try:
                #         #         # Pivot for Elasticity values
                #         #         elasticity_pivot = elasticity_df.pivot_table(
                #         #             index=['Model Type', 'Segment'],
                #         #             columns='Feature',
                #         #             values='Elasticity',
                #         #             aggfunc='first'
                #         #         ).reset_index()
                                
                #         #         # Pivot for p-values
                #         #         pvalue_pivot = elasticity_df.pivot_table(
                #         #             index=['Model Type', 'Segment'],
                #         #             columns='Feature',
                #         #             values='p-value',
                #         #             aggfunc='first'
                #         #         ).reset_index()
                                
                #         #         # Add suffixes to distinguish metrics
                #         #         elasticity_pivot.columns = [f"{col}_elasticity" if col not in ['Model Type', 'Segment'] else col for col in elasticity_pivot.columns]
                #         #         pvalue_pivot.columns = [f"{col}_pvalue" if col not in ['Model Type', 'Segment'] else col for col in pvalue_pivot.columns]
                                
                #         #         # Merge the two dataframes
                #         #         pivoted_df = pd.merge(elasticity_pivot, pvalue_pivot, on=['Model Type', 'Segment'])
                                
                #         #         # Display the pivoted dataframe
                #         #         st.dataframe(
                #         #             pivoted_df,
                #         #             use_container_width=True,
                #         #             column_config={
                #         #                 col: st.column_config.NumberColumn(format="%.4f") 
                #         #                 for col in pivoted_df.columns 
                #         #                 if 'elasticity' in col or 'pvalue' in col
                #         #             }
                #         #         )
                                
                #         #     except Exception as e:
                #         #         st.warning(f"Could not pivot data. Showing raw format. Error: {e}")
                #         #         st.dataframe(
                #         #             elasticity_df,
                #         #             use_container_width=True,
                #         #             column_config={
                #         #                 "Elasticity": st.column_config.NumberColumn(format="%.4f"),
                #         #                 "p-value": st.column_config.NumberColumn(format="%.4f")
                #         #             }
                #         #         )
                #         # else:
                #         #     st.warning("No feature elasticities data available")


                # if 'saved_models' in st.session_state:


                #     frequency_options=st.session_state.frequency_options

                #     fiscal_start_month=st.session_state.fiscal_start_month


                #     def calculate_feature_growth(forecast_results, fiscal_start_month=1):
                #         """Calculate annual growth rates for each feature based on fiscal year averages"""
                #         growth_data = []
                        
                #         for model_key, model_data in forecast_results.items():
                #             if 'feature_forecasts' not in model_data:
                #                 continue
                                
                #             for feature_name, feature_data in model_data['feature_forecasts'].items():
                #                 # Combine actual and forecasted data
                #                 all_dates = pd.to_datetime(feature_data['actual_dates'] + feature_data['future_dates'])
                #                 all_values = feature_data['actual_values'] + feature_data['future_forecast']
                                
                #                 # Create DataFrame
                #                 df = pd.DataFrame({
                #                     'date': all_dates,
                #                     'value': all_values,
                #                     'type': ['actual']*len(feature_data['actual_dates']) + ['forecast']*len(feature_data['future_dates'])
                #                 })
                                
                #                 # Calculate fiscal year as string (e.g. "2023")
                #                 df['fiscal_year'] = (df['date'] - pd.offsets.DateOffset(months=fiscal_start_month-1)).dt.year.astype(int)

                #                 # Adjust fiscal year label to match the starting year
                #                 if fiscal_start_month > 6:  # e.g., July (7) to Dec (12)
                #                     df['fiscal_year'] = df['fiscal_year'] + 1

                #                 df['fiscal_year'] = df['fiscal_year'].astype(str)
                                
                #                 # Calculate annual averages
                #                 annual_df = df.groupby('fiscal_year')['value'].mean().reset_index()
                                
                #                 # Calculate year-over-year growth rates
                #                 annual_df['prev_year_mean'] = annual_df['value'].shift(1)
                #                 annual_df['growth_rate'] = ((annual_df['value'] - annual_df['prev_year_mean']) / 
                #                                         annual_df['prev_year_mean']) * 100
                #                 annual_df['growth_rate'] = annual_df['growth_rate'].fillna(0).round(2)

                                


                                
                #                 # Prepare output with year as string
                #                 for _, row in annual_df.iterrows():
                #                     growth_data.append({
                #                         'Feature': feature_name,
                #                         'Model': model_data['Model'],
                #                         'Fiscal Year': row['fiscal_year'],  # This is now a string like "2023"
                #                         'Growth Rate (%)': row['growth_rate'],
                #                         'Fiscal Start Month': fiscal_start_month
                #                     })
                        
                #         return growth_data


                if 'saved_models' in st.session_state:
                    frequency_options = st.session_state.frequency_options
                    fiscal_start_month = st.session_state.fiscal_start_month

                    def calculate_feature_growth(forecast_results, fiscal_start_month=1):
                        """Calculate annual growth rates for each feature based on fiscal year averages"""
                        growth_data = []
                        
                        for model_key, model_data in forecast_results.items():
                            if 'feature_forecasts' not in model_data:
                                continue
                                
                            for feature_name, feature_data in model_data['feature_forecasts'].items():
                                # Get the model name from the forecast data (added in your update)
                                model_name = feature_data.get('Model', 'Unknown')
                                
                                # Combine actual and forecasted data
                                all_dates = pd.to_datetime(feature_data['actual_dates'] + feature_data['future_dates'])
                                all_values = feature_data['actual_values'] + feature_data['future_forecast']
                                
                                # Create DataFrame
                                df = pd.DataFrame({
                                    'date': all_dates,
                                    'value': all_values,
                                    'type': ['actual']*len(feature_data['actual_dates']) + ['forecast']*len(feature_data['future_dates'])
                                })
                                
                                # Calculate fiscal year as string (e.g. "2023")
                                df['fiscal_year'] = (df['date'] - pd.offsets.DateOffset(months=fiscal_start_month-1)).dt.year.astype(int)

                                # Adjust fiscal year label to match the starting year
                                if fiscal_start_month > 6:  # e.g., July (7) to Dec (12)
                                    df['fiscal_year'] = df['fiscal_year'] + 1

                                df['fiscal_year'] = df['fiscal_year'].astype(str)
                                
                                # Calculate annual averages
                                annual_df = df.groupby('fiscal_year')['value'].mean().reset_index()
                                
                                # Calculate year-over-year growth rates
                                annual_df['prev_year_mean'] = annual_df['value'].shift(1)
                                annual_df['growth_rate'] = ((annual_df['value'] - annual_df['prev_year_mean']) / 
                                                        annual_df['prev_year_mean']) * 100
                                annual_df['growth_rate'] = annual_df['growth_rate'].fillna(0).round(2)

                                # Prepare output with year as string
                                for _, row in annual_df.iterrows():
                                    growth_data.append({
                                        'Feature': feature_name,
                                        'Model': model_name,  # Using the model name from forecast data
                                        'Fiscal Year': row['fiscal_year'],  # This is now a string like "2023"
                                        'Growth Rate (%)': row['growth_rate'],
                                        'Fiscal Start Month': fiscal_start_month
                                    })
                        
                        return growth_data










                    # st.subheader("📈 Model Evaluation")
                    st.write("#### Model Evaluation")

                    from copy import deepcopy


                    # st.subheader("📈 Combined Feature Forecasts")

                    with st.expander("All Features Forecast and Growth Rates"):

                        st.markdown("#### Feature Forecasts")

                        # Collect all feature forecast data with dates
                        all_forecast_data = []
                        for model_key, model_data in st.session_state.saved_models.items():
                            if 'feature_forecasts' in model_data:
                                for feature_name, forecast_data in model_data['feature_forecasts'].items():
                                    # Convert dates to datetime objects
                                    actual_dates = pd.to_datetime(forecast_data['actual_dates'])
                                    future_dates = pd.to_datetime(forecast_data['future_dates'])
                                    
                                    # Create continuous date range
                                    all_dates = list(actual_dates) + list(future_dates)
                                    
                                    # Prepare data for plotting
                                    all_forecast_data.append({
                                        'feature': feature_name,
                                        'model_type': model_data['model_type'],
                                        'custom_name': model_data['custom_name'],
                                        'dates': all_dates,
                                        'values': np.concatenate([
                                            forecast_data['actual_values'],
                                            forecast_data['future_forecast']
                                        ]),
                                        'is_actual': [True]*len(forecast_data['actual_values']) + [False]*len(forecast_data['future_forecast'])
                                    })
                        if all_forecast_data:
                            # Create combined plot
                            fig = go.Figure()
                            
                            # Track which features we've added to legend
                            added_features = set()
                            
                            # Add traces for each feature
                            for data in all_forecast_data:
                                # Only add to legend once per feature
                                show_legend = data['feature'] not in added_features
                                added_features.add(data['feature'])
                                
                                # Create a color for this feature (consistent across models)
                                feature_color = f'hsl({hash(data["feature"]) % 360}, 70%, 50%)'
                                
                                # Plot actual values (solid line)
                                fig.add_trace(go.Scatter(
                                    x=data['dates'][:sum(data['is_actual'])],
                                    y=data['values'][:sum(data['is_actual'])],
                                    name=f"{data['feature']} (Actual)",
                                    line=dict(color=feature_color, width=2),
                                    legendgroup=data['feature'],
                                    showlegend=show_legend,
                                    mode='lines'
                                ))
                                
                                # Plot forecast values (dots connected with thin line)
                                fig.add_trace(go.Scatter(
                                    x=data['dates'][sum(data['is_actual'])-1:],  # Connect to last actual point
                                    y=data['values'][sum(data['is_actual'])-1:],
                                    name=f"{data['feature']} Forecast ({data['model_type']})",
                                    line=dict(color=feature_color, width=1, dash='dot'),
                                    marker=dict(size=6, symbol='circle'),  # Add dots
                                    legendgroup=data['feature'],
                                    showlegend=show_legend,
                                    mode='lines+markers',  # Both lines and markers
                                    opacity=0.8
                                ))
                            
                            # Update layout with right-side scrollable legend
                            fig.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Value",
                                hovermode="x unified",
                                legend=dict(
                                    orientation="v",
                                    yanchor="auto",  # Changed to auto for better positioning
                                    y=0.5,  # Center vertically
                                    xanchor="left",
                                    x=1.02,  # Position just outside the plot area
                                    bgcolor='rgba(255,255,255,0.7)',
                                    bordercolor='rgba(0,0,0,0.2)',
                                    borderwidth=1,
                                    font=dict(size=10),
                                    itemwidth=30  # Ensures consistent width for legend items
                                ),
                                margin=dict(l=50, r=150, b=50, t=30, pad=4),  # Adjusted margins
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No feature forecast data available")


                    

                        # growth_data = calculate_feature_growth(
                        #     st.session_state.saved_models,
                        #     # frequency=frequency_options,
                        #     fiscal_start_month=fiscal_start_month
                        # )

                        # if growth_data:
                        #     # Create pivot table
                        #     pivot_df = pd.DataFrame(growth_data).pivot_table(
                        #         index=['Feature', 'Model'],
                        #         columns='Fiscal Year',
                        #         values='Growth Rate (%)',
                        #         aggfunc='first'
                        #     ).reset_index()
                            
                        #     # Format display
                        #     st.dataframe(
                        #         pivot_df.style.format("{:.3f}", subset=pivot_df.columns[2:]),
                        #         use_container_width=True,
                        #         height=min(600, 50 + len(pivot_df) * 35)
                        #     )


                        # Calculate initial growth data
                        growth_data = calculate_feature_growth(
                            st.session_state.saved_models,
                            fiscal_start_month=fiscal_start_month
                        )

                        if growth_data:
                            # Create pivot table for editing
                            pivot_df = pd.DataFrame(growth_data).pivot_table(
                                index=['Feature', 'Model'],
                                columns='Fiscal Year',
                                values='Growth Rate (%)',
                                aggfunc='first'
                            ).reset_index()
                            
                            # Make the growth rates editable - use a unique key based on fiscal_start_month
                            editor_key = f"growth_rate_editor_{fiscal_start_month}"
                            edited_pivot = st.data_editor(
                                pivot_df.style.format("{:.3f}", subset=pivot_df.columns[2:]),
                                use_container_width=True,
                                height=min(600, 50 + len(pivot_df) * 35),
                                key=editor_key
                            )
                            
                            # # Check if any growth rates were modified
                            # if not edited_pivot.equals(pivot_df):
                            #     # Create a copy of the session state to modify
                            #     modified_models = deepcopy(st.session_state.saved_models)
                                
                            #     # Track if any changes were made
                            #     changes_made = False
                                
                            #     # Get the changes by comparing row by row
                            #     for i in range(len(pivot_df)):
                            #         original_row = pivot_df.iloc[i]
                            #         edited_row = edited_pivot.iloc[i]
                                    
                            #         if not original_row.equals(edited_row):
                            #             feature = original_row['Feature']
                            #             model = original_row['Model']
                                        
                            #             # Find changed columns
                            #             changed_cols = edited_row.index[original_row != edited_row]
                            #             for col in changed_cols:
                            #                 if col not in ['Feature', 'Model']:  # Skip index columns
                            #                     year = col
                            #                     old_value = original_row[col]
                            #                     new_value = edited_row[col]
                                                
                            #                     # Find the corresponding model and feature
                            #                     for model_key, model_data in modified_models.items():
                            #                         if model_data['model_type'] == model and 'feature_forecasts' in model_data:
                            #                             if feature in model_data['feature_forecasts']:
                            #                                 # Adjust the forecasted values based on growth rate change
                            #                                 forecast_data = model_data['feature_forecasts'][feature]
                                                            
                            #                                 # Convert dates to find the relevant forecast period
                            #                                 future_dates = pd.to_datetime(forecast_data['future_dates'])
                                                            
                            #                                 # Calculate fiscal years (matching the calculation function)
                            #                                 date_series = pd.Series(future_dates)
                            #                                 fiscal_years = (date_series - pd.offsets.DateOffset(months=fiscal_start_month-1)).dt.year.astype(int)
                            #                                 if fiscal_start_month > 6:
                            #                                     fiscal_years = fiscal_years + 1
                            #                                 fiscal_years = fiscal_years.astype(str)
                                                            
                            #                                 # Calculate adjustment factor
                            #                                 growth_diff = new_value - old_value
                            #                                 adjustment_factor = 1 + (growth_diff / 100)
                                                            
                            #                                 # Apply adjustment to values in this fiscal year
                            #                                 for i, fy in enumerate(fiscal_years):
                            #                                     if fy == year:
                            #                                         forecast_data['future_forecast'][i] *= adjustment_factor
                            #                                         changes_made = True
                                                            
                            #                                 # Update the session state
                            #                                 st.session_state.saved_models[model_key]['feature_forecasts'][feature] = forecast_data
                                
                            #     if changes_made:
                            #         # Show success message and force update
                            #         st.success("Forecast values updated based on growth rate changes!")
                            #         # time.sleep(0.5)  # Brief pause to show message
                            #         st.rerun()  # Force refresh to show updated forecasts


                            # Check if any growth rates were modified
                            if not edited_pivot.equals(pivot_df):
                                # Create a copy of the session state to modify
                                modified_models = deepcopy(st.session_state.saved_models)
                                
                                # Track changes
                                changes_made = False
                                
                                # Convert all numeric values to precise decimals first
                                original_data = pivot_df.apply(pd.to_numeric, errors='ignore')
                                edited_data = edited_pivot.apply(pd.to_numeric, errors='ignore')
                                
                                # Compare each cell precisely
                                for (i, row), (_, edited_row) in zip(original_data.iterrows(), edited_data.iterrows()):
                                    feature = row['Feature']
                                    model = row['Model']
                                    
                                    # Compare each year column
                                    for year_col in [col for col in row.index if col not in ['Feature', 'Model']]:
                                        original_val = row[year_col]
                                        edited_val = edited_row[year_col]
                                        
                                        # Skip if no meaningful change (accounting for floating point)
                                        if abs(edited_val - original_val) < 0.001:
                                            continue
                                            
                                        # Find matching models
                                        for model_key, model_data in modified_models.items():
                                            if (model_data['model_type'] == model and 
                                                'feature_forecasts' in model_data and 
                                                feature in model_data['feature_forecasts']):
                                                
                                                forecast_data = model_data['feature_forecasts'][feature]
                                                future_dates = pd.to_datetime(forecast_data['future_dates'])
                                                
                                                # Calculate fiscal years
                                                date_series = pd.Series(future_dates)
                                                fiscal_years = (date_series - pd.offsets.DateOffset(months=fiscal_start_month-1)).dt.year.astype(int)
                                                if fiscal_start_month > 6:
                                                    fiscal_years += 1
                                                fiscal_years = fiscal_years.astype(str)
                                                
                                                # Calculate adjustment while preserving integer nature
                                                is_integer_input = edited_val.is_integer() and original_val.is_integer()
                                                growth_diff = edited_val - original_val
                                                
                                                if is_integer_input:
                                                    # Use integer math when both inputs are whole numbers
                                                    adjustment = 1 + int(growth_diff)/100
                                                else:
                                                    # Use decimal math otherwise
                                                    adjustment = 1 + growth_diff/100
                                                
                                                # Apply adjustment
                                                for idx, fy in enumerate(fiscal_years):
                                                    if fy == year_col:
                                                        original_forecast = forecast_data['future_forecast'][idx]
                                                        
                                                        if isinstance(original_forecast, (int, np.integer)):
                                                            # For integer forecasts, round to nearest whole number
                                                            adjusted_value = round(original_forecast * adjustment)
                                                            # Preserve original type
                                                            if isinstance(original_forecast, int):
                                                                adjusted_value = int(adjusted_value)
                                                        else:
                                                            # For float forecasts, round to 2 decimals
                                                            adjusted_value = round(original_forecast * adjustment, 2)
                                                        
                                                        forecast_data['future_forecast'][idx] = adjusted_value
                                                        changes_made = True
                                
                                if changes_made:
                                    # Update session state
                                    st.session_state.saved_models = modified_models
                                    st.success("Forecasts updated successfully!")
                                    # time.sleep(0.3)  # Brief display before refresh
                                    st.rerun()

















                            


                    # Multiselect for custom names
                    custom_names = sorted(list({data['custom_name'] for data in st.session_state.saved_models.values()}))
                    selected_custom_names = st.multiselect(
                        "Filter by Custom Name(s):",
                        options=custom_names,
                        default=custom_names[0] if custom_names else None
                    )


                    # Filter models by selected custom names
                    filtered_models = {
                        key: data for key, data in st.session_state.saved_models.items() 
                        if not selected_custom_names or data['custom_name'] in selected_custom_names
                    }

                    if filtered_models:
                        # Show Performance and Elasticities (filtered)
                        # with st.expander("Performance Summary and Feature Elasticities", expanded=True):
                        #     # 1. Show Performance Summary
                        #     st.write(f"Performance Summary for selected models")
                            
                        #     performance_data = []
                        #     for data in filtered_models.values():
                        #         performance_data.append({
                        #             'Base Name':data['base_name'],
                        #             'Custom Name': data['custom_name'],
                        #             'Model Type': data['model_type'],
                        #             'Segment': data['segment'],
                        #             'MAPE': data.get('MAPE'),
                        #             'R-squared': data.get('R_squared'),
                        #             'Adj R-squared': data.get('Adjusted_R_squared'),
                        #             'Features Used': ', '.join(data.get('features_used', []))
                        #         })
                            
                        #     performance_df = pd.DataFrame(performance_data)
                        #     st.dataframe(
                        #         performance_df.sort_values(['Custom Name', 'Model Type', 'Segment']),
                        #         use_container_width=True,
                        #         height=min(400, 50 + len(performance_df) * 35)
                        #     )


                        with st.expander("Performance Summary and Feature Elasticities", expanded=True):
                            # 1. Show Performance Summary
                            st.write(f"Performance Summary for selected models")
                            
                            performance_data = []
                            for data in filtered_models.values():
                                # Helper function to safely format values as percentages
                                def format_as_percent(value):
                                    if value is None:
                                        return None
                                    if isinstance(value, str):
                                        # If already a string, try converting to float first (e.g., "0.95" → 95.00%)
                                        try:
                                            value = float(value)
                                        except ValueError:
                                            return value  # Return as-is if not a number
                                    return f"{float(value) * 100:.1f}%"  # Format as percentage
                                
                                performance_data.append({
                                    'Base Name': data['base_name'],
                                    'Custom Name': data['custom_name'],
                                    'Model Type': data['model_type'],
                                    'Segment': data['segment'],
                                    'MAPE': format_as_percent(data.get('MAPE')),
                                    'R-squared': format_as_percent(data.get('R_squared')),
                                    'Adj R-squared': format_as_percent(data.get('Adjusted_R_squared')),
                                    'Features Used': ', '.join(data.get('features_used', []))
                                })
                            
                            performance_df = pd.DataFrame(performance_data)
                            st.dataframe(
                                performance_df.sort_values(['Custom Name', 'Model Type', 'Segment']),
                                use_container_width=True,
                                height=min(400, 50 + len(performance_df) * 35))









                            
                            # # 2. Show Elasticities from session state
                            # st.write(f"Feature Elasticities for selected models")

                            # elasticity_data = []
                            # for model_key in filtered_models:
                            #     if model_key in st.session_state.elasticity_data:
                            #         data = st.session_state.saved_models[model_key]
                            #         for feature_name, feature_data in st.session_state.elasticity_data[model_key].items():
                            #             p_value = feature_data.get('p-value')
                            #             significance = ('***' if isinstance(p_value, (int, float)) and p_value < 0.001 
                            #                         else '**' if isinstance(p_value, (int, float)) and p_value < 0.01 
                            #                         else '*' if isinstance(p_value, (int, float)) and p_value < 0.05 
                            #                         else '')
                                        
                            #             elasticity_data.append({
                            #                 'Base Name':data['base_name'],
                            #                 'Custom Name': data['custom_name'],
                            #                 'Model Type': data['model_type'],
                            #                 'Segment': data['segment'],
                            #                 'Feature': feature_name,
                            #                 'Elasticity': feature_data.get('Elasticity'),
                            #                 'p-value': p_value,
                            #                 'Significance': significance
                            #             })

                            # if elasticity_data:
                            #     elasticity_df = pd.DataFrame(elasticity_data)
                                
                            #     # Convert to numeric safely
                            #     for col in ['Elasticity', 'p-value']:
                            #         if col in elasticity_df:
                            #             elasticity_df[col] = pd.to_numeric(elasticity_df[col], errors='coerce')
                                
                            #     # Sort by absolute elasticity
                            #     elasticity_df['abs_elasticity'] = elasticity_df['Elasticity'].abs()
                            #     elasticity_df = elasticity_df.sort_values(
                            #         ['Custom Name', 'Model Type', 'Segment', 'abs_elasticity'],
                            #         ascending=[True, True, True, False]
                            #     ).drop('abs_elasticity', axis=1)
                                
                            #     st.dataframe(
                            #         elasticity_df,
                            #         use_container_width=True,
                            #         column_config={
                            #             "Elasticity": st.column_config.NumberColumn(format="%.4f"),
                            #             "p-value": st.column_config.NumberColumn(format="%.4f")
                            #         },
                            #         height=min(400, 50 + len(elasticity_df) * 35)
                            #     )
                            # else:
                            #     st.warning("No feature elasticities data available for selected models")


                            # 2. Show Elasticities from session state
                            st.write(f"Feature Elasticities for selected models")

                            elasticity_data = []
                            for model_key in filtered_models:
                                if model_key in st.session_state.elasticity_data:
                                    data = st.session_state.saved_models[model_key]
                                    for feature_name, feature_data in st.session_state.elasticity_data[model_key].items():
                                        p_value = feature_data.get('p-value')
                                        significance = ('***' if isinstance(p_value, (int, float)) and p_value < 0.001 
                                                    else '**' if isinstance(p_value, (int, float)) and p_value < 0.01 
                                                    else '*' if isinstance(p_value, (int, float)) and p_value < 0.05 
                                                    else '')
                                        
                                        elasticity_data.append({
                                            'Base Name': data['base_name'],
                                            'Custom Name': data['custom_name'],
                                            'Model Type': data['model_type'],
                                            'Segment': data['segment'],
                                            'Feature': feature_name,
                                            'Elasticity': feature_data.get('Elasticity'),
                                            'p-value': p_value,
                                            'Significance': significance
                                        })

                            if elasticity_data:
                                elasticity_df = pd.DataFrame(elasticity_data)
                                
                                # Convert to numeric safely
                                for col in ['Elasticity', 'p-value']:
                                    if col in elasticity_df:
                                        elasticity_df[col] = pd.to_numeric(elasticity_df[col], errors='coerce')
                                
                                # Pivot the table to have features as columns
                                pivoted_df = elasticity_df.pivot_table(
                                    index=['Base Name', 'Custom Name', 'Model Type', 'Segment'],
                                    columns='Feature',
                                    values=['Elasticity', 'p-value'],
                                    aggfunc='first'
                                )
                                
                                # Flatten the multi-index columns
                                pivoted_df.columns = [f"{col[1]}_{col[0]}" for col in pivoted_df.columns]
                                pivoted_df = pivoted_df.reset_index()
                                
                                st.dataframe(
                                    pivoted_df,
                                    use_container_width=True,
                                    column_config={
                                        col: st.column_config.NumberColumn(format="%.4f") 
                                        for col in pivoted_df.columns 
                                        if 'Elasticity' in col or 'p-value' in col
                                    },
                                    height=min(400, 50 + len(pivoted_df) * 35))
                            else:
                                st.warning("No feature elasticities data available for selected models")











                            # Collect all relevant data from saved_models
                            complete_metrics = []
                            for data in filtered_models.values():
                                if 'feature_metrics' in data:
                                    for feature in data['feature_metrics']:
                                        complete_metrics.append({
                                            'Base Name': data['base_name'],
                                            'Custom Name': data['custom_name'],
                                            'Model Type': data['model_type'],
                                            'Segment': data['segment'],
                                            **feature  # Unpack all feature metrics
                                        })
                            
                            if complete_metrics:
                                # Create DataFrame from collected metrics
                                overall_df = pd.DataFrame(complete_metrics)
                                
                                # Reorder columns to match your desired display
                                column_order = [
                                    'Base Name', 'Custom Name', 'Model Type', 'Segment', 'Variable',
                                    'Elasticity', 'p_value', 'Feature_Mean', 'Feature_Std',
                                    'Target_Column', 'Target_Mean', 'intercept',
                                    'Prev_Year_Target_Mean', 'Prev_Year_Feature_Mean'
                                ]
                                
                                # Only keep columns that exist in the data
                                column_order = [col for col in column_order if col in overall_df.columns]
                                
                                st.dataframe(
                                    overall_df[column_order],
                                    use_container_width=True,
                                    column_config={
                                        "Elasticity": st.column_config.NumberColumn(format="%.4f"),
                                        "p_value": st.column_config.NumberColumn(format="%.4f"),
                                        "Feature_Mean": st.column_config.NumberColumn(format="%.4f"),
                                        "Feature_Std": st.column_config.NumberColumn(format="%.4f"),
                                        "Target_Mean": st.column_config.NumberColumn(format="%.4f"),
                                        "intercept": st.column_config.NumberColumn(format="%.4f"),
                                        "Prev_Year_Target_Mean": st.column_config.NumberColumn(format="%.4f"),
                                        "Prev_Year_Feature_Mean": st.column_config.NumberColumn(format="%.4f")
                                    },
                                    height=min(600, 50 + len(overall_df) * 35)
                                )
                            else:
                                st.warning("No complete metrics data available for selected models")
                    else:
                        st.warning("No models match the current filters")





















                # with st.expander("Model Results:"):

                #     if valid_features:


                #         if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in models:


                #             # Create a figure for each group
                #             for group in selected_group:
                #                 # Check if we have forecast results for this group
                #                 if group in forecast_results:
                #                     # Create the figure for this group
                #                     fig = go.Figure()
                #                     data = forecast_results[group]
                                    
                #                     # Actual Volume Data (always shown)
                #                     fig.add_trace(go.Scatter(
                #                         x=data['actual_dates'], 
                #                         y=data['actual_volume'],
                #                         mode='lines', 
                #                         line=dict(color='blue', width=2.5), 
                #                         name='Actual Volume',
                #                         legendgroup="actual"
                #                     ))

                #                     # Handle non-Prophet models if they exist
                #                     try:
                #                         if not df_final.empty and 'predicted_df' in locals():
                #                             # Filter predictions for this group
                #                             group_predictions = predicted_df[predicted_df['Segment'] == group]
                #                             for model_name, model_group in group_predictions.groupby('Model_type'):
                #                                 fig.add_trace(go.Scatter(
                #                                     x=model_group['Date'],
                #                                     y=model_group['Volume'],
                #                                     mode='lines',
                #                                     name=f'{model_name} Forecast',
                #                                     line=dict(dash='dot', width=3)
                #                                 ))
                #                     except NameError:
                #                         st.warning("`df_final_sp` is not defined.")
                #                     except Exception as e:
                #                         st.warning(f"An error occurred while displaying predictions: {e}")

                #                     # Handle Prophet forecasts if selected
                #                     if "Prophet" in models:
                #                         # Prophet Future Forecasts
                #                         fig.add_trace(go.Scatter(
                #                             x=future_dates, 
                #                             y=data['prophet_future_forecast'],
                #                             mode='lines', 
                #                             name='Prophet Forecast', 
                #                             line=dict(dash='dot', color='peru', width=3),
                #                             legendgroup="prophet"
                #                         ))

                #                     # Update layout for this group's figure
                #                     fig.update_layout(
                #                         title=f"Forecast for {group}", 
                #                         xaxis_title="Date", 
                #                         yaxis_title=target_col, 
                #                         template="plotly_dark",
                #                         legend=dict(
                #                             traceorder="normal",
                #                             itemsizing="constant"
                #                         )
                #                     )

                #                     # Display the figure for this group
                #                     st.plotly_chart(fig, use_container_width=True)
                #                     st.markdown('<hr class="thin">', unsafe_allow_html=True)


                            









                # with st.expander("Growth Rates:"):

                #     if valid_features:


                #         def calculate_and_display_growth_analysis(forecast_results, predicted_df, models, 
                #                                                 forecast_horizon=12, start_year=2020, 
                #                                                 fiscal_start_month=1, frequency="M"):
                            
                #             # Calculate and display growth rates for yearly, half-yearly, and quarterly periods
                #             # with all groups plotted together for each model.
                            
                #             # Add frequency selector
                #             analysis_freq = st.radio("Growth Rate Frequency:", 
                #                                     ["Yearly", "Half-Yearly", "Quarterly"],
                #                                     horizontal=True)
                            
                #             # Calculate growth rates for all available models
                #             growth_results = calculate_growth_rates_with_forecasts(
                #                 forecast_results=forecast_results,
                #                 predicted_df=predicted_df,
                #                 forecast_horizon=forecast_horizon,
                #                 start_year=start_year,
                #                 fiscal_start_month=fiscal_start_month,
                #                 frequency=frequency,
                #                 analysis_freq=analysis_freq.lower()
                #             )
                            
                #             # Prepare data for table and chart - keeping same DataFrame structure
                #             table_data = []
                #             chart_data = []
                            
                #             for segment, model_data in growth_results.items():
                #                 for model_name, result_df in model_data.items():
                #                     if model_name in models or (model_name == "Prophet" and "Prophet" in models):
                #                         table_entry = result_df.copy()
                #                         table_entry['Segment'] = segment
                #                         table_entry['Model'] = model_name
                #                         table_data.append(table_entry)
                                        
                #                         for _, row in result_df.iterrows():
                #                             chart_data.append({
                #                                 'Segment': segment,
                #                                 'Model': model_name,
                #                                 'period': row['period'],
                #                                 'growth_rate': row['growth_rate'],
                #                                 'volume': row['volume']
                #                             })
                            
                #             if not table_data:
                #                 st.warning("No growth rates available for selected models.")
                #                 return
                            
                #             # Create combined DataFrame (same structure as original)
                #             combined_df = pd.concat(table_data, ignore_index=True)
                #             chart_df = pd.DataFrame(chart_data)
                            
                #             # Display in columns
                #             col1, col2 = st.columns(2)
                            
                #             with col1:
                #                 # Create one chart with all groups for each model
                #                 fig = px.line(
                #                     chart_df,
                #                     x='period',
                #                     y='growth_rate',
                #                     color='Model',
                #                     line_dash='Segment',  # Different dash patterns for segments
                #                     labels={'growth_rate': 'Growth Rate (%)', 'period': 'Period'},
                #                     title=f'{analysis_freq} Growth Rate Comparison by Model',
                #                     markers=True,
                #                     text='growth_rate'
                #                 )
                                
                #                 fig.update_yaxes(tickformat=".1%")
                #                 fig.update_traces(
                #                     hovertemplate="<b>%{fullData.name}</b><br>" +
                #                                 "Segment: %{customdata[0]}<br>" +
                #                                 "Period: %{x}<br>" +
                #                                 "Growth Rate: %{y:.2%}<extra></extra>",
                #                     texttemplate='%{y:.1%}',
                #                     textposition='top center',
                #                     customdata=chart_df[['Segment']]
                #                 )
                                
                #                 fig.update_layout(
                #                     hovermode='x unified',
                #                     legend_title_text='Model/Segment',
                #                     template='plotly_white',
                #                     height=500,
                #                     uniformtext_minsize=8,
                #                     uniformtext_mode='hide'
                #                 )
                                
                #                 st.plotly_chart(fig, use_container_width=True)
                            
                #             with col2:
                #                 # Table remains the same
                #                 combined_df = combined_df.dropna()
                #                 st.dataframe(combined_df.style.format({
                #                     'growth_rate': '{:.4f}',
                #                     'volume': '{:,.2f}'
                #                 }), use_container_width=True)


                #         def calculate_growth_rates_with_forecasts(forecast_results, predicted_df=None, forecast_horizon=12, 
                #                                     start_year=2020, fiscal_start_month=1, frequency="M", 
                #                                     analysis_freq="yearly"):
                            
                #             # Calculate growth rates for different time periods with proper frequency handling
                        
                #             growth_results = {}
                            
                #             for segment, data in forecast_results.items():
                #                 actual_dates = pd.Series(pd.to_datetime(data['actual_dates']))
                #                 actual_volume = pd.Series(data['actual_volume'])
                                
                #                 segment_growth = {}
                                
                #                 # Process Prophet model if available
                #                 if "prophet_future_forecast" in data:
                #                     # Calculate proper date offset based on frequency
                #                     if frequency == "D":
                #                         offset = pd.DateOffset(days=1)
                #                     elif frequency == "W":
                #                         offset = pd.DateOffset(weeks=1)
                #                     elif frequency == "M":
                #                         offset = pd.DateOffset(months=1)
                #                     elif frequency == "Q":
                #                         offset = pd.DateOffset(months=3)
                #                     elif frequency == "Y":
                #                         offset = pd.DateOffset(years=1)
                #                     else:
                #                         offset = pd.DateOffset(months=1)  # Default to monthly
                                    
                #                     future_dates = pd.Series(pd.date_range(
                #                         start=actual_dates.max() + offset,  # Use the calculated offset
                #                         periods=forecast_horizon,
                #                         freq=frequency
                #                     ))
                                    
                #                     prophet_df = pd.DataFrame({
                #                         'date': pd.concat([actual_dates, future_dates], ignore_index=True),
                #                         'volume': pd.concat([actual_volume, 
                #                                         pd.Series(data['prophet_future_forecast'])], 
                #                                         ignore_index=True),
                #                         'Model_type': 'Prophet',
                #                         'Segment': segment  # Add segment information
                #                     })
                                    
                #                     prophet_result = _calculate_model_growth_by_freq(
                #                         prophet_df, start_year, fiscal_start_month, frequency, analysis_freq
                #                     )
                #                     segment_growth['Prophet'] = prophet_result
                                
                #                 # Process other models
                #                 if predicted_df is not None and not predicted_df.empty:
                #                     predicted_df['date'] = pd.to_datetime(predicted_df['Date'])
                                    
                #                     # Filter predicted_df for current segment if Segment column exists
                #                     if 'Segment' in predicted_df.columns:
                #                         segment_predicted_df = predicted_df[predicted_df['Segment'] == segment]
                #                     else:
                #                         segment_predicted_df = predicted_df
                                    
                #                     for model_name, model_df in segment_predicted_df.groupby('Model_type'):
                #                         model_combined = pd.DataFrame({
                #                             'date': pd.concat([actual_dates, 
                #                                             pd.Series(model_df['date'])], 
                #                                             ignore_index=True),
                #                             'volume': pd.concat([actual_volume, 
                #                                             pd.Series(model_df['Volume'])], 
                #                                             ignore_index=True),
                #                             'Model_type': model_name,
                #                             'Segment': segment  # Ensure segment is preserved
                #                         })
                                        
                #                         model_result = _calculate_model_growth_by_freq(
                #                             model_combined, start_year, fiscal_start_month, frequency, analysis_freq
                #                         )
                #                         segment_growth[model_name] = model_result
                                
                #                 growth_results[segment] = segment_growth
                            
                #             return growth_results

                #         def _calculate_model_growth_by_freq(df, start_year, fiscal_start_month, frequency_options, analysis_freq):
                #             """Calculate growth rates with proper period handling for all frequencies"""
                #             df = df.copy()
                #             df['date'] = pd.to_datetime(df['date'])
                #             df = df.sort_values('date')
                #             df = df[df['date'].dt.year >= start_year - 1]
                #             df.set_index('date', inplace=True)
                            
                #             # Determine period grouping based on analysis frequency
                #             if analysis_freq == "yearly":
                #                 # Handle fiscal year offset
                #                 if frequency_options in ["D", "W", "M", "Q"]:
                #                     df['period'] = (df.index - pd.offsets.DateOffset(months=fiscal_start_month-1)).year
                #                 else:
                #                     df['period'] = df.index.year
                                
                #                 # if not (1 <= fiscal_start_month <= 5):
                #                 #     df['period'] = df['period'] + 1
                #                 period_name = 'Year'
                                
                #             elif analysis_freq == "half-yearly":
                #                 # Create proper half-year periods considering fiscal year
                #                 if fiscal_start_month == 1:  # Calendar year
                #                     df['half'] = np.where(df.index.month <= 6, 'H1', 'H2')
                #                     df['period'] = df.index.year.astype(str) + '-' + df['half']
                #                 else:
                #                     # Adjust for fiscal year - convert to numpy array for modification
                #                     adjusted_month = (df.index.month - (fiscal_start_month - 1)).to_numpy()
                #                     adjusted_month[adjusted_month <= 0] += 12
                #                     df['half'] = np.where(adjusted_month <= 6, 'H1', 'H2')
                #                     year = np.where(adjusted_month <= 12, df.index.year, df.index.year + 1)
                #                     df['period'] = year.astype(str) + '-' + df['half']
                                
                #                 period_name = 'Half-Year'
                                
                #             elif analysis_freq == "quarterly":
                #                 # Create proper quarterly periods considering fiscal year
                #                 if fiscal_start_month == 1:  # Calendar year
                #                     df['period'] = df.index.to_period('Q').astype(str)
                #                 else:
                #                     # Adjust for fiscal year - convert to numpy array for modification
                #                     adjusted_month = (df.index.month - fiscal_start_month).to_numpy()
                #                     adjusted_month[adjusted_month < 0] += 12
                #                     df['quarter'] = (adjusted_month // 3) + 1
                #                     year = np.where(df.index.month >= fiscal_start_month, df.index.year, df.index.year - 1)
                #                     df['period'] = year.astype(str) + 'Q' + df['quarter'].astype(str)
                                
                #                 period_name = 'Quarter'
                            
                #             # Group by period and calculate mean volume
                #             period_df = df.groupby('period')['volume'].mean().reset_index()
                            
                #             # Calculate growth rates
                #             model_name = df['Model_type'].iloc[0]
                #             period_df['growth_rate'] = period_df['volume'].pct_change()
                #             period_df['Model'] = model_name
                #             period_df['period_type'] = period_name
                            
                #             # Sort periods chronologically
                #             if analysis_freq == "half-yearly":
                #                 period_df['sort_year'] = period_df['period'].str[:4].astype(int)
                #                 period_df['sort_half'] = np.where(period_df['period'].str.contains('H1'), 1, 2)
                #                 period_df = period_df.sort_values(['sort_year', 'sort_half']).drop(['sort_year', 'sort_half'], axis=1)
                #             elif analysis_freq == "quarterly":
                #                 period_df['sort_year'] = period_df['period'].str.extract(r'(\d+)Q')[0].astype(int)
                #                 period_df['sort_qtr'] = period_df['period'].str.extract(r'Q(\d+)')[0].astype(int)
                #                 period_df = period_df.sort_values(['sort_year', 'sort_qtr']).drop(['sort_year', 'sort_qtr'], axis=1)
                            
                #             return period_df


                            
                #         if any(m in models for m in ["Generalized Constrained Ridge", "Generalized Constrained Lasso", "Ridge", "Linear Regression"]) or "Prophet" in models:
                #             calculate_and_display_growth_analysis(
                #                 forecast_results=forecast_results,
                #                 predicted_df=predicted_df if 'predicted_df' in locals() else None,
                #                 models=models,
                #                 forecast_horizon=forecast_horizon,
                #                 start_year=min_year,
                #                 fiscal_start_month=fiscal_start_month
                #             )









    if __name__ == "__main__":
        main()


