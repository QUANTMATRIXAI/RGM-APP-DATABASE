



    def build_page():
        """
        COMPLETE: Price/Promo Elasticity code + Model Selection & Filtering in a single page.

        This function:
        1) Runs the aggregator pipeline, modeling, storing results in session_state.
        2) Lets the user filter & select models with st_aggrid.
        3) Displays contribution, radar, and bar charts.
        4) Allows final saving of models.
        """

        import streamlit as st
        import pandas as pd
        import numpy as np
        import math
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from statsmodels.tsa.seasonal import STL
        from pykalman import KalmanFilter

        from sklearn.base import BaseEstimator, RegressorMixin
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
        from sklearn.metrics import r2_score, mean_absolute_percentage_error

        from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
        import statsmodels.formula.api as smf
        # ─────────────────────────────────────────────────────────────────────────────
        # 1) NAVIGATION BUTTONS
        # ─────────────────────────────────────────────────────────────────────────────
        nav_home,  nav_transform,nav_model = st.columns(3)

        with nav_home:
            if st.button("🏠 Home", key="build_nav_home"):
                go_home()

        with nav_transform:
            if st.button("🔄 Back to select", key="build_nav_transform"):
                go_to("select_section3")
        with nav_model:
            if st.button("🧪 Evaluate", key="build_nav_model"):
                go_to("model_selection")


        # ─── END DATE FILTER ───────────────────────────────────────────

        class MixedEffectsModelWrapper(BaseEstimator, RegressorMixin):
            def __init__(self, random_effects=None, random_slopes=None):
                """
                Parameters:
                -----------
                random_effects : list
                    List of column names to use as random intercepts
                random_slopes : dict
                    Dictionary mapping group variables to lists of variables 
                    that have random slopes for that group
                    Example: {'Brand': ['PPU', 'D1']} - Both PPU and D1 have random slopes by Brand
                """
                self.random_effects = random_effects if random_effects else []
                self.random_slopes = random_slopes if random_slopes else {}
                self.coef_ = None
                self.intercept_ = None
                self.model = None
                self.result = None
                
            def fit(self, X, y, feature_names=None, groups=None):
                import statsmodels.formula.api as smf
                import pandas as pd
                
                # Create dataframe for statsmodels
                data = pd.DataFrame(X, columns=feature_names if feature_names is not None else X.columns)
                data['target'] = y
                
                # Add group variables if they're not in X
                if groups is not None:
                    for col in self.random_effects:
                        if col not in data.columns and col in groups.columns:
                            data[col] = groups[col].values
                
                # Construct formula
                fixed_effects = " + ".join(data.columns.drop('target'))
                
                # Add random intercepts
                random_parts = []
                for re in self.random_effects:
                    if re in data.columns:
                        random_parts.append(f"(1|{re})")
                
                # Add random slopes
                for group, slopes in self.random_slopes.items():
                    if group in data.columns:
                        slope_terms = " + ".join(slopes)
                        random_parts.append(f"(1 + {slope_terms}|{group})")
                
                random_formula = " + ".join(random_parts)
                formula = f"target ~ {fixed_effects}" + (f" + {random_formula}" if random_formula else "")
                
                try:
                    # Use first random effect as group
                    first_group = self.random_effects[0] if self.random_effects else None
                    self.model = smf.mixedlm(formula, data=data, groups=data[first_group] if first_group else None)
                    self.result = self.model.fit()
                    
                    # Extract coefficients for scikit-learn compatibility
                    self.intercept_ = self.result.params.get('Intercept', 0)
                    self.coef_ = np.zeros(X.shape[1])
                    
                    # Map coefficient names to indices
                    for i, col in enumerate(data.columns.drop('target')):
                        if col in self.result.params:
                            self.coef_[i] = self.result.params[col]
                    
                    return self
                except Exception as e:
                    print(f"Mixed model fitting error: {str(e)}")
                    # Fall back to linear regression
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(X, y)
                    self.intercept_ = lr.intercept_
                    self.coef_ = lr.coef_
                    return self
            
            def predict(self, X):
                # For prediction, we use fixed effects only
                return X @ self.coef_ + self.intercept_


        # ─────────────────────────────────────────────────────────────────────────────
        # 2) CUSTOM MODEL CLASSES
        # ─────────────────────────────────────────────────────────────────────────────
        class CustomConstrainedRidge(BaseEstimator, RegressorMixin):
            def __init__(self, l2_penalty=0.1, learning_rate=0.001, iterations=100000,
                        adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
                self.learning_rate = learning_rate
                self.iterations = iterations
                self.l2_penalty = l2_penalty
                self.adam = adam
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon

            def fit(self, X, Y, feature_names):
                self.m, self.n = X.shape
                self.W = np.zeros(self.n)
                self.b = 0
                self.X = X
                self.Y = Y
                self.feature_names = feature_names
                self.rpi_ppu_indices = [
                    i for i, name in enumerate(feature_names)
                    if name.endswith("_RPI") or name == "PPU"
                ]
                self.d1_index = next((i for i, name in enumerate(feature_names) if name == "D1"), None)

                if self.adam:
                    self.m_W = np.zeros(self.n)
                    self.v_W = np.zeros(self.n)
                    self.m_b = 0
                    self.v_b = 0
                    self.t = 0

                for _ in range(self.iterations):
                    self.update_weights()

                self.intercept_ = self.b
                self.coef_ = self.W
                return self

            def update_weights(self):
                Y_pred = self.predict(self.X)
                grad_w = (
                    -(2 * (self.X.T).dot(self.Y - Y_pred))
                    + 2 * self.l2_penalty * self.W
                ) / self.m
                grad_b = -(2 / self.m) * np.sum(self.Y - Y_pred)

                if self.adam:
                    self.t += 1
                    self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_w
                    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                    self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_w ** 2)
                    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)

                    m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
                    m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
                    v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
                    v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

                    self.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                    self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                else:
                    self.W -= self.learning_rate * grad_w
                    self.b -= self.learning_rate * grad_b

                # Constraints
                for i in range(self.n):
                    if i in self.rpi_ppu_indices and self.W[i] > 0:
                        self.W[i] = 0
                    if i == self.d1_index and self.W[i] < 0:
                        self.W[i] = 0

            def predict(self, X):
                return X.dot(self.W) + self.b


        class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
            def __init__(self, learning_rate=0.001, iterations=10000,
                        adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
                self.learning_rate = learning_rate
                self.iterations = iterations
                self.adam = adam
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon

            def fit(self, X, Y, feature_names):
                self.m, self.n = X.shape
                self.W = np.zeros(self.n)
                self.b = 0
                self.X = X
                self.Y = Y
                self.feature_names = feature_names
                self.rpi_ppu_indices = [
                    i for i, name in enumerate(feature_names)
                    if name.endswith('_RPI') or name == 'PPU'
                ]
                self.d1_index = next((i for i, name in enumerate(feature_names) if name == 'D1'), None)

                if self.adam:
                    self.m_W = np.zeros(self.n)
                    self.v_W = np.zeros(self.n)
                    self.m_b = 0
                    self.v_b = 0
                    self.t = 0

                for _ in range(self.iterations):
                    self.update_weights()

                self.intercept_ = self.b
                self.coef_ = self.W
                return self

            def update_weights(self):
                Y_pred = self.predict(self.X)
                dW = -(2 * self.X.T.dot(self.Y - Y_pred)) / self.m
                db = -2 * np.sum(self.Y - Y_pred) / self.m

                if self.adam:
                    self.t += 1
                    self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
                    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
                    self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
                    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

                    m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
                    m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
                    v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
                    v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

                    self.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                    self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                else:
                    self.W -= self.learning_rate * dW
                    self.b -= self.learning_rate * db

                # enforce constraints
                self.W[self.rpi_ppu_indices] = np.minimum(self.W[self.rpi_ppu_indices], 0)
                if self.d1_index is not None:
                    self.W[self.d1_index] = max(self.W[self.d1_index], 0)

            def predict(self, X):
                return X.dot(self.W) + self.b

        # ─────────────────────────────────────────────────────────────────────────────
        # 3) MODELS DICTIONARY
        # ─────────────────────────────────────────────────────────────────────────────
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
            "Bayesian Ridge Regression": BayesianRidge(),
            "Custom Constrained Ridge": CustomConstrainedRidge(l2_penalty=0.1, learning_rate=0.001, iterations=10000),
            "Constrained Linear Regression": ConstrainedLinearRegression(learning_rate=0.001, iterations=10000)
        }
    
        # ─────────────────────────────────────────────────────────────────────────────
        # 4) HELPER FUNCTIONS
        # ─────────────────────────────────────────────────────────────────────────────
        def safe_mape(y_true, y_pred):
            y_true = np.array(y_true, dtype=float)
            y_pred = np.array(y_pred, dtype=float)
            nonzero_mask = (y_true != 0)
            y_true_nonzero = y_true[nonzero_mask]
            y_pred_nonzero = y_pred[nonzero_mask]
            if len(y_true_nonzero) == 0:
                return float("nan")
            return np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
        
    
        
    
        def run_full_pipeline(raw_df, group_keys, pivot_keys, use_kalman=True, use_ratio_flag=False):
            """
            Cleans data, computes PPU, brand shares, outliers, etc.
            Returns a final df with 'FilteredVolume' for modeling.
            """
            import streamlit as st
            import pandas as pd
            import numpy as np
            from statsmodels.tsa.seasonal import STL
            from pykalman import KalmanFilter

            st.write("Preparing Data for Model....")

            # ── 1) Identify & convert your date column ─────────────────────────────────
            date_col = next((c for c in raw_df.columns if c.strip().lower() == 'date'), None)
            if not date_col:
                st.error("DataFrame must have a 'date' column.")
                st.stop()
            raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
            all_dates = raw_df[date_col].dt.date.dropna()
            if all_dates.empty:
                st.error("No valid dates in your data.")
                st.stop()
            min_date, max_date = all_dates.min(), all_dates.max()



            # ── 3) Helper functions ────────────────────────────────────────────────────
            def adjust_volume_column(df, chosen_col):
                df = df.copy()
                c = chosen_col.strip().lower()
                if c == 'volume':
                    df.drop(columns=['VolumeUnits'], errors='ignore', inplace=True)
                elif c == 'volumeunits':
                    df.drop(columns=['Volume'], errors='ignore', inplace=True)
                    df.rename(columns={'VolumeUnits': 'Volume'}, inplace=True, errors='ignore')
                else:
                    st.warning(f"Unrecognized volume column '{chosen_col}'.")
                return df

            def compute_category_weighted_price(df, d_date, d_channel):
                df = df.copy()
                df[d_date] = pd.to_datetime(df[d_date], errors='coerce')
                grp = df.groupby([d_channel, d_date])
                return (
                    grp.apply(lambda g: (g['PPU']*g['Volume']).sum()/g['Volume'].sum()
                            if g['Volume'].sum() else 0)
                    .reset_index(name='Cat_Weighted_Price')
                )

            def compute_cat_down_up(df, d_date, d_channel, l0=None, l2=None):
                df = df.copy()
                df[d_date] = pd.to_datetime(df[d_date], errors='coerce')
                mean_keys = [d_channel] + ([l0] if l0 else []) + ([l2] if l2 else [])
                daily_keys = [d_channel, d_date] + ([l0] if l0 else []) + ([l2] if l2 else [])
                mean_df = (
                    df.groupby(mean_keys)['PPU']
                    .mean().reset_index()
                    .rename(columns={'PPU':'mean_ppu'})
                )
                daily = (
                    df.groupby(daily_keys)['Volume']
                    .sum().reset_index()
                    .merge(mean_df, on=mean_keys, how='left')
                )
                total = (
                    daily.groupby([d_channel, d_date])['Volume']
                        .sum().reset_index().rename(columns={'Volume':'total_volume'})
                )
                daily = daily.merge(total, on=[d_channel, d_date], how='left')
                daily['weighted_contrib'] = daily['mean_ppu'] * (daily['Volume']/daily['total_volume'])
                return (
                    daily.groupby([d_channel, d_date])['weighted_contrib']
                        .sum().reset_index()
                        .rename(columns={'weighted_contrib':'Cat_Down_Up'})
                )

            # ── 4) Adjust volume column ────────────────────────────────────────────────
            df_proc = raw_df.copy()   # start with the already-filtered data
            selected_volume = st.session_state.get("selected_volume", "Volume")
            df_proc = adjust_volume_column(df_proc, selected_volume)

            # ── 5) Identify date & channel columns ────────────────────────────────────
            d_date = next((c for c in df_proc.columns if c.strip().lower()=='date'), date_col)
            d_channel = next((c for c in df_proc.columns if c.strip().lower()=='channel'), 'Channel')

            # ── 6) Aggregate to PPU ───────────────────────────────────────────────────
            if "Price" in df_proc.columns and "SalesValue" in df_proc.columns:
                agg_df = (
                    df_proc.groupby(group_keys)
                        .agg({"Volume":"sum","Price":"mean","SalesValue":"sum"})
                        .reset_index()
                        .rename(columns={"Price":"PPU"})
                )
            elif "Price" in df_proc.columns:
                agg_df = (
                    df_proc.groupby(group_keys)
                        .agg({"Volume":"sum","Price":"mean"})
                        .reset_index()
                        .rename(columns={"Price":"PPU"})
                )
            else:
                agg_df = (
                    df_proc.groupby(group_keys)
                        .agg({"Volume":"sum","SalesValue":"sum"})
                        .reset_index()
                )
                agg_df["PPU"] = np.where(
                    agg_df["Volume"] != 0,
                    agg_df["SalesValue"]/agg_df["Volume"],
                    0
                )

            # Add Year/Month/Week and a datetime64 'Date' for internal joins
            agg_df[d_date] = pd.to_datetime(agg_df[d_date], errors='coerce')
            agg_df['Year']  = agg_df[d_date].dt.year
            agg_df['Month'] = agg_df[d_date].dt.month
            agg_df['Week']  = agg_df[d_date].dt.isocalendar().week
            agg_df['Date']  = agg_df[d_date].dt.normalize()    # datetime64[ns]

            # pivot competitor PPU
            if pivot_keys:
                pivot_df = agg_df.pivot_table(index=[d_date, d_channel], columns=pivot_keys, values='PPU')
                agg_df = pd.concat([agg_df.set_index([d_date, d_channel]), pivot_df], axis=1).reset_index()
                if isinstance(pivot_df.columns, pd.MultiIndex):
                    for col_tuple in pivot_df.columns:
                        comp_col = "_".join(map(str, col_tuple)) + "_PPU"
                        agg_df[comp_col] = agg_df[col_tuple]
                        cond = True
                        for i, key in enumerate(pivot_keys):
                            cond &= (agg_df[key] == col_tuple[i])
                        agg_df.loc[cond, comp_col] = np.nan
                else:
                    for val in pivot_df.columns:
                        comp_col = f"{val}_PPU"
                        agg_df[comp_col] = agg_df[val]
                        cond = (agg_df[pivot_keys[0]] == val)
                        agg_df.loc[cond, comp_col] = np.nan

                try:
                    agg_df.drop(columns=pivot_df.columns, inplace=True)
                except Exception as e:
                    st.warning("Could not drop pivot columns: " + str(e))

                # rename pivoted -> RPI
                agg_df.columns = [
                    c.replace('_PPU','_RPI') if isinstance(c,str) and c.endswith('_PPU') else c
                    for c in agg_df.columns
                ]
                own_ppu = agg_df["PPU"]
                for col in agg_df.columns:
                    if isinstance(col, str) and col.endswith('_RPI') and col != "PPU_RPI":
                        agg_df[col] = np.where(agg_df[col] != 0, own_ppu / agg_df[col], 0)


            # ── 8) Category & market‐share metrics ────────────────────────────────────
            catvol = (
                agg_df.groupby([d_channel, d_date])['Volume']
                    .sum().reset_index(name='CatVol')
            )
            agg_df = agg_df.merge(catvol, on=[d_channel, d_date], how='left')
            agg_df['NetCatVol'] = agg_df['CatVol'] - agg_df['Volume']

            # Prepare brand_totals for later
            keys_for_brand = [d_channel] + (pivot_keys or [])
            brand_totals = (
                raw_df.groupby(keys_for_brand)['SalesValue']
                    .sum().reset_index(name='BrandSales')
            )
            channel_totals = (
                raw_df.groupby(d_channel)['SalesValue']
                    .sum().reset_index(name='ChannelSales')
            )
            brand_totals = brand_totals.merge(channel_totals, on=[d_channel], how='left')
            brand_totals['MarketShare_overall'] = (
                brand_totals['BrandSales']/brand_totals['ChannelSales']*100
            ).fillna(0)

            # ── 9) Seasonality & price trend ─────────────────────────────────────────
            season = (
                agg_df.groupby([d_channel,'Month'])['Volume']
                    .mean().reset_index(name='CatSeasonality')
            )
            agg_df = agg_df.merge(season, on=[d_channel,'Month'], how='left')

            cwp = compute_category_weighted_price(agg_df, d_date, d_channel)
            cdu = compute_cat_down_up(
                agg_df, d_date, d_channel,
                pivot_keys[0] if pivot_keys else None,
                pivot_keys[1] if len(pivot_keys or [])>1 else None
            )
            trend = pd.merge(cwp, cdu, on=[d_channel, d_date], how='inner')
            trend['mean_cat_down_up'] = trend.groupby(d_channel)['Cat_Down_Up'].transform('mean')
            trend['Cat_Price_trend_over_time'] = (
                trend['Cat_Weighted_Price'] *
                (trend['mean_cat_down_up']/trend['Cat_Down_Up'])
            )
            agg_df = agg_df.merge(
                trend[[d_channel,d_date,'Cat_Weighted_Price','Cat_Down_Up','Cat_Price_trend_over_time']],
                on=[d_channel,d_date], how='left'
            )

            # ── 10) Outlier detection ─────────────────────────────────────────────────
            final_df = agg_df.copy().set_index(d_date)
            final_df[['residual','z_score_residual','is_outlier']] = np.nan, np.nan, 0
            outlier_keys = [d_channel] + (pivot_keys or [])
            for name, grp in final_df.groupby(outlier_keys):
                if len(grp)<2: continue
                grp0 = grp.reset_index()
                try:
                    res = STL(grp0['Volume'], seasonal=13, period=13).fit()
                    grp0['residual'] = res.resid
                    grp0['z_score_residual'] = (
                        (grp0['residual']-grp0['residual'].mean())/grp0['residual'].std()
                    )
                    grp0['is_outlier'] = (grp0['z_score_residual'].abs()>3).astype(int)
                    for _, row in grp0.iterrows():
                        dt = row[d_date]
                        final_df.at[dt,'residual'] = row['residual']
                        final_df.at[dt,'z_score_residual'] = row['z_score_residual']
                        final_df.at[dt,'is_outlier'] = row['is_outlier']
                except Exception as e:
                    st.warning(f"STL failed for {name}: {e}")

            final_df.reset_index(inplace=True)
            final_df.sort_values(by=d_date, inplace=True)

            # ── 11) Kalman smoothing ───────────────────────────────────────────────────
            if use_kalman:
                def apply_kf(vals):
                    kf = KalmanFilter(initial_state_mean=vals[0], n_dim_obs=1)
                    means,_ = kf.filter(vals)
                    return means.flatten()

                final_df['FilteredVolume'] = np.nan
                for _, grp in final_df.groupby([d_channel] + (pivot_keys or [])):
                    grp_s = grp.sort_values(d_date).reset_index()
                    filt = apply_kf(grp_s['Volume'].values)
                    final_df.loc[grp_s['index'],'FilteredVolume'] = filt
            else:
                final_df['FilteredVolume'] = final_df['Volume']

            if use_ratio_flag:
                final_df['FilteredVolume'] = np.where(
                    final_df['CatVol']!=0,
                    final_df['FilteredVolume']/final_df['CatVol'],
                    0
                )

            final_df.fillna(0, inplace=True)

            # ── 12) Merge back extra raw columns ────────────────────────────────────────
            raw_copy = raw_df.copy()
            raw_copy[date_col] = pd.to_datetime(raw_copy[date_col], errors='coerce')
            # both sides use datetime64 normalized to midnight
            raw_copy['Date'] = raw_copy[date_col].dt.normalize()
            final_df['Date']   = final_df['Date'].dt.normalize()

            merge_keys = [d_channel, 'Date'] + (pivot_keys or [])
            used = list(final_df.columns) + merge_keys
            extras = [c for c in raw_copy.columns if c not in used]
            extra_df = raw_copy[merge_keys + extras].drop_duplicates(subset=merge_keys)

            final_df = final_df.merge(extra_df, on=merge_keys, how='left').fillna(0)

            # ── 13) Attach market share as 'Contribution' ─────────────────────────────
            final_df = final_df.merge(
                brand_totals[keys_for_brand + ['MarketShare_overall']],
                on=keys_for_brand, how='left'
            )
            final_df.rename(columns={'MarketShare_overall':'Contribution'}, inplace=True)
            final_df['Contribution'] = final_df['Contribution'].fillna(0)

            return final_df


        # ────────────────────────────────────────────────────────────────────────
        #  PER‑FOLD MODEL PIPELINE  •  uses in‑fold own‑price for MCV
        # ────────────────────────────────────────────────────────────────────────
        def run_model_pipeline(
                final_df,
                grouping_keys,
                X_columns,
                target_col,
                k_folds,
                chosen_std_cols,
                model_dict=None
            ):
            """
            Returns one row per fold with columns:
                grouping_keys + ["Model","Fold","CSF","MCV","SelfElasticity",
                                "PPU_at_Elasticity",
                                "B0 (Original)","R2 Train","R2 Test","MAPE Train","MAPE Test",
                                "MSE Train","MSE Test",
                                <mean‑X columns>, <Beta_… columns>, "ElasticityFlag"]
            Fold numbering restarts at 1 for every model‑within‑group.
            """

            import numpy as np, pandas as pd, streamlit as st
            from sklearn.model_selection import KFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score

            rows, preds_records = [], []          # collect results here

            # helper to append Beta_… cols
            def _add_beta_cols(d, names, coefs):
                for c, b in zip(names, coefs):
                    d[f"Beta_{c}"] = b

            grouped = final_df.groupby(grouping_keys) if grouping_keys else [((None,), final_df)]

            for gvals, gdf in grouped:

                gvals = (gvals,) if not isinstance(gvals, tuple) else gvals
                contrib = gdf.get("Contribution", np.nan).iloc[0]

                present_cols = [c for c in X_columns if c in gdf.columns]
                if len(present_cols) < len(X_columns):
                    st.warning(f"Skipping {gvals} — missing predictors.");  continue

                X_full = gdf[present_cols].fillna(0).copy()
                y_full = gdf[target_col].copy()
                if len(X_full) < k_folds:
                    continue

                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

                # Use provided model dictionary or global models
                models_to_use = model_dict if model_dict is not None else models
                for mname, mdl in models_to_use.items():
                    fold_id = 0                                         # restart per model

                    for tr_idx, te_idx in kf.split(X_full, y_full):
                        fold_id += 1

                        X_tr, X_te = X_full.iloc[tr_idx].copy(), X_full.iloc[te_idx].copy()
                        y_tr, y_te = y_full.iloc[tr_idx], y_full.iloc[te_idx]

                        # optional standardisation
                        scaler = {}
                        if chosen_std_cols:
                            sc = StandardScaler().fit(X_tr[chosen_std_cols])
                            X_tr[chosen_std_cols] = sc.transform(X_tr[chosen_std_cols])
                            X_te[chosen_std_cols] = sc.transform(X_te[chosen_std_cols])
                            scaler = {c: (m, s) for c, m, s
                                    in zip(chosen_std_cols, sc.mean_, sc.scale_)}

                        # fit / predict
                        # fit / predict
                        if isinstance(mdl, MixedEffectsModelWrapper):
                            # Extract grouping data from original dataframe
                            groups_data = gdf.loc[tr_idx, grouping_keys].copy()
                            mdl.fit(X_tr, y_tr, present_cols, groups_data)
                            y_tr_pred, y_te_pred = mdl.predict(X_tr), mdl.predict(X_te)
                            B0_std, B1_std = mdl.intercept_, mdl.coef_
                        elif mname in ["Custom Constrained Ridge", "Constrained Linear Regression"]:
                            mdl.fit(X_tr.values, y_tr.values, X_tr.columns.tolist())
                            y_tr_pred, y_te_pred = mdl.predict(X_tr.values), mdl.predict(X_te.values)
                            B0_std, B1_std = mdl.intercept_, mdl.coef_
                        else:
                            mdl.fit(X_tr, y_tr)
                            y_tr_pred, y_te_pred = mdl.predict(X_tr), mdl.predict(X_te)
                            B0_std, B1_std = mdl.intercept_, mdl.coef_

                        # metrics
                        r2_tr, r2_te = r2_score(y_tr, y_tr_pred), r2_score(y_te, y_te_pred)
                        mape_tr, mape_te = safe_mape(y_tr, y_tr_pred), safe_mape(y_te, y_te_pred)
                        mse_tr,  mse_te  = np.mean((y_tr - y_tr_pred)**2), np.mean((y_te - y_te_pred)**2)

                        # back‑transform coefs if std‑ised
                        raw_int, raw_coefs = B0_std, B1_std.copy()
                        for i, col in enumerate(present_cols):
                            if col in scaler:
                                mu, sd = scaler[col]
                                raw_coefs[i] = raw_coefs[i] / sd
                                raw_int     -= raw_coefs[i] * mu

                        # elasticity
                        mean_x = X_full.mean(numeric_only=True).to_dict()
                        q_hat  = raw_int + sum(raw_coefs[i] * mean_x.get(c, 0) for i, c in enumerate(present_cols))

                        dQdP = 0.0
                        if "PPU" in present_cols:
                            dQdP += raw_coefs[present_cols.index("PPU")]
                        for c in [c for c in present_cols if c.endswith("_RPI")]:
                            idx, ratio = present_cols.index(c), mean_x.get(c, 0)
                            P_own = mean_x.get("PPU", 0)
                            if ratio and P_own:
                                dQdP += raw_coefs[idx] / (P_own / ratio)

                        self_elas = (dQdP * mean_x.get("PPU", 0) / q_hat
                                    if (q_hat > 0 and mean_x.get("PPU", 0) > 0) else np.nan)
                        elas_flag = "ELASTICITY>100" if np.isfinite(self_elas) and abs(self_elas) > 100 else ""

                        # assemble row
                        d = {k: v for k, v in zip(grouping_keys, gvals)}
                        d.update({
                            "Model": mname,
                            "Fold":  fold_id,
                            "SelfElasticity": self_elas,
                            "PPU_at_Elasticity": mean_x.get("PPU", np.nan),
                            "B0 (Original)": raw_int,
                            "R2 Train": r2_tr, "R2 Test": r2_te,
                            "MAPE Train": mape_tr, "MAPE Test": mape_te,
                            "MSE Train": mse_tr,  "MSE Test": mse_te,
                            "ElasticityFlag": elas_flag,
                            "Contribution": contrib
                        })
                        # mean‑X cols
                        for c, v in mean_x.items(): d[c] = v
                        # Beta_… cols
                        _add_beta_cols(d, present_cols, raw_coefs)
                        rows.append(d)

                        # predictions
                        pr = gdf.loc[X_te.index].copy()
                        pr["Actual"], pr["Predicted"] = y_te.values, y_te_pred
                        pr["Model"], pr["Fold"] = mname, fold_id
                        preds_records.append(pr)

            if not rows:
                st.warning("No fold‑level results.");  return None, None

            df = pd.DataFrame(rows)

            # KPI columns
            df["CSF"] = df["SelfElasticity"].apply(lambda x: 1 - (1/x) if x and x != 0 else np.nan)
            df["MCV"] = df["CSF"] * df["PPU_at_Elasticity"]

            # tidy order (optional)
            front = grouping_keys + ["Model","Fold","CSF","MCV","SelfElasticity","PPU_at_Elasticity"]
            metric_block = ["B0 (Original)","R2 Train","R2 Test","MAPE Train","MAPE Test",
                            "MSE Train","MSE Test","Contribution","ElasticityFlag"]
            mean_x_cols  = [c for c in df.columns
                            if c not in front + metric_block and not c.startswith("Beta_")]
            beta_cols    = [c for c in df.columns if c.startswith("Beta_")]
            df = df[front + metric_block + mean_x_cols + beta_cols]

            df.sort_values(by=grouping_keys + ["Model","Fold"], inplace=True, ignore_index=True)

            preds_df = pd.concat(preds_records, ignore_index=True) if preds_records else None
            return df, preds_df



        # ─────────────────────────────────────────────────────────────────────────────
        # 5) MAIN STREAMLIT UI LOGIC
        # ─────────────────────────────────────────────────────────────────────────────
        st.subheader("Price/Promo Elasticity – Aggregation & Modeling")

        # Retrieve main data
        dataframe = st.session_state.get("D0", None)
        if dataframe is None:
            st.error("No data found (st.session_state['D0']). Please upload a file.")
            return


        # ─── START DATE FILTER ─────────────────────────────────────────
        date_col = next(c for c in dataframe.columns if c.strip().lower() == "date")
        dataframe[date_col] = pd.to_datetime(dataframe[date_col], errors="coerce")

        valid_dates = dataframe[date_col].dt.date.dropna()
        min_date, max_date = valid_dates.min(), valid_dates.max()

        if "filter_start_date" not in st.session_state:
            st.session_state["filter_start_date"] = min_date
        if "filter_end_date" not in st.session_state:
            st.session_state["filter_end_date"] = max_date

        st.sidebar.subheader("⏳ Time Period Filter")
        st.sidebar.date_input(
            "Start Date",
            value=st.session_state["filter_start_date"],
            min_value=min_date,
            max_value=max_date,
            key="filter_start_date",
        )
        st.sidebar.date_input(
            "End Date",
            value=st.session_state["filter_end_date"],
            min_value=min_date,
            max_value=max_date,
            key="filter_end_date",
        )

        df_filtered = dataframe[
            (dataframe[date_col].dt.date >= st.session_state["filter_start_date"])
            & (dataframe[date_col].dt.date <= st.session_state["filter_end_date"])
        ].copy()

        if df_filtered.empty:
            st.error("No data in the chosen date range.")
            return
        # ─── END DATE FILTER ───────────────────────────────────────────
        
            # ─── RESET CACHES WHEN DATE FILTER CHANGES ─────────────────────
        start_new = st.session_state["filter_start_date"]
        end_new   = st.session_state["filter_end_date"]

        # remember last dates across reruns
        if "last_filter_start" not in st.session_state:
            st.session_state["last_filter_start"] = start_new
            st.session_state["last_filter_end"]   = end_new

        # if user changed either start or end, clear cached results
        if (start_new != st.session_state["last_filter_start"] or
            end_new   != st.session_state["last_filter_end"]):

            st.session_state["last_filter_start"] = start_new
            st.session_state["last_filter_end"]   = end_new

            # wipe cached aggregations & model outputs
            st.session_state["final_df"]          = None
            st.session_state["combined_results"]  = None
            st.session_state["predictions_df"]    = None
            st.session_state["type2_dfs"]         = {}
            st.session_state["type2_results"]     = {}
            st.session_state["type2_predictions"] = {}
        # ─── END RESET BLOCK ───────────────────────────────────────────
        
        # ─── Choose aggregator parameters ───────────────────────────────
        col1, col2 = st.columns(2)

        # Volume selector
        with col1:
            vol_options = []
            if "Volume" in dataframe.columns:
                vol_options.append("Volume")
            if "VolumeUnits" in dataframe.columns:
                vol_options.append("VolumeUnits")
            selected_volume = st.selectbox("Select Volume column:", options=vol_options)
            st.session_state["selected_volume"] = selected_volume

        # Model-type selector
        with col2:
            model_type = st.radio(
                "Select Model Type:",
                options=["Type 1 (Three Distinct Keys)", "Type 2 (Multiple Single Keys)"]
            )

        # <-- NEW LOGIC: If user changes model_type, reset old stored results
        if "previous_model_type" not in st.session_state:
            st.session_state["previous_model_type"] = model_type
        if st.session_state["previous_model_type"] != model_type:
            st.session_state["previous_model_type"] = model_type
            st.session_state["final_df"] = None
            st.session_state["combined_results"] = None
            st.session_state["predictions_df"] = None
            st.session_state["type2_dfs"] = {}
            st.session_state["type2_results"] = {}
            st.session_state["type2_predictions"] = {}

        use_kalman = st.checkbox("Use Kalman Filter?", value=True)
        use_ratio  = st.checkbox("Use FilteredVolume as Ratio?", value=False)

        if model_type == "Type 1 (Three Distinct Keys)":
            possible_keys = [c for c in ["Brand","Variant","PackType","PPG","PackSize"] if c in dataframe.columns]
            c1,c2,c3 = st.columns(3)
            with c1:
                key1 = st.selectbox("Key 1:", options=possible_keys)
            with c2:
                remainA = [x for x in possible_keys if x!=key1]
                key2 = st.selectbox("Key 2:", options=remainA)
            with c3:
                remainB = [x for x in remainA if x!=key2]
                key3 = st.selectbox("Key 3:", options=remainB)

            selected_keys = [key1,key2,key3]
            group_keys = [
                next((c for c in dataframe.columns if c.strip().lower()=='date'), 'date'),
                next((c for c in dataframe.columns if c.strip().lower()=='channel'),'Channel')
            ] + selected_keys
            
            

            # RUN FULL PIPELINE only if we haven't done it yet (or if user wants to refresh)
            if "final_df" not in st.session_state or st.session_state["final_df"] is None:
                # Actually run the aggregator
                final_agg_df = run_full_pipeline(
                    df_filtered,
                    group_keys=group_keys,
                    pivot_keys=selected_keys,
                    use_kalman=use_kalman,
                    use_ratio_flag=use_ratio
                )
                st.session_state["final_df"] = final_agg_df
            else:
                final_agg_df = st.session_state["final_df"]

            with st.expander("📊 Aggregated Data (Type 1)", expanded=False):
                st.dataframe(final_agg_df, height=600, use_container_width=True)

            st.session_state.model_type = "Type 1"

            # MODELING
            st.title("Modeling")
            modeling_df = st.session_state.get("final_df", None)
            if modeling_df is None:
                st.warning("No aggregated DataFrame found for Type 1.")
                return

            available_cols = sorted(modeling_df.columns)
            default_predictors = [
                c for c in available_cols
                if c.endswith("_RPI") or c in ["PPU","D1","is_outlier","NetCatVol","Cat_Down_Up"]
            ]
            selected_predictors = st.multiselect(
                "Select Predictor Columns:",
                options=available_cols,
                default=default_predictors
            )
            grouping_keys_model = [
                next((col for col in modeling_df.columns if col.strip().lower()=='channel'),'Channel')
            ] + selected_keys
            X_columns = [c for c in selected_predictors if c not in grouping_keys_model]
            target_col = "FilteredVolume"
            k_folds = st.number_input("Number of folds (k):", min_value=2, max_value=20, value=5)

            numeric_in_X = [
                c for c in X_columns
                if c in modeling_df.columns and pd.api.types.is_numeric_dtype(modeling_df[c])
            ]
            default_std = [
                c for c in numeric_in_X
                if c in ["D1","PPU","NetCatVol","Cat_Weighted_Price","Cat_Down_Up"]
            ]
            chosen_std_cols = st.multiselect(
                "Select columns to standardize:",
                numeric_in_X,
                default=default_std
            )


            # Add option for mixed models
            st.subheader("Mixed Models Options")
            use_mixed_models = st.checkbox("Use Mixed Effects Models", value=False, 
                                        help="Mixed effects models can estimate different price elasticities for different brands/channels")

            if use_mixed_models:
                st.write("Mixed models allow you to estimate how price sensitivity varies across brands and channels:")
                
                # User-friendly explanation
                st.markdown("""
                **Random Intercepts**: Allow different baseline volumes for groups  
                **Random Slopes**: Allow different price/promo sensitivity for groups
                """)
                
                # Random intercepts selection
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Step 1:** Select groups that have different baseline volumes:")
                    random_intercepts = st.multiselect(
                        "Groups with different baselines:",
                        options=grouping_keys_model,
                        default=[grouping_keys_model[0]] if grouping_keys_model else []
                    )
                
                # Random slopes selection
                with col2:
                    if random_intercepts:
                        st.write("**Step 2:** Select which effects vary by group:")
                        
                        slope_options = [
                            {"group": ri, "variable": "PPU", "label": f"Price sensitivity varies by {ri}"} 
                            for ri in random_intercepts
                        ] + [
                            {"group": ri, "variable": "D1", "label": f"Promo sensitivity varies by {ri}"}
                            for ri in random_intercepts if "D1" in X_columns
                        ]
                        
                        selected_slopes = []
                        for option in slope_options:
                            if st.checkbox(option["label"], 
                                        value=option["variable"]=="PPU" and option["group"]==random_intercepts[0] if random_intercepts else False):
                                selected_slopes.append(option)
                
                # Create random_slopes dictionary
                random_slopes = {}
                for slope in selected_slopes:
                    group = slope["group"]
                    var = slope["variable"]
                    if group not in random_slopes:
                        random_slopes[group] = []
                    random_slopes[group].append(var)
                
                # Add mixed models to the models dictionary
                if random_intercepts:
                    # Basic mixed model with just random intercepts
                    intercepts_str = "+".join(random_intercepts)
                    models[f"Mixed({intercepts_str})"] = MixedEffectsModelWrapper(
                        random_effects=random_intercepts
                    )
                    
                    # Add models with random slopes
                    for group, slopes in random_slopes.items():
                        slopes_str = "+".join(slopes)
                        models[f"Mixed({group}, {slopes_str})"] = MixedEffectsModelWrapper(
                            random_effects=random_intercepts,
                            random_slopes={group: slopes}
                        )
                        
            
            
            # <-- NEW LOGIC: If we already have results in session, show them; else give button to run
            if "combined_results" in st.session_state and st.session_state["combined_results"] is not None:
                st.write("### Existing Model Results (Type 1)")
                st.dataframe(st.session_state["combined_results"], height=500, use_container_width=True)

                # Add a button to re-run if desired
                if st.button("Re-run Models"):
                    res, preds = run_model_pipeline(
                        modeling_df,
                        grouping_keys_model,
                        X_columns,
                        target_col,
                        k_folds,
                        chosen_std_cols
                    )
                    st.session_state["combined_results"] = res
                    st.session_state["predictions_df"]   = preds
                    if res is not None:
                        st.dataframe(res, height=500, use_container_width=True)

            else:
                # We don't have results yet, show "Run Models" button
                if st.button("Run Models"):
                    res, preds = run_model_pipeline(
                        modeling_df,
                        grouping_keys_model,
                        X_columns,
                        target_col,
                        k_folds,
                        chosen_std_cols
                    )
                    st.session_state["combined_results"] = res
                    st.session_state["predictions_df"]   = preds
                    if res is not None:
                        st.dataframe(res, height=500, use_container_width=True)
        
            
            
            
            
            
        else:
            # TYPE 2 LOGIC
            st.session_state.model_type = "Type 2"
            if "type2_dfs" not in st.session_state:
                st.session_state["type2_dfs"] = {}
            multi_keys = st.multiselect(
                "Select L0 keys to aggregate separately:",
                options=[c for c in ["Brand","Variant","PackType","PPG","PackSize"] if c in dataframe.columns]
            )

            for key in multi_keys:
                group_keys = [
                    next((c for c in dataframe.columns if c.strip().lower()=='date'), 'date'),
                    next((c for c in dataframe.columns if c.strip().lower()=='channel'),'Channel'),
                    key
                ]
                # If we haven't built an agg df for this key yet, do so
                if key not in st.session_state["type2_dfs"]:
                    agg_df_key = run_full_pipeline(
                        df_filtered,
                        group_keys,
                        [key],
                        use_kalman=use_kalman,
                        use_ratio_flag=use_ratio
                    )
                    st.session_state["type2_dfs"][key] = agg_df_key

                with st.expander(f"📊 Aggregated Data — {key}", expanded=False):
                    st.dataframe(st.session_state["type2_dfs"][key], height=600, use_container_width=True)

            st.markdown("## Type 2 Modeling Parameters")
            type2_params = {}
            for key in multi_keys:
                agg_df = st.session_state["type2_dfs"][key]
                available_cols = sorted(agg_df.columns)
                default_predictors = [
                    c for c in available_cols
                    if c.endswith("_RPI") or c in ["PPU","D1","is_outlier","NetCatVol","Cat_Down_Up","Cat_Price_trend_over_time"]
                ]
                selected_predictors = st.multiselect(
                    f"Select Predictor Columns for '{key}':",
                    options=available_cols,
                    default=default_predictors,
                    key=f"pred_cols_{key}"
                )
                grouping_keys_model = [
                    next((col for col in agg_df.columns if col.strip().lower()=='channel'),'Channel'),
                    key
                ]
                X_cols = [c for c in selected_predictors if c not in grouping_keys_model]
                target_col = "FilteredVolume"
                k_folds = st.number_input(
                    f"Number of folds (k) for {key}:",
                    min_value=2, max_value=20, value=5,
                    key=f"kfold_{key}"
                )
                numeric_in_X = [
                    c for c in X_cols
                    if c in agg_df.columns and pd.api.types.is_numeric_dtype(agg_df[c])
                ]
                default_std = [
                    c for c in numeric_in_X
                    if c in ["D1","PPU","NetCatVol","Cat_Weighted_Price",
                            "Cat_Down_Up","Cat_Price_trend_over_time","is_outlier"]
                ]
                chosen_std = st.multiselect(
                    f"Select columns to standardize for {key}:",
                    numeric_in_X,
                    default=default_std,
                    key=f"std_{key}"
                )
                type2_params[key] = {
                    "agg_df": agg_df,
                    "grouping_keys_model": grouping_keys_model,
                    "X_cols": X_cols,
                    "target_col": target_col,
                    "k_folds": k_folds,
                    "chosen_std": chosen_std
                }
                
                # Mixed model options for Type 2
                st.write(f"Mixed Models Options for {key}:")
                use_mixed_models_type2 = st.checkbox(f"Use Mixed Effects for {key}", value=False, key=f"use_mixed_{key}")

                if use_mixed_models_type2:
                    random_intercepts = st.multiselect(
                        f"Groups with different baselines for {key}:",
                        options=["Channel", key],
                        default=["Channel"],
                        key=f"ri_{key}"
                    )
                    
                    random_slopes = {}
                    use_price_random_slope = st.checkbox(
                        f"Price sensitivity varies by {key}", 
                        value=True,
                        key=f"rs_price_{key}"
                    )
                    
                    use_promo_random_slope = st.checkbox(
                        f"Promotion sensitivity varies by Channel", 
                        value=False,
                        key=f"rs_promo_{key}"
                    )
                    
                    if use_price_random_slope and key in random_intercepts:
                        if key not in random_slopes:
                            random_slopes[key] = []
                        random_slopes[key].append("PPU")
                        
                    if use_promo_random_slope and "Channel" in random_intercepts and "D1" in X_cols:
                        if "Channel" not in random_slopes:
                            random_slopes["Channel"] = []
                        random_slopes["Channel"].append("D1")
                    
                    # Store mixed model configs in type2_params
                    type2_params[key]["use_mixed"] = use_mixed_models_type2
                    type2_params[key]["random_intercepts"] = random_intercepts
                    type2_params[key]["random_slopes"] = random_slopes

            # Check if we already have results stored
            if "type2_results" not in st.session_state:
                st.session_state["type2_results"] = {}
            if "type2_predictions" not in st.session_state:
                st.session_state["type2_predictions"] = {}

            # If results are found, let user see them or re-run
            if st.session_state["type2_results"]:
                st.write("### Existing Model Results (Type 2)")
                for key, df_res in st.session_state["type2_results"].items():
                    if df_res is not None:
                        st.write(f"**Results for {key}**:")
                        st.dataframe(df_res, height=500, use_container_width=True)
                        if key in st.session_state["type2_predictions"]:
                            st.write(f"**Predictions sample for {key}**:")
                            st.dataframe(st.session_state["type2_predictions"][key].head(10))

                if st.button("Re-run Type 2 Models"):
                    type2_results = {}
                    for key, params in type2_params.items():
                        # Add mixed models if enabled for this key
                        if params.get("use_mixed", False):
                            # Create a copy of the global models dictionary
                            type2_models = models.copy()
                            
                            # Add mixed models specific to this key
                            random_intercepts = params.get("random_intercepts", [])
                            random_slopes = params.get("random_slopes", {})
                            
                            if random_intercepts:
                                # Basic mixed model with just random intercepts
                                intercepts_str = "+".join(random_intercepts)
                                type2_models[f"Mixed({intercepts_str})"] = MixedEffectsModelWrapper(
                                    random_effects=random_intercepts
                                )
                                
                                # Add models with random slopes
                                for group, slopes in random_slopes.items():
                                    slopes_str = "+".join(slopes)
                                    type2_models[f"Mixed({group}, {slopes_str})"] = MixedEffectsModelWrapper(
                                        random_effects=random_intercepts,
                                        random_slopes={group: slopes}
                                    )
                            
                            # Pass models dict to run_model_pipeline (need to modify function signature too)
                            res, preds = run_model_pipeline(
                                params["agg_df"],
                                params["grouping_keys_model"],
                                params["X_cols"],
                                params["target_col"],
                                params["k_folds"],
                                params["chosen_std"],
                                model_dict=type2_models  # Add this parameter
                            )
                        else:
                            # Original call without mixed models
                            res, preds = run_model_pipeline(
                                params["agg_df"],
                                params["grouping_keys_model"],
                                params["X_cols"],
                                params["target_col"],
                                params["k_folds"],
                                params["chosen_std"]
                            )
                        type2_results[key] = res
                        if preds is not None:
                            st.session_state["type2_predictions"][key] = preds

                    st.session_state.type2_results = type2_results
                    # show updated results
                    for key, df_res in type2_results.items():
                        if df_res is not None:
                            st.write(f"**Results for {key}** (updated):")
                            st.dataframe(df_res, height=500, use_container_width=True)
                            if key in st.session_state["type2_predictions"]:
                                st.write(f"**Predictions sample for {key}**:")
                                st.dataframe(st.session_state["type2_predictions"][key].head(10))

            else:
                # No existing type2 results: show a button to run
                if st.button("Run Models for all Type 2 Keys"):
                    type2_results = {}
                    for key, params in type2_params.items():
                        res, preds = run_model_pipeline(
                            params["agg_df"],
                            params["grouping_keys_model"],
                            params["X_cols"],
                            params["target_col"],
                            params["k_folds"],
                            params["chosen_std"]
                        )
                        type2_results[key] = res
                        if preds is not None:
                            st.session_state["type2_predictions"][key] = preds

                    st.session_state.type2_results = type2_results
                    # display
                    for key, df_res in type2_results.items():
                        st.markdown(f"### Results for **{key}**")
                        if df_res is not None:
                            st.dataframe(df_res, height=500, use_container_width=True)
                        if key in st.session_state["type2_predictions"]:
                            st.markdown(f"#### Sample Actual vs. Predicted for **{key}**")
                            st.dataframe(st.session_state["type2_predictions"][key].head(20))

