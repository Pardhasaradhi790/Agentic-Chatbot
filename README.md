
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
            from sklearn.linear_model import Ridge, Lasso
            from sklearn.neural_network import MLPRegressor
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            import joblib

            # Optional libraries
            try:
                import shap
            except ImportError:
                shap = None

            try:
                import xgboost as xgb
            except ImportError:
                xgb = None

            try:
                from catboost import CatBoostRegressor
            except ImportError:
                CatBoostRegressor = None

            try:
                from lightgbm import LGBMRegressor
            except ImportError:
                LGBMRegressor = None

            # -----------------------------
            # 1. Load CSV files
            # -----------------------------
            claims_df = pd.read_csv('Claims_Rework_ProcessTable last 10 days.csv')
            reporting_df = pd.read_csv('ReportingTable_claims_rework last 10 days.csv')

            # -----------------------------
            # 2. Clean column names and values
            # -----------------------------
            claims_df.columns = claims_df.columns.str.strip()
            reporting_df.columns = reporting_df.columns.str.strip()
            claims_df = claims_df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
            reporting_df = reporting_df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

            # -----------------------------
            # 3. Convert datetime columns
            # -----------------------------
            reporting_df['Start_Time'] = pd.to_datetime(reporting_df['Start_Time'], errors='coerce')
            reporting_df['End_Time'] = pd.to_datetime(reporting_df['End_Time'], errors='coerce')
            reporting_df['Duration'] = (reporting_df['End_Time'] - reporting_df['Start_Time']).dt.total_seconds()

            # -----------------------------
            # 4. Merge tables on ClaimID
            # -----------------------------
            merged_df = pd.merge(claims_df, reporting_df, left_on='ClaimID', right_on='Claim_ID', how='inner')
            merged_df['Adjusted_Claim_Status'] = merged_df['Adjusted_Claim_Status'].str.strip().str.upper().fillna('UNKNOWN')

            # -----------------------------
            # 5. Aggregate by RequestID (BOT_User_Count removed)
            # -----------------------------
            status_counts = merged_df.pivot_table(
                index='RequestID',
                columns='Adjusted_Claim_Status',
                values='ClaimID',
                aggfunc='count',
                fill_value=0
            )

            agg_metrics = merged_df.groupby('RequestID').agg(
                Total_Claims=('ClaimID', 'count'),
                PriorityID=('PriorityID', 'first'),
                Total_Time_Seconds=('Duration', 'sum')
            )

            agg_df = pd.concat([agg_metrics, status_counts], axis=1).reset_index()
            agg_df.to_csv('aggregated_data.csv', index=False)

            # Encode PriorityID
            label_enc = LabelEncoder()
            agg_df['PriorityID'] = label_enc.fit_transform(agg_df['PriorityID'].astype(str))

            # -----------------------------
            # 6. Prepare Features and Target
            # -----------------------------
            X_original = agg_df.drop(['RequestID', 'Total_Time_Seconds'], axis=1)
            y = agg_df['Total_Time_Seconds']

            # Scale features for training
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_original)

            # Train/Test split (keep original and scaled)
            X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            X_train_orig, X_test_orig, _, _ = train_test_split(X_original, y, test_size=0.2, random_state=42)

            def seconds_to_hms(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

            # -----------------------------
            # 7. Models and Hyperparameter Tuning
            # -----------------------------
            models = {
                'RandomForest': RandomForestRegressor(random_state=42),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'ExtraTrees': ExtraTreesRegressor(random_state=42),
                'HistGB': HistGradientBoostingRegressor(random_state=42),
                'MLP': MLPRegressor(max_iter=500, random_state=42),
                'Ridge': Ridge(),
                'Lasso': Lasso()
            }
            if xgb:
                models['XGBoost'] = xgb.XGBRegressor(random_state=42)
            if CatBoostRegressor:
                models['CatBoost'] = CatBoostRegressor(verbose=0, random_state=42)
            if LGBMRegressor:
                models['LightGBM'] = LGBMRegressor(random_state=42)

            param_grid = {
                'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
                'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
                'ExtraTrees': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
                'HistGB': {'max_iter': [100, 200], 'learning_rate': [0.05, 0.1]},
                'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
                'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                'Lasso': {'alpha': [0.001, 0.01, 0.1]}
            }
            if xgb:
                param_grid['XGBoost'] = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
            if CatBoostRegressor:
                param_grid['CatBoost'] = {'iterations': [200, 500], 'learning_rate': [0.05, 0.1]}
            if LGBMRegressor:
                param_grid['LightGBM'] = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}

            results = {}
            test_results = pd.DataFrame()

            for name, model in models.items():
                print(f"Training {name}...")
                grid = GridSearchCV(model, param_grid[name], cv=3, scoring='r2', n_jobs=-1)
                grid.fit(X_train_scaled, y_train)
                best_model = grid.best_estimator_

                # Predict using DataFrame to avoid LightGBM warning
                y_pred = best_model.predict(pd.DataFrame(X_test_scaled, columns=X_original.columns))

                # Combine predictions with original features
                X_test_df = X_test_orig.copy()
                X_test_df['Actual_Seconds'] = y_test.values
                X_test_df['Predicted_Seconds'] = y_pred
                X_test_df['Actual_HHMMSS'] = [seconds_to_hms(x) for x in y_test.values]
                X_test_df['Predicted_HHMMSS'] = [seconds_to_hms(x) for x in y_pred]
                X_test_df['Model'] = name

                # Append to combined results
                test_results = pd.concat([test_results, X_test_df], axis=0)

                # Store performance metrics
                results[name] = {
                    'model': best_model,
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred)
                }

            # -----------------------------
            # 8. Save Results
            # -----------------------------
            performance_df = pd.DataFrame(results).T[['MAE', 'RMSE', 'R2']]
            performance_df.to_csv('model_performance_results.csv', index=True)
            test_results.to_csv('predictions_with_original_features.csv', index=False)
            performance_df.to_csv('performance_summary.csv', index=True)

            # Plot performance
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sns.barplot(x=performance_df.index, y=performance_df['MAE'], ax=ax[0])
            ax[0].set_title('MAE by Model')
            sns.barplot(x=performance_df.index, y=performance_df['RMSE'], ax=ax[1])
            ax[1].set_title('RMSE by Model')
            sns.barplot(x=performance_df.index, y=performance_df['R2'], ax=ax[2])
            ax[2].set_title('R² by Model')
            plt.tight_layout()
            plt.savefig('model_performance.png')

            # Save best model
            best_model_name = performance_df['R2'].idxmax()
            best_model = results[best_model_name]['model']
            joblib.dump(best_model, 'best_model.pkl')

