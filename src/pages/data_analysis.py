import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    'Inj Gas Meter Volume Instantaneous',
    'Inj Gas Meter Volume Setpoint',
    'Inj Gas Valve Percent Open',
    'Rolling Std',
    'Volume_Diff',
    'Volume_Ratio',
    'Hour',
    'Day',
]

REQUIRED_TIME_SERIES_COLS = [
    'Inj Gas Meter Volume Instantaneous',
    'Inj Gas Meter Volume Setpoint',
    'Inj Gas Valve Percent Open',
]

CHART_OPTIONS = [
    "Time Series - All Variables",
    "Correlation Heatmap",
    "Hydrate Risk Distribution",
    "Valve vs Volume Relationship",
    "Risk Alert Timeline",
]

CRITICAL_RISK_THRESHOLD = 7.0
HIGH_RISK_THRESHOLD = 5.0
MEDIUM_RISK_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def data_analysis():
    """Main data analysis page."""
    st.header("Data Analysis & Hydrate Formation Prediction")

    from .data_upload import get_uploaded_datasets

    model, scaler, feature_columns = _render_model_section()
    uploaded_datasets = get_uploaded_datasets()

    if not uploaded_datasets:
        _render_no_datasets_message()
        return

    selected_dataset = _render_dataset_selector(uploaded_datasets)
    if selected_dataset:
        df = uploaded_datasets[selected_dataset].copy()
        df = _render_predictions_section(df, model, scaler, feature_columns)
        _render_visualization_section(df, selected_dataset)
        _render_data_table_section(df)
        _render_export_section(df, selected_dataset)

    _render_manage_datasets_section(uploaded_datasets)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def _render_model_section():
    st.subheader("Machine Learning Model")
    with st.expander("Model Training Information", expanded=False):
        st.info(
            "The model is trained using final.csv data with features like gas "
            "volume, valve position, and rolling statistics to predict hydrate "
            "formation likelihood."
        )
        if st.button("Retrain Model"):
            st.cache_resource.clear()
    return train_hydrate_model()


def _render_no_datasets_message():
    st.warning(
        "No datasets uploaded yet. Please upload CSV files in the Data Upload "
        "page to proceed."
    )
    st.info(
        "**Tip**: Upload your pipeline data CSV files to get started with "
        "hydrate formation analysis and predictions!"
    )


def _render_dataset_selector(uploaded_datasets):
    st.subheader("Dataset Selection")
    return st.selectbox(
        "Select dataset for analysis:",
        options=list(uploaded_datasets.keys()),
        key="dataset_selector",
    )


def _render_predictions_section(df, model, scaler, feature_columns):
    st.subheader("Hydrate Formation Predictions")
    if model is None or scaler is None:
        return df

    with st.spinner("Generating predictions..."):
        predictions = predict_hydrate_likelihood(df, model, scaler, feature_columns)

    if predictions is None:
        return df

    df['Predicted_Hydrate_Likelihood'] = predictions
    _render_prediction_metrics(predictions)
    _render_risk_alerts(predictions)
    return df


def _render_prediction_metrics(predictions):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Risk", f"{predictions.max():.2f}")
    col2.metric("Avg Risk", f"{predictions.mean():.2f}")
    col3.metric("High Risk Points", int(np.sum(predictions > HIGH_RISK_THRESHOLD)))
    col4.metric("Total Points", len(predictions))


def _render_risk_alerts(predictions):
    max_risk = predictions.max()
    if max_risk > CRITICAL_RISK_THRESHOLD:
        st.error("CRITICAL: Very high hydrate formation risk detected!")
    elif max_risk > HIGH_RISK_THRESHOLD:
        st.warning("WARNING: High hydrate formation risk detected!")
    else:
        st.success("Hydrate formation risk is within acceptable limits")


def _render_visualization_section(df, dataset_name):
    st.subheader("Data Visualization")
    selected_chart = st.selectbox(
        "Select visualization type:",
        options=CHART_OPTIONS,
        key="chart_selector",
    )
    if not selected_chart:
        return
    fig = create_visualization(df, selected_chart, dataset_name)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)


def _render_data_table_section(df):
    st.subheader("Data Table")
    display_columns = st.multiselect(
        "Select columns to display:",
        options=df.columns.tolist(),
        default=df.columns.tolist()[:5],
        key="column_selector",
    )
    if display_columns:
        st.dataframe(df[display_columns], use_container_width=True)


def _render_export_section(df, dataset_name):
    st.subheader("Export Results")
    if 'Predicted_Hydrate_Likelihood' in df.columns:
        _render_download_with_predictions(df, dataset_name)
    else:
        _render_download_without_predictions(df, dataset_name)


def _render_download_with_predictions(df, dataset_name):
    st.success("Dataset includes ML predictions - ready for download")

    with st.expander("Preview Download Data", expanded=False):
        st.write(f"**Columns to be included:** {len(df.columns)}")
        st.write(f"**Rows:** {len(df)}")
        st.write("**Column Names:**")
        for i, col in enumerate(df.columns, 1):
            st.write(f"{i}. {col}")

    try:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download data with predictions",
            data=csv_data,
            file_name=f"{dataset_name}_with_predictions.csv",
            mime="text/csv",
            help="Download the dataset with ML predictions included",
        )
        st.info(f"File will be saved as: {dataset_name}_with_predictions.csv")
    except Exception as e:
        st.error(f"Error preparing download: {e}")
        st.info("Please try selecting the dataset again or contact support.")


def _render_download_without_predictions(df, dataset_name):
    st.warning("No predictions available for this dataset")
    st.info("Please wait for the ML model to generate predictions, then try again.")
    if st.button("Download original data (without predictions)"):
        try:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download original data",
                data=csv_data,
                file_name=f"{dataset_name}_original.csv",
                mime="text/csv",
                help="Download the original dataset without predictions",
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")


def _render_manage_datasets_section(uploaded_datasets):
    if not uploaded_datasets:
        return

    st.subheader("Manage Datasets")
    dataset_to_remove = st.selectbox(
        "Select dataset to remove",
        options=list(uploaded_datasets.keys()),
        key="remove_dataset_analysis",
    )
    if st.button("Remove Dataset"):
        if dataset_to_remove in st.session_state.uploaded_datasets:
            del st.session_state.uploaded_datasets[dataset_to_remove]
            st.session_state.pop('remove_dataset_analysis', None)
            st.success(f"Removed {dataset_to_remove}")
            st.rerun()

# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------

def _safe_ratio(numerator, denominator):
    """Element-wise division that returns 0 where denominator is 0/NaN."""
    safe_denom = denominator.replace(0, np.nan)
    return (numerator / safe_denom).fillna(0)


def _engineer_features(df):
    """Add engineered features the model expects, in-place on a copy."""
    df = df.copy()
    df['Volume_Diff'] = (
        df['Inj Gas Meter Volume Instantaneous']
        - df['Inj Gas Meter Volume Setpoint']
    )
    df['Volume_Ratio'] = _safe_ratio(
        df['Inj Gas Meter Volume Instantaneous'],
        df['Inj Gas Meter Volume Setpoint'],
    )

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df['Hour'] = df['Time'].dt.hour
        df['Day'] = df['Time'].dt.day
    else:
        df['Hour'] = df.get('Hour', 0)
        df['Day'] = df.get('Day', 1)

    if 'Rolling Std' not in df.columns:
        df['Rolling Std'] = (
            df['Inj Gas Meter Volume Instantaneous']
            .rolling(window=5).std().fillna(0)
        )
    return df


@st.cache_data
def load_training_data():
    """Load the training data from final.csv."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', '..', 'data', 'final.csv')
        df = pd.read_csv(data_path)
        df['Time'] = pd.to_datetime(df['Time'])
        return df
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None


@st.cache_resource
def train_hydrate_model():
    """Train the hydrate formation prediction model."""
    df = load_training_data()
    if df is None:
        return None, None, None

    df = _engineer_features(df)

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df['Likelihood of Hydrate']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success(f"Model trained successfully! MSE: {mse:.4f}, R²: {r2:.4f}")

    return model, scaler, FEATURE_COLUMNS


def predict_hydrate_likelihood(df, model, scaler, feature_columns):
    """Predict hydrate formation likelihood for uploaded data."""
    if model is None or scaler is None:
        return None

    df_processed = _engineer_features(df)
    X = df_processed[feature_columns].fillna(0)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _get_time_axis(df):
    """Return a usable time axis, falling back to the dataframe index."""
    if 'Time' not in df.columns:
        return df.index
    try:
        return pd.to_datetime(df['Time'])
    except (ValueError, TypeError):
        return df.index


def _check_required_columns(df, required):
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(f"Available columns: {list(df.columns)}")
        return False
    return True


def _classify_risk(value):
    if value > HIGH_RISK_THRESHOLD:
        return 'High'
    if value > MEDIUM_RISK_THRESHOLD:
        return 'Medium'
    return 'Low'


def _chart_time_series(df, dataset_name):
    if not _check_required_columns(df, REQUIRED_TIME_SERIES_COLS):
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Volume Instantaneous', 'Volume Setpoint',
            'Valve Percent Open', 'Hydrate Likelihood',
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    time_data = _get_time_axis(df)
    try:
        fig.add_trace(go.Scatter(
            x=time_data, y=df['Inj Gas Meter Volume Instantaneous'],
            name='Volume Instantaneous', line=dict(color='blue'),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=time_data, y=df['Inj Gas Meter Volume Setpoint'],
            name='Volume Setpoint', line=dict(color='red'),
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=time_data, y=df['Inj Gas Valve Percent Open'],
            name='Valve % Open', line=dict(color='green'),
        ), row=2, col=1)

        if 'Predicted_Hydrate_Likelihood' in df.columns:
            fig.add_trace(go.Scatter(
                x=time_data, y=df['Predicted_Hydrate_Likelihood'],
                name='Predicted Hydrate Likelihood',
                line=dict(color='orange'),
            ), row=2, col=2)
        else:
            fig.add_trace(go.Scatter(
                x=time_data, y=[0] * len(time_data),
                name='No Predictions Available', line=dict(color='gray'),
            ), row=2, col=2)

        fig.update_layout(
            height=600,
            title_text=f"Time Series Analysis - {dataset_name}",
            showlegend=True,
        )
        return fig
    except Exception as e:
        st.error(f"Error creating time series plot: {e}")
        return None


def _chart_correlation_heatmap(df, dataset_name):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    return px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title=f"Correlation Matrix - {dataset_name}",
    )


def _chart_risk_distribution(df, dataset_name):
    if 'Predicted_Hydrate_Likelihood' not in df.columns:
        st.warning("No hydrate predictions available for this dataset")
        return None
    return px.histogram(
        df, x='Predicted_Hydrate_Likelihood',
        title=f"Hydrate Risk Distribution - {dataset_name}",
        nbins=30,
    )


def _chart_valve_vs_volume(df, dataset_name):
    color = (
        'Predicted_Hydrate_Likelihood'
        if 'Predicted_Hydrate_Likelihood' in df.columns else None
    )
    return px.scatter(
        df,
        x='Inj Gas Valve Percent Open',
        y='Inj Gas Meter Volume Instantaneous',
        color=color,
        title=f"Valve vs Volume Relationship - {dataset_name}",
    )


def _chart_risk_timeline(df, dataset_name):
    if 'Predicted_Hydrate_Likelihood' not in df.columns:
        st.warning("No hydrate predictions available for this dataset")
        return None

    df = df.copy()
    df['Risk_Level'] = df['Predicted_Hydrate_Likelihood'].apply(_classify_risk)
    time_col = 'Time' if 'Time' in df.columns else df.index

    fig = px.line(
        df, x=time_col, y='Predicted_Hydrate_Likelihood',
        color='Risk_Level',
        title=f"Hydrate Risk Timeline - {dataset_name}",
    )
    fig.add_hline(
        y=HIGH_RISK_THRESHOLD,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk Threshold",
    )
    return fig


_CHART_BUILDERS = {
    "Time Series - All Variables": _chart_time_series,
    "Correlation Heatmap": _chart_correlation_heatmap,
    "Hydrate Risk Distribution": _chart_risk_distribution,
    "Valve vs Volume Relationship": _chart_valve_vs_volume,
    "Risk Alert Timeline": _chart_risk_timeline,
}


def create_visualization(df, chart_type, dataset_name):
    """Dispatch to the appropriate chart builder."""
    builder = _CHART_BUILDERS.get(chart_type)
    if builder is None:
        return None
    return builder(df, dataset_name)