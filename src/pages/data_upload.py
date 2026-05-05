import streamlit as st
import pandas as pd
from typing import Dict


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def upload_data():
    st.header("Upload Your Pipeline Data")
    _init_state()

    tab1, tab2 = st.tabs(["Single Upload", "Batch Upload"])
    with tab1:
        _render_single_upload_tab()
    with tab2:
        _render_batch_upload_tab()

    _render_summary()


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _init_state():
    if 'uploaded_datasets' not in st.session_state:
        st.session_state.uploaded_datasets = {}


def get_uploaded_datasets() -> Dict[str, pd.DataFrame]:
    """Return the uploaded datasets from session state."""
    return st.session_state.get('uploaded_datasets', {})


def _try_read_csv(uploaded_file):
    """Read a CSV upload safely, surfacing errors in the UI."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Upload tabs
# ---------------------------------------------------------------------------

def _render_single_upload_tab():
    st.subheader("Upload Individual Pipeline Data")

    pipeline_name = st.text_input(
        "Pipeline Name", placeholder="Enter pipeline identifier"
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=['csv'], key="single_upload"
    )

    if uploaded_file and not pipeline_name:
        st.warning("Please enter a pipeline name.")
        return

    if uploaded_file and pipeline_name:
        df = _try_read_csv(uploaded_file)
        if df is None:
            return
        st.session_state.uploaded_datasets[pipeline_name] = df
        st.success(f"Successfully uploaded data for {pipeline_name}")
        st.dataframe(df.head())
        st.info(f"Dataset shape: {df.shape}")


def _render_batch_upload_tab():
    st.subheader("Upload Multiple Pipeline Files")

    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        key="batch_upload",
    )
    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        pipeline_name = uploaded_file.name.replace('.csv', '')
        df = _try_read_csv(uploaded_file)
        if df is not None:
            st.session_state.uploaded_datasets[pipeline_name] = df

    st.success(f"Successfully processed {len(uploaded_files)} files please hold.")


# ---------------------------------------------------------------------------
# Summary section
# ---------------------------------------------------------------------------

def _render_summary():
    datasets = st.session_state.uploaded_datasets
    if not datasets:
        st.info("No datasets uploaded yet. Please upload CSV files to proceed.")
        return

    st.subheader("Uploaded Datasets Summary")
    summary_df = _build_summary_dataframe(datasets)
    st.dataframe(summary_df)

    if len(datasets) > 1:
        _render_combined_export(datasets)


def _build_summary_dataframe(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {
            'Pipeline': name,
            'Rows': df.shape[0],
            'Columns': df.shape[1],
            'Memory Usage (MB)': round(
                df.memory_usage(deep=True).sum() / 1024 / 1024, 2
            ),
        }
        for name, df in datasets.items()
    ]
    return pd.DataFrame(rows)


def _render_combined_export(datasets: Dict[str, pd.DataFrame]):
    st.subheader("Export Combined Dataset Info")
    combined_info = get_combined_dataset_info(datasets)
    if st.download_button(
        label="Download Combined Dataset Info",
        data=combined_info.to_csv(index=False),
        file_name="combined_pipeline_info.csv",
        mime="text/csv",
    ):
        st.success("Download started!")


def get_combined_dataset_info(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a per-column summary of all uploaded datasets."""
    rows = []
    for name, df in datasets.items():
        for col in df.columns:
            rows.append({
                'Pipeline': name,
                'Column': col,
                'Data Type': str(df[col].dtype),
                'Non-Null Count': df[col].count(),
                'Null Count': df[col].isnull().sum(),
                'Unique Values': df[col].nunique(),
            })
    return pd.DataFrame(rows)
