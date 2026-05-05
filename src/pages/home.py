import streamlit as st

from . import style
from .data_upload import get_uploaded_datasets

def home_page():
    style.style()
    uploaded_datasets = get_uploaded_datasets()

    _render_welcome()
    _render_section_heading("System Overview")
    _render_metrics(uploaded_datasets)
    _render_quick_actions()

    if uploaded_datasets:
        _render_section_heading("Recent Datasets")
        _render_recent_datasets(uploaded_datasets)

    _render_section_heading("System Health")
    _render_system_health()
    _render_section_heading("Navigation Guide")
    _render_navigation_guide()
    _render_pro_tip()

def _render_section_heading(title: str):
    st.markdown(
        f"""
        <div style='text-align: center; margin: 40px 0;'>
            <h4 style='color: #1f4e79; margin-bottom: 30px;'>{title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_welcome():
    st.markdown(
        """
        <div class='welcome-section'>
            <h3>Welcome to Hydrate Alert System</h3>
            <p style='font-size: 18px; margin: 20px 0;'>
                Your comprehensive pipeline monitoring solution is ready. Monitor gas injection systems,
                analyze data trends, and receive intelligent alerts for potential hydrate formation.
            </p>
            <div class='alert-badge'>System Status: Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_metrics(uploaded_datasets):
    total_rows = sum(df.shape[0] for df in uploaded_datasets.values())
    total_size_mb = sum(
        df.memory_usage(deep=True).sum() for df in uploaded_datasets.values()
    ) / (1024 * 1024)

    col1, col2, col3, col4 = st.columns(4)
    _centered_metric(
        col1, " Datasets Uploaded", len(uploaded_datasets),
        "Total number of datasets currently loaded",
    )
    _centered_metric(
        col2, " Data Points", f"{total_rows:,}",
        "Total number of data points across all datasets",
    )
    _centered_metric(
        col3, " Memory Usage", f"{total_size_mb:.1f} MB",
        "Total memory usage of uploaded datasets",
    )
    _centered_metric(
        col4, " ML Model", "Active",
        "Machine learning model status",
    )

def _centered_metric(col, label, value, help_text):
    with col:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.metric(label=label, value=value, help=help_text)
        st.markdown("</div>", unsafe_allow_html=True)

def _render_quick_actions():
    st.markdown(
        """
        <div style='text-align: center; margin: 40px 0;'>
            <h4 style='color: #1f4e79; margin-bottom: 30px;'>Quick Actions</h4>
            <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin-top: 20px;'>
                <div class='feature-card' style='flex: 1; min-width: 250px; max-width: 300px;'>
                    <h5 style='color: #667eea;'>Upload New Data</h5>
                    <p style='color: #555;'>Upload CSV files containing pipeline data for analysis and monitoring</p>
                </div>
                <div class='feature-card' style='flex: 1; min-width: 250px; max-width: 300px;'>
                    <h5 style='color: #667eea;'>Analyze Data</h5>
                    <p style='color: #555;'>View predictions, visualizations, and detailed analysis of your pipeline data</p>
                </div>
                <div class='feature-card' style='flex: 1; min-width: 250px; max-width: 300px;'>
                    <h5 style='color: #667eea;'>Get Help</h5>
                    <p style='color: #555;'>Access documentation, FAQs, and troubleshooting guides</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_recent_datasets(uploaded_datasets, limit: int = 3):
    for name, df in list(uploaded_datasets.items())[:limit]:
        with st.expander(name, expanded=False):
            _render_dataset_card(df)
            st.write("**Columns:** " + ", ".join(df.columns.tolist()))

    remaining = len(uploaded_datasets) - limit
    if remaining > 0:
        st.info(f"And {remaining} more datasets available in Data Analysis page")

def _render_dataset_card(df):
    col1, col2, col3 = st.columns(3)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    _centered_text(col1, f"**Rows:** {df.shape[0]:,}")
    _centered_text(col2, f"**Columns:** {df.shape[1]}")
    _centered_text(col3, f"**Size:** {memory_mb:.1f} MB")

def _centered_text(col, text):
    with col:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.write(text)
        st.markdown("</div>", unsafe_allow_html=True)

def _render_system_health():
    statuses = [
        ("Data Processing", "All systems operational"),
        ("ML Engine", "Model ready for predictions"),
        ("Visualization", "Charts and graphs available"),
    ]
    columns = st.columns(len(statuses))
    for col, (title, description) in zip(columns, statuses):
        with col:
            st.markdown(
                f"""
                <div class='feature-card' style='text-align: center;'>
                    <h5 style='color: #27ae60;'>{title}</h5>
                    <p style='color: #555;'>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

def _render_navigation_guide():
    steps = [
        ("Step 1", "Upload your CSV data files"),
        ("Step 2", "Let ML model analyze your data"),
        ("Step 3", "View predictions and insights"),
        ("Step 4", "Export results for action"),
    ]
    cards = "".join(
        f"""
        <div class='feature-card' style='flex: 1; min-width: 200px; max-width: 240px; text-align: center;'>
            <h6 style='color: #667eea;'>{title}</h6>
            <p style='color: #555;'>{body}</p>
        </div>
        """
        for title, body in steps
    )
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 15px; margin-top: 20px;'>
            {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_pro_tip():
    st.markdown(
        """
        <div class='login-box' style='margin-top: 50px;'>
         <strong>Pro Tip:</strong> Upload multiple datasets to compare pipeline performance across different time periods or locations.
            Use the Data Analysis page to generate comprehensive reports and identify trends.
        </div>
        """,
        unsafe_allow_html=True,
    )