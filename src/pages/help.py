import streamlit as st


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def help_page():
    st.header("Help & Documentation")
    st.markdown(
        "Welcome to the Hydrate Alert System! This guide will help you get "
        "started and make the most of the application."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Getting Started",
        "Data Analysis",
        "Machine Learning",
        "FAQs",
        "Troubleshooting",
    ])

    with tab1:
        _render_getting_started_tab()
    with tab2:
        _render_data_analysis_tab()
    with tab3:
        _render_machine_learning_tab()
    with tab4:
        _render_faqs_tab()
    with tab5:
        _render_troubleshooting_tab()

    _render_footer()


# ---------------------------------------------------------------------------
# Tab content
# ---------------------------------------------------------------------------

def _render_getting_started_tab():
    st.subheader("Getting Started")
    st.markdown(
        """
        ### Welcome to Hydrate Alert!

        This application helps you monitor and predict hydrate formation in gas pipelines using machine learning technology.

        #### **Step-by-Step Guide:**

        1. **Login**: Use your Google account to authenticate
        2. **Upload Data**: Go to the "Data Upload" page to upload your CSV files
        3. **Analyze**: Visit the "Data Analysis" page to view predictions and visualizations
        4. **Export**: Download your results with predictions included

        #### **Required Data Format:**
        Your CSV files should contain these columns:
        - `Time` - Timestamp of measurements
        - `Inj Gas Meter Volume Instantaneous` - Current gas volume
        - `Inj Gas Meter Volume Setpoint` - Target gas volume
        - `Inj Gas Valve Percent Open` - Valve opening percentage

        #### **Supported File Types:**
        - CSV files (.csv)
        - Maximum file size: 200MB
        - Multiple file upload supported
        """
    )


def _render_data_analysis_tab():
    st.subheader("Data Analysis Guide")
    st.markdown(
        """
        ### Understanding Your Data

        #### **Available Visualizations:**

        1. **Time Series - All Variables**
           - Shows all pipeline variables over time
           - Includes volume, setpoint, valve position, and hydrate predictions
           - Best for: Understanding overall pipeline behavior

        2. **Correlation Heatmap**
           - Shows relationships between different variables
           - Color-coded correlation matrix
           - Best for: Identifying which factors influence hydrate formation

        3. **Hydrate Risk Distribution**
           - Histogram of predicted hydrate likelihood values
           - Shows distribution of risk levels
           - Best for: Understanding overall risk profile

        4. **Valve vs Volume Relationship**
           - Scatter plot showing valve position vs gas volume
           - Color-coded by hydrate risk
           - Best for: Identifying operational patterns

        5. **Risk Alert Timeline**
           - Time-based view of hydrate risk levels
           - Shows risk threshold violations
           - Best for: Monitoring and alerting

        #### **Risk Levels:**
        - **Low Risk**: < 2.0
        - **Medium Risk**: 2.0 - 5.0
        - **High Risk**: > 5.0
        - **Critical Risk**: > 7.0
        """
    )


def _render_machine_learning_tab():
    st.subheader("Machine Learning Information")
    st.markdown(
        """
        ### How Our ML Model Works

        #### **Training Data:**
        - Model is trained on historical pipeline data
        - Uses `final.csv` as the training dataset
        - Continuously updated for better accuracy

        #### **Features Used:**
        - Gas volume measurements (instantaneous & setpoint)
        - Valve position data
        - Rolling standard deviation
        - Time-based features (hour, day)
        - Calculated ratios and differences

        #### **Model Details:**
        - **Algorithm**: Random Forest Regressor
        - **Features**: 8 engineered features
        - **Output**: Hydrate formation likelihood (0-10 scale)
        - **Accuracy**: R² score displayed during training

        #### **Prediction Process:**
        1. **Data Preprocessing**: Clean and prepare your data
        2. **Feature Engineering**: Create derived features
        3. **Scaling**: Normalize data for consistent predictions
        4. **Prediction**: Generate hydrate likelihood scores
        5. **Risk Classification**: Categorize into risk levels

        #### **Model Performance:**
        - Performance metrics are shown after training
        - MSE (Mean Squared Error) and R² values provided
        - Model can be retrained with new data
        """
    )


def _render_faqs_tab():
    st.subheader("Frequently Asked Questions")
    st.markdown(
        """
        ### Common Questions

        #### **Q: What file format should I use?**
        A: Upload CSV files with the required columns. Ensure your data includes timestamps and the four main pipeline variables.

        #### **Q: How accurate are the predictions?**
        A: Model accuracy depends on data quality and similarity to training data. Check the R² score shown during training for current performance.

        #### **Q: Can I upload multiple files?**
        A: Yes! Use the "Batch Upload" tab to upload multiple CSV files at once. Each file will be treated as a separate dataset.

        #### **Q: Why are my predictions not showing?**
        A: Ensure your data has the required columns and the ML model has finished training. Check the "Machine Learning Model" section for status.

        #### **Q: How do I interpret the risk levels?**
        A: Risk levels are color-coded:
        - Green (Low): Normal operation
        - Yellow (Medium): Monitor closely
        - Red (High): Take preventive action
        - Dark Red (Critical): Immediate intervention required

        #### **Q: Can I download my results?**
        A: Yes! Use the "Export Results" section to download your data with predictions included.

        #### **Q: What if my data is missing some columns?**
        A: The system will attempt to calculate missing values (like Rolling Std) automatically. However, the main pipeline variables are required.

        #### **Q: How often should I retrain the model?**
        A: Retrain when you have new representative data or notice decreased accuracy. The model is cached for performance.
        """
    )


def _render_troubleshooting_tab():
    st.subheader("Troubleshooting")
    st.markdown(
        """
        ### Common Issues & Solutions

        #### **Upload Issues**

        **Problem**: File won't upload
        - Check file size (must be < 200MB)
        - Ensure file is in CSV format
        - Verify file isn't corrupted

        **Problem**: Missing columns error
        - Check your CSV has required column names
        - Ensure no extra spaces in column names
        - Verify data types are correct

        #### **Visualization Issues**

        **Problem**: Charts not displaying
        - Check if dataset is selected
        - Verify required columns exist
        - Try refreshing the page

        **Problem**: Time series not working
        - Ensure Time column is in proper format
        - Check for missing data points
        - Verify time format is consistent

        #### **ML Model Issues**

        **Problem**: No predictions generated
        - Wait for model training to complete
        - Check if required packages are installed
        - Try retraining the model

        **Problem**: Poor prediction accuracy
        - Check data quality and completeness
        - Ensure data is similar to training data
        - Consider retraining with more data

        #### **Download Issues**

        **Problem**: Download button not working
        - Ensure predictions are generated
        - Check browser download settings
        - Try using a different browser

        #### **Performance Issues**

        **Problem**: App running slowly
        - Reduce dataset size if possible
        - Close other browser tabs
        - Clear browser cache

        ### **Need More Help?**

        If you're still experiencing issues:
        1. Check the browser console for error messages
        2. Try refreshing the page
        3. Clear your browser cache
        4. Contact your system administrator

        ### **Technical Requirements**

        - **Browser**: Chrome, Firefox, Safari, or Edge
        - **Internet**: Stable connection required
        - **Files**: CSV format with proper column names
        - **Size**: Files under 200MB recommended
        """
    )


def _render_footer():
    st.markdown("---")
    st.markdown(
        """
        ### Quick Links

        - **[Upload Data](/)** - Start by uploading your CSV files
        - **[Data Analysis](/)** - View predictions and visualizations
        - **[Home](/)** - Return to main dashboard

        ### Support

        For technical support or questions about the Hydrate Alert system, please contact your system administrator.

        **Version**: 1.0.0 | **Last Updated**: July 2025
    """)