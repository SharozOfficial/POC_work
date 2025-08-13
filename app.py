import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import modified functions from model_serving_utils
try:
    from model_serving_utils import (
        get_spark_session,
        run_databricks_healthcare_pipeline,
        HealthcareAIQueryProcessor,
        analyze_capacity_impact_optimized,
        identify_high_risk_patients_optimized,
        optimize_schedule_optimized,
        analyze_capacity_impact_pandas  
    )
    print("☑️ Successfully imported functions from model_serving_utils")
except ImportError as e:
    st.error(f"❌ Error importing functions: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'spark_session' not in st.session_state:
        st.session_state.spark_session = None
    if 'ai_processor' not in st.session_state:
        st.session_state.ai_processor = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def validate_dataframe_columns(df, required_columns=None):
    """Validate that DataFrame has required columns"""
    if df is None or len(df) == 0:
        return False, "DataFrame is empty or None"
    
    if required_columns is None:
        # Core columns that should always be present
        core_required = ['patient_id', 'age', 'cancel_risk', 'los', 'cost_estimate', 
                        'priority_score', 'readmit_risk', 'wait_days']
    else:
        core_required = required_columns
    
    missing_columns = [col for col in core_required if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    return True, "All required columns present"

def reconstruct_categorical_columns(df):
    """Reconstruct original categorical columns from one-hot encoded columns"""
    df_reconstructed = df.copy()
    
    # Reconstruct gender
    if 'gender_M' in df.columns:
        df_reconstructed['gender'] = df['gender_M'].map({1: 'M', 0: 'F'})
    
    # Reconstruct specialty
    specialty_cols = [col for col in df.columns if col.startswith('specialty_')]
    if specialty_cols:
        def get_specialty(row):
            for col in specialty_cols:
                if row[col] == 1:
                    return col.replace('specialty_', '')
            return 'Cardiology'  # Default fallback
        
        df_reconstructed['specialty'] = df[specialty_cols].apply(get_specialty, axis=1)
    
    # Reconstruct season
    season_cols = [col for col in df.columns if col.startswith('season_')]
    if season_cols:
        def get_season(row):
            for col in season_cols:
                if row[col] == 1:
                    return col.replace('season_', '')
            return 'Autumn'  # Default fallback (since it's not in dummy variables)
        
        df_reconstructed['season'] = df[season_cols].apply(get_season, axis=1)
    
    return df_reconstructed

def load_data_and_pipeline():
    """Load data and initialize pipeline - now uses pandas backend"""
    if not st.session_state.data_loaded:
        with st.spinner("Initializing Healthcare Analytics Pipeline..."):
            try:
                # Try to get Spark session (might not work in Streamlit)
                try:
                    st.session_state.spark_session = get_spark_session()
                except:
                    st.session_state.spark_session = None
                    st.info("Running in pandas mode")
                
                # Run pipeline (now uses pandas backend)
                st.session_state.pipeline_results = run_databricks_healthcare_pipeline()
                
                # Initialize AI Query Processor (works with or without Spark)
                st.session_state.ai_processor = HealthcareAIQueryProcessor(
                    spark_session=st.session_state.spark_session
                )
                
                st.session_state.data_loaded = True
                #st.success("✅ Pipeline initialized successfully!")
                
                # Validate the data
                df_test = get_pandas_dataframe()
                if df_test is not None:
                    is_valid, message = validate_dataframe_columns(df_test)
                    if not is_valid:
                        st.warning(f" Data validation warning: {message}")
                        st.info(f"Available columns: {list(df_test.columns)}")
                    else:
                        #st.success("Data structure validated successfully!")
                        # Check if we successfully reconstructed categorical columns
                        if all(col in df_test.columns for col in ['gender', 'specialty', 'season']):
                            pass
                            #st.info("Categorical columns reconstructed from one-hot encoding")
                
            except Exception as e:
                st.error(f"Error initializing pipeline: {str(e)}")
                st.exception(e)  # Show full traceback for debugging
                return False
    return True

def create_overview_metrics(df_pandas):
    """Create overview metrics cards with column validation"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Validate required columns exist
    required_cols = ['wait_days', 'readmit_risk', 'cost_estimate']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"Missing columns for metrics: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    with col1:
        total_patients = len(df_pandas)
        st.metric(
            label="📊 Total Patients",
            value=f"{total_patients:,}",
            #delta=f"+{total_patients//10} this month"
        )
    
    with col2:
        avg_wait = df_pandas['wait_days'].mean()
        st.metric(
            label="⏱️ Avg Wait Time",
            value=f"{avg_wait:.1f} days",
            #delta=f"-{avg_wait*0.1:.1f} from last month"
        )
    
    with col3:
        high_risk = len(df_pandas[df_pandas['readmit_risk'] > 0.7])
        st.metric(
            label="⚠️ High Risk Patients",
            value=f"{high_risk:,}",
            #delta=f"+{high_risk//20} this week"
        )
    
    with col4:
        total_cost = df_pandas['cost_estimate'].sum()
        st.metric(
            label="💰 Total Cost Estimate",
            value=f"£{total_cost/1000000:.1f}M",
            delta=f"+£{total_cost/10000000:.1f}M this quarter"
        )

def create_specialty_analysis(df_pandas):
    """Create specialty analysis visualizations with column validation"""
    st.subheader("🏥 Specialty Analysis")
    
    # Check for required columns
    required_cols = ['specialty', 'wait_days', 'cost_estimate', 'cancel_risk', 'readmit_risk', 'patient_id']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns for specialty analysis: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    # Specialty distribution
    col1, col2 = st.columns(2)
    
    with col1:
        specialty_counts = df_pandas['specialty'].value_counts()
        fig_pie = px.pie(
            values=specialty_counts.values,
            names=specialty_counts.index,
            title="Patient Distribution by Specialty"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Average wait time by specialty
        specialty_wait = df_pandas.groupby('specialty')['wait_days'].mean().sort_values(ascending=False)
        fig_bar = px.bar(
            x=specialty_wait.values,
            y=specialty_wait.index,
            orientation='h',
            title="Average Wait Time by Specialty",
            labels={'x': 'Average Wait Days', 'y': 'Specialty'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Specialty performance table
    st.subheader("📈 Specialty Performance Summary")
    specialty_summary = df_pandas.groupby('specialty').agg({
        'patient_id': 'count',
        'wait_days': 'mean',
        'cost_estimate': 'mean',
        'cancel_risk': 'mean',
        'readmit_risk': 'mean'
    }).round(2)
    specialty_summary.columns = ['Patient Count', 'Avg Wait Days', 'Avg Cost', 'Avg Cancel Risk', 'Avg Readmit Risk']
    st.dataframe(specialty_summary, use_container_width=True)

def create_risk_analysis(df_pandas):
    """Create risk analysis visualizations with column validation"""
    st.subheader("⚠️ Risk Analysis")
    
    # Check for required columns
    required_cols = ['readmit_risk', 'specialty', 'cancel_risk']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns for risk analysis: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Readmission risk distribution
        fig_hist = px.histogram(
            df_pandas,
            x='readmit_risk',
            nbins=20,
            title="Readmission Risk Distribution",
            labels={'x': 'Readmission Risk', 'y': 'Count'}
        )
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                          annotation_text="High Risk Threshold")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Cancellation risk by specialty
        cancel_risk = df_pandas.groupby('specialty')['cancel_risk'].mean().sort_values(ascending=False)
        fig_cancel = px.bar(
            x=cancel_risk.index,
            y=cancel_risk.values,
            title="Average Cancellation Risk by Specialty",
            labels={'x': 'Specialty', 'y': 'Average Cancellation Risk'}
        )
        fig_cancel.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cancel, use_container_width=True)

def create_capacity_planning(df_pandas):
    """Create improved capacity planning interface with real-time analysis"""
    st.subheader("📊 Capacity Planning & Backlog Analysis")
    
    # Calculate current backlog
    total_patients = len(df_pandas)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Capacity Settings")
        
        # Current capacity settings
        baseline_capacity = st.number_input(
            "Current Weekly Capacity", 
            min_value=50, 
            max_value=500, 
            value=100,
            help="Number of patients that can be treated per week currently"
        )
        
        # Capacity uplift slider for real-time analysis
        uplift_percentage = st.slider(
            "Capacity Uplift (%)", 
            min_value=0, 
            max_value=100, 
            value=20,
            help="Percentage increase in weekly capacity"
        )
        
        # Calculate new capacity
        new_capacity = int(baseline_capacity * (1 + uplift_percentage/100))
        
        # Display capacity comparison
        st.info(f"**Current Capacity:** {baseline_capacity} patients/week")
        st.success(f"**New Capacity:** {new_capacity} patients/week")
        st.metric("Additional Weekly Capacity", f"+{new_capacity - baseline_capacity}", f"{uplift_percentage}% increase")
        
        # Calculate basic metrics
        current_weeks_needed = np.ceil(total_patients / baseline_capacity)
        new_weeks_needed = np.ceil(total_patients / new_capacity)
        weeks_saved = current_weeks_needed - new_weeks_needed
        
        st.markdown("### Impact Summary")
        st.metric("Weeks to Clear Backlog", f"{int(new_weeks_needed)}", f"-{int(weeks_saved)} weeks")
        
        # Calculate patients moved earlier (those who would be treated sooner)
        patients_moved_earlier = 0
        for week in range(1, int(current_weeks_needed) + 1):
            current_week_patients = min(baseline_capacity, total_patients - (week-1) * baseline_capacity)
            new_week_patients = min(new_capacity, total_patients - (week-1) * new_capacity)
            if new_week_patients > current_week_patients:
                patients_moved_earlier += (new_week_patients - current_week_patients)
        
        st.metric("Patients Treated Earlier", f"{int(patients_moved_earlier):,}")
    
    with col2:
        st.markdown("### Backlog Reduction Timeline")
        
        # Create timeline data for visualization
        max_weeks = min(int(current_weeks_needed) + 5, 52)  # Cap at 52 weeks for display
        weeks = list(range(1, max_weeks + 1))
        
        current_backlog = []
        new_backlog = []
        
        for week in weeks:
            # Calculate remaining patients after each week
            current_treated = week * baseline_capacity
            new_treated = week * new_capacity
            
            current_remaining = max(0, total_patients - current_treated)
            new_remaining = max(0, total_patients - new_treated)
            
            current_backlog.append(current_remaining)
            new_backlog.append(new_remaining)
        
        # Create interactive chart
        fig_timeline = go.Figure()
        
        # Current capacity line
        fig_timeline.add_trace(go.Scatter(
            x=weeks,
            y=current_backlog,
            mode='lines+markers',
            name=f'Current Capacity ({baseline_capacity}/week)',
            line=dict(color='#ff6b6b', width=3),
            hovertemplate='<b>Week %{x}</b><br>' +
                         'Remaining Patients: %{y:,}<br>' +
                         f'Weekly Capacity: {baseline_capacity}<extra></extra>'
        ))
        
        # New capacity line
        fig_timeline.add_trace(go.Scatter(
            x=weeks,
            y=new_backlog,
            mode='lines+markers',
            name=f'Increased Capacity ({new_capacity}/week)',
            line=dict(color='#4ecdc4', width=3),
            hovertemplate='<b>Week %{x}</b><br>' +
                         'Remaining Patients: %{y:,}<br>' +
                         f'Weekly Capacity: {new_capacity}<extra></extra>'
        ))
        
        # Add area fill between lines to show improvement
        fig_timeline.add_trace(go.Scatter(
            x=weeks + weeks[::-1],
            y=new_backlog + current_backlog[::-1],
            fill='toself',
            fillcolor='rgba(78, 205, 196, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Improvement Area'
        ))
        
        fig_timeline.update_layout(
            title=f"Patient Backlog Over Time (Total: {total_patients:,} patients)",
            xaxis_title="Week",
            yaxis_title="Remaining Patients in Backlog",
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add annotations for key points
        if int(current_weeks_needed) <= max_weeks:
            fig_timeline.add_annotation(
                x=int(current_weeks_needed),
                y=0,
                text=f"Current: {int(current_weeks_needed)} weeks",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#ff6b6b",
                bgcolor="#ff6b6b",
                bordercolor="#ff6b6b",
                font=dict(color="white")
            )
        
        if int(new_weeks_needed) <= max_weeks:
            fig_timeline.add_annotation(
                x=int(new_weeks_needed),
                y=0,
                text=f"Improved: {int(new_weeks_needed)} weeks",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#4ecdc4",
                bgcolor="#4ecdc4",
                bordercolor="#4ecdc4",
                font=dict(color="white")
            )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Additional insights
        st.markdown("### 📈 Key Insights")
        
        efficiency_gain = ((current_weeks_needed - new_weeks_needed) / current_weeks_needed) * 100
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info(f"🎯 **Time Reduction:** {int(weeks_saved)} weeks faster")
            st.info(f"📊 **Efficiency Gain:** {efficiency_gain:.1f}%")
            
        with insights_col2:
            additional_weekly = new_capacity - baseline_capacity
            st.success(f"⚡ **Additional Weekly Throughput:** +{additional_weekly}")
            monthly_additional = additional_weekly * 4
            st.success(f"📅 **Additional Monthly Capacity:** +{monthly_additional}")
        
        # Cost-benefit analysis (simple estimation)
        if 'cost_estimate' in df_pandas.columns:
            avg_cost = df_pandas['cost_estimate'].mean()
            monthly_revenue_increase = monthly_additional * avg_cost
            
            st.markdown("### 💰 Financial Impact Estimate")
            st.metric(
                "Additional Monthly Revenue Potential", 
                f"£{monthly_revenue_increase:,.0f}" 
            )
        
        # # Specialty breakdown if available
        # if 'specialty' in df_pandas.columns:
        #     st.markdown("### 🏥 Impact by Specialty")
            
        #     # Calculate backlog by specialty
        #     specialty_counts = df_pandas['specialty'].value_counts()
        #     specialty_backlog = pd.DataFrame({
        #         'Specialty': specialty_counts.index,
        #         'Current_Patients': specialty_counts.values,
        #         'Current_Weeks': np.ceil(specialty_counts.values / baseline_capacity),
        #         'New_Weeks': np.ceil(specialty_counts.values / new_capacity)
        #     })
        #     specialty_backlog['Weeks_Saved'] = specialty_backlog['Current_Weeks'] - specialty_backlog['New_Weeks']
            
            # Display as table
            # st.dataframe(
            #     specialty_backlog.style.format({
            #         'Current_Patients': '{:,}',
            #         'Current_Weeks': '{:.0f}',
            #         'New_Weeks': '{:.0f}',
            #         'Weeks_Saved': '{:.0f}'
            #     }),
            #     use_container_width=True
            # )

# def create_capacity_planning(df_pandas):
#     """Create capacity planning interface - now using pandas backend"""
#     st.subheader("📊 Capacity Planning & Analysis")
    
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.markdown("### Capacity Settings")
#         baseline_capacity = st.number_input("Baseline Weekly Capacity", min_value=50, max_value=500, value=100)
#         uplift_percentage = st.slider("Capacity Uplift (%)", min_value=5, max_value=100, value=20)
        
#         if st.button("🚀 Run Capacity Analysis", type="primary"):
#             with st.spinner("Analyzing capacity impact..."):
#                 try:
#                     # Use pandas analysis directly
#                     summary_df, weekly_df, scheduled_df = analyze_capacity_impact_pandas(
#                         df_pandas, baseline_capacity, uplift_percentage
#                     )
                    
#                     # Store in session state
#                     st.session_state.capacity_results = {
#                         'summary': summary_df,
#                         'weekly': weekly_df if weekly_df is not None else None
#                     }
                    
#                     st.success("✅ Capacity analysis completed!")
                    
#                 except Exception as e:
#                     st.error(f"❌ Error in capacity analysis: {str(e)}")
#                     st.exception(e)
    
#     with col2:
#         if 'capacity_results' in st.session_state:
#             st.markdown("### Capacity Impact Results")
#             summary = st.session_state.capacity_results['summary'].iloc[0]
            
#             # Display key metrics
#             metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
#             with metrics_col1:
#                 st.metric(
#                     "Weeks Saved",
#                     f"{summary['weeks_saved']} weeks",
#                     f"-{summary['weeks_saved']} weeks sooner"
#                 )
            
#             with metrics_col2:
#                 st.metric(
#                     "Patients Moved Earlier",
#                     f"{summary['patients_moved_earlier']:,}",
#                     f"+{summary['new_capacity'] - summary['baseline_capacity']} weekly capacity"
#                 )
            
#             with metrics_col3:
#                 efficiency_gain = ((summary['baseline_weeks'] - summary['new_weeks']) / summary['baseline_weeks']) * 100
#                 st.metric(
#                     "Efficiency Gain",
#                     f"{efficiency_gain:.1f}%",
#                     f"+{((summary['new_capacity'] - summary['baseline_capacity']) / summary['baseline_capacity']) * 100:.1f}% improvement"
#                 )
            
#             # Capacity timeline chart
#             if st.session_state.capacity_results['weekly'] is not None:
#                 weekly_data = st.session_state.capacity_results['weekly']
#                 fig_timeline = go.Figure()
                
#                 fig_timeline.add_trace(go.Scatter(
#                     x=weekly_data['week'],
#                     y=weekly_data['baseline_backlog'],
#                     mode='lines+markers',
#                     name='Baseline Backlog',
#                     line=dict(color='red', width=3)
#                 ))
                
#                 fig_timeline.add_trace(go.Scatter(
#                     x=weekly_data['week'],
#                     y=weekly_data['new_backlog'],
#                     mode='lines+markers',
#                     name='New Capacity Backlog',
#                     line=dict(color='green', width=3)
#                 ))
                
#                 fig_timeline.update_layout(
#                     title="Backlog Reduction Timeline",
#                     xaxis_title="Week",
#                     yaxis_title="Backlog Count",
#                     hovermode='x unified'
#                 )
                
#                 st.plotly_chart(fig_timeline, use_container_width=True)

def create_ai_query_interface():
    """Create AI-powered query interface - now works with pandas backend"""
    st.subheader("🤖 AI-Powered Query Interface")
    st.markdown("Ask questions about your healthcare data in natural language!")
    
    # Sample queries
    with st.expander("📝 Sample Queries"):
        sample_queries = [
            "What is the average wait time for Cardiology and Orthopedics?",
            "Show me high risk patients with readmission risk above 70%",
            "Reduce cancellations by 15%",
            "What is the cost impact for Orthopedics?",
            "What is the cost impact by specialty?",
            "Show seasonal analysis for Winter",
            "What is the average cost for young adults?",
            "Impact of Baseline capacity 200 and Capacity Increase by 20%"
        ]
        for i, query in enumerate(sample_queries, 1):
            st.write(f"{i}. {query}")
    
    # Query input
    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., What is the average wait time for Orthopedics?",
        help="Ask questions about patient data, wait times, risks, costs, etc."
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        query_button = st.button("🔍 Process Query", type="primary")
    
    if query_button and user_query:
        with st.spinner("🧠 Processing your query..."):
            try:
                if st.session_state.ai_processor:
                    # Get the reconstructed DataFrame instead of raw pipeline results
                    df_reconstructed = get_pandas_dataframe()
                    if df_reconstructed is not None:
                        # Create a wrapper object that the AI processor expects
                        class DataFrameWrapper:
                            def __init__(self, df):
                                self.df = df
                                self.raw_pandas_df = df
                        
                        df_data = DataFrameWrapper(df_reconstructed)
                        result = st.session_state.ai_processor.process_natural_language_query(user_query, df_data)
                        
                        # Display query type
                        st.info(f"🏷️ **Detected Query Type:** {result.query_type}")
                        
                        # Check if this is a function-related query
                        is_function_query = (hasattr(result, 'query_type') and 
                                           result.query_type.lower() in ['cancellation_reduction', 'function', 'capacity_analysis'])
                        # Format and display response
                        if is_function_query:
                            response = st.session_state.ai_processor.format_response(result)
                            st.markdown("### 📊 Query Results")
                            st.markdown(response)
                        else:
                            # Show raw data if available
                            if hasattr(result, 'result_data') and isinstance(result.result_data, pd.DataFrame):
                                if len(result.result_data) > 0:
                                    st.markdown("### 📋 Resultant Data")
                                    st.dataframe(result.result_data, use_container_width=True)
                    else:
                        st.error("❌ Could not get reconstructed data for query processing")
                else:
                    st.error("❌ AI Query Processor not initialized")
                    
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)

def create_seasonal_analysis(df_pandas):
    """Create seasonal analysis with column validation"""
    st.subheader("🌤️ Seasonal Analysis")
    
    # Check for required columns
    required_cols = ['season', 'cancel_risk', 'readmit_risk']
    missing_cols = [col for col in required_cols if col not in df_pandas.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns for seasonal analysis: {missing_cols}")
        st.info(f"Available columns: {list(df_pandas.columns)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal distribution
        seasonal_counts = df_pandas['season'].value_counts()
        fig_seasonal = px.bar(
            x=seasonal_counts.index,
            y=seasonal_counts.values,
            title="Patient Count by Season",
            labels={'x': 'Season', 'y': 'Patient Count'}
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        # Seasonal risk analysis
        seasonal_risk = df_pandas.groupby('season').agg({
            'cancel_risk': 'mean',
            'readmit_risk': 'mean'
        }).round(3)
        
        fig_risk_seasonal = go.Figure()
        fig_risk_seasonal.add_trace(go.Bar(
            name='Cancel Risk',
            x=seasonal_risk.index,
            y=seasonal_risk['cancel_risk']
        ))
        fig_risk_seasonal.add_trace(go.Bar(
            name='Readmit Risk',
            x=seasonal_risk.index,
            y=seasonal_risk['readmit_risk']
        ))
        
        fig_risk_seasonal.update_layout(
            title='Average Risk by Season',
            xaxis_title='Season',
            yaxis_title='Risk Score',
            barmode='group'
        )
        st.plotly_chart(fig_risk_seasonal, use_container_width=True)

def get_pandas_dataframe():
    """Get pandas DataFrame from pipeline results with better error handling"""
    if st.session_state.pipeline_results:
        try:
            # Try different ways to get the pandas DataFrame
            df = None
            if 'raw_pandas_df' in st.session_state.pipeline_results:
                df = st.session_state.pipeline_results['raw_pandas_df']
            elif hasattr(st.session_state.pipeline_results.get('df_final'), 'df'):
                df = st.session_state.pipeline_results['df_final'].df
            elif hasattr(st.session_state.pipeline_results.get('df_final'), 'toPandas'):
                df = st.session_state.pipeline_results['df_final'].toPandas()
            else:
                # If all else fails, return the first 10k records for display
                df_final = st.session_state.pipeline_results.get('df_final')
                if df_final and hasattr(df_final, 'limit'):
                    df = df_final.limit(10000).toPandas()
                else:
                    st.error("❌ Unable to extract DataFrame from pipeline results")
                    return None
            
            # If we got a DataFrame, reconstruct categorical columns from one-hot encoding
            if df is not None:
                df_reconstructed = reconstruct_categorical_columns(df)
                return df_reconstructed
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error extracting DataFrame: {str(e)}")
            return None
    return None

def create_debug_info():
    """Create debug information panel"""
    st.subheader("🔍 Debug Information")
    
    if st.session_state.pipeline_results:
        st.write("**Pipeline Results Keys:**", list(st.session_state.pipeline_results.keys()))
        
        for key, value in st.session_state.pipeline_results.items():
            st.write(f"**{key}:** {type(value)}")
            
            # If it's a DataFrame wrapper, try to get more info
            if hasattr(value, 'df'):
                try:
                    st.write(f"  - DataFrame shape: {value.df.shape}")
                    st.write(f"  - DataFrame columns: {list(value.df.columns)}")
                except:
                    st.write(f"  - Could not access DataFrame info")
            elif hasattr(value, 'toPandas'):
                try:
                    sample_df = value.limit(5).toPandas()
                    st.write(f"  - DataFrame columns: {list(sample_df.columns)}")
                    st.write(f"  - Sample data:")
                    st.dataframe(sample_df.head())
                except:
                    st.write(f"  - Could not convert to pandas")
    else:
        st.write("No pipeline results available")
    
    # Test DataFrame extraction
    df_test = get_pandas_dataframe()
    if df_test is not None:
        st.write("**Successfully extracted DataFrame:**")
        st.write(f"Shape: {df_test.shape}")
        st.write(f"Columns: {list(df_test.columns)}")
        st.dataframe(df_test.head())
    else:
        st.write("**Failed to extract DataFrame**")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Healthcare Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🕹️ Dashboard Controls")
    
    # Load data
    if st.sidebar.button("🔄 Initialize/Refresh Data", type="primary"):
        st.session_state.data_loaded = False
        st.session_state.pipeline_results = None
        load_data_and_pipeline()
    
    #Debug toggle
    # show_debug = st.sidebar.checkbox("🔍 Show Debug Info", value=False)
    
    # Load data if not already loaded
    if not load_data_and_pipeline():
        st.stop()
    
    # Show debug info if requested
    # if show_debug:
    #     create_debug_info()
    #     st.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📊 Select Analysis",
        [
            "📈 Overview & Metrics",
            "📊 Capacity Planning",
            "🏥 Specialty Analysis", 
            "⚠️ Risk Analysis",
            "🌤️ Seasonal Analysis",
            "🤖 AI Query Interface"
        ]
    )
    
    # Get DataFrame for analysis
    try:
        df_pandas = get_pandas_dataframe()
        
        if df_pandas is not None and len(df_pandas) > 0:
            # Validate DataFrame structure
            is_valid, validation_message = validate_dataframe_columns(df_pandas)
            if not is_valid:
                st.warning(f"⚠️ Data validation warning: {validation_message}")
                st.info(f"Available columns: {list(df_pandas.columns)}")
                # Continue anyway since we now reconstruct the missing columns
            else:
                pass
                #st.success("✅ Data validation passed")
            
            # Sample data for performance if too large
            if len(df_pandas) > 100000:
                #st.sidebar.info(f"📊 Using sample of {min(100000, len(df_pandas)):,} records for performance")
                df_display = df_pandas.sample(n=min(100000, len(df_pandas)), random_state=42)
            else:
                df_display = df_pandas
            
            # Page routing
            if page == "📈 Overview & Metrics":
                create_overview_metrics(df_display)
                st.markdown("---")
                
                # Quick insights with column validation
                st.subheader("🔍 Quick Insights")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'specialty' in df_display.columns:
                        most_common_specialty = df_display['specialty'].mode().iloc[0]
                        st.info(f"📊 **Most Common Specialty:** {most_common_specialty}")
                        
                        highest_risk_specialty = df_display.groupby('specialty')['readmit_risk'].mean().idxmax()
                        st.warning(f"⚠️ **Highest Risk Specialty:** {highest_risk_specialty}")
                    else:
                        st.warning("⚠️ Specialty information not available")
                
                with col2:
                    if 'age' in df_display.columns:
                        avg_age = df_display['age'].mean()
                        st.info(f"👥 **Average Patient Age:** {avg_age:.1f} years")
                    else:
                        st.warning("⚠️ Age information not available")
                    
                    if 'season' in df_display.columns:
                        winter_patients = len(df_display[df_display['season'] == 'Winter'])
                        st.info(f"❄️ **Winter Patients:** {winter_patients:,}")
                    else:
                        st.warning("⚠️ Seasonal information not available")
            
            elif page == "🏥 Specialty Analysis":
                create_specialty_analysis(df_display)
            
            elif page == "⚠️ Risk Analysis":
                create_risk_analysis(df_display)
            
            elif page == "📊 Capacity Planning":
                create_capacity_planning(df_pandas)  # Use full dataset for analysis
            
            elif page == "🌤️ Seasonal Analysis":
                create_seasonal_analysis(df_display)
            
            elif page == "🤖 AI Query Interface":
                create_ai_query_interface()
        
        else:
            st.warning("⚠️ No data available. Please initialize the pipeline first.")
    
    except Exception as e:
        st.error(f"❌ Error loading data for display: {str(e)}")
        st.exception(e)
        
        # Show debug info automatically on error
        st.markdown("---")
        create_debug_info()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            🏩 Healthcare Analytics Dashboard | Built with Streamlit & Pandas | 
            Powered by Advanced ML & AI | Compatible with Databricks
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()