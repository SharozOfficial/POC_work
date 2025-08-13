from mlflow.deployments import get_deploy_client
from faker import Faker
import random
import pandas as pd
import numpy as np
import time
import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import math

# For Databricks environment
try:
    from databricks import sql as databricks_sql
    from databricks.connect import DatabricksSession
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

# Fallback imports for local development
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    import pyspark.sql.functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import openai
from dataclasses import dataclass

# Global flag to determine execution mode
USE_DATABRICKS_SQL = True

def get_spark_session():
    """Get Spark session - compatible with both environments"""
    if DATABRICKS_AVAILABLE and USE_DATABRICKS_SQL:
        try:
            # Use Databricks Connect for Streamlit apps
            spark = DatabricksSession.builder.getOrCreate()
            return spark
        except Exception as e:
            print(f"Warning: Could not create Databricks session: {e}")
            # Fall back to reading from Delta tables via SQL
            return None
    
    if PYSPARK_AVAILABLE:
        try:
            spark = SparkSession.getActiveSession()
            if spark is None:
                spark = SparkSession.builder \
                    .appName("DatabricksHealthcareAnalytics") \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
                    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                    .getOrCreate()
            return spark
        except Exception as e:
            print(f"Warning: Could not configure Spark: {e}")
            return None
    
    return None

def get_databricks_connection():
    """Get connection to Databricks SQL warehouse"""
    try:
        import os
        # These should be set as environment variables or in secrets
        connection = databricks_sql.connect(
            server_hostname=os.getenv('DATABRICKS_SERVER_HOSTNAME'),
            http_path=os.getenv('DATABRICKS_HTTP_PATH'),
            access_token=os.getenv('DATABRICKS_TOKEN')
        )
        return connection
    except Exception as e:
        print(f"Could not connect to Databricks SQL: {e}")
        return None

def execute_sql_query(query: str):
    """Execute SQL query using Databricks SQL connector"""
    if USE_DATABRICKS_SQL:
        connection = get_databricks_connection()
        if connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=columns)
            except Exception as e:
                print(f"Error executing SQL query: {e}")
                return pd.DataFrame()
            finally:
                connection.close()
    
    # Fallback to Spark if available
    spark = get_spark_session()
    if spark:
        try:
            return spark.sql(query).toPandas()
        except Exception as e:
            print(f"Error executing Spark SQL: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

def create_synthetic_patient_data_pandas(num_records=100000):
    """Create synthetic patient data using pandas instead of Spark"""
    print(f"Generating {num_records:,} synthetic patient records with pandas...")
    
    # Define all data structures
    specialties = ["Orthopedics", "ENT", "General Surgery", "Gynaecology", 
                  "Urology", "Ophthalmology", "Cardiology"]
    specialty_weights = [30, 15, 25, 10, 10, 5, 5]
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    
    # Set seeds for reproducibility
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Generate data efficiently with numpy
    data = {
        'patient_id': range(1, num_records + 1),
        'age': np.random.randint(18, 91, num_records),
        'gender': np.random.choice(["M", "F"], num_records, p=[0.48, 0.52]),
        'specialty': np.random.choice(specialties, num_records, p=np.array(specialty_weights)/sum(specialty_weights)),
        'cancel_risk': np.round(np.random.uniform(0.05, 0.5, num_records), 2),
        'los': np.random.randint(1, 11, num_records),
        'cost_estimate': np.random.randint(2000, 10001, num_records),
        'season': np.random.choice(seasons, num_records)
    }
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df):,} records")
    return df

def calculate_priority_score_pandas(df):
    """Calculate priority score using pandas operations"""
    print("Computing priority scores...")
    
    # Initialize score
    df['priority_score'] = 0.0
    
    # Age factor
    df.loc[df['age'] > 70, 'priority_score'] += 0.15
    
    # LOS factor
    df.loc[df['los'] > 3, 'priority_score'] += 0.15
    
    # Cost factor
    df.loc[df['cost_estimate'] > 5000, 'priority_score'] += 0.15
    
    # Specialty weights
    specialty_weights = {
        "Cardiology": 0.15, "General Surgery": 0.13, "Urology": 0.11,
        "Gynaecology": 0.09, "Ophthalmology": 0.07, "Orthopedics": 0.05, "ENT": 0.03
    }
    
    for specialty, weight in specialty_weights.items():
        df.loc[df['specialty'] == specialty, 'priority_score'] += weight
    
    # Season factor
    df.loc[df['season'] == 'Winter', 'priority_score'] += 0.07
    df.loc[df['season'] == 'Autumn', 'priority_score'] += 0.03
    
    # Cap at 1.0
    df['priority_score'] = df['priority_score'].clip(upper=1.0)
    
    return df

def calculate_readmission_risk_pandas(df):
    """Calculate readmission risk using pandas operations"""
    print("Computing readmission risks...")
    
    # Initialize readmission risk
    df['readmit_risk'] = 0.0
    
    # Age factor
    df.loc[df['age'] > 70, 'readmit_risk'] += 0.2
    
    # LOS factor  
    df.loc[df['los'] > 5, 'readmit_risk'] += 0.2
    
    # Cancel risk factor
    df['readmit_risk'] += 0.2 * df['cancel_risk']
    
    # Cost factor
    df.loc[df['cost_estimate'] > 6000, 'readmit_risk'] += 0.2
    
    # Season factor
    df.loc[df['season'] == 'Winter', 'readmit_risk'] += 0.2
    
    # Cap at 1.0
    df['readmit_risk'] = df['readmit_risk'].clip(upper=1.0)
    
    return df

def train_optimized_ml_model_pandas(df):
    """Train ML model using pandas DataFrame"""
    print("Training XGBoost model with pandas data...")
    
    # Create label
    df['label'] = (df['readmit_risk'] > 0.5).astype(int)
    
    # Prepare features
    feature_cols = ['age', 'los', 'cost_estimate', 'cancel_risk', 'priority_score']
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['gender', 'specialty', 'season'], drop_first=True)
    
    # Get all feature columns (original + encoded)
    encoded_feature_cols = [col for col in df_encoded.columns if 
                           col.startswith(tuple(feature_cols)) or 
                           col.startswith('gender_') or 
                           col.startswith('specialty_') or 
                           col.startswith('season_')]
    
    X = df_encoded[encoded_feature_cols].fillna(0)
    y = df_encoded['label']
    
    print(f"Training on {len(X):,} samples with {X.shape[1]} features")
    
    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=50,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )
    
    # Train model
    start_time = time.time()
    xgb_model.fit(X, y)
    training_time = time.time() - start_time
    
    # Generate predictions
    predictions = xgb_model.predict(X)
    probabilities = xgb_model.predict_proba(X)[:, 1]
    
    # Add predictions to DataFrame
    df_encoded['prediction'] = predictions
    df_encoded['score'] = probabilities
    df_encoded['percentage'] = np.round(probabilities * 100, 2)
    
    # Calculate metrics
    auc = roc_auc_score(y, probabilities)
    accuracy = accuracy_score(y, predictions)
    
    print(f"XGBoost completed in {training_time:.2f}s")
    print(f"XGB Performance: AUC={auc:.4f}, Accuracy={accuracy:.4f}")
    
    return df_encoded, xgb_model

def generate_wait_days_pandas(df):
    """Generate wait days using pandas"""
    print("Generating wait days...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate wait days based on priority score
    df['wait_days'] = (100 - (df['priority_score'] * 100)).astype(int) + \
                      np.random.randint(0, 31, len(df))
    
    return df

def save_to_delta_table(df, table_name="patient_data"):
    """Save DataFrame to Delta table"""
    try:
        # If we have Spark available, convert pandas to Spark and save
        spark = get_spark_session()
        if spark:
            spark_df = spark.createDataFrame(df)
            spark_df.write.format("delta") \
                  .mode("overwrite") \
                  .option("overwriteSchema", "true") \
                  .saveAsTable(table_name)
            print(f"Data saved to Delta table: {table_name}")
            return True
        else:
            # Fallback: save as parquet or use Databricks SQL
            print("Spark not available, saving to parquet file...")
            df.to_parquet(f"/tmp/{table_name}.parquet")
            return True
    except Exception as e:
        print(f"Error saving to Delta table: {e}")
        return False

def analyze_capacity_impact_pandas(df, baseline_capacity=100, uplift_percentage=20):
    """Analyze capacity impact using pandas operations"""
    print(f"Analyzing capacity impact: {uplift_percentage}% uplift...")
    
    # Calculate new capacity
    new_capacity = int(baseline_capacity * (1 + uplift_percentage/100))
    
    # Sort by priority score and wait days
    df_sorted = df.sort_values(['priority_score', 'wait_days'], ascending=[False, False]).reset_index(drop=True)
    
    # Calculate queue positions and scheduling weeks
    df_sorted['queue_position'] = range(1, len(df_sorted) + 1)
    df_sorted['baseline_week'] = np.ceil(df_sorted['queue_position'] / baseline_capacity)
    df_sorted['new_week'] = np.ceil(df_sorted['queue_position'] / new_capacity)
    
    # Calculate summary metrics
    baseline_weeks_needed = int(df_sorted['baseline_week'].max())
    new_weeks_needed = int(df_sorted['new_week'].max())
    weeks_saved = baseline_weeks_needed - new_weeks_needed
    patients_moved_earlier = len(df_sorted[df_sorted['new_week'] < df_sorted['baseline_week']])
    avg_weeks_saved = (df_sorted['baseline_week'] - df_sorted['new_week']).mean()
    
    # Create summary DataFrame
    summary_data = {
        'total_patients': len(df),
        'baseline_capacity': baseline_capacity,
        'new_capacity': new_capacity,
        'baseline_weeks': baseline_weeks_needed,
        'new_weeks': new_weeks_needed,
        'weeks_saved': weeks_saved,
        'patients_moved_earlier': patients_moved_earlier,
        'avg_weeks_saved': avg_weeks_saved or 0.0
    }
    
    summary_df = pd.DataFrame([summary_data])
    
    # Create weekly backlog analysis
    weekly_data = []
    for week in range(1, min(baseline_weeks_needed + 1, 52)):
        baseline_backlog = len(df_sorted[df_sorted['baseline_week'] > week])
        new_backlog = len(df_sorted[df_sorted['new_week'] > week])
        weekly_data.append({
            'week': week,
            'baseline_backlog': baseline_backlog,
            'new_backlog': new_backlog
        })
    
    weekly_df = pd.DataFrame(weekly_data)
    
    return summary_df, weekly_df, df_sorted

def identify_high_risk_patients_pandas(df, risk_threshold=0.7):
    """Identify high-risk patients using pandas"""
    print(f"Identifying high-risk patients (threshold: {risk_threshold})...")
    
    high_risk_df = df[df['readmit_risk'] > risk_threshold].sort_values('readmit_risk', ascending=False)
    high_risk_count = len(high_risk_df)
    
    print(f"Found {high_risk_count:,} high-risk patients")
    return high_risk_df

def optimize_schedule_pandas(df, weekly_capacity=100, risk_weight=0.3):
    """Optimize schedule using pandas operations"""
    print("Creating optimized schedule...")
    
    # Calculate combined score
    df['combined_score'] = (df['priority_score'] * (1 - risk_weight)) + (df['readmit_risk'] * risk_weight)
    
    # Sort by combined score and wait days
    df_scheduled = df.sort_values(['combined_score', 'wait_days'], ascending=[False, False]).reset_index(drop=True)
    
    # Calculate queue position and scheduled week
    df_scheduled['queue_position'] = range(1, len(df_scheduled) + 1)
    df_scheduled['scheduled_week'] = np.ceil(df_scheduled['queue_position'] / weekly_capacity)
    
    return df_scheduled

def run_databricks_healthcare_pipeline():
    """Execute the complete healthcare analytics pipeline with pandas fallback"""
    print("Starting Healthcare Analytics Pipeline (Pandas Mode)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Generate synthetic data
        df = create_synthetic_patient_data_pandas(num_records=100000)
        
        # Step 2: Calculate priority scores
        df = calculate_priority_score_pandas(df)
        
        # Step 3: Calculate readmission risks
        df = calculate_readmission_risk_pandas(df)
        
        # Step 4: Train ML model
        df_with_predictions, ml_model = train_optimized_ml_model_pandas(df)
        
        # Step 5: Generate wait days
        df_final = generate_wait_days_pandas(df_with_predictions)
        
        # Step 6: Save to Delta table (if possible)
        save_to_delta_table(df_final)
        
        # Step 7: Run some analyses
        high_risk_patients = identify_high_risk_patients_pandas(df_final, risk_threshold=0.7)
        
        print(f"High-Risk Patients: {len(high_risk_patients):,}")
        print(f"Total Pipeline Execution Time: {time.time() - start_time:.2f}s")
        print("=" * 60)
        
        # Return results in a format compatible with existing code
        class DataFrameWrapper:
            def __init__(self, df):
                self.df = df
            
            def limit(self, n):
                return DataFrameWrapper(self.df.head(n))
            
            def toPandas(self):
                return self.df
            
            def count(self):
                return len(self.df)
            
            def filter(self, condition):
                # This would need more complex logic for real Spark-like filtering
                return DataFrameWrapper(self.df)
        
        return {
            'df_final': DataFrameWrapper(df_final),
            'predictions': DataFrameWrapper(df_with_predictions),
            'ml_model': ml_model,
            'high_risk_patients': DataFrameWrapper(high_risk_patients),
            'raw_pandas_df': df_final  # For direct pandas access
        }
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise

# Compatibility functions for existing code
def analyze_capacity_impact_optimized(df, baseline_capacity=100, uplift_percentage=20):
    """Wrapper to maintain compatibility with existing code"""
    if hasattr(df, 'df'):  # DataFrameWrapper
        pandas_df = df.df
    else:
        pandas_df = df
    
    summary_df, weekly_df, scheduled_df = analyze_capacity_impact_pandas(
        pandas_df, baseline_capacity, uplift_percentage
    )
    
    # Return as DataFrameWrapper for compatibility
    class DataFrameWrapper:
        def __init__(self, df):
            self.df = df
        
        def toPandas(self):
            return self.df
    
    return DataFrameWrapper(summary_df), DataFrameWrapper(weekly_df), DataFrameWrapper(scheduled_df)

def identify_high_risk_patients_optimized(df, risk_threshold=0.7):
    """Wrapper to maintain compatibility"""
    if hasattr(df, 'df'):
        pandas_df = df.df
    else:
        pandas_df = df
    
    high_risk_df = identify_high_risk_patients_pandas(pandas_df, risk_threshold)
    
    class DataFrameWrapper:
        def __init__(self, df):
            self.df = df
        
        def count(self):
            return len(self.df)
        
        def toPandas(self):
            return self.df
    
    return DataFrameWrapper(high_risk_df)

def optimize_schedule_optimized(df, weekly_capacity=100, risk_weight=0.3):
    """Wrapper to maintain compatibility"""
    if hasattr(df, 'df'):
        pandas_df = df.df
    else:
        pandas_df = df
    
    scheduled_df = optimize_schedule_pandas(pandas_df, weekly_capacity, risk_weight)
    
    class DataFrameWrapper:
        def __init__(self, df):
            self.df = df
        
        def toPandas(self):
            return self.df
    
    return DataFrameWrapper(scheduled_df)

# Keep the existing HealthcareAIQueryProcessor class but modify to use pandas
@dataclass
class QueryResult:
    """Structure for query results"""
    query_type: str
    sql_query: str = None
    function_name: str = None
    parameters: Dict = None
    result_data: Any = None
    visualization_type: str = None
    cost_impact: float = None
    bed_days_freed: int = None

class HealthcareAIQueryProcessor:
    def __init__(self, spark_session=None, openai_api_key: str = None):
        self.spark = spark_session
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key

        # Define available functions and their descriptions
        self.available_functions = {
            'analyze_capacity_impact_optimized': {
                'description': 'Analyze impact of capacity increase on waiting list',
                'parameters': ['baseline_capacity', 'uplift_percentage'],
                'returns': 'weeks_saved, patients_moved_earlier, cost_impact'
            },
            'identify_high_risk_patients_optimized': {
                'description': 'Find patients with high readmission risk',
                'parameters': ['risk_threshold'],
                'returns': 'high_risk_patient_count, potential_cost_savings'
            },
            'optimize_schedule_optimized': {
                'description': 'Create optimized schedule balancing priority and risk',
                'parameters': ['weekly_capacity', 'risk_weight'],
                'returns': 'optimized_schedule, efficiency_gain'
            },
            'simulate_cancellation_reduction': {
                'description': 'Simulate impact of reducing cancellations',
                'parameters': ['reduction_percentage'],
                'returns': 'cost_savings, bed_days_freed, capacity_freed'
            }
        }
        
        # Define common query patterns (keeping existing patterns)
        self.query_patterns = {
            'average_wait_time': r'(average|avg|mean).*(wait|waiting).*(time|days|period)',
            'capacity_analysis': r'(capacity|increase|uplift|boost).*((\d+)%?|\d+)',
            'cancellation_reduction': r'(reduce|decrease|lower|reduction).*(cancel|cancellation).*((\d+)%?|\d+)',
            'high_risk_patients': r'(high.risk|risky|dangerous).*(patient|case)',
            'specialty_analysis': r'(specialty|department|ward|category).*(analysis|breakdown|summary)',
            'cost_impact': r'(cost|expense|budget|saving).*(impact|effect|analysis)',
            'seasonal_readmission': r'(season|seasonal|winter|summer|spring|autumn).*(readmission|readmit)',
            'seasonal_cost': r'(season|seasonal|winter|summer|spring|autumn).*(cost|expense|budget)',
            'seasonal_analysis': r'(season|seasonal|winter|summer|spring|autumn).*(analysis|impact|effect|comparison)',
            'seasonal_comparison': r'(which|what).*(season|seasonal).*(highest|most|best|worst)',
            'age_based_risk': r'(risk|readmit|readmission).*(age|elderly|young|adult|child|children|senior|group|segment)|(age|elderly|young|adult|child|children|senior|group|segment).*(risk|readmit|readmission)',
            'age_based_cancellation': r'(age|elderly|young|children|child|senior|middle-aged|group|segment).*(cancel|cancellation)',
            'age_based_cost': r'(age|elderly|young|child|children|senior|middle-aged|group|segment).*(cost|expense|budget)'
        }
    
    def classify_query(self, user_query: str) -> str:
        """Classify the type of query based on patterns"""
        user_query_lower = user_query.lower()
        
        for query_type, pattern in self.query_patterns.items():
            if re.search(pattern, user_query_lower):
                return query_type
        
        return 'general_analysis'
    
    def extract_parameters(self, user_query: str) -> Dict:
        """Extract numerical parameters from user query"""
        parameters = {}
        
        # # Extract percentages
        # # percent_match = re.search(r'(\d+(?:\.\d+)?)%?', user_query)
        # percent_match = re.search(r'(\d+(?:\.\d+)?)%', user_query)
        # if percent_match:
        #     parameters['percentage'] = float(percent_match.group(1))
        
        # # Extract specific numbers
        # number_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', user_query)
        # if number_matches:
        #     parameters['numbers'] = [float(n) for n in number_matches]
        
        # # Extract capacity numbers
        # # capacity_match = re.search(r'capacity.*?(\d+)', user_query.lower())
        # capacity_match = re.search(r'capacity.*?(\d+)(?!%)', user_query.lower())
        # if capacity_match:
        #     # parameters['baseline_capacity'] = int(capacity_match.group(1))
        #     capacity_num = int(capacity_match.group(1))
        #     if 'percentage' not in parameters or capacity_num != parameters['percentage']:
        #         parameters['baseline_capacity'] = capacity_num
        #         parameters['weekly_capacity'] = capacity_num
        
        # if ('capacity' in user_query.lower() or 'increase' in user_query.lower()) and 'baseline_capacity' not in parameters:
        #     parameters['baseline_capacity'] = 100
        #     parameters['weekly_capacity'] = 100

        # Extract percentages first
        percent_match = re.search(r'(\d+(?:\.\d+)?)%', user_query)
        if percent_match:
            parameters['percentage'] = float(percent_match.group(1))
    
        # Extract specific numbers
        number_matches = re.findall(r'\b(\d+(?:\.\d+)?)\b', user_query)
        if number_matches:
            parameters['numbers'] = [float(n) for n in number_matches]
    
        # Extract baseline capacity - look for explicit baseline capacity mentions
        baseline_capacity_match = re.search(r'baseline\s+capacity.*?(\d+)', user_query.lower())
        if baseline_capacity_match:
            parameters['baseline_capacity'] = int(baseline_capacity_match.group(1))
            parameters['weekly_capacity'] = int(baseline_capacity_match.group(1))
    
        # If no baseline capacity mentioned but it's a capacity query, use default
        if ('capacity' in user_query.lower() or 'increase' in user_query.lower()) and 'baseline_capacity' not in parameters:
            parameters['baseline_capacity'] = 100
            parameters['weekly_capacity'] = 100
        
        # Extract risk weight
        risk_weight_match = re.search(r'risk.weight.*?(\d+(?:\.\d+)?)', user_query.lower())
        if risk_weight_match:
            parameters['risk_weight'] = float(risk_weight_match.group(1))
        
        # Extract threshold values
        threshold_match = re.search(r'threshold.*?(\d+(?:\.\d+)?)', user_query.lower())
        if threshold_match:
            parameters['threshold'] = float(threshold_match.group(1))
        
        # Extract specialties
        specialties = ["orthopedics", "ent", "general surgery", "gynaecology", 
                      "urology", "ophthalmology", "cardiology"]
        
        found_specialties = []
        for specialty in specialties:
            pattern = r'\b' + re.escape(specialty.lower()) + r'\b'
            if re.findall(pattern, user_query.lower()):
                found_specialties.append(specialty)
        
        if found_specialties:
            titled_specialties = [s.title() for s in found_specialties]
            parameters['specialties'] = titled_specialties

        # Extract age information (keeping existing logic)
        age_match = re.search(r'\bage(?:d)?\s*(\d{1,3})\b', user_query.lower())
        if age_match:
            parameters['age'] = int(age_match.group(1))

        age_range_match = re.search(r'between\s+(\d{1,3})\s+(?:and|to)\s+(\d{1,3})', user_query.lower())
        if age_range_match:
            parameters['age_min'] = int(age_range_match.group(1))
            parameters['age_max'] = int(age_range_match.group(2))

        age_groups = {
            "child": (0, 17),
            "young": (18, 35),
            "young adult": (18, 35),
            "adult": (36, 60),
            "elderly": (61, 120),
            "senior": (61, 120),
            "middle-aged": (40, 60),
        }

        for group in age_groups:
            if re.search(r'\b' + re.escape(group) + r'\b', user_query.lower()):
                parameters['age_group'] = group
                parameters['age_min'], parameters['age_max'] = age_groups[group]
        
        return parameters
    
    def execute_pandas_query(self, query_type: str, parameters: Dict, df: pd.DataFrame) -> pd.DataFrame:
        """Execute queries using pandas operations instead of SQL"""
        
        if query_type == 'average_wait_time':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            result = filtered_df.groupby('specialty').agg({
                'wait_days': 'mean',
                'patient_id': 'count'
            }).rename(columns={'wait_days': 'avg_wait_days', 'patient_id': 'patient_count'})
            return result.reset_index()
        
        elif query_type == 'specialty_analysis':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            result = filtered_df.groupby('specialty').agg({
                'patient_id': 'count',
                'wait_days': 'mean',
                'cost_estimate': 'mean',
                'cancel_risk': 'mean',
                'readmit_risk': 'mean'
            }).rename(columns={
                'patient_id': 'patient_count',
                'wait_days': 'avg_wait_days',
                'cost_estimate': 'avg_cost',
                'cancel_risk': 'avg_cancel_risk',
                'readmit_risk': 'avg_readmit_risk'
            })
            return result.reset_index()
        
        elif query_type == 'high_risk_patients':
            threshold = parameters.get('percentage', 70) / 100
            high_risk_df = df[df['readmit_risk'] > threshold]
            
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                high_risk_df = high_risk_df[high_risk_df['specialty'].isin(parameters['specialties'])]
            
            result = high_risk_df.groupby('specialty').agg({
                'patient_id': 'count',
                'readmit_risk': 'mean',
                'cost_estimate': 'mean'
            }).rename(columns={
                'patient_id': 'high_risk_count',
                'readmit_risk': 'avg_risk',
                'cost_estimate': 'avg_cost'
            })
            return result.reset_index()
        
        elif query_type == 'cost_impact':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            filtered_df['potential_loss_from_cancellations'] = filtered_df['cost_estimate'] * filtered_df['cancel_risk']
            
            result = filtered_df.groupby('specialty').agg({
                'cost_estimate': ['sum', 'mean', 'count'],
                'potential_loss_from_cancellations': 'sum'
            })
            result.columns = ['total_cost', 'avg_cost', 'patient_count', 'potential_loss_from_cancellations']
            return result.reset_index()
        
        elif query_type == 'seasonal_analysis':
            if 'specialties' in parameters and len(parameters['specialties']) > 0:
                filtered_df = df[df['specialty'].isin(parameters['specialties'])]
            else:
                filtered_df = df
            
            result = filtered_df.groupby('season').agg({
                'cancel_risk': 'mean',
                'patient_id': 'count'
            }).rename(columns={
                'cancel_risk': 'avg_cancel_risk',
                'patient_id': 'total_cases'
            })
            return result.reset_index().sort_values('avg_cancel_risk', ascending=False)
        
        elif query_type == 'age_based_risk':
            filtered_df = self.apply_age_filter(df, parameters)
            
            # Create age groups
            filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                            bins=[0, 17, 35, 60, 120], 
                                            labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
            
            result = filtered_df.groupby('age_group').agg({
                'readmit_risk': 'mean',
                'patient_id': 'count'
            }).rename(columns={
                'readmit_risk': 'avg_readmit_risk',
                'patient_id': 'total_patients'
            })
            result = result[result['total_patients'] > 0]
            return result.reset_index()
        
        elif query_type == 'age_based_cancellation':
            filtered_df = self.apply_age_filter(df, parameters)
            
            filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                            bins=[0, 17, 35, 60, 120], 
                                            labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
            
            result = filtered_df.groupby('age_group').agg({
                'cancel_risk': 'mean',
                'patient_id': 'count'
            }).rename(columns={
                'cancel_risk': 'avg_cancel_risk',
                'patient_id': 'total_patients'
            })
            result = result[result['total_patients'] > 0]
            return result.reset_index().sort_values('avg_cancel_risk', ascending=False)
        
        elif query_type == 'age_based_cost':
            filtered_df = self.apply_age_filter(df, parameters)
            
            filtered_df['age_group'] = pd.cut(filtered_df['age'], 
                                            bins=[0, 17, 35, 60, 120], 
                                            labels=['Child', 'Young Adult', 'Adult', 'Elderly'])
            
            result = filtered_df.groupby('age_group').agg({
                'cost_estimate': ['mean', 'sum', 'count']
            })
            result.columns = ['avg_cost', 'total_cost', 'patient_count']
            result = result[result['patient_count'] > 0]
            return result.reset_index().sort_values('avg_cost', ascending=False)
        
        # Default query
        return df.head(10)
    
    def apply_age_filter(self, df, parameters):
        """Apply age filtering based on parameters"""
        filtered_df = df.copy()
        
        if 'age' in parameters:
            filtered_df = filtered_df[filtered_df['age'] == parameters['age']]
        elif 'age_min' in parameters and 'age_max' in parameters:
            filtered_df = filtered_df[(filtered_df['age'] >= parameters['age_min']) & 
                                    (filtered_df['age'] <= parameters['age_max'])]
        elif 'age_group' in parameters:
            group = parameters['age_group']
            if group == "child":
                filtered_df = filtered_df[(filtered_df['age'] >= 0) & (filtered_df['age'] <= 17)]
            elif group in ["young", "young adult"]:
                filtered_df = filtered_df[(filtered_df['age'] >= 18) & (filtered_df['age'] <= 35)]
            elif group == "adult":
                filtered_df = filtered_df[(filtered_df['age'] >= 36) & (filtered_df['age'] <= 60)]
            elif group in ["elderly", "senior"]:
                filtered_df = filtered_df[filtered_df['age'] > 60]
            elif group == "middle-aged":
                filtered_df = filtered_df[(filtered_df['age'] >= 40) & (filtered_df['age'] <= 60)]
        
        if 'specialties' in parameters and len(parameters['specialties']) > 0:
            filtered_df = filtered_df[filtered_df['specialty'].isin(parameters['specialties'])]
        
        return filtered_df
    
    def simulate_cancellation_reduction(self, df: pd.DataFrame, reduction_percentage: float) -> Dict:
        """Simulate the impact of reducing cancellations using pandas"""
        
        # Calculate current cancellation impact
        total_patients = len(df)
        current_cancellations = df['cancel_risk'].sum()
        avg_cost = df['cost_estimate'].mean()
        avg_los = df['los'].mean()
        
        # Calculate reduction impact
        cancellations_prevented = current_cancellations * (reduction_percentage / 100)
        cost_savings = cancellations_prevented * avg_cost
        bed_days_freed = cancellations_prevented * avg_los
        
        # Calculate capacity freed (assuming 5 days per week operation)
        weekly_capacity_freed = bed_days_freed / 7 * 5
        
        return {
            'current_expected_cancellations': round(current_cancellations, 2),
            'cancellations_prevented': round(cancellations_prevented, 2),
            'cost_savings': round(cost_savings, 2),
            'bed_days_freed': round(bed_days_freed, 2),
            'weekly_capacity_freed': round(weekly_capacity_freed, 2),
            'reduction_percentage': reduction_percentage
        }
    
    def execute_function_call(self, function_name: str, parameters: Dict, df) -> Dict:
        """Execute specific function calls based on the query"""
        
        # Get pandas DataFrame from wrapper or direct DataFrame
        if hasattr(df, 'df'):
            pandas_df = df.df
        elif hasattr(df, 'raw_pandas_df'):
            pandas_df = df.raw_pandas_df  
        else:
            pandas_df = df
        
        if function_name == 'simulate_cancellation_reduction':
            reduction_pct = parameters.get('percentage', 20)
            return self.simulate_cancellation_reduction(pandas_df, reduction_pct)
        
        elif function_name == 'analyze_capacity_impact_optimized':
            baseline_cap = parameters.get('baseline_capacity', 100)
            uplift_pct = parameters.get('percentage', 20)
            # uplift_pct = parameters.get('uplift_percentage', parameters.get('percentage', 20))
            total_patients = len(pandas_df)
            new_capacity = int(baseline_cap * (1 + uplift_pct/100))
            # Calculate basic timing metrics
            current_weeks_needed = np.ceil(total_patients / baseline_cap)
            new_weeks_needed = np.ceil(total_patients / new_capacity)
            weeks_saved = current_weeks_needed - new_weeks_needed

            # Calculate patients moved earlier (same logic as app.py)
            patients_moved_earlier = 0
            for week in range(1, int(current_weeks_needed) + 1):
                current_week_patients = min(baseline_cap, total_patients - (week-1) * baseline_cap)
                new_week_patients = min(new_capacity, total_patients - (week-1) * new_capacity)
                if new_week_patients > current_week_patients:
                    patients_moved_earlier += (new_week_patients - current_week_patients)
            
            # Calculate additional metrics for comprehensive analysis
            additional_weekly_capacity = new_capacity - baseline_cap
            monthly_additional_capacity = additional_weekly_capacity * 4
            efficiency_gain = ((current_weeks_needed - new_weeks_needed) / current_weeks_needed) * 100 if current_weeks_needed > 0 else 0
        
            summary_df, weekly_df, scheduled_df = analyze_capacity_impact_pandas(
                pandas_df, baseline_cap, uplift_pct
            )
            
            # Extract key metrics
            summary_data = summary_df.iloc[0]
            return {
                # 'weeks_saved': summary_data['weeks_saved'],
                # 'patients_moved_earlier': summary_data['patients_moved_earlier'],
                # 'baseline_weeks': summary_data['baseline_weeks'],
                # 'new_weeks': summary_data['new_weeks'],
                # 'capacity_increase': f"{uplift_pct}%"
                'weeks_saved': int(weeks_saved),
                'patients_moved_earlier': int(patients_moved_earlier),
                'baseline_weeks': int(current_weeks_needed),
                'new_weeks': int(new_weeks_needed),
                'baseline_capacity': baseline_cap,
                'new_capacity': new_capacity,
                'additional_weekly_capacity': additional_weekly_capacity,
                'monthly_additional_capacity': monthly_additional_capacity,
                'efficiency_gain': round(efficiency_gain, 1),
                'capacity_increase': f"{uplift_pct}%",
                'total_patients': total_patients
            }
        
        elif function_name == 'identify_high_risk_patients_optimized':
            threshold = parameters.get('percentage', 70) / 100
            high_risk_df = pandas_df[pandas_df['readmit_risk'] > threshold]
            high_risk_count = len(high_risk_df)
            avg_cost_high_risk = high_risk_df['cost_estimate'].mean() if high_risk_count > 0 else 0
            
            return {
                'high_risk_patient_count': high_risk_count,
                'risk_threshold': threshold,
                'avg_cost_high_risk': round(avg_cost_high_risk, 2),
                'potential_intervention_needed': high_risk_count > 0
            }
        
        return {}
    
    def process_natural_language_query(self, user_query: str, df) -> QueryResult:
        """Main method to process natural language queries"""
        
        # Step 1: Classify the query
        query_type = self.classify_query(user_query)
        
        # Step 2: Extract parameters
        parameters = self.extract_parameters(user_query)
        
        # Step 3: Determine if it's a SQL query or function call
        result = QueryResult(query_type=query_type)
        
        if query_type in ['cancellation_reduction', 'capacity_analysis']:
            # Function call approach
            if 'cancel' in user_query.lower():
                result.function_name = 'simulate_cancellation_reduction'
            elif 'capacity' in user_query.lower() or 'increase' in user_query.lower():
                result.function_name = 'analyze_capacity_impact_optimized'
            elif query_type == 'high_risk_analysis':
                result.function_name = 'identify_high_risk_patients_optimized'
            elif query_type == 'schedule_optimization':
                result.function_name = 'optimize_schedule_optimized'
            
        if result.function_name:
            result.parameters = parameters
            result.result_data = self.execute_function_call(result.function_name, parameters, df)
            
        else:
            # Pandas query approach instead of SQL
            pandas_df = df.df if hasattr(df, 'df') else df.raw_pandas_df if hasattr(df, 'raw_pandas_df') else df
            result.result_data = self.execute_pandas_query(query_type, parameters, pandas_df)
        
        return result
    
    def format_response(self, result: QueryResult) -> str:
        """Format the response in a user-friendly way"""
        
        if result.function_name == 'simulate_cancellation_reduction':
            data = result.result_data
            return f"""
            **Cancellation Reduction Analysis**
            
            By reducing cancellations by {data['reduction_percentage']}%:
            \n• **Cost Savings**: £{data['cost_savings']:,.2f}
            \n• **Bed Days Freed**: {data['bed_days_freed']:.0f} days
            \n• **Weekly Capacity Freed**: {data['weekly_capacity_freed']:.1f} procedures
            \n• **Cancellations Prevented**: {data['cancellations_prevented']:.0f} cases
            
            This represents significant operational improvement and cost efficiency.
            """
        
        elif result.function_name == 'analyze_capacity_impact_optimized':
            data = result.result_data
            return f"""
            **Capacity Impact Analysis**
            
            With {data['capacity_increase']} capacity increase:
            \n• **Time Saved**: {data['weeks_saved']} weeks
            \n• **Patients Benefiting**: {data['patients_moved_earlier']} moved to earlier slots
            \n• **Original Timeline**: {data['baseline_weeks']} weeks
            \n• **New Timeline**: {data['new_weeks']} weeks to clear backlog
            \n• **Current Capacity**: {data['baseline_capacity']} patients/week
            \n• **New Capacity**: {data['new_capacity']} patients/week
            \n• **Additional Weekly Capacity**: +{data['additional_weekly_capacity']} patients
            \n• **Additional Monthly Capacity**: +{data['monthly_additional_capacity']} patients
            \n• **Efficiency Gain**: {data['efficiency_gain']}%
            
            This capacity boost significantly improves patient flow.
            """
        
        elif isinstance(result.result_data, pd.DataFrame):
            # Format pandas DataFrame results
            if len(result.result_data) > 0:
                return f"**Query Results**: Found {len(result.result_data)} records\n\n" + \
                       result.result_data.to_string(index=False)
            else:
                return "No results found for your query."
        
        return "Query processed successfully."