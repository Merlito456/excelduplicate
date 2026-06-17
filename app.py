import streamlit as st
import pandas as pd
import json
from io import StringIO, BytesIO
from utils.duplicate_detector import DuplicateDetector, Project
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import openpyxl

# Page configuration
st.set_page_config(
    page_title="Duplicate Project Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .duplicate-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff4b4b;
    }
    .similarity-high {
        color: #28a745;
        font-weight: bold;
    }
    .similarity-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .similarity-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stats-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .cluster-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .similarity-heatmap {
        margin: 1rem 0;
    }
    .header-match {
        background-color: #d4edda;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        color: #155724;
    }
    .header-mismatch {
        background-color: #f8d7da;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'duplicates' not in st.session_state:
        st.session_state.duplicates = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.75
    if 'row_threshold' not in st.session_state:
        st.session_state.row_threshold = 0.7
    if 'weights' not in st.session_state:
        st.session_state.weights = {
            'site_name': 0.25,
            'plaid': 0.30,
            'project': 0.30,
            'keyword_overlap': 0.15
        }
    if 'additional_fields' not in st.session_state:
        st.session_state.additional_fields = []
    if 'excel_sheets' not in st.session_state:
        st.session_state.excel_sheets = []
    if 'selected_sheet' not in st.session_state:
        st.session_state.selected_sheet = None
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    if 'auto_detected_headers' not in st.session_state:
        st.session_state.auto_detected_headers = []
    if 'raw_excel_data' not in st.session_state:
        st.session_state.raw_excel_data = None

def detect_headers_and_sheets(uploaded_file):
    """Auto-detect headers and sheets in Excel file."""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        st.session_state.excel_sheets = sheet_names
        st.session_state.raw_excel_data = excel_file
        
        # Auto-detect which sheet has project data
        best_sheet = None
        best_score = 0
        
        for sheet in sheet_names:
            try:
                df = pd.read_excel(uploaded_file, sheet_name=sheet, nrows=5)
                score = 0
                
                # Check for common project-related column names
                common_columns = ['site', 'project', 'plaid', 'id', 'name', 'description', 'location', 'status']
                for col in df.columns:
                    col_lower = str(col).lower()
                    for common in common_columns:
                        if common in col_lower:
                            score += 1
                
                # Check if first row looks like headers
                first_row = df.iloc[0].astype(str).tolist() if len(df) > 0 else []
                if any('project' in str(cell).lower() for cell in first_row):
                    score += 2
                
                if score > best_score:
                    best_score = score
                    best_sheet = sheet
            except:
                continue
        
        return best_sheet, sheet_names
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None, []

def auto_detect_columns(df):
    """Auto-detect which columns correspond to site_name, plaid, and project."""
    column_mapping = {
        'site_name': None,
        'plaid': None,
        'project': None
    }
    
    # Keywords for each field
    keywords = {
        'site_name': ['site', 'location', 'address', 'city', 'region', 'area', 'site name'],
        'plaid': ['plaid', 'id', 'identifier', 'code', 'reference', 'project id', 'project code'],
        'project': ['project', 'name', 'title', 'description', 'project name', 'work', 'scope']
    }
    
    # Score each column for each field
    scores = {field: {} for field in keywords}
    
    for col in df.columns:
        col_lower = str(col).lower()
        for field, field_keywords in keywords.items():
            score = 0
            for keyword in field_keywords:
                if keyword in col_lower:
                    score += 1
            # Check if column contains relevant data
            if len(df) > 0:
                sample_values = df[col].head(5).astype(str).tolist()
                # Check if values look like site names, IDs, etc.
                if field == 'site_name' and any(len(str(v)) > 3 for v in sample_values):
                    score += 0.5
                elif field == 'plaid' and any(any(c.isdigit() for c in str(v)) for v in sample_values):
                    score += 0.5
                elif field == 'project' and any(len(str(v)) > 5 for v in sample_values):
                    score += 0.5
            
            scores[field][col] = score
    
    # Select best column for each field
    for field in column_mapping:
        if scores[field]:
            best_col = max(scores[field], key=scores[field].get)
            if scores[field][best_col] > 0:
                column_mapping[field] = best_col
    
    return column_mapping

def display_stats(duplicates, total_projects, cluster_analysis=None):
    """Display statistics about duplicates."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_duplicates = sum(len(dup_list) for dup_list in duplicates.values())
    unique_duplicates = len(duplicates)
    
    with col1:
        st.markdown(f"""
            <div class="stats-card">
                <h3>📊 Total Projects</h3>
                <h2>{total_projects}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stats-card">
                <h3>🔍 Duplicate Groups</h3>
                <h2>{unique_duplicates}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stats-card">
                <h3>📋 Duplicate Entries</h3>
                <h2>{total_duplicates}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        dup_percentage = (total_duplicates / total_projects * 100) if total_projects > 0 else 0
        st.markdown(f"""
            <div class="stats-card">
                <h3>📈 Duplicate Rate</h3>
                <h2>{dup_percentage:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        if cluster_analysis:
            avg_cluster = cluster_analysis.get('average_cluster_size', 0)
            st.markdown(f"""
                <div class="stats-card">
                    <h3>🏷️ Avg Cluster Size</h3>
                    <h2>{avg_cluster:.1f}</h2>
                </div>
            """, unsafe_allow_html=True)

def display_duplicate_groups(duplicates, detector):
    """Display duplicate groups with expandable sections."""
    st.subheader("🔍 Duplicate Groups by Row Similarity")
    st.info("Each group shows projects that are similar across the entire row (site_name + plaid + project + additional fields)")
    
    for idx, (main_idx, dup_list) in enumerate(duplicates.items(), 1):
        main_proj = detector.projects[main_idx]
        
        with st.expander(f"📌 Group {idx} - Main: {main_proj.project} ({len(dup_list)} duplicates)", expanded=False):
            # Main project card with full row details
            st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <strong>🎯 Main Project (Row {main_idx + 1}):</strong><br>
                    <strong>Site:</strong> {main_proj.site_name}<br>
                    <strong>Plaid:</strong> {main_proj.plaid}<br>
                    <strong>Project:</strong> {main_proj.project}
                    {f'<br><strong>Additional Fields:</strong> {main_proj.additional_fields}' if main_proj.additional_fields else ''}
                </div>
            """, unsafe_allow_html=True)
            
            # Duplicate projects
            for dup_idx, score, scores in dup_list:
                dup_proj = detector.projects[dup_idx]
                
                # Determine color based on similarity
                if score >= 0.8:
                    color_class = "similarity-high"
                elif score >= 0.6:
                    color_class = "similarity-medium"
                else:
                    color_class = "similarity-low"
                
                st.markdown(f"""
                    <div class="duplicate-card">
                        <strong>📎 Duplicate (Row {dup_idx + 1}) - Similarity: <span class="{color_class}">{score:.1%}</span></strong><br>
                        <strong>Site:</strong> {dup_proj.site_name}<br>
                        <strong>Plaid:</strong> {dup_proj.plaid}<br>
                        <strong>Project:</strong> {dup_proj.project}
                        {f'<br><strong>Additional Fields:</strong> {dup_proj.additional_fields}' if dup_proj.additional_fields else ''}
                        <details>
                            <summary style="color: #666; cursor: pointer; margin-top: 0.5rem;">📊 View Detailed Match Analysis</summary>
                            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                                <strong>Field-wise Similarity:</strong><br>
                                Site Name: {scores['site_name']:.1%}<br>
                                Plaid: {scores['plaid']:.1%}<br>
                                Project: {scores['project']:.1%}<br>
                                Keyword Overlap: {scores.get('keyword_overlap', 0):.1%}<br>
                                Text Similarity: {scores.get('text_similarity', 0):.1%}<br>
                                Match Significance: {scores.get('match_significance', 0):.1%}
                            </div>
                        </details>
                    </div>
                """, unsafe_allow_html=True)

def display_visualizations(duplicates, detector):
    """Display visualization charts."""
    st.subheader("📊 Similarity Visualizations")
    
    col1, col2 = st.columns(2)
    
    # Similarity distribution
    with col1:
        similarities = []
        for dup_list in duplicates.values():
            for _, score, _ in dup_list:
                similarities.append(score)
        
        if similarities:
            fig = go.Figure(data=[go.Histogram(
                x=similarities,
                nbinsx=20,
                marker_color='#667eea',
                name='Similarity Distribution'
            )])
            fig.update_layout(
                title='Duplicate Similarity Distribution',
                xaxis_title='Similarity Score',
                yaxis_title='Count',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Field match rates
    with col2:
        field_scores = {
            'Site Name': [],
            'Plaid': [],
            'Project': [],
            'Keyword': [],
            'Text': [],
            'Overall': []
        }
        
        for dup_list in duplicates.values():
            for _, _, scores in dup_list:
                field_scores['Site Name'].append(scores.get('site_name', 0))
                field_scores['Plaid'].append(scores.get('plaid', 0))
                field_scores['Project'].append(scores.get('project', 0))
                field_scores['Keyword'].append(scores.get('keyword_overlap', 0))
                field_scores['Text'].append(scores.get('text_similarity', 0))
                field_scores['Overall'].append(scores.get('overall', 0))
        
        if any(field_scores.values()):
            avg_scores = {k: sum(v)/len(v) if v else 0 for k, v in field_scores.items()}
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(avg_scores.keys()),
                    y=list(avg_scores.values()),
                    marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a'],
                    text=[f'{v:.1%}' for v in avg_scores.values()],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title='Average Match Rate by Field',
                yaxis_title='Average Match Rate',
                yaxis_tickformat='.0%',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Similarity Matrix Heatmap (for a subset of projects)
    if len(detector.projects) <= 20:  # Only show for smaller datasets
        st.subheader("🌐 Row Similarity Heatmap")
        st.info("Shows similarity between all project rows. Darker colors indicate higher similarity.")
        
        similarity_matrix = detector.get_similarity_matrix()
        
        fig = px.imshow(
            similarity_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='Blues',
            title="Project Similarity Matrix"
        )
        fig.update_layout(
            height=500,
            xaxis_title="Projects",
            yaxis_title="Projects"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    if len(duplicates) > 0:
        st.subheader("🏷️ Cluster Analysis")
        cluster_analysis = detector.get_cluster_analysis(duplicates)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clusters", cluster_analysis['total_clusters'])
        with col2:
            st.metric("Avg Cluster Size", f"{cluster_analysis['average_cluster_size']:.1f}")
        with col3:
            st.metric("Max Cluster Size", cluster_analysis['max_cluster_size'])
        
        # Cluster size distribution
        cluster_sizes = [c['size'] for c in cluster_analysis['clusters']]
        if cluster_sizes:
            fig = go.Figure(data=[go.Histogram(
                x=cluster_sizes,
                nbinsx=10,
                marker_color='#764ba2',
                name='Cluster Size Distribution'
            )])
            fig.update_layout(
                title='Duplicate Cluster Sizes',
                xaxis_title='Cluster Size',
                yaxis_title='Count',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🔍 Duplicate Project Detector</h1>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="color: #666; font-size: 1.1rem;">
                Upload your Excel or CSV file to find duplicates. The tool automatically detects headers,
                allows sheet selection, and intelligently matches similar projects across rows.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Similarity thresholds
        st.subheader("Thresholds")
        threshold = st.slider(
            "Duplicate Detection Threshold",
            min_value=0.5,
            max_value=0.95,
            value=st.session_state.threshold,
            step=0.05,
            help="Higher values mean stricter matching for exact duplicates."
        )
        st.session_state.threshold = threshold
        
        row_threshold = st.slider(
            "Row Similarity Threshold",
            min_value=0.5,
            max_value=0.95,
            value=st.session_state.row_threshold,
            step=0.05,
            help="Minimum overall similarity for a row to be considered a match."
        )
        st.session_state.row_threshold = row_threshold
        
        # Field weights
        st.subheader("Field Weights")
        st.info("Adjust the importance of each field in duplicate detection")
        
        weights = {
            'site_name': st.slider("Site Name Weight", 0.0, 1.0, st.session_state.weights.get('site_name', 0.25), 0.05),
            'plaid': st.slider("Plaid ID Weight", 0.0, 1.0, st.session_state.weights.get('plaid', 0.30), 0.05),
            'project': st.slider("Project Name Weight", 0.0, 1.0, st.session_state.weights.get('project', 0.30), 0.05),
            'keyword_overlap': st.slider("Keyword Overlap Weight", 0.0, 1.0, st.session_state.weights.get('keyword_overlap', 0.15), 0.05)
        }
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        st.session_state.weights = weights
        
        # Sample data
        st.divider()
        st.subheader("📥 Sample Data")
        if st.button("📂 Load Sample Data", use_container_width=True):
            sample_data = [
                {"site_name": "Site A", "plaid": "PLD001", "project": "2025 FTTH Deployment", "location": "City A", "status": "Active"},
                {"site_name": "Site A", "plaid": "PLD001", "project": "FTTH Project", "location": "City A", "status": "Active"},
                {"site_name": "Site B", "plaid": "PLD002", "project": "Fiber Optic Installation", "location": "City B", "status": "Planning"},
                {"site_name": "Site B", "plaid": "PLD003", "project": "Fiber Optic Network", "location": "City B", "status": "Planning"},
                {"site_name": "Site C", "plaid": "PLD004", "project": "5G Tower Setup", "location": "City C", "status": "Active"},
                {"site_name": "Site C", "plaid": "PLD004", "project": "5G Tower Installation", "location": "City C", "status": "Active"},
                {"site_name": "Site D", "plaid": "PLD005", "project": "Network Upgrade", "location": "City D", "status": "Planning"},
                {"site_name": "Site E", "plaid": "PLD006", "project": "Different Project", "location": "City E", "status": "Active"},
                {"site_name": "Site F", "plaid": "PLD007", "project": "2024 Fiber Network", "location": "City F", "status": "Planning"},
                {"site_name": "Site F", "plaid": "PLD007", "project": "Fiber Network Expansion", "location": "City F", "status": "Planning"},
            ]
            
            detector = DuplicateDetector(
                similarity_threshold=threshold,
                row_similarity_threshold=row_threshold,
                weights=weights
            )
            detector.load_from_list(sample_data, additional_fields=['location', 'status'])
            duplicates = detector.find_duplicates()
            
            st.session_state.detector = detector
            st.session_state.duplicates = duplicates
            st.session_state.processed_data = pd.DataFrame(sample_data)
            
            st.success("✅ Sample data loaded successfully!")
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "🔍 Duplicates", "📊 Analytics", "📥 Export"])
    
    # Tab 1: Upload & Process
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload your project data (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Supports CSV, Excel (.xlsx, .xls) files. For Excel, all sheets will be detected."
        )
        
        if uploaded_file is not None:
            try:
                # Check if it's an Excel file
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # Auto-detect sheets and headers
                    best_sheet, sheet_names = detect_headers_and_sheets(uploaded_file)
                    
                    if sheet_names:
                        st.success(f"✅ Detected {len(sheet_names)} sheets in the Excel file")
                        
                        # Sheet selection
                        selected_sheet = st.selectbox(
                            "Select sheet to analyze:",
                            options=sheet_names,
                            index=sheet_names.index(best_sheet) if best_sheet in sheet_names else 0,
                            help="Choose which sheet contains your project data"
                        )
                        st.session_state.selected_sheet = selected_sheet
                        
                        # Load the selected sheet
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                        
                        # Auto-detect headers
                        st.subheader("🔍 Auto-Detected Columns")
                        
                        # Check if first row looks like headers
                        first_row = df.iloc[0].astype(str).tolist() if len(df) > 0 else []
                        header_indicators = ['site', 'project', 'plaid', 'id', 'name', 'location', 'status']
                        has_headers = any(any(ind in str(cell).lower() for ind in header_indicators) for cell in first_row)
                        
                        if has_headers and len(df) > 1:
                            st.info("📌 Headers detected in the first row. Using them as column names.")
                            # Use first row as headers
                            df.columns = df.iloc[0]
                            df = df.iloc[1:].reset_index(drop=True)
                        
                        # Auto-detect column mappings
                        column_mapping = auto_detect_columns(df)
                        
                        # Display detected mappings
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**🏷️ Site Name Column**")
                            if column_mapping['site_name']:
                                st.markdown(f'<span class="header-match">✅ {column_mapping["site_name"]}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="header-mismatch">❌ Not detected</span>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**🆔 Plaid ID Column**")
                            if column_mapping['plaid']:
                                st.markdown(f'<span class="header-match">✅ {column_mapping["plaid"]}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="header-mismatch">❌ Not detected</span>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("**📋 Project Column**")
                            if column_mapping['project']:
                                st.markdown(f'<span class="header-match">✅ {column_mapping["project"]}</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="header-mismatch">❌ Not detected</span>', unsafe_allow_html=True)
                        
                        # Manual column mapping option
                        with st.expander("✏️ Manual Column Mapping (if auto-detection is incorrect)"):
                            st.info("Manually select which columns correspond to each field")
                            all_columns = df.columns.tolist()
                            
                            manual_site = st.selectbox("Site Name Column", [''] + all_columns, 
                                                      index=0 if not column_mapping['site_name'] else all_columns.index(column_mapping['site_name']) + 1)
                            manual_plaid = st.selectbox("Plaid ID Column", [''] + all_columns,
                                                      index=0 if not column_mapping['plaid'] else all_columns.index(column_mapping['plaid']) + 1)
                            manual_project = st.selectbox("Project Column", [''] + all_columns,
                                                        index=0 if not column_mapping['project'] else all_columns.index(column_mapping['project']) + 1)
                            
                            # Additional fields
                            additional_fields = st.multiselect(
                                "Additional Fields to Include",
                                options=[col for col in all_columns if col not in [manual_site, manual_plaid, manual_project]],
                                default=st.session_state.additional_fields
                            )
                            
                            if st.button("Apply Manual Mapping"):
                                if manual_site and manual_plaid and manual_project:
                                    column_mapping = {
                                        'site_name': manual_site,
                                        'plaid': manual_plaid,
                                        'project': manual_project
                                    }
                                    st.session_state.column_mapping = column_mapping
                                    st.session_state.additional_fields = additional_fields
                                    st.success("✅ Mapping applied successfully!")
                                    st.rerun()
                        
                        # Use auto-detected or manual mapping
                        if 'column_mapping' in st.session_state and st.session_state.column_mapping:
                            col_mapping = st.session_state.column_mapping
                        else:
                            col_mapping = column_mapping
                        
                        # Display data preview
                        st.subheader("📋 Data Preview")
                        preview_cols = [col for col in df.columns if col in col_mapping.values() or col in st.session_state.additional_fields]
                        if preview_cols:
                            st.dataframe(df[preview_cols].head(10), use_container_width=True)
                        else:
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Process button
                        if col_mapping['site_name'] and col_mapping['plaid'] and col_mapping['project']:
                            if st.button("🚀 Detect Duplicates", use_container_width=True):
                                with st.spinner("Processing data..."):
                                    # Prepare data
                                    data = []
                                    for _, row in df.iterrows():
                                        entry = {
                                            'site_name': str(row.get(col_mapping['site_name'], '')),
                                            'plaid': str(row.get(col_mapping['plaid'], '')),
                                            'project': str(row.get(col_mapping['project'], ''))
                                        }
                                        # Add additional fields
                                        for field in st.session_state.additional_fields:
                                            if field in row:
                                                entry[field] = str(row[field])
                                        data.append(entry)
                                    
                                    detector = DuplicateDetector(
                                        similarity_threshold=threshold,
                                        row_similarity_threshold=row_threshold,
                                        weights=weights
                                    )
                                    detector.load_from_list(data, additional_fields=st.session_state.additional_fields)
                                    duplicates = detector.find_duplicates()
                                    
                                    st.session_state.detector = detector
                                    st.session_state.duplicates = duplicates
                                    st.session_state.processed_data = pd.DataFrame(data)
                                    
                                    st.success(f"✅ Analysis complete! Found {len(duplicates)} duplicate groups.")
                                    st.balloons()
                        else:
                            st.warning("⚠️ Please ensure all required columns are mapped before processing.")
                    else:
                        st.error("Could not read sheets from the Excel file.")
                
                else:  # CSV file
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ CSV file loaded successfully! Found {len(df)} records.")
                    
                    # Auto-detect columns for CSV
                    column_mapping = auto_detect_columns(df)
                    
                    # Display data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Process button for CSV
                    if st.button("🚀 Detect Duplicates", use_container_width=True):
                        with st.spinner("Processing data..."):
                            detector = DuplicateDetector(
                                similarity_threshold=threshold,
                                row_similarity_threshold=row_threshold,
                                weights=weights
                            )
                            detector.load_from_dataframe(df)
                            duplicates = detector.find_duplicates()
                            
                            st.session_state.detector = detector
                            st.session_state.duplicates = duplicates
                            st.session_state.processed_data = df
                            
                            st.success(f"✅ Analysis complete! Found {len(duplicates)} duplicate groups.")
                            st.balloons()
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please make sure your file is properly formatted.")
        
        # Manual data entry option
        with st.expander("✏️ Or enter data manually"):
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_site = st.text_input("Site Name")
            with col2:
                manual_plaid = st.text_input("Plaid ID")
            with col3:
                manual_project = st.text_input("Project Name")
            
            # Additional fields for manual entry
            additional_values = {}
            if st.session_state.additional_fields:
                for field in st.session_state.additional_fields:
                    additional_values[field] = st.text_input(f"{field}")
            
            if st.button("➕ Add Entry"):
                if manual_site and manual_plaid and manual_project:
                    entry = {
                        'site_name': manual_site,
                        'plaid': manual_plaid,
                        'project': manual_project
                    }
                    if st.session_state.additional_fields:
                        for field in st.session_state.additional_fields:
                            entry[field] = additional_values.get(field, '')
                    
                    if 'manual_data' not in st.session_state:
                        st.session_state.manual_data = []
                    st.session_state.manual_data.append(entry)
                    st.success("✅ Entry added!")
                    st.rerun()
            
            if 'manual_data' in st.session_state and st.session_state.manual_data:
                st.write("Current entries:")
                st.dataframe(pd.DataFrame(st.session_state.manual_data))
                
                if st.button("🚀 Process Manual Data", use_container_width=True):
                    with st.spinner("Processing data..."):
                        detector = DuplicateDetector(
                            similarity_threshold=threshold,
                            row_similarity_threshold=row_threshold,
                            weights=weights
                        )
                        detector.load_from_list(st.session_state.manual_data, additional_fields=st.session_state.additional_fields)
                        duplicates = detector.find_duplicates()
                        
                        st.session_state.detector = detector
                        st.session_state.duplicates = duplicates
                        st.session_state.processed_data = pd.DataFrame(st.session_state.manual_data)
                        
                        st.success(f"✅ Analysis complete! Found {len(duplicates)} duplicate groups.")
                        st.balloons()
    
    # Tab 2: Duplicates
    with tab2:
        if st.session_state.detector and st.session_state.duplicates:
            # Stats
            cluster_analysis = st.session_state.detector.get_cluster_analysis(st.session_state.duplicates)
            display_stats(st.session_state.duplicates, len(st.session_state.detector.projects), cluster_analysis)
            st.divider()
            
            # Display duplicates
            display_duplicate_groups(st.session_state.duplicates, st.session_state.detector)
        else:
            st.info("📊 No data processed yet. Please upload data or load sample data from the sidebar.")
    
    # Tab 3: Analytics
    with tab3:
        if st.session_state.detector and st.session_state.duplicates:
            display_visualizations(st.session_state.duplicates, st.session_state.detector)
            
            # Additional metrics
            st.subheader("📈 Detailed Metrics")
            
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                avg_similarity = 0
                count = 0
                for dup_list in st.session_state.duplicates.values():
                    for _, score, _ in dup_list:
                        avg_similarity += score
                        count += 1
                if count > 0:
                    st.metric("Average Similarity Score", f"{avg_similarity/count:.1%}")
            
            with metrics_cols[1]:
                unique_projects = len(set(p.project for p in st.session_state.detector.projects))
                st.metric("Unique Projects", unique_projects)
            
            with metrics_cols[2]:
                duplicate_projects = len(st.session_state.duplicates)
                st.metric("Duplicate Groups", duplicate_projects)
            
            with metrics_cols[3]:
                total_dupes = sum(len(dup_list) for dup_list in st.session_state.duplicates.values())
                st.metric("Total Duplicates", total_dupes)
            
            # Show all data with similarity scores
            st.subheader("📊 All Projects with Duplicate Status")
            data_with_scores = []
            for i, project in enumerate(st.session_state.detector.projects):
                is_duplicate = any(i in dup_list for dup_list in st.session_state.duplicates.values())
                data_with_scores.append({
                    'Index': i + 1,
                    'Site': project.site_name,
                    'Plaid': project.plaid,
                    'Project': project.project,
                    'Is Duplicate': '⚠️ Yes' if is_duplicate else '✅ No'
                })
            
            st.dataframe(
                pd.DataFrame(data_with_scores),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("📊 No data processed yet. Please upload data or load sample data.")
    
    # Tab 4: Export
    with tab4:
        if st.session_state.detector and st.session_state.duplicates:
            st.subheader("📥 Export Results")
            
            export_options = st.radio(
                "Select export format:",
                ['JSON', 'CSV', 'Excel']
            )
            
            if st.button("📥 Download Report", use_container_width=True):
                # Prepare export data
                export_data = {
                    "analysis_date": datetime.now().isoformat(),
                    "settings": {
                        "similarity_threshold": st.session_state.threshold,
                        "row_similarity_threshold": st.session_state.row_threshold,
                        "weights": st.session_state.weights
                    },
                    "summary": {
                        "total_projects": len(st.session_state.detector.projects),
                        "duplicate_groups": len(st.session_state.duplicates),
                        "total_duplicates": sum(len(dup_list) for dup_list in st.session_state.duplicates.values())
                    },
                    "duplicate_groups": []
                }
                
                for main_idx, dup_list in st.session_state.duplicates.items():
                    main_proj = st.session_state.detector.projects[main_idx]
                    group = {
                        "main_project": {
                            "site_name": main_proj.site_name,
                            "plaid": main_proj.plaid,
                            "project": main_proj.project
                        },
                        "duplicates": []
                    }
                    
                    for dup_idx, score, scores in dup_list:
                        dup_proj = st.session_state.detector.projects[dup_idx]
                        group["duplicates"].append({
                            "site_name": dup_proj.site_name,
                            "plaid": dup_proj.plaid,
                            "project": dup_proj.project,
                            "similarity_score": score,
                            "details": {
                                "site_name_match": scores.get('site_name', 0),
                                "plaid_match": scores.get('plaid', 0),
                                "project_match": scores.get('project', 0),
                                "keyword_overlap": scores.get('keyword_overlap', 0),
                                "text_similarity": scores.get('text_similarity', 0)
                            }
                        })
                    
                    export_data["duplicate_groups"].append(group)
                
                # Export based on selected format
                if export_options == 'JSON':
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"duplicates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                elif export_options == 'CSV':
                    # Flatten the data for CSV
                    rows = []
                    for group in export_data["duplicate_groups"]:
                        main = group["main_project"]
                        for dup in group["duplicates"]:
                            rows.append({
                                "main_site": main["site_name"],
                                "main_plaid": main["plaid"],
                                "main_project": main["project"],
                                "dup_site": dup["site_name"],
                                "dup_plaid": dup["plaid"],
                                "dup_project": dup["project"],
                                "similarity": dup["similarity_score"]
                            })
                    df = pd.DataFrame(rows)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"duplicates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:  # Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Summary sheet
                        summary_df = pd.DataFrame([{
                            "Total Projects": export_data["summary"]["total_projects"],
                            "Duplicate Groups": export_data["summary"]["duplicate_groups"],
                            "Total Duplicates": export_data["summary"]["total_duplicates"],
                            "Analysis Date": export_data["analysis_date"]
                        }])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        
                        # Details sheet
                        rows = []
                        for group in export_data["duplicate_groups"]:
                            main = group["main_project"]
                            for dup in group["duplicates"]:
                                rows.append({
                                    "Main Site": main["site_name"],
                                    "Main Plaid": main["plaid"],
                                    "Main Project": main["project"],
                                    "Duplicate Site": dup["site_name"],
                                    "Duplicate Plaid": dup["plaid"],
                                    "Duplicate Project": dup["project"],
                                    "Similarity Score": f"{dup['similarity_score']:.1%}",
                                    "Site Match": f"{dup['details']['site_name_match']:.1%}",
                                    "Plaid Match": f"{dup['details']['plaid_match']:.1%}",
                                    "Project Match": f"{dup['details']['project_match']:.1%}",
                                    "Keyword Overlap": f"{dup['details']['keyword_overlap']:.1%}"
                                })
                        df = pd.DataFrame(rows)
                        df.to_excel(writer, sheet_name='Duplicates', index=False)
                    
                    output.seek(0)
                    st.download_button(
                        label="Download Excel",
                        data=output,
                        file_name=f"duplicates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Export configuration
            st.subheader("⚙️ Export Configuration")
            if st.button("📋 Copy Analysis Summary"):
                summary = f"""
                Duplicate Project Analysis Summary
                ==================================
                Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Settings:
                - Similarity Threshold: {st.session_state.threshold}
                - Row Similarity Threshold: {st.session_state.row_threshold}
                - Field Weights: {st.session_state.weights}
                
                Results:
                - Total Projects: {len(st.session_state.detector.projects)}
                - Duplicate Groups: {len(st.session_state.duplicates)}
                - Total Duplicates: {sum(len(dup_list) for dup_list in st.session_state.duplicates.values())}
                - Duplicate Rate: {sum(len(dup_list) for dup_list in st.session_state.duplicates.values()) / len(st.session_state.detector.projects) * 100:.1f}%
                """
                st.code(summary, language="text")
        else:
            st.info("📥 No data to export. Please process data first.")

if __name__ == "__main__":
    main()