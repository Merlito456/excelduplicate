import streamlit as st
import pandas as pd
from difflib import SequenceMatcher
import plotly.express as px
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Duplicate Detector",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Duplicate Project Detector")
st.markdown("Upload your data to find duplicate projects")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'duplicates' not in st.session_state:
    st.session_state.duplicates = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Higher = stricter matching"
    )
    
    st.divider()
    
    # Sample data button
    if st.button("📂 Load Sample Data", use_container_width=True):
        sample_data = [
            {"site_name": "Site A", "plaid": "PLD001", "project": "2025 FTTH Deployment"},
            {"site_name": "Site A", "plaid": "PLD001", "project": "FTTH Project"},
            {"site_name": "Site B", "plaid": "PLD002", "project": "Fiber Optic Installation"},
            {"site_name": "Site B", "plaid": "PLD003", "project": "Fiber Optic Network"},
            {"site_name": "Site C", "plaid": "PLD004", "project": "5G Tower Setup"},
            {"site_name": "Site C", "plaid": "PLD004", "project": "5G Tower Installation"},
            {"site_name": "Site D", "plaid": "PLD005", "project": "Network Upgrade"},
            {"site_name": "Site E", "plaid": "PLD006", "project": "Different Project"},
        ]
        st.session_state.data = pd.DataFrame(sample_data)
        st.session_state.duplicates = None
        st.success("✅ Sample data loaded!")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "🔍 Find Duplicates", "📊 Results"])

# Tab 1: Upload
with tab1:
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="File must have columns: site_name, plaid, project"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check required columns
            required = ['site_name', 'plaid', 'project']
            if all(col in df.columns for col in required):
                st.session_state.data = df
                st.success(f"✅ Loaded {len(df)} records")
                st.dataframe(df.head())
            else:
                st.error(f"❌ Missing columns. Need: {', '.join(required)}")
                st.write("Found:", df.columns.tolist())
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tab 2: Find Duplicates
with tab2:
    if st.session_state.data is not None:
        st.subheader("Current Data")
        st.dataframe(st.session_state.data, use_container_width=True)
        
        if st.button("🚀 Find Duplicates", use_container_width=True):
            with st.spinner("Analyzing..."):
                data = st.session_state.data
                duplicates = find_duplicates(data, threshold)
                st.session_state.duplicates = duplicates
                st.success(f"✅ Found {len(duplicates)} duplicate groups!")
                st.balloons()
    else:
        st.info("📊 Please upload data or load sample data first")

# Tab 3: Results
with tab3:
    if st.session_state.duplicates and len(st.session_state.duplicates) > 0:
        display_results(st.session_state.duplicates, st.session_state.data)
    elif st.session_state.duplicates is not None:
        st.success("🎉 No duplicates found!")
    else:
        st.info("📊 Run duplicate detection first")

# Helper functions
def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def find_duplicates(df, threshold):
    """Find duplicate entries"""
    duplicates = {}
    seen = set()
    
    for i in range(len(df)):
        if i in seen:
            continue
            
        current_group = []
        for j in range(i + 1, len(df)):
            # Calculate similarity scores
            site_sim = calculate_similarity(
                str(df.iloc[i]['site_name']), 
                str(df.iloc[j]['site_name'])
            )
            plaid_match = str(df.iloc[i]['plaid']) == str(df.iloc[j]['plaid'])
            project_sim = calculate_similarity(
                str(df.iloc[i]['project']), 
                str(df.iloc[j]['project'])
            )
            
            # Overall similarity (weighted)
            overall = (site_sim * 0.3 + plaid_match * 0.4 + project_sim * 0.3)
            
            # Check if similar
            if overall >= threshold:
                current_group.append({
                    'index': j,
                    'site_name': df.iloc[j]['site_name'],
                    'plaid': df.iloc[j]['plaid'],
                    'project': df.iloc[j]['project'],
                    'similarity': overall,
                    'site_match': site_sim,
                    'plaid_match': plaid_match,
                    'project_match': project_sim
                })
                seen.add(j)
        
        if current_group:
            duplicates[i] = {
                'main': {
                    'site_name': df.iloc[i]['site_name'],
                    'plaid': df.iloc[i]['plaid'],
                    'project': df.iloc[i]['project']
                },
                'duplicates': current_group
            }
            seen.add(i)
    
    return duplicates

def display_results(duplicates, df):
    """Display duplicate results"""
    st.subheader("📋 Duplicate Groups")
    
    # Stats
    total_dupes = sum(len(v['duplicates']) for v in duplicates.values())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Projects", len(df))
    with col2:
        st.metric("Duplicate Groups", len(duplicates))
    with col3:
        st.metric("Duplicate Entries", total_dupes)
    
    st.divider()
    
    # Display each group
    for idx, (main_idx, group) in enumerate(duplicates.items(), 1):
        with st.expander(f"📌 Group {idx}: {group['main']['project']} ({len(group['duplicates'])} duplicates)", expanded=True):
            # Main project
            st.markdown(f"""
                **🎯 Main Project:**
                - Site: {group['main']['site_name']}
                - Plaid: {group['main']['plaid']}
                - Project: {group['main']['project']}
            """)
            
            st.markdown("---")
            
            # Duplicates
            for dup in group['duplicates']:
                color = "🟢" if dup['similarity'] >= 0.8 else "🟡" if dup['similarity'] >= 0.6 else "🔴"
                st.markdown(f"""
                    **{color} Duplicate (Similarity: {dup['similarity']:.1%})**
                    - Site: {dup['site_name']}
                    - Plaid: {dup['plaid']}
                    - Project: {dup['project']}
                    - Details:
                        - Site Match: {dup['site_match']:.1%}
                        - Plaid Match: {'✅ Yes' if dup['plaid_match'] else '❌ No'}
                        - Project Match: {dup['project_match']:.1%}
                """)
                st.markdown("---")
    
    # Export
    st.subheader("📥 Export Results")
    if st.button("📥 Download Report"):
        export_data = []
        for group in duplicates.values():
            main = group['main']
            for dup in group['duplicates']:
                export_data.append({
                    'Main Site': main['site_name'],
                    'Main Plaid': main['plaid'],
                    'Main Project': main['project'],
                    'Duplicate Site': dup['site_name'],
                    'Duplicate Plaid': dup['plaid'],
                    'Duplicate Project': dup['project'],
                    'Similarity': f"{dup['similarity']:.1%}"
                })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="duplicates_report.csv",
            mime="text/csv"
        )