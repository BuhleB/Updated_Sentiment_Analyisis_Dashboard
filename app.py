import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import json
import csv
from datetime import datetime
import os
import sys

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sentiment_analyzer import analyze_sentiment_textblob, analyze_sentiment_huggingface, extract_keywords, batch_analyze_sentiment, get_sentiment_explanation, generate_accuracy_report
from visualizations import (
    create_sentiment_distribution_chart,
    create_sentiment_over_time_chart,
    create_sentiment_by_source_chart,
    create_confidence_distribution_chart,
    create_polarity_vs_subjectivity_scatter,
    create_wordcloud,
    create_keyword_frequency_chart,
    create_sentiment_metrics_summary,
    display_comparative_analysis
)
from export_utils import export_to_csv, export_to_json, export_to_excel, create_pdf_report, batch_process_files
from comparative_analysis import analyze_with_both_models, display_comparative_analysis_tab, create_agreement_matrix

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()
if 'comparative_results' not in st.session_state:
    st.session_state.comparative_results = pd.DataFrame()

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔧 Analysis Options</div>', unsafe_allow_html=True)
        
        # Navigation
        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            ["Single Text Analysis", "Batch Text Analysis", "File Upload Analysis", "Compare Analyzers"]
        )

        
        st.session_state.selected_analyzer = st.selectbox(
            "Choose Sentiment Analyzer",
            ["TextBlob", "HuggingFace"]
        )
        st.markdown("---")
        
        # Filters (will be populated based on data)
        if not st.session_state.analysis_results.empty:
            st.markdown('<div class="sidebar-header">🔍 Filters</div>', unsafe_allow_html=True)
            
            # Sentiment filter
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=st.session_state.analysis_results['sentiment'].unique(),
                default=st.session_state.analysis_results['sentiment'].unique()
            )
            
            # Source filter
            if 'source' in st.session_state.analysis_results.columns:
                source_filter = st.multiselect(
                    "Filter by Source",
                    options=st.session_state.analysis_results['source'].unique(),
                    default=st.session_state.analysis_results['source'].unique()
                )
            else:
                source_filter = None
    
    # Main content area
    if analysis_mode == "Single Text Analysis":
        single_text_analysis()
    elif analysis_mode == "Batch Text Analysis":
        batch_text_analysis()
    elif analysis_mode == "File Upload Analysis":
        file_upload_analysis()
    elif analysis_mode == "Compare Analyzers":
        compare_analyzers_analysis()

    
    # Display results if available
    display_analysis_results()

def single_text_analysis():
    st.subheader("🔍 Single Text Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        # Source and date inputs
        col_source, col_date = st.columns(2)
        with col_source:
            source = st.text_input("Source (optional)", placeholder="e.g., Twitter, Survey")
        with col_date:
            date = st.date_input("Date (optional)", value=datetime.now())
    
    with col2:
        st.markdown("### Analysis Options")
        num_keywords = st.slider("Number of keywords to extract", 3, 10, 5)
        
        if st.button("🚀 Analyze Text", type="primary"):
            if text_input.strip():
                # Perform analysis
                if st.session_state.selected_analyzer == 'HuggingFace':
                    sentiment, confidence, polarity, subjectivity = analyze_sentiment_huggingface(text_input)
                else:
                    sentiment, confidence, polarity, subjectivity = analyze_sentiment_textblob(text_input)

                keywords = extract_keywords(text_input, num_keywords)
                explanation = get_sentiment_explanation(text_input, sentiment, polarity)
                
                # Create result dataframe
                result = pd.DataFrame([{
                    'text': text_input,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'keywords': keywords,
                    'source': source if source else 'Manual Input',
                    'date': str(date),
                    'explanation': explanation
                }])
                
                # Update session state
                st.session_state.analysis_results = pd.concat([st.session_state.analysis_results, result], ignore_index=True)
                
                st.success("✅ Analysis completed!")
                st.rerun()
            else:
                st.error("Please enter some text to analyze.")

def batch_text_analysis():
    st.subheader("📝 Batch Text Analysis")
    
    # Text area for multiple texts
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Text 1\nText 2\nText 3..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        default_source = st.text_input("Default source for all texts", placeholder="e.g., Survey Responses")
    with col2:
        default_date = st.date_input("Default date for all texts", value=datetime.now())
    
    if st.button("🚀 Analyze Batch", type="primary"):
        if batch_text.strip():
            # Split texts by lines
            texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
            
            # Prepare data for batch analysis
            text_data = []
            for i, text in enumerate(texts):
                text_data.append({
                    'text': text,
                    'source': default_source if default_source else f'Batch Input {i+1}',
                    'date': str(default_date)
                })
            
            # Perform batch analysis
                with st.spinner("Analyzing texts..."):
                    results = batch_analyze_sentiment(text_data, analyzer_type=st.session_state.selected_analyzer)
                    
                    # Add explanations
                    results["explanation"] = results.apply(
                        lambda row: get_sentiment_explanation(row["text"], row["sentiment"], row["polarity"], analyzer_type=st.session_state.selected_analyzer),
                        axis=1
                    )

                
                # Update session state
                st.session_state.analysis_results = pd.concat([st.session_state.analysis_results, results], ignore_index=True)
            
            st.success(f"✅ Analyzed {len(texts)} texts successfully!")
            st.rerun()
        else:
            st.error("Please enter some texts to analyze.")

def file_upload_analysis():
    st.subheader("📁 File Upload Analysis")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="CSV files should have a 'text' column. TXT files will be processed line by line. Optional columns for CSV: 'source', 'date'"
    )
    
    if uploaded_files:
        st.write(f"### {len(uploaded_files)} file(s) uploaded")
        
        # Show file details
        for uploaded_file in uploaded_files:
            st.write(f"- **{uploaded_file.name}** ({uploaded_file.size} bytes)")
        
        # Process files button
        if st.button("🚀 Analyze All Files", type="primary"):
            try:
                # Use batch processing utility
                combined_df = batch_process_files(uploaded_files)
                
                if combined_df.empty:
                    st.error("No valid data found in uploaded files.")
                    return
                
                st.write("### Combined File Preview")
                st.dataframe(combined_df.head())
                
                # Prepare data for analysis
                text_data = []
                for _, row in combined_df.iterrows():
                    text_data.append({
                        'text': str(row['text']),
                        'source': str(row.get('source', row.get('file_source', 'File Upload'))),
                        'date': str(row.get('date', datetime.now().date()))
                    })
                
                # Perform batch analysis
                with st.spinner(f"Analyzing {len(text_data)} texts from {len(uploaded_files)} file(s)..."):
                    results = batch_analyze_sentiment(text_data, analyzer_type=st.session_state.selected_analyzer)
                    
                    # Add explanations
                    results["explanation"] = results.apply(
                        lambda row: get_sentiment_explanation(row["text"], row["sentiment"], row["polarity"], analyzer_type=st.session_state.selected_analyzer),
                        axis=1
                    )

                    
                    # Update session state
                    st.session_state.analysis_results = pd.concat([st.session_state.analysis_results, results], ignore_index=True)
                
                st.success(f"✅ Analyzed {len(text_data)} texts from {len(uploaded_files)} file(s) successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
    
    # Single file upload (legacy support)
    st.markdown("---")
    st.subheader("📄 Single CSV File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a single CSV file",
        type=['csv'],
        help="CSV file should have a 'text' column. Optional columns: 'source', 'date'",
        key="single_file_upload"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.write("### File Preview")
            st.dataframe(df.head())
            
            # Check if 'text' column exists
            if 'text' not in df.columns:
                st.error("CSV file must contain a 'text' column.")
                return
            
            # Map columns
            text_col = 'text'
            source_col = st.selectbox("Source column (optional)", ['None'] + list(df.columns), key="source_col_select")
            date_col = st.selectbox("Date column (optional)", ['None'] + list(df.columns), key="date_col_select")
            
            if st.button("🚀 Analyze Single File", type="primary", key="analyze_single_file"):
                # Prepare data for analysis
                text_data = []
                for _, row in df.iterrows():
                    text_data.append({
                        'text': str(row[text_col]),
                        'source': str(row[source_col]) if source_col != 'None' else 'File Upload',
                        'date': str(row[date_col]) if date_col != 'None' else str(datetime.now().date())
                    })
                
                # Perform batch analysis
                with st.spinner(f"Analyzing {len(text_data)} texts from file..."):
                    results = batch_analyze_sentiment(text_data, analyzer_type=st.session_state.selected_analyzer)
                    
                    # Add explanations
                    results["explanation"] = results.apply(
                        lambda row: get_sentiment_explanation(row["text"], row["sentiment"], row["polarity"], analyzer_type=st.session_state.selected_analyzer),
                        axis=1
                    )

                    
                    # Update session state
                    st.session_state.analysis_results = pd.concat([st.session_state.analysis_results, results], ignore_index=True)
                
                st.success(f"✅ Analyzed {len(text_data)} texts from file successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

def compare_analyzers_analysis():
    """
    Function for comparing analyzers analysis mode
    """
    st.subheader("🔄 Compare Analyzers")
    st.write("This mode allows you to analyze text with both TextBlob and HuggingFace analyzers simultaneously to compare their results.")
    
    # Text input options
    comparison_mode = st.radio(
        "Choose comparison input method:",
        ["Single Text", "Batch Text", "File Upload"],
        horizontal=True
    )
    
    if comparison_mode == "Single Text":
        # Single text comparison
        text_input = st.text_area(
            "Enter text to analyze with both models:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source (optional)", placeholder="e.g., Twitter, Survey")
        with col2:
            date = st.date_input("Date (optional)", value=datetime.now())
        
        if st.button("🔄 Compare Analyzers", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing with both models..."):
                    text_data = [{
                        'text': text_input,
                        'source': source if source else 'Manual Input',
                        'date': str(date)
                    }]
                    
                    comparative_results = analyze_with_both_models(text_data)
                    st.session_state.comparative_results = pd.concat([
                        st.session_state.comparative_results, 
                        comparative_results
                    ], ignore_index=True)
                
                st.success("✅ Comparative analysis completed!")
                st.rerun()
            else:
                st.error("Please enter some text to analyze.")
    
    elif comparison_mode == "Batch Text":
        # Batch text comparison
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Text 1\nText 2\nText 3..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            default_source = st.text_input("Default source for all texts", placeholder="e.g., Survey Responses")
        with col2:
            default_date = st.date_input("Default date for all texts", value=datetime.now())
        
        if st.button("🔄 Compare Batch", type="primary"):
            if batch_text.strip():
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                
                text_data = []
                for i, text in enumerate(texts):
                    text_data.append({
                        'text': text,
                        'source': default_source if default_source else f'Batch Input {i+1}',
                        'date': str(default_date)
                    })
                
                with st.spinner(f"Analyzing {len(texts)} texts with both models..."):
                    comparative_results = analyze_with_both_models(text_data)
                    st.session_state.comparative_results = pd.concat([
                        st.session_state.comparative_results, 
                        comparative_results
                    ], ignore_index=True)
                
                st.success(f"✅ Comparative analysis completed for {len(texts)} texts!")
                st.rerun()
            else:
                st.error("Please enter some texts to analyze.")
    
    elif comparison_mode == "File Upload":
        # File upload comparison
        uploaded_file = st.file_uploader(
            "Choose a CSV file for comparative analysis",
            type=['csv'],
            help="CSV file should have a 'text' column. Optional columns: 'source', 'date'",
            key="comparative_file_upload"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.write("### File Preview")
                st.dataframe(df.head())
                
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                    return
                
                # Map columns
                source_col = st.selectbox("Source column (optional)", ['None'] + list(df.columns), key="comp_source_col")
                date_col = st.selectbox("Date column (optional)", ['None'] + list(df.columns), key="comp_date_col")
                
                if st.button("🔄 Compare File Data", type="primary", key="compare_file_data"):
                    text_data = []
                    for _, row in df.iterrows():
                        text_data.append({
                            'text': str(row['text']),
                            'source': str(row[source_col]) if source_col != 'None' else 'File Upload',
                            'date': str(row[date_col]) if date_col != 'None' else str(datetime.now().date())
                        })
                    
                    with st.spinner(f"Analyzing {len(text_data)} texts with both models..."):
                        comparative_results = analyze_with_both_models(text_data)
                        st.session_state.comparative_results = pd.concat([
                            st.session_state.comparative_results, 
                            comparative_results
                        ], ignore_index=True)
                    
                    st.success(f"✅ Comparative analysis completed for {len(text_data)} texts!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Show current comparative results summary
    if not st.session_state.comparative_results.empty:
        st.markdown("---")
        st.subheader("📊 Current Comparative Results Summary")
        
        total_comparisons = len(st.session_state.comparative_results)
        discrepancies = st.session_state.comparative_results['sentiment_discrepancy'].sum()
        agreement_rate = ((total_comparisons - discrepancies) / total_comparisons * 100) if total_comparisons > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Comparisons", total_comparisons)
        
        with col2:
            st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
        
        with col3:
            if st.button("🗑️ Clear Comparative Results"):
                st.session_state.comparative_results = pd.DataFrame()
                st.rerun()

def display_analysis_results():
    st.markdown("---")
    st.subheader("📊 Analysis Results")

    # Create tabs
    tabs = ["📊 Overview", "📈 Trends", "🔍 Keywords", "📋 Data", "🎯 Accuracy Report", "🔄 Analyzer Comparison"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs)

    if not st.session_state.analysis_results.empty:
        # Apply filters if they exist
        filtered_data = st.session_state.analysis_results.copy()
        

        
        with tab1:
            # Get summary metrics
            metrics = create_sentiment_metrics_summary(filtered_data)
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Texts", metrics["total_texts"])
            
            with col2:
                st.metric("Positive", metrics["positive_count"], f"{metrics["positive_pct"]:.1f}%")
            
            with col3:
                st.metric("Negative", metrics["negative_count"], f"{metrics["negative_pct"]:.1f}%")
            
            with col4:
                st.metric("Neutral", metrics["neutral_count"], f"{metrics["neutral_pct"]:.1f}%")
            
            # Additional metrics row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Avg Confidence", f"{metrics["avg_confidence"]:.3f}")
            
            with col6:
                st.metric("Avg Polarity", f"{metrics["avg_polarity"]:.3f}")
            
            with col7:
                st.metric("Avg Subjectivity", f"{metrics["avg_subjectivity"]:.3f}")
            
            with col8:
                # Clear results button
                if st.button("🗑️ Clear All Results"):
                    st.session_state.analysis_results = pd.DataFrame()
                    st.rerun()

            # Overview visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution pie chart
                fig_pie = create_sentiment_distribution_chart(filtered_data)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Confidence distribution histogram
                fig_conf = create_confidence_distribution_chart(filtered_data)
                if fig_conf:
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            # Sentiment by source (if multiple sources exist)
            if 'source' in filtered_data.columns and len(filtered_data['source'].unique()) > 1:
                fig_source = create_sentiment_by_source_chart(filtered_data)
                if fig_source:
                    st.plotly_chart(fig_source, use_container_width=True)
            
            # Polarity vs Subjectivity scatter plot
            fig_scatter = create_polarity_vs_subjectivity_scatter(filtered_data)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab2:
            # Trends over time
            if 'date' in filtered_data.columns:
                fig_time = create_sentiment_over_time_chart(filtered_data)
                if fig_time:
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("No time-based data available for trend analysis.")
            else:
                st.info("Date information not available for trend analysis.")
        
        with tab3:
            # Keywords analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Word cloud
                sentiment_filter = st.multiselect(
                    "Filter word cloud by sentiment:",
                    options=filtered_data['sentiment'].unique(),
                    default=filtered_data['sentiment'].unique(),
                    key="wordcloud_filter"
                )
                
                fig_wordcloud = create_wordcloud(filtered_data, sentiment_filter)
                if fig_wordcloud:
                    st.pyplot(fig_wordcloud)
                else:
                    st.info("No keywords available for word cloud.")
            
            with col2:
                # Top keywords bar chart
                top_n = st.slider("Number of top keywords to show:", 5, 20, 10, key="top_keywords_slider")
                fig_keywords = create_keyword_frequency_chart(filtered_data, top_n)
                if fig_keywords:
                    st.plotly_chart(fig_keywords, use_container_width=True)
                else:
                    st.info("No keywords available for frequency analysis.")
        
        with tab4:
            # Detailed data table
            st.subheader("📋 Detailed Results")
            
            # Show dataframe with key columns
            display_columns = ['text', 'sentiment', 'confidence', 'polarity', 'source', 'date']
            available_columns = [col for col in display_columns if col in filtered_data.columns]
            
            st.dataframe(
                filtered_data[available_columns],
                use_container_width=True,
                hide_index=True
            )
            
            # Show explanations
            with st.expander("🔍 View Explanations"):
                for idx, row in filtered_data.iterrows():
                    if 'explanation' in row:
                        st.write(f"**Text {idx + 1}:** {row['explanation']}")
            
            # Export options
            st.subheader("📥 Export Data")
            export_format = st.selectbox("Choose export format:", ["CSV", "JSON", "Excel", "PDF Report"])
            
            # Option to include visuals in PDF
            include_visuals = False
            if export_format == "PDF Report":
                include_visuals = st.checkbox("Include visualizations in PDF", value=True)
            
            if st.button("📥 Download Data"):
                if export_format == "CSV":
                    csv_data, filename = export_to_csv(filtered_data)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                elif export_format == "JSON":
                    json_data, filename = export_to_json(filtered_data)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=filename,
                        mime="application/json"
                    )
                elif export_format == "Excel":
                    excel_data, filename = export_to_excel(filtered_data)
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "PDF Report":
                    with st.spinner("Generating PDF report..."):
                        image_paths = None
                        if include_visuals:
                            # Generate and save visualizations as images
                            import tempfile
                            import os
                            
                            temp_dir = tempfile.mkdtemp()
                            image_paths = []
                            
                            # Save sentiment distribution chart
                            dist_path = os.path.join(temp_dir, "sentiment_distribution.png")
                            create_sentiment_distribution_chart(filtered_data, save_path=dist_path)
                            image_paths.append(dist_path)
                            
                            # Save confidence distribution chart
                            conf_path = os.path.join(temp_dir, "confidence_distribution.png")
                            create_confidence_distribution_chart(filtered_data, save_path=conf_path)
                            image_paths.append(conf_path)
                            
                            # Save polarity vs subjectivity scatter
                            scatter_path = os.path.join(temp_dir, "polarity_subjectivity.png")
                            create_polarity_vs_subjectivity_scatter(filtered_data, save_path=scatter_path)
                            image_paths.append(scatter_path)
                            
                            # Save sentiment by source if available
                            if 'source' in filtered_data.columns and len(filtered_data['source'].unique()) > 1:
                                source_path = os.path.join(temp_dir, "sentiment_by_source.png")
                                create_sentiment_by_source_chart(filtered_data, save_path=source_path)
                                image_paths.append(source_path)
                        
                        pdf_data, filename = create_pdf_report(filtered_data, image_paths=image_paths)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=filename,
                            mime="application/pdf"
                        )
    else:
        with tab1:
            st.info("No analysis results yet. Please analyze some text first.")

    with tab5:
        st.subheader("🎯 Model Accuracy Report")
        st.write("Upload a CSV file with labeled data to evaluate the sentiment analysis model.")
        st.info("The CSV file should contain at least two columns: `text` (for the content to analyze) and `true_sentiment` (for the ground truth sentiment: Positive, Negative, or Neutral).")

        uploaded_file = st.file_uploader(
            "Upload Labeled Data (CSV)",
            type=["csv"],
            key="accuracy_report_uploader_tab"
        )

        if uploaded_file is not None:
            try:
                df_labeled = pd.read_csv(uploaded_file)
                if 'text' not in df_labeled.columns or 'true_sentiment' not in df_labeled.columns:
                    st.error("Uploaded CSV must contain 'text' and 'true_sentiment' columns.")
                else:
                    st.write("### Labeled Data Preview")
                    st.dataframe(df_labeled.head())

                    if st.button("Generate Accuracy Report", type="primary", key="generate_accuracy_report_button"):
                        with st.spinner(f"Generating accuracy report using {st.session_state.selected_analyzer}..."):
                            report_data, error = generate_accuracy_report(df_labeled, analyzer_type=st.session_state.selected_analyzer)
                            
                            if error:
                                st.error(f"Error: {error}")
                            else:
                                st.write("### Classification Report")
                                report_df = pd.DataFrame(report_data["classification_report"]).transpose()
                                st.dataframe(report_df)

                                st.write("### Confusion Matrix")
                                cm_df = pd.DataFrame(
                                    report_data["confusion_matrix"],
                                    index=[f"True {label}" for label in report_data["labels"]],
                                    columns=[f"Predicted {label}" for label in report_data["labels"]]
                                )
                                st.dataframe(cm_df)

                                st.metric("Overall Accuracy", f"{report_data['accuracy']:.2f}")

            except Exception as e:
                st.error(f"Error processing file: {e}")
        
        with tab6:
            # Analyzer Comparison tab
            display_comparative_analysis_tab(st.session_state.comparative_results)
            
            # Show agreement matrix if comparative data exists
            if not st.session_state.comparative_results.empty:
                st.subheader("📊 Analyzer Agreement Matrix")
                agreement_fig = create_agreement_matrix(st.session_state.comparative_results)
                if agreement_fig:
                    st.plotly_chart(agreement_fig, use_container_width=True)

if __name__ == "__main__":
    main()