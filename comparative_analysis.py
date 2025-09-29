import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentiment_analyzer import analyze_sentiment_textblob, analyze_sentiment_huggingface, get_sentiment_explanation

def analyze_with_both_models(text_data):
    """
    Analyze text data with both TextBlob and HuggingFace models
    
    Args:
        text_data: List of dictionaries with 'text', 'source', 'date' keys
    
    Returns:
        DataFrame with results from both analyzers
    """
    results = []
    
    for item in text_data:
        text = item['text']
        source = item.get('source', 'Unknown')
        date = item.get('date', 'Unknown')
        
        # Analyze with TextBlob
        tb_sentiment, tb_confidence, tb_polarity, tb_subjectivity = analyze_sentiment_textblob(text)
        tb_explanation = get_sentiment_explanation(text, tb_sentiment, tb_polarity, analyzer_type='TextBlob')
        
        # Analyze with HuggingFace
        hf_sentiment, hf_confidence, hf_polarity, hf_subjectivity = analyze_sentiment_huggingface(text)
        hf_explanation = get_sentiment_explanation(text, hf_sentiment, hf_polarity, analyzer_type='HuggingFace')
        
        # Check for discrepancy
        discrepancy = tb_sentiment != hf_sentiment
        confidence_diff = abs(tb_confidence - hf_confidence)
        
        results.append({
            'text': text,
            'source': source,
            'date': date,
            'textblob_sentiment': tb_sentiment,
            'textblob_confidence': tb_confidence,
            'textblob_polarity': tb_polarity,
            'textblob_subjectivity': tb_subjectivity,
            'textblob_explanation': tb_explanation,
            'huggingface_sentiment': hf_sentiment,
            'huggingface_confidence': hf_confidence,
            'huggingface_polarity': hf_polarity,
            'huggingface_subjectivity': hf_subjectivity,
            'huggingface_explanation': hf_explanation,
            'sentiment_discrepancy': discrepancy,
            'confidence_difference': confidence_diff
        })
    
    return pd.DataFrame(results)

def display_comparative_analysis_tab(df):
    """
    Display comparative analysis between TextBlob and HuggingFace analyzers
    
    Args:
        df: DataFrame with comparative analysis results
    """
    if df.empty:
        st.info("No comparative analysis data available. Use the 'Compare Analyzers' feature to generate comparative results.")
        return
    
    st.subheader("🔍 Analyzer Comparison")
    
    # Summary metrics
    total_texts = len(df)
    discrepancies = df['sentiment_discrepancy'].sum()
    agreement_rate = ((total_texts - discrepancies) / total_texts * 100) if total_texts > 0 else 0
    avg_confidence_diff = df['confidence_difference'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Texts", total_texts)
    
    with col2:
        st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
    
    with col3:
        st.metric("Discrepancies", discrepancies)
    
    with col4:
        st.metric("Avg Confidence Diff", f"{avg_confidence_diff:.3f}")
    
    # Sentiment distribution comparison
    st.subheader("📊 Sentiment Distribution Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tb_sentiment_counts = df['textblob_sentiment'].value_counts()
        fig_tb = px.pie(
            values=tb_sentiment_counts.values,
            names=tb_sentiment_counts.index,
            title="TextBlob Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#4682B4'
            }
        )
        fig_tb.update_layout(height=400)
        st.plotly_chart(fig_tb, use_container_width=True)
    
    with col2:
        hf_sentiment_counts = df['huggingface_sentiment'].value_counts()
        fig_hf = px.pie(
            values=hf_sentiment_counts.values,
            names=hf_sentiment_counts.index,
            title="HuggingFace Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2E8B57',
                'Negative': '#DC143C',
                'Neutral': '#4682B4'
            }
        )
        fig_hf.update_layout(height=400)
        st.plotly_chart(fig_hf, use_container_width=True)
    
    # Confidence comparison
    st.subheader("📈 Confidence Score Comparison")
    
    fig_conf = go.Figure()
    
    fig_conf.add_trace(go.Histogram(
        x=df['textblob_confidence'],
        name='TextBlob',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig_conf.add_trace(go.Histogram(
        x=df['huggingface_confidence'],
        name='HuggingFace',
        opacity=0.7,
        nbinsx=20
    ))
    
    fig_conf.update_layout(
        title="Confidence Score Distribution",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_conf, use_container_width=True)
    
    # Discrepancy analysis
    if discrepancies > 0:
        st.subheader("⚠️ Discrepancy Analysis")
        
        discrepancy_df = df[df['sentiment_discrepancy'] == True].copy()
        
        st.write(f"Found {len(discrepancy_df)} texts where analyzers disagree:")
        
        # Show discrepancy details
        display_df = discrepancy_df[['text', 'textblob_sentiment', 'textblob_confidence', 
                                   'huggingface_sentiment', 'huggingface_confidence', 
                                   'confidence_difference']].copy()
        
        display_df.columns = ['Text', 'TextBlob Sentiment', 'TextBlob Confidence', 
                             'HuggingFace Sentiment', 'HuggingFace Confidence', 'Confidence Diff']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Highlight high-confidence discrepancies
        high_conf_discrepancies = discrepancy_df[
            (discrepancy_df['textblob_confidence'] > 0.7) & 
            (discrepancy_df['huggingface_confidence'] > 0.7)
        ]
        
        if not high_conf_discrepancies.empty:
            st.warning(f"⚠️ {len(high_conf_discrepancies)} high-confidence discrepancies found. These texts may require manual review.")
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    # Allow filtering
    show_discrepancies_only = st.checkbox("Show only discrepancies")
    
    if show_discrepancies_only:
        filtered_df = df[df['sentiment_discrepancy'] == True]
    else:
        filtered_df = df
    
    if not filtered_df.empty:
        # Select columns to display
        display_columns = [
            'text', 'textblob_sentiment', 'textblob_confidence', 'textblob_explanation',
            'huggingface_sentiment', 'huggingface_confidence', 'huggingface_explanation',
            'sentiment_discrepancy', 'confidence_difference'
        ]
        
        st.dataframe(filtered_df[display_columns], use_container_width=True)
    else:
        st.info("No data to display with current filters.")

def create_agreement_matrix(df):
    """
    Create a confusion matrix showing agreement between analyzers
    
    Args:
        df: DataFrame with comparative analysis results
    
    Returns:
        Plotly figure
    """
    if df.empty:
        return None
    
    # Create confusion matrix data
    sentiments = ['Positive', 'Negative', 'Neutral']
    matrix_data = []
    
    for tb_sent in sentiments:
        row = []
        for hf_sent in sentiments:
            count = len(df[(df['textblob_sentiment'] == tb_sent) & (df['huggingface_sentiment'] == hf_sent)])
            row.append(count)
        matrix_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix_data,
        x=sentiments,
        y=sentiments,
        colorscale='Blues',
        text=matrix_data,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Analyzer Agreement Matrix",
        xaxis_title="HuggingFace Predictions",
        yaxis_title="TextBlob Predictions",
        height=400
    )
    
    return fig
