import pandas as pd


def analyze_forecast(df) -> dict:
    """
    Analyzes forecast data and returns statistics about predicted power generation.
    
    Args:
        df: DataFrame containing a 'Predicted Power (kW)' column
    
    Returns:
        dict with keys:
        - 'mean': average predicted power (kW)
        - 'max': maximum predicted power (kW)
        - 'min': minimum predicted power (kW)
        - 'std': standard deviation of predicted power (kW)
        - 'peak_indices': list of indices where power is in top 25%
        - 'low_indices': list of indices where power is in bottom 25%
    """
    if 'Predicted Power (kW)' not in df.columns:
        raise ValueError("DataFrame must contain 'Predicted Power (kW)' column")
    
    power_col = df['Predicted Power (kW)'].dropna()
    
    if len(power_col) == 0:
        raise ValueError("No valid power data found")
    
    # Calculate statistics
    mean_power = float(power_col.mean())
    max_power = float(power_col.max())
    min_power = float(power_col.min())
    std_power = float(power_col.std())
    
    # Calculate 75th and 25th percentiles
    q75 = power_col.quantile(0.75)
    q25 = power_col.quantile(0.25)
    
    # Find indices for peak and low periods
    peak_indices = df[df['Predicted Power (kW)'] >= q75].index.tolist()
    low_indices = df[df['Predicted Power (kW)'] <= q25].index.tolist()
    
    return {
        'mean': mean_power,
        'max': max_power,
        'min': min_power,
        'std': std_power,
        'peak_indices': peak_indices,
        'low_indices': low_indices
    }


def identify_risks(df, threshold=500) -> list[dict]:
    """
    Identifies risk periods where predicted power generation falls below threshold.
    
    A risk period is a contiguous sequence of time steps where power < threshold.
    NaN values are excluded from risk detection.
    
    Args:
        df: DataFrame containing a 'Predicted Power (kW)' column
        threshold: power threshold in kW below which to flag as risk (default: 500)
    
    Returns:
        list of dicts, each representing a risk period:
        - 'start_index': first index label of low-power period
        - 'end_index': last index label of low-power period  
        - 'avg_power': average power during this period (kW)
        - 'risk_level': 'HIGH' if avg < 200, 'MEDIUM' if 200 <= avg < 500, 'LOW' if avg >= 500
    """
    if 'Predicted Power (kW)' not in df.columns:
        raise ValueError("DataFrame must contain 'Predicted Power (kW)' column")
    
    # Drop NaN values for consistent risk detection (excludes missing data, doesn't artificially fill)
    power_col = df['Predicted Power (kW)'].dropna()
    
    # Create boolean mask for below-threshold periods
    below_threshold = power_col < threshold
    
    # Find contiguous regions, tracking both current and previous index
    risks = []
    in_risk_period = False
    start_idx = None
    prev_idx = None
    
    for idx, is_risky in below_threshold.items():
        if is_risky and not in_risk_period:
            # Start of new risk period
            start_idx = idx
            in_risk_period = True
        elif not is_risky and in_risk_period:
            # End of risk period: use prev_idx (last risky index, not current non-risky)
            # Use .loc for label-based slicing to work with any index type (DatetimeIndex, RangeIndex, etc.)
            risk_data = power_col.loc[start_idx:prev_idx]
            avg_power = float(risk_data.mean())
            
            # Determine risk level per specification
            if avg_power < 200:
                risk_level = 'HIGH'
            elif avg_power < 500:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            risks.append({
                'start_index': start_idx,
                'end_index': prev_idx,
                'avg_power': avg_power,
                'risk_level': risk_level
            })
            in_risk_period = False
        
        if is_risky:
            prev_idx = idx
    
    # Handle case where risk period extends to end of dataframe
    if in_risk_period:
        # Use .loc for label-based slicing (works with any index type)
        risk_data = power_col.loc[start_idx:prev_idx]
        avg_power = float(risk_data.mean())
        
        if avg_power < 200:
            risk_level = 'HIGH'
        elif avg_power < 500:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        risks.append({
            'start_index': start_idx,
            'end_index': prev_idx,
            'avg_power': avg_power,
            'risk_level': risk_level
        })
    
    return risks


def retrieve_guidelines(query: str) -> str:
    """
    Retrieves relevant knowledge base documents using RAG (Retrieval-Augmented Generation).
    
    Searches the knowledge base for documents matching the query and returns
    the most relevant chunks concatenated as a single string.
    
    Args:
        query: user's question or topic to search for
    
    Returns:
        concatenated text from top-3 most relevant knowledge chunks,
        separated by newlines
    
    Raises:
        ImportError: if RAG module is not available
    """
    try:
        from src.rag import get_retriever
    except ImportError as exc:
        raise ImportError("RAG module not available. Ensure src/rag.py exists and sentence-transformers is installed.") from exc
    
    # Get the retriever singleton
    retriever = get_retriever()
    
    # Retrieve top 3 most relevant chunks
    results = retriever.retrieve(query, k=3)
    
    # Concatenate the text from all results
    guidelines_text = "\n\n".join([
        f"[Source: {r['source']}]\n{r['text']}"
        for r in results
    ])
    
    return guidelines_text


if __name__ == '__main__':
    # Quick test (requires data and models to exist)
    import sys
    
    try:
        import joblib
        
        # Load model and data
        model = joblib.load('models/random_forest_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        df_test = pd.read_csv('data/spg.csv').head(100)
        
        # Test feature alignment
        try:
            from src.feature_aligner import align_features
        except ImportError:
            from feature_aligner import align_features
        aligned = align_features(df_test)
        scaled = scaler.transform(aligned)
        predictions = model.predict(scaled)
        
        # Create test dataframe with predictions
        test_df = pd.DataFrame({'Predicted Power (kW)': predictions})
        
        # Test analyze_forecast
        print("=" * 60)
        print("Testing analyze_forecast():")
        print("=" * 60)
        summary = analyze_forecast(test_df)
        for key, value in summary.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Test identify_risks
        print("\n" + "=" * 60)
        print("Testing identify_risks():")
        print("=" * 60)
        risks = identify_risks(test_df, threshold=500)
        for i, risk in enumerate(risks, 1):
            print(f"\nRisk Period {i}:")
            for key, value in risk.items():
                print(f"  {key}: {value}")
        
        # Test retrieve_guidelines
        print("\n" + "=" * 60)
        print("Testing retrieve_guidelines():")
        print("=" * 60)
        query = "How should I manage solar variability during peak hours?"
        guidelines = retrieve_guidelines(query)
        print(f"Query: {query}")
        print(f"Retrieved {len(guidelines)} characters of guidelines")
        print(f"First 300 chars:\n{guidelines[:300]}...")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)