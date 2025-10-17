import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import streamlit as st

load_dotenv()

st.title("🔮 Embedding Vector Visualizer")
st.markdown("Generate and visualize text embeddings in multiple formats")

# Sidebar for configuration
st.sidebar.header("Visualization Settings")

# Example texts section
st.markdown("### 📚 Example Texts for Visualization")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("🔬 Scientific Texts", help="Compare scientific concepts", key="sci_btn"):
        st.session_state.example_texts = {
            "main": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "additional": [
                "Deep learning uses neural networks with multiple layers to process complex data patterns.",
                "Natural language processing enables computers to understand and generate human language.",
                "Computer vision allows machines to interpret and analyze visual information from images."
            ]
        }
        st.rerun()

with example_col2:
    if st.button("🌍 Different Topics", help="Compare diverse subjects", key="diff_btn"):
        st.session_state.example_texts = {
            "main": "The sun is a star located at the center of our solar system.",
            "additional": [
                "Cooking involves preparing food using various techniques and ingredients.",
                "Music is an art form that combines rhythm, melody, and harmony.",
                "Programming is the process of creating software applications using code."
            ]
        }
        st.rerun()

with example_col3:
    if st.button("📖 Similar Concepts", help="Compare related ideas", key="sim_btn"):
        st.session_state.example_texts = {
            "main": "Democracy is a system of government where citizens have the power to choose their leaders.",
            "additional": [
                "Republic is a form of government where power is held by elected representatives.",
                "Constitutional monarchy combines democratic principles with a hereditary monarch.",
                "Federalism divides power between central and regional governments."
            ]
        }
        st.rerun()

# Load example texts if selected
if hasattr(st.session_state, 'example_texts'):
    example_texts = st.session_state.example_texts
    input_text = example_texts["main"]
    additional_examples = example_texts["additional"]
else:
    input_text = "LangChain is a framework for developing applications powered by large language models."
    additional_examples = []

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area(
    label="Enter Text",
        value=input_text,
        height=100,
        key="main_text"
    )

with col2:
    st.markdown("**Visualization Options:**")
    show_2d_scatter = st.checkbox("2D Scatter Plot", value=True)
    show_3d_scatter = st.checkbox("3D Scatter Plot", value=False, help="Requires WebGL support")
    show_heatmap = st.checkbox("Heatmap", value=True)
    show_vector_magnitude = st.checkbox("Vector Magnitude", value=True)
    show_comparison = st.checkbox("Multi-text Comparison", value=False)
    show_radar_chart = st.checkbox("Radar Chart", value=True, help="Alternative to 3D visualization")
    show_vector_field = st.checkbox("Vector Field", value=True, help="kNN/ANN style vector visualization")
    show_dataframe = st.checkbox("DataFrame View", value=True, help="Show all embedding values in table format")

# Additional text inputs for comparison
if show_comparison:
    st.markdown("### Additional Texts for Comparison")
    additional_texts = []
    
    # Pre-fill with example texts if available
    for i in range(3):
        default_value = additional_examples[i] if i < len(additional_examples) else ""
        text = st.text_input(f"Text {i+2}:", value=default_value, key=f"text_{i+2}")
        if text:
            additional_texts.append(text)

if st.button("🚀 GENERATE EMBEDDINGS", type="primary"):
    try:
        # Generate embeddings
        embeddings_model = OpenAIEmbeddings()
        
        # Main embedding
        main_embedding = embeddings_model.embed_query(input_text)
        embeddings_data = {
            "text": [input_text],
            "embedding": [main_embedding],
            "label": ["Main Text"]
        }
        
        # Additional embeddings for comparison
        if show_comparison and additional_texts:
            for i, text in enumerate(additional_texts):
                if text.strip():
                    embedding = embeddings_model.embed_query(text)
                    embeddings_data["text"].append(text)
                    embeddings_data["embedding"].append(embedding)
                    embeddings_data["label"].append(f"Text {i+2}")
        
        embeddings_array = np.array(embeddings_data["embedding"])
        
        st.success(f"✅ Generated embeddings with {len(main_embedding)} dimensions")
        
        # Display basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vector Dimensions", len(main_embedding))
        with col2:
            st.metric("Magnitude", f"{np.linalg.norm(main_embedding):.3f}")
        with col3:
            st.metric("Min Value", f"{np.min(main_embedding):.3f}")
        
        # Visualization tabs
        tab_names = ["📊 2D Visualization", "🔥 Heatmap", "📏 Vector Analysis", "📈 Comparison"]
        if show_3d_scatter:
            tab_names.insert(1, "🌐 3D Visualization")
        if show_radar_chart:
            tab_names.append("🎯 Radar Chart")
        if show_vector_field:
            tab_names.append("🎯 Vector Field")
        if show_dataframe:
            tab_names.append("📊 DataFrame View")
        
        tabs = st.tabs(tab_names)
        tab_index = 0
        
        with tabs[tab_index]:
            if show_2d_scatter:
                st.subheader("2D Scatter Plot (PCA)")
                
                if len(embeddings_data["embedding"]) > 1:
                    # PCA for 2D visualization (multiple embeddings)
                    pca_2d = PCA(n_components=2)
                    embeddings_2d = pca_2d.fit_transform(embeddings_array)
                    
                    # Create DataFrame for plotting
                    df_2d = pd.DataFrame({
                        'PC1': embeddings_2d[:, 0],
                        'PC2': embeddings_2d[:, 1],
                        'Label': embeddings_data["label"],
                        'Text': embeddings_data["text"]
                    })
                    
                    # Plotly scatter plot
                    fig_2d = px.scatter(
                        df_2d, 
                        x='PC1', 
                        y='PC2', 
                        color='Label',
                        hover_data=['Text'],
                        title="2D Embedding Visualization (PCA)",
                        labels={'PC1': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', 
                               'PC2': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)'}
                    )
                    fig_2d.update_layout(height=500)
                    st.plotly_chart(fig_2d, use_container_width=True, key="pca_2d")
                    
                    # Explained variance
                    st.info(f"PCA explains {sum(pca_2d.explained_variance_ratio_):.1%} of the variance")
                else:
                    # Single embedding visualization - show first 2 dimensions
                    st.info("📊 Single embedding detected. Showing first 2 dimensions instead of PCA.")
                    
                    embedding_2d = main_embedding[:2]  # Take first 2 dimensions
                    
                    fig_2d = px.scatter(
                        x=[embedding_2d[0]], 
                        y=[embedding_2d[1]],
                        title="2D Embedding Visualization (First 2 Dimensions)",
                        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                        hover_data=[['Main Text']]
                    )
                    fig_2d.update_layout(height=500)
                    fig_2d.update_traces(marker=dict(size=15, color='red'))
                    st.plotly_chart(fig_2d, use_container_width=True, key="single_2d")
                    
                    # Add dimension distribution for single embedding
                    st.subheader("Dimension Value Distribution")
                    dim_df = pd.DataFrame({
                        'Dimension': range(len(main_embedding)),
                        'Value': main_embedding
                    })
                    
                    fig_dist = px.line(
                        dim_df.head(50),  # Show first 50 dimensions
                        x='Dimension',
                        y='Value',
                        title="Embedding Values Across Dimensions (First 50)",
                        labels={'Dimension': 'Dimension Index', 'Value': 'Embedding Value'}
                    )
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True, key="dim_dist")
                    
                    st.info(f"Showing dimensions 1 and 2 of {len(main_embedding)} total dimensions")
        
        # 3D Visualization tab (only if enabled)
        if show_3d_scatter:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("3D Scatter Plot (PCA)")
                
                st.warning("⚠️ 3D visualization requires WebGL support. If you see a blank chart, your browser doesn't support WebGL.")
                
                if len(embeddings_data["embedding"]) > 2:
                    # PCA for 3D visualization (3+ embeddings)
                    pca_3d = PCA(n_components=3)
                    embeddings_3d = pca_3d.fit_transform(embeddings_array)
                    
                    # Create DataFrame for plotting
                    df_3d = pd.DataFrame({
                        'PC1': embeddings_3d[:, 0],
                        'PC2': embeddings_3d[:, 1],
                        'PC3': embeddings_3d[:, 2],
                        'Label': embeddings_data["label"],
                        'Text': embeddings_data["text"]
                    })
                    
                    # Plotly 3D scatter plot
                    fig_3d = px.scatter_3d(
                        df_3d, 
                        x='PC1', 
                        y='PC2', 
                        z='PC3',
                        color='Label',
                        hover_data=['Text'],
                        title="3D Embedding Visualization (PCA)"
                    )
                    fig_3d.update_layout(height=600)
                    st.plotly_chart(fig_3d, use_container_width=True, key="pca_3d")
                    
                    # Explained variance
                    st.info(f"PCA explains {sum(pca_3d.explained_variance_ratio_):.1%} of the variance")
                elif len(embeddings_data["embedding"]) == 2:
                    # 2 embeddings - use 2D PCA and add a zero z-axis
                    pca_2d = PCA(n_components=2)
                    embeddings_2d = pca_2d.fit_transform(embeddings_array)
                    
                    # Add zero z-axis to make it 3D
                    embeddings_3d = np.column_stack([embeddings_2d, np.zeros(len(embeddings_2d))])
                    
                    # Create DataFrame for plotting
                    df_3d = pd.DataFrame({
                        'PC1': embeddings_3d[:, 0],
                        'PC2': embeddings_3d[:, 1],
                        'PC3': embeddings_3d[:, 2],
                        'Label': embeddings_data["label"],
                        'Text': embeddings_data["text"]
                    })
                    
                    # Plotly 3D scatter plot
                    fig_3d = px.scatter_3d(
                        df_3d, 
                        x='PC1', 
                        y='PC2', 
                        z='PC3',
                        color='Label',
                        hover_data=['Text'],
                        title="3D Embedding Visualization (2D PCA + Zero Z-axis)"
                    )
                    fig_3d.update_layout(height=600)
                    st.plotly_chart(fig_3d, use_container_width=True, key="pca_3d_2d")
                    
                    st.info("📊 Only 2 embeddings available. Showing 2D PCA with zero z-axis.")
                    st.info(f"PCA explains {sum(pca_2d.explained_variance_ratio_):.1%} of the variance")
                else:
                    # Single embedding visualization - show first 3 dimensions
                    st.info("🌐 Single embedding detected. Showing first 3 dimensions instead of PCA.")
                    
                    embedding_3d = main_embedding[:3]  # Take first 3 dimensions
                    
                    fig_3d = px.scatter_3d(
                        x=[embedding_3d[0]], 
                        y=[embedding_3d[1]], 
                        z=[embedding_3d[2]],
                        title="3D Embedding Visualization (First 3 Dimensions)",
                        labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'},
                        hover_data=[['Main Text']]
                    )
                    fig_3d.update_layout(height=600)
                    fig_3d.update_traces(marker=dict(size=15, color='red'))
                    st.plotly_chart(fig_3d, use_container_width=True, key="single_3d")
                    
                    st.info(f"Showing dimensions 1, 2, and 3 of {len(main_embedding)} total dimensions")
        
        # Heatmap tab
        tab_index += 1
        with tabs[tab_index]:
            if show_heatmap:
                st.subheader("Embedding Heatmap")
                
                # Create heatmap of first 50 dimensions
                max_dims = min(50, len(main_embedding))
                heatmap_data = embeddings_array[:, :max_dims]
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    title=f"Embedding Heatmap (First {max_dims} dimensions)",
                    labels=dict(x="Dimension", y="Text", color="Value"),
                    aspect="auto"
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
                
                # Dimension statistics
                st.subheader("Dimension Statistics")
                dim_stats = pd.DataFrame({
                    'Dimension': range(max_dims),
                    'Mean': np.mean(heatmap_data, axis=0),
                    'Std': np.std(heatmap_data, axis=0),
                    'Min': np.min(heatmap_data, axis=0),
                    'Max': np.max(heatmap_data, axis=0)
                })
                st.dataframe(dim_stats, use_container_width=True)
        
        # Vector Analysis tab
        tab_index += 1
        with tabs[tab_index]:
            if show_vector_magnitude:
                st.subheader("Vector Magnitude Analysis")
                
                # Magnitude comparison
                magnitudes = [np.linalg.norm(emb) for emb in embeddings_data["embedding"]]
                
                fig_mag = px.bar(
                    x=embeddings_data["label"],
                    y=magnitudes,
                    title="Vector Magnitudes",
                    labels={'x': 'Text', 'y': 'Magnitude'}
                )
                st.plotly_chart(fig_mag, use_container_width=True, key="magnitude")
                
                # Vector direction analysis (first 20 dimensions)
                st.subheader("Vector Direction (First 20 Dimensions)")
                max_dims_viz = min(20, len(main_embedding))
                
                fig_direction = go.Figure()
                for i, (label, embedding) in enumerate(zip(embeddings_data["label"], embeddings_data["embedding"])):
                    fig_direction.add_trace(go.Scatter(
                        x=list(range(max_dims_viz)),
                        y=embedding[:max_dims_viz],
                        mode='lines+markers',
                        name=label,
                        line=dict(width=2)
                    ))
                
                fig_direction.update_layout(
                    title="Vector Direction Visualization",
                    xaxis_title="Dimension",
                    yaxis_title="Value",
                    height=400
                )
                st.plotly_chart(fig_direction, use_container_width=True, key="direction")
        
        # Comparison tab
        tab_index += 1
        with tabs[tab_index]:
            if show_comparison and len(embeddings_data["embedding"]) > 1:
                st.subheader("Embedding Comparison")
                
                # Cosine similarity matrix
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(embeddings_array)
                
                fig_similarity = px.imshow(
                    similarity_matrix,
                    title="Cosine Similarity Matrix",
                    labels=dict(x="Text", y="Text", color="Similarity"),
                    x=embeddings_data["label"],
                    y=embeddings_data["label"],
                    color_continuous_scale="RdYlBu_r"
                )
                st.plotly_chart(fig_similarity, use_container_width=True, key="similarity")
                
                # Similarity scores
                st.subheader("Similarity Scores")
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=embeddings_data["label"],
                    columns=embeddings_data["label"]
                )
                st.dataframe(similarity_df, use_container_width=True)
                
            else:
                st.info("Add additional texts above to enable comparison visualization")
        
        # Radar Chart tab (WebGL alternative)
        if show_radar_chart:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("🎯 Radar Chart (WebGL Alternative)")
                st.info("This radar chart shows embedding patterns without requiring WebGL support.")
                
                # Create radar chart for first 8 dimensions
                max_radar_dims = min(8, len(main_embedding))
                radar_dims = [f"Dim {i+1}" for i in range(max_radar_dims)]
                
                fig_radar = go.Figure()
                
                for i, (label, embedding) in enumerate(zip(embeddings_data["label"], embeddings_data["embedding"])):
                    # Normalize values for radar chart (0-1 range)
                    radar_values = np.array(embedding[:max_radar_dims])
                    radar_values = (radar_values - radar_values.min()) / (radar_values.max() - radar_values.min())
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_values,
                        theta=radar_dims,
                        fill='toself',
                        name=label,
                        line=dict(width=2)
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Embedding Pattern Comparison (First 8 Dimensions)",
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True, key="radar")
                
                # Additional 2D projection alternatives
                st.subheader("Alternative 2D Projections")
                
                # Create multiple 2D projections using different dimension pairs
                col1, col2 = st.columns(2)
                
                with col1:
                    # Dimensions 1-2
                    fig_proj1 = px.scatter(
                        x=[main_embedding[0]], 
                        y=[main_embedding[1]],
                        title="Dimensions 1-2",
                        labels={'x': 'Dimension 1', 'y': 'Dimension 2'}
                    )
                    fig_proj1.update_traces(marker=dict(size=15, color='blue'))
                    st.plotly_chart(fig_proj1, use_container_width=True, key="proj1")
                
                with col2:
                    # Dimensions 3-4
                    if len(main_embedding) >= 4:
                        fig_proj2 = px.scatter(
                            x=[main_embedding[2]], 
                            y=[main_embedding[3]],
                            title="Dimensions 3-4",
                            labels={'x': 'Dimension 3', 'y': 'Dimension 4'}
                        )
                        fig_proj2.update_traces(marker=dict(size=15, color='green'))
                        st.plotly_chart(fig_proj2, use_container_width=True, key="proj2")
                    else:
                        st.info("Not enough dimensions for 3-4 projection")
        
        # Vector Field tab (kNN/ANN style visualization)
        if show_vector_field:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("🎯 Vector Field Visualization")
                st.info("This shows vectors radiating from the origin, similar to kNN/ANN search visualizations.")
                
                # Create vector field visualization for all texts
                fig_vector_field = go.Figure()
                
                # Colors for different texts
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                
                # Generate vectors for each embedding
                for text_idx, (label, embedding) in enumerate(zip(embeddings_data["label"], embeddings_data["embedding"])):
                    # Generate vectors from embedding dimensions
                    num_vectors = min(20, len(embedding))  # Show up to 20 vectors per text
                    
                    # Create vectors from first num_vectors dimensions
                    vectors = []
                    for i in range(num_vectors):
                        # Use embedding values as vector components
                        x_component = embedding[i] if i < len(embedding) else 0
                        y_component = embedding[(i + 1) % len(embedding)] if (i + 1) < len(embedding) else 0
                        
                        # Normalize vector length for better visualization
                        magnitude = np.sqrt(x_component**2 + y_component**2)
                        if magnitude > 0:
                            scale_factor = 0.4  # Scale down for better visualization
                            x_component = (x_component / magnitude) * scale_factor
                            y_component = (y_component / magnitude) * scale_factor
                        
                        vectors.append({
                            'x': x_component,
                            'y': y_component,
                            'magnitude': magnitude,
                            'dimension': i
                        })
                    
                    # Sort vectors by magnitude for color coding
                    vectors.sort(key=lambda v: v['magnitude'], reverse=True)
                    
                    # Color coding: top 30% in bright color, rest in faded color
                    top_percent = 0.3
                    top_count = max(1, int(len(vectors) * top_percent))
                    
                    # Get color for this text
                    text_color = colors[text_idx % len(colors)]
                    
                    # Add vectors with different colors
                    for i, vector in enumerate(vectors):
                        if i < top_count:
                            # Bright color for top vectors (like ANN active vectors)
                            color = text_color
                            opacity = 1.0
                            line_width = 3
                        else:
                            # Faded color for others (like ANN inactive vectors)
                            color = text_color
                            opacity = 0.3
                            line_width = 1
                        
                        # Add arrow from origin to vector endpoint
                        fig_vector_field.add_trace(go.Scatter(
                            x=[0, vector['x']],
                            y=[0, vector['y']],
                            mode='lines+markers',
                            line=dict(color=color, width=line_width),
                            marker=dict(size=6, color=color),
                            opacity=opacity,
                            name=f"{label} - Vector {vector['dimension']}" if i < 3 else "",
                            showlegend=False,
                            hovertemplate=f"{label}<br>Dimension {vector['dimension']}<br>Magnitude: {vector['magnitude']:.3f}<extra></extra>"
                        ))
                
                # Add coordinate axes
                fig_vector_field.add_trace(go.Scatter(
                    x=[-0.6, 0.6], y=[0, 0],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_vector_field.add_trace(go.Scatter(
                    x=[0, 0], y=[-0.6, 0.6],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add origin point
                fig_vector_field.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers',
                    marker=dict(size=10, color='black'),
                    showlegend=False,
                    hovertemplate="Origin<extra></extra>"
                ))
                
                fig_vector_field.update_layout(
                    title="Vector Field Visualization (kNN/ANN Style)",
                    xaxis=dict(range=[-0.7, 0.7], showgrid=True, zeroline=True),
                    yaxis=dict(range=[-0.7, 0.7], showgrid=True, zeroline=True),
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig_vector_field, use_container_width=True, key="vector_field")
                
                # Add explanation
                st.markdown("""
                **Vector Field Explanation:**
                - **Bright Colored Vectors**: Top 30% by magnitude for each text (like active ANN vectors)
                - **Faded Colored Vectors**: Remaining vectors for each text (like inactive ANN vectors)
                - **Different Colors**: Each text gets its own color (red, blue, green, etc.)
                - **Origin**: Central point where all vectors start
                - **Hover**: Shows text label, dimension index, and magnitude
                """)
                
                # Add color legend
                if len(embeddings_data["label"]) > 1:
                    st.markdown("**Color Legend:**")
                    legend_cols = st.columns(min(len(embeddings_data["label"]), 4))
                    for i, (label, color) in enumerate(zip(embeddings_data["label"], colors)):
                        with legend_cols[i % 4]:
                            st.markdown(f"🔴 **{label}**" if color == 'red' else 
                                       f"🔵 **{label}**" if color == 'blue' else
                                       f"🟢 **{label}**" if color == 'green' else
                                       f"🟠 **{label}**" if color == 'orange' else
                                       f"🟣 **{label}**" if color == 'purple' else
                                       f"🟤 **{label}**" if color == 'brown' else
                                       f"🩷 **{label}**" if color == 'pink' else
                                       f"⚫ **{label}**")
                
                # Add comparison with multiple embeddings if available
                if len(embeddings_data["embedding"]) > 1:
                    st.subheader("Multi-Text Vector Field Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Text 1 Vectors**")
                        # Show first text vectors
                        fig_comp1 = go.Figure()
                        # Similar logic but for first embedding
                        # ... (implementation similar to above)
                        st.plotly_chart(fig_comp1, use_container_width=True, key="comp1")
                    
                    with col2:
                        st.markdown("**Text 2 Vectors**")
                        # Show second text vectors
                        fig_comp2 = go.Figure()
                        # Similar logic but for second embedding
                        # ... (implementation similar to above)
                        st.plotly_chart(fig_comp2, use_container_width=True, key="comp2")
        
        # DataFrame View tab
        if show_dataframe:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("📊 Embedding Values DataFrame")
                st.info("View all embedding values in a structured table format for detailed analysis.")
                
                # Create comprehensive dataframe
                max_dims = min(50, len(main_embedding))  # Show first 50 dimensions
                
                # Create dataframe with all embeddings
                df_data = pd.DataFrame()
                
                for i, (label, embedding) in enumerate(zip(embeddings_data["label"], embeddings_data["embedding"])):
                    # Create row for this text
                    row_data = {
                        'Text_Label': label,
                        'Text_Content': embeddings_data["text"][i][:100] + "..." if len(embeddings_data["text"][i]) > 100 else embeddings_data["text"][i],
                        'Vector_Magnitude': np.linalg.norm(embedding),
                        'Min_Value': np.min(embedding),
                        'Max_Value': np.max(embedding),
                        'Mean_Value': np.mean(embedding),
                        'Std_Value': np.std(embedding)
                    }
                    
                    # Add individual dimension values
                    for dim in range(max_dims):
                        row_data[f'Dim_{dim+1}'] = embedding[dim] if dim < len(embedding) else 0
                    
                    # Add to dataframe
                    df_data = pd.concat([df_data, pd.DataFrame([row_data])], ignore_index=True)
                
                # Display main statistics
                st.subheader("📈 Summary Statistics")
                summary_cols = st.columns(2)
                
                with summary_cols[0]:
                    st.markdown("**Basic Statistics:**")
                    basic_stats = df_data[['Text_Label', 'Vector_Magnitude', 'Min_Value', 'Max_Value', 'Mean_Value', 'Std_Value']].copy()
                    basic_stats.columns = ['Text', 'Magnitude', 'Min', 'Max', 'Mean', 'Std']
                    st.dataframe(basic_stats, use_container_width=True)
                
                with summary_cols[1]:
                    st.markdown("**Dimension Statistics:**")
                    dim_stats = pd.DataFrame({
                        'Dimension': [f'Dim_{i+1}' for i in range(max_dims)],
                        'Mean_Across_Texts': [df_data[f'Dim_{i+1}'].mean() for i in range(max_dims)],
                        'Std_Across_Texts': [df_data[f'Dim_{i+1}'].std() for i in range(max_dims)],
                        'Min_Across_Texts': [df_data[f'Dim_{i+1}'].min() for i in range(max_dims)],
                        'Max_Across_Texts': [df_data[f'Dim_{i+1}'].max() for i in range(max_dims)]
                    })
                    st.dataframe(dim_stats.head(10), use_container_width=True)
                
                # Display full embedding values
                st.subheader("🔍 Full Embedding Values")
                
                # Create dimension columns for display
                dim_cols = [f'Dim_{i+1}' for i in range(max_dims)]
                display_cols = ['Text_Label', 'Text_Content', 'Vector_Magnitude'] + dim_cols
                
                df_display = df_data[display_cols].copy()
                df_display.columns = ['Text', 'Content', 'Magnitude'] + [f'D{i+1}' for i in range(max_dims)]
                
                # Format the dataframe for better display
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=400
                )
                
                # Add dimension analysis
                st.subheader("📊 Dimension Analysis")
                
                # Find most important dimensions (highest variance)
                dim_variance = []
                for i in range(max_dims):
                    variance = df_data[f'Dim_{i+1}'].var()
                    dim_variance.append({
                        'Dimension': f'Dim_{i+1}',
                        'Variance': variance,
                        'Mean': df_data[f'Dim_{i+1}'].mean(),
                        'Range': df_data[f'Dim_{i+1}'].max() - df_data[f'Dim_{i+1}'].min()
                    })
                
                dim_analysis_df = pd.DataFrame(dim_variance)
                dim_analysis_df = dim_analysis_df.sort_values('Variance', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 10 Most Variable Dimensions:**")
                    st.dataframe(dim_analysis_df.head(10), use_container_width=True)
                
                with col2:
                    st.markdown("**Dimension Value Distribution:**")
                    # Create a simple histogram of dimension values
                    all_values = []
                    for i in range(max_dims):
                        all_values.extend(df_data[f'Dim_{i+1}'].tolist())
                    
                    fig_hist = px.histogram(
                        x=all_values,
                        title="Distribution of All Embedding Values",
                        nbins=30,
                        labels={'x': 'Embedding Value', 'y': 'Count'}
                    )
                    fig_hist.update_layout(height=300)
                    st.plotly_chart(fig_hist, use_container_width=True, key="embedding_hist")
                
                # Download options
                st.subheader("💾 Download Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download full dataframe as CSV
                    csv_full = df_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Data (CSV)",
                        data=csv_full,
                        file_name="embedding_data_full.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download summary statistics
                    csv_summary = basic_stats.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv_summary,
                        file_name="embedding_summary.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    # Download dimension analysis
                    csv_analysis = dim_analysis_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis (CSV)",
                        data=csv_analysis,
                        file_name="dimension_analysis.csv",
                        mime="text/csv"
                    )
        
        # Raw embedding data
        with st.expander("🔍 Raw Embedding Data"):
            st.subheader("Full Embedding Vector")
            st.code(f"Dimensions: {len(main_embedding)}")
            st.code(f"Values: {main_embedding[:10]}...")  # Show first 10 values
            
            # Download option
            embedding_df = pd.DataFrame({
                'dimension': range(len(main_embedding)),
                'value': main_embedding
            })
            csv = embedding_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Embedding as CSV",
                data=csv,
                file_name="embedding_data.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error generating embeddings: {str(e)}")
        st.info("Make sure your OpenAI API key is properly set in your environment variables.")