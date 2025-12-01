"""Comparison Dashboard for multiple places."""

from pathlib import Path
from typing import Any

import dash
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from .dashboard import load_results


def load_all_places(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load all place results from outputs directory.
    
    Returns:
        Dictionary mapping run_id to results dict
    """
    all_results = {}
    
    if not outputs_dir.exists():
        return all_results
    
    # Find all run directories
    run_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    for run_dir in sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            results = load_results(run_dir)
            if "config" in results:
                # Extract site info
                config = results["config"]
                site_metadata = config.get("site_metadata", {})
                site_id = site_metadata.get("meta_info", {}).get("site_id", "unknown")
                if site_id == "unknown":
                    site_id = site_metadata.get("site_id", run_dir.name)
                
                # Create display name
                display_name = site_id
                if "data_dir" in config:
                    data_dir = Path(config["data_dir"])
                    if data_dir.exists():
                        # Try to get location from resolved_places.json
                        places_file = Path("data/resolved_places.json")
                        if places_file.exists():
                            import json
                            with open(places_file) as f:
                                places = json.load(f)
                            for place in places:
                                place_site_id = f"{place['latitude']}_{place['longitude']}"
                                if place_site_id == site_id:
                                    display_name = place.get("display_name", site_id)
                                    break
                
                results["display_name"] = display_name
                results["site_id"] = site_id
                all_results[run_dir.name] = results
        except Exception as e:
            print(f"Error loading {run_dir}: {e}")
            continue
    
    return all_results


def create_comparison_mfa_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create comparison figure for MFA across multiple places."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Multifractal Spectrum f(α)",
            "Mass Exponent τ(q)",
            "Generalized Dimensions D(q)",
            "D(0), D(1), D(2) Comparison"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F39C12", "#9B59B6",
        "#E74C3C", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22",
        "#95A5A6", "#34495E", "#16A085", "#27AE60", "#2980B9",
        "#8E44AD", "#C0392B", "#D35400"
    ]
    
    for idx, (run_id, results) in enumerate(all_results.items()):
        if "mfa_spectrum" not in results:
            continue
        
        display_name = results.get("display_name", run_id)
        color = colors[idx % len(colors)]
        
        df = results["mfa_spectrum"]
        
        # f(α) vs α
        fig.add_trace(
            go.Scatter(
                x=df["alpha"],
                y=df["f_alpha"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=display_name,
                showlegend=True,
            ),
            row=1, col=1
        )
        
        # τ(q) vs q
        fig.add_trace(
            go.Scatter(
                x=df["q"],
                y=df["tau"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=display_name,
                showlegend=False,
            ),
            row=1, col=2
        )
        
        # D(q) vs q
        if "mfa_dimensions" in results:
            dq = results["mfa_dimensions"]
            fig.add_trace(
                go.Scatter(
                    x=dq["q"],
                    y=dq["D_q"],
                    mode="lines+markers",
                    name=display_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    legendgroup=display_name,
                    showlegend=False,
                ),
                row=2, col=1
            )
    
    # D(0), D(1), D(2) comparison bar chart
    place_names = []
    d0_values = []
    d1_values = []
    d2_values = []
    
    for run_id, results in all_results.items():
        if "mfa_dimensions" not in results:
            continue
        
        display_name = results.get("display_name", run_id)
        dq = results["mfa_dimensions"]
        
        d0 = dq[dq["q"] == 0]["D_q"].values[0] if 0 in dq["q"].values else None
        d1 = dq[abs(dq["q"] - 1) < 0.5]["D_q"].values[0] if len(dq[abs(dq["q"] - 1) < 0.5]) > 0 else None
        d2 = dq[dq["q"] == 2]["D_q"].values[0] if 2 in dq["q"].values else None
        
        if d0 is not None and d1 is not None and d2 is not None:
            place_names.append(display_name)
            d0_values.append(d0)
            d1_values.append(d1)
            d2_values.append(d2)
    
    if place_names:
        fig.add_trace(
            go.Bar(name="D(0)", x=place_names, y=d0_values, marker_color="#FF6B6B"),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name="D(1)", x=place_names, y=d1_values, marker_color="#4ECDC4"),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name="D(2)", x=place_names, y=d2_values, marker_color="#45B7D1"),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(x=1.02, y=1, xanchor="left"),
    )
    
    fig.update_xaxes(title_text="α (Hölder exponent)", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="f(α)", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="q (moment order)", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="τ(q)", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="q", row=2, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="D(q)", row=2, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="Place", row=2, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Dimension", row=2, col=2, gridcolor="#2a2a4e")
    
    return fig


def create_comparison_lacunarity_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create comparison figure for Lacunarity across multiple places."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Lacunarity Curve Λ(r)", "Power Law Exponent β Comparison"),
        horizontal_spacing=0.12,
    )
    
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F39C12", "#9B59B6",
        "#E74C3C", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22",
        "#95A5A6", "#34495E", "#16A085", "#27AE60", "#2980B9",
        "#8E44AD", "#C0392B", "#D35400"
    ]
    
    place_names = []
    beta_values = []
    
    for idx, (run_id, results) in enumerate(all_results.items()):
        if "lacunarity" not in results:
            continue
        
        display_name = results.get("display_name", run_id)
        color = colors[idx % len(colors)]
        
        df = results["lacunarity"]
        
        # Lacunarity curve
        fig.add_trace(
            go.Scatter(
                x=df["r"],
                y=df["lambda"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            ),
            row=1, col=1
        )
        
        # Collect beta values
        if "lacunarity_fit" in results:
            fit = results["lacunarity_fit"]
            place_names.append(display_name)
            beta_values.append(fit.get("beta", 0))
    
    # Beta comparison bar chart
    if place_names:
        fig.add_trace(
            go.Bar(
                x=place_names,
                y=beta_values,
                marker_color="#F39C12",
                name="β",
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(x=1.02, y=1, xanchor="left"),
    )
    
    fig.update_xaxes(title_text="Box size r [px]", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Λ(r)", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="Place", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="β", row=1, col=2, gridcolor="#2a2a4e")
    
    return fig


def create_comparison_percolation_figure(all_results: dict[str, dict[str, Any]]) -> go.Figure:
    """Create comparison figure for Percolation across multiple places."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Giant Component Fraction", "Critical Distance d_c Comparison"),
        horizontal_spacing=0.12,
    )
    
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#F39C12", "#9B59B6",
        "#E74C3C", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22",
        "#95A5A6", "#34495E", "#16A085", "#27AE60", "#2980B9",
        "#8E44AD", "#C0392B", "#D35400"
    ]
    
    place_names = []
    d_critical_values = []
    
    for idx, (run_id, results) in enumerate(all_results.items()):
        if "percolation" not in results:
            continue
        
        display_name = results.get("display_name", run_id)
        color = colors[idx % len(colors)]
        
        df = results["percolation"]
        
        # Giant component fraction
        fig.add_trace(
            go.Scatter(
                x=df["d"],
                y=df["giant_fraction"],
                mode="lines+markers",
                name=display_name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ),
            row=1, col=1
        )
        
        # Collect critical distances
        if "percolation_stats" in results:
            stats = results["percolation_stats"]
            d_critical = stats.get("d_critical_50", None)
            if d_critical is not None:
                place_names.append(display_name)
                d_critical_values.append(d_critical)
    
    # Critical distance comparison bar chart
    if place_names:
        fig.add_trace(
            go.Bar(
                x=place_names,
                y=d_critical_values,
                marker_color="#9B59B6",
                name="d_c (50%)",
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
        legend=dict(x=1.02, y=1, xanchor="left"),
    )
    
    fig.update_xaxes(title_text="Distance threshold d [m]", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="Giant fraction", row=1, col=1, gridcolor="#2a2a4e")
    fig.update_xaxes(title_text="Place", row=1, col=2, gridcolor="#2a2a4e")
    fig.update_yaxes(title_text="d_c [m]", row=1, col=2, gridcolor="#2a2a4e")
    
    return fig


def create_raster_gallery(all_results: dict[str, dict[str, Any]], max_places: int = 17) -> list[go.Figure]:
    """Create gallery of raster images for all places."""
    figures = []
    
    # Limit number of places for performance
    places_to_show = list(all_results.items())[:max_places]
    
    for run_id, results in places_to_show:
        display_name = results.get("display_name", run_id)
        
        if "buildings_raster" not in results and "roads_raster" not in results:
            continue
        
        # Create subplot for this place
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{display_name} - Buildings", f"{display_name} - Roads"),
            horizontal_spacing=0.1,
        )
        
        # Buildings
        if "buildings_raster" in results:
            buildings = results["buildings_raster"]
            # Downsample for gallery view
            step = max(1, buildings.shape[0] // 200)
            buildings_small = buildings[::step, ::step]
            fig.add_trace(
                go.Heatmap(
                    z=buildings_small,
                    colorscale="gray",
                    showscale=False,
                ),
                row=1, col=1
            )
        
        # Roads
        if "roads_raster" in results:
            roads = results["roads_raster"]
            # Downsample for gallery view
            step = max(1, roads.shape[0] // 200)
            roads_small = roads[::step, ::step]
            fig.add_trace(
                go.Heatmap(
                    z=roads_small,
                    colorscale="viridis",
                    showscale=False,
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(family="Noto Sans JP, sans-serif", color="#eee", size=10),
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
        
        figures.append(fig)
    
    return figures


def create_network_gallery(all_results: dict[str, dict[str, Any]], max_places: int = 17) -> list[go.Figure]:
    """Create gallery of network graphs for all places."""
    figures = []
    
    # Limit number of places for performance
    places_to_show = list(all_results.items())[:max_places]
    
    for run_id, results in places_to_show:
        display_name = results.get("display_name", run_id)
        
        if "network_path" not in results:
            continue
        
        try:
            network_path = Path(results["network_path"])
            if not network_path.exists():
                continue
            
            G = nx.read_graphml(str(network_path))
            
            if G.number_of_nodes() == 0:
                continue
            
            # Sample nodes for gallery view (smaller than full view)
            max_nodes = 1000
            if G.number_of_nodes() > max_nodes:
                import random
                nodes_to_keep = random.sample(list(G.nodes()), max_nodes)
                G = G.subgraph(nodes_to_keep).copy()
            
            # Use spring layout
            pos = nx.spring_layout(G, k=0.1, iterations=20, seed=42)
            
            # Extract coordinates
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.3, color="#888"),
                hoverinfo='none',
                mode='lines',
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(size=2, color="#4ECDC4"),
                hoverinfo='skip',
            ))
            
            fig.update_layout(
                title=dict(text=display_name, font=dict(size=12)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                height=300,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(family="Noto Sans JP, sans-serif", color="#eee"),
            )
            
            figures.append(fig)
            
        except Exception as e:
            print(f"Error creating network for {display_name}: {e}")
            continue
    
    return figures


def create_comparison_dashboard(outputs_dir: Path | str) -> dash.Dash:
    """Create comparison dashboard for multiple places."""
    outputs_dir = Path(outputs_dir)
    
    # Load all places
    all_results = load_all_places(outputs_dir)
    
    if not all_results:
        # Fallback to single place dashboard
        from .dashboard import create_dashboard
        return create_dashboard(outputs_dir)
    
    app = dash.Dash(
        __name__,
        title="Urban Structure Analysis - Comparison",
        suppress_callback_exceptions=True,
    )
    
    # Custom CSS (same as original)
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
            <style>
                body {
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    min-height: 100vh;
                    margin: 0;
                    font-family: 'Noto Sans JP', sans-serif;
                    color: #eee;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    text-align: center;
                    background: linear-gradient(120deg, #FF6B6B, #4ECDC4, #45B7D1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }
                .subtitle {
                    text-align: center;
                    color: #888;
                    margin-bottom: 30px;
                }
                .section {
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 16px;
                    padding: 20px;
                    margin-bottom: 30px;
                }
                .section h2 {
                    color: #4ECDC4;
                    border-bottom: 2px solid #4ECDC4;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .control-panel {
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 30px;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Create comparison figures
    mfa_fig = create_comparison_mfa_figure(all_results)
    lac_fig = create_comparison_lacunarity_figure(all_results)
    perc_fig = create_comparison_percolation_figure(all_results)
    
    # Create galleries
    raster_gallery = create_raster_gallery(all_results)
    network_gallery = create_network_gallery(all_results)
    
    # Build layout
    place_list = [results.get("display_name", run_id) for run_id, results in all_results.items()]
    
    # Build gallery sections
    gallery_sections = []
    
    # Raster Gallery
    if raster_gallery:
        raster_items = []
        for fig in raster_gallery:
            raster_items.append(
                html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "300px"}
                    )
                ], style={
                    "display": "inline-block",
                    "width": "48%",
                    "margin": "1%",
                    "verticalAlign": "top"
                })
            )
        
        gallery_sections.append(
            html.Div([
                html.H2("Raster Data Gallery"),
                html.Div(raster_items, style={"textAlign": "center"}),
            ], className="section")
        )
    
    # Network Gallery
    if network_gallery:
        network_items = []
        for fig in network_gallery:
            network_items.append(
                html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "300px"}
                    )
                ], style={
                    "display": "inline-block",
                    "width": "48%",
                    "margin": "1%",
                    "verticalAlign": "top"
                })
            )
        
        gallery_sections.append(
            html.Div([
                html.H2("Network Graph Gallery"),
                html.Div(network_items, style={"textAlign": "center"}),
            ], className="section")
        )
    
    app.layout = html.Div([
        html.Div([
            html.H1("🏙️ Urban Structure Analysis - Comparison"),
            html.P(f"Comparing {len(all_results)} places", className="subtitle"),
            
            # Control panel
            html.Div([
                html.H3("Places Included", style={"color": "#4ECDC4", "marginTop": 0}),
                html.Ul([
                    html.Li(name, style={"margin": "5px 0"}) for name in place_list
                ]),
            ], className="control-panel"),
            
            # Galleries (before analysis comparisons)
            *gallery_sections,
            
            # MFA Comparison
            html.Div([
                html.H2("Multifractal Analysis Comparison"),
                dcc.Graph(
                    id="mfa-comparison",
                    figure=mfa_fig,
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ], className="section"),
            
            # Lacunarity Comparison
            html.Div([
                html.H2("Lacunarity Analysis Comparison"),
                dcc.Graph(
                    id="lacunarity-comparison",
                    figure=lac_fig,
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ], className="section"),
            
            # Percolation Comparison
            html.Div([
                html.H2("Percolation Analysis Comparison"),
                dcc.Graph(
                    id="percolation-comparison",
                    figure=perc_fig,
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
            ], className="section"),
            
        ], className="container"),
    ])
    
    return app

