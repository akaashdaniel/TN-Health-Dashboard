import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json
from sklearn.preprocessing import MinMaxScaler

#  Page config 
st.set_page_config(
    page_title="TN Health Dashboard",
    page_icon="🏥",
    layout="wide"
)

#  Load data 
@st.cache_data
def load_data():
    df_long    = pd.read_csv("data/processed/hmis_long_clean.csv")
    df_cluster = pd.read_csv("data/processed/district_clusters.csv")
    with open("data/tamilnadu_districts.geojson") as f:
        geojson = json.load(f)
    return df_long, df_cluster, geojson

df_long, df_cluster, geojson = load_data()

#  Sidebar 
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Emblem_of_Tamil_Nadu.svg/200px-Emblem_of_Tamil_Nadu.svg.png", width=80)
st.sidebar.title("TN Health Dashboard")
st.sidebar.markdown("**Data:** NHM HMIS 2018-19")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "District Map", "Indicator Explorer", "ML Clusters"]
)

# PAGE 1: OVERVIEW
if page == "Overview":
    st.title("🏥 Tamil Nadu Public Health Dashboard")
    st.markdown("District-level analysis of NHM HMIS data • 2018-19 • 32 Districts")
    st.markdown("---")

    # KPI cards
    anc_total = df_long[
        df_long['Parameters'].str.contains("Total number of pregnant women registered for ANC", na=False)
    ]['Value'].sum()

    delivery_total = df_long[
        df_long['Indicator'].str.contains("M2", na=False)
    ]['Value'].sum()

    bcg_total = df_long[
        df_long['Parameters'].str.contains("BCG", na=False)
    ]['Value'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total ANC Registrations", f"{anc_total/1e6:.2f}M")
    col2.metric("Total Deliveries", f"{delivery_total/1e6:.2f}M")
    col3.metric("BCG Immunisations", f"{bcg_total/1e6:.2f}M")
    col4.metric("Districts Covered", "32")

    st.markdown("---")

    # ANC bar chart
    st.subheader("ANC Registrations by District")
    anc = df_long[
        df_long['Parameters'].str.contains("Total number of pregnant women registered for ANC", na=False)
    ].dropna(subset=['Value']).sort_values('Value', ascending=True)

    fig = px.bar(anc, x='Value', y='District', orientation='h',
                 color='Value', color_continuous_scale='YlOrRd',
                 labels={'Value': 'Registrations', 'District': ''},
                 height=650)
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: DISTRICT MAP
elif page == "District Map":
    st.title("🗺️ District Health Map")

    indicator_choice = st.selectbox(
        "Select Indicator to Map",
        ["ANC Registrations", "BCG Immunisation", "Family Planning", "C-Section"]
    )

    param_map = {
        "ANC Registrations":  ("Parameters", "Total number of pregnant women registered for ANC"),
        "BCG Immunisation":   ("Parameters", "BCG"),
        "Family Planning":    ("Indicator",  "M8|M18"),
        "C-Section":          ("Indicator",  "M3"),
    }

    col_type, search_str = param_map[indicator_choice]
    if col_type == "Parameters":
        map_data = df_long[
            df_long['Parameters'].str.contains(search_str, na=False)
        ].groupby('District')['Value'].sum().reset_index()
    else:
        map_data = df_long[
            df_long['Indicator'].str.contains(search_str, na=False)
        ].groupby('District')['Value'].sum().reset_index()

    m = folium.Map(location=[11.0, 78.5], zoom_start=7, tiles="CartoDB positron")
    folium.Choropleth(
        geo_data=geojson,
        data=map_data,
        columns=['District', 'Value'],
        key_on='feature.properties.DISTRICT',
        fill_color='YlOrRd',
        fill_opacity=0.75,
        line_opacity=0.5,
        legend_name=indicator_choice,
        nan_fill_color='lightgray'
    ).add_to(m)
    folium.GeoJson(
        geojson,
        tooltip=folium.GeoJsonTooltip(fields=['DISTRICT'], aliases=['District:'])
    ).add_to(m)

    st_folium(m, width=900, height=500)

# PAGE 3: INDICATOR EXPLORER

elif page == "Indicator Explorer":
    st.title("📊 Indicator Explorer")

    indicators = df_long['Indicator'].dropna().unique().tolist()
    selected_indicator = st.selectbox("Select Health Indicator", sorted(indicators))

    params = df_long[df_long['Indicator'] == selected_indicator]['Parameters'].dropna().unique()
    selected_param = st.selectbox("Select Parameter", params)

    filtered = df_long[
        (df_long['Indicator'] == selected_indicator) &
        (df_long['Parameters'] == selected_param)
    ].dropna(subset=['Value']).sort_values('Value', ascending=True)

    if len(filtered) > 0:
        fig = px.bar(
            filtered,
            x='Value', y='District', orientation='h',
            color='Value', color_continuous_scale='Blues',
            title=f"{selected_param[:80]}...",
            height=650
        )
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Table")
        st.dataframe(
            filtered[['District', 'Value']].sort_values('Value', ascending=False),
            use_container_width=True
        )
    else:
        st.warning("No data found for this selection.")

# PAGE 4: ML CLUSTERS

elif page == "ML Clusters":
    st.title("🤖 District Health Clusters (K-Means)")
    st.markdown("Districts grouped by health performance using K-Means clustering (K=3, Silhouette=0.519)")

    # Scatter
    fig = px.scatter(
        df_cluster,
        x='ANC_Registrations', y='Total_Deliveries',
        color='Cluster_Label', text='District',
        title='District Clusters: ANC vs Total Deliveries',
        color_discrete_map={
            'High Volume Metro':           '#2ecc71',
            'Urban Outlier (Coimbatore)':  '#f39c12',
            'Developing Districts':         '#e74c3c'
        },
        size='C_Section', size_max=35, height=550
    )
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.subheader("Health Profile by Cluster")
    categories = ['ANC_Registrations', 'Total_Deliveries',
                  'BCG_Immunisation', 'Family_Planning', 'C_Section']
    cat_labels  = ['ANC', 'Deliveries', 'BCG', 'Family Planning', 'C-Section']

    radar_data = df_cluster.groupby('Cluster_Label')[categories].mean()
    mms = MinMaxScaler()
    radar_scaled = pd.DataFrame(
        mms.fit_transform(radar_data),
        index=radar_data.index, columns=cat_labels
    )

    colors = {
        'High Volume Metro':           '#2ecc71',
        'Urban Outlier (Coimbatore)':  '#f39c12',
        'Developing Districts':         '#e74c3c'
    }
    fig_r = go.Figure()
    for name, row in radar_scaled.iterrows():
        vals = row.tolist() + [row.tolist()[0]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=cat_labels + [cat_labels[0]],
            fill='toself', name=name,
            line_color=colors[name], opacity=0.6
        ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=450
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Cluster table
    st.subheader("District Assignments")
    for label in df_cluster['Cluster_Label'].unique():
        districts = df_cluster[df_cluster['Cluster_Label'] == label]['District'].tolist()
        st.markdown(f"**{label}** ({len(districts)} districts)")
        st.write(", ".join(districts))