import streamlit as st
import pandas as pd
import folium
import numpy as np
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from PIL import Image
import os
import colorsys
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from folium.features import DivIcon
from sklearn.preprocessing import MinMaxScaler
import random
import io


# --- CONFIGURATION ---
st.set_page_config(
    page_title="Indonesia Observation Network",
    page_icon="🗺️",
    layout="wide"
)

# --- LOAD DATA ---
@st.cache_data
def load_site_metadata():
    """Load site metadata from CSV"""
    try:
        # --- Memuat Data ---
        file_path = 'Metadata ALL - Sheet.xlsx'
        xls = pd.ExcelFile(file_path)
        sheets = pd.read_excel(file_path, sheet_name=None)

        # Ambil dan gabungkan data dari semua sheet
        dfs = {
            sheet: xls.parse(sheet)[[
                'id_station', 'name_station', 'nama_propinsi', 'nama_kota', 'kecamatan', 'kelurahan', 'latt_station', 'long_station', 'elv_station', 'status_operasional', 'hp_petugas', 'tgl_pasang', 'addr_instansi', 'data_transport', 'instansi', 'nama_vendor'
            ]].dropna(subset=['latt_station', 'long_station']).assign(JENIS=sheet)
            for sheet in sheets
        }

        # Gabungkan semua data
        df = pd.concat(dfs.values(), ignore_index=True)
        
        return df

    except Exception as e:
        st.error(f"Error loading site metadata: {str(e)}")
        return None

def create_clustered_map(df, selected_station_type='AAWS'):
    """Create a clustered map with individual markers color-coded by province using unique hex colors."""

    # --- Base map setup ---
    center_lat = -2.5
    center_lon = 129.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4.2, tiles=None)
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron').add_to(m)

    # --- Filter by selected station type ---
    filtered_df = df[df['JENIS'] == selected_station_type].copy()

    # --- Assign unique hex colors per province ---
    provinces = sorted(filtered_df['nama_propinsi'].dropna().unique())
    hex_palette = px.colors.qualitative.Alphabet
    color_pool = hex_palette * ((len(provinces) // len(hex_palette)) + 1)
    province_hex = {
        province: color_pool[i]
        for i, province in enumerate(provinces)
    }

    # --- Group by province and add clustered markers ---
    grouped = filtered_df.groupby('nama_propinsi')

    for province, group in grouped:
        color = province_hex[province]
        cluster_group = folium.FeatureGroup(name=province)
        cluster = MarkerCluster().add_to(cluster_group)

        for _, site in group.iterrows():
            popup_content = f"""
            <div style="width: 300px;">
                <h4>{site['name_station']}</h4>
                <hr>
                <b>Site ID:</b> {site['id_station']}<br>
                <b>Province:</b> {site['nama_propinsi']}<br>
                <b>District:</b> {site['nama_kota']}<br>
                <b>Coordinates:</b> {site['latt_station']:.3f}, {site['long_station']:.3f}<br>
                <b>Elevation:</b> {site['elv_station']:.3f} m <br>
                <b>Installation Year:</b> {site['tgl_pasang'].strftime("%m/%d/%Y") if pd.notna(site['tgl_pasang']) else 'N/A'}<br>
                <b>Equipment:</b> {site['nama_vendor'] if pd.notna(site['nama_vendor']) else 'N/A'}<br>
                <b>Address:</b> {site['addr_instansi'] if pd.notna(site['addr_instansi']) else 'N/A'}
            </div>
            """
            folium.CircleMarker(
                location=[site['latt_station'], site['long_station']],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=folium.Popup(popup_content, max_width=350),
                tooltip=f"{site['name_station']} (ID: {site['id_station']})"
            ).add_to(cluster)

        cluster_group.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m

def create_selective_map(df, selected_station_type='AAWS', selected_id_station='10001'):
    """Create an interactive map centered on Indonesia showing stations of the selected type and highlighting the selected station."""

    # --- Center map on Indonesia ---
    center_lat = -2.5
    center_lon = 129.0

    # --- Base map ---
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4.2,
        tiles='OpenStreetMap'
    )

    folium.TileLayer(
        tiles='CartoDB positron',
        attr='© OpenStreetMap contributors © CARTO'
    ).add_to(m)

    # --- Add markers ---
    for idx, site in df.iterrows():
        site_id_str = str(site['id_station'])
        selected_id_str = str(selected_id_station)

        # Decide marker style
        if site_id_str == selected_id_str:
            color = 'red'
            icon = 'star'
        elif site['JENIS'] == selected_station_type:
            color = 'blue'
            icon = 'info-sign'
        else:
            continue  # Skip stations outside selected type

        # Handle possible missing date or address
        if pd.notna(site.get('tgl_pasang')) and hasattr(site['tgl_pasang'], 'strftime'):
            install_date = site['tgl_pasang'].strftime("%m/%d/%Y")
        else:
            install_date = 'N/A'

        popup_content = f"""
        <div style="width: 300px;">
            <h4>{site['name_station']}</h4>
            <hr>
            <b>Site ID:</b> {site_id_str}<br>
            <b>Province:</b> {site.get('nama_propinsi', 'N/A')}<br>
            <b>District:</b> {site.get('nama_kota', 'N/A')}<br>
            <b>Coordinates:</b> {site['latt_station']:.3f}, {site['long_station']:.3f}<br>
            <b>Elevation:</b> {site['elv_station']:.3f} m <br>
            <b>Installation Year:</b> {install_date}<br>
            <b>Equipment:</b> {site.get('nama_vendor', 'N/A')}<br>
            <b>Address:</b> {site.get('addr_instansi', 'N/A')}
        </div>
        """

        folium.Marker(
            location=[site['latt_station'], site['long_station']],
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"{site['name_station']} (ID: {site_id_str})",
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    return m
    
def create_province_distribution_chart(df):
    """Create bar chart showing Station distribution by province"""
    province_counts = df['nama_propinsi'].value_counts()
    
    fig = px.bar(
        x=province_counts.values,
        y=province_counts.index,
        orientation='h',
        title="Stations Distribution by Province",
        labels={'x': 'Number of Sites', 'y': 'Province'},
        color=province_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_equipment_distribution_chart(df):
    """Create pie chart showing equipment brand distribution"""
    df_clean = df.dropna(subset=['nama_vendor'])
    brand_counts = df_clean['nama_vendor'].value_counts()
    
    fig = px.pie(
        values=brand_counts.values,
        names=brand_counts.index,
        title="Equipment Brand Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=400)
    
    return fig

@st.cache_data
def get_filtered_data(df, station_type):
    """Return filtered site data by station type."""
    return df[df["JENIS"] == station_type].copy()

def main():
    st.title("🗺️ Indonesia Observation Network")
    st.markdown("---")
    
    # Load combined site metadata
    df = load_site_metadata()
    
    if df is None:
        return

    # Initialize session state
    if "selected_id_station" not in st.session_state:
        st.session_state.selected_id_station = "10001"  # default site ID as string

    # --- Sidebar: Station Type Summary with Icons ---
    st.sidebar.markdown("### 📊 Station Type Summary")

    # Icon mapping based on context
    type_icons = {
        "AAWS": "🌾",   # Agroclimate
        "AWS": "🌦️",   # Weather
        "ARG": "🌧️",   # Rain Gauge
        "ASRS": "☀️",   # Solar Radiation
        "IKRO": "🌱"    # Micro Climate
    }

    type_counts = df['JENIS'].value_counts().sort_index()

    for station_type, count in type_counts.items():
        icon = type_icons.get(station_type, "📡")
        st.sidebar.write(f"{icon} **{station_type}**: {count} sites")
    
    # --- Sidebar: Filter Controls ---
    st.sidebar.header("🧭 Station Filtering")

    # ✅ Use radio for selecting station type
    station_types = df["JENIS"].unique().tolist()
    selected_type = st.sidebar.radio("Select Station Type", station_types)

    # Filter sites based on selected station type
    filtered_df = get_filtered_data(df, selected_type)

    # Build display names for dropdown
    filtered_df["id_station_str"] = filtered_df["id_station"].astype(str)
    filtered_df["display"] = filtered_df["id_station_str"] + " - " + filtered_df["name_station"]

    # Recalculate default index from session_state
    selected_id_str = st.session_state.selected_id_station
    filtered_display_list = filtered_df["display"].tolist()

    if not filtered_df.empty:
        try:
            default_index = filtered_df[filtered_df['id_station_str'] == selected_id_str].index[0]
            # Convert index to position in the filtered list
            default_index = filtered_df.index.get_loc(default_index)
        except (IndexError, KeyError):
            default_index = 0

        selected_site_display = st.sidebar.selectbox(
            "Select Site for Detailed Analysis:",
            options=filtered_display_list,
            index=default_index,
            key="selected_site_dropdown"
            )

        selected_id_station = selected_site_display.split(" - ")[0]

        # ✅ Only update if the user actually changed it
        if st.session_state.selected_id_station != selected_id_station:
            st.session_state.selected_id_station = selected_id_station

        # Display info
        selected_id_station = selected_site_display.split(" - ")[0]
        selected_site_row = filtered_df[filtered_df["id_station_str"] == selected_id_station]

    else:
        st.sidebar.warning("No sites available for the selected station type.")
        st.info("Please select a different station type.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "🎯 Static Map", "📊 Statistics", "📋 Station Directory"])
    
    with tab1:
        st.subheader("🌍 Indonesia Observation Sites Map")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sites", len(df))
        with col2:
            st.metric("Provinces Covered", df['nama_propinsi'].nunique())
        with col3:
            earliest_date = df['tgl_pasang'].min()
            formatted_date = earliest_date.strftime('%d/%m/%Y') if pd.notna(earliest_date) else "N/A"
            st.metric("Active Since", formatted_date)

        selected_id = st.session_state.selected_id_station
        
        view_mode = st.radio("🗺️ Map Mode", ["Individual Markers", "Clustered Markers"], horizontal=True)

        if view_mode == "Individual Markers":
            map_obj = create_selective_map(
                df=df,
                selected_station_type=selected_type,
                selected_id_station=selected_id
            )
        else:
            map_obj = create_clustered_map(
                df=df,
                selected_station_type=selected_type
            )

        with st.container():
            map_data = st_folium(map_obj, use_container_width=True, height=500)

        st.markdown("---")
        
        # Handle map click
        clicked_station = None
        if map_data:
            last_clicked = map_data.get("last_object_clicked")
            if last_clicked and "lat" in last_clicked and "lng" in last_clicked:
                clicked_lat = last_clicked["lat"]
                clicked_lon = last_clicked["lng"]

                clicked_station = df.loc[
                    ((df['latt_station'] - clicked_lat) ** 2 + (df['long_station'] - clicked_lon) ** 2).idxmin()
                ]

        # Show clicked info and selection button BETWEEN the map and image
        with st.container():
            if clicked_station is not None:
                st.info(f"📍 You clicked: **{clicked_station['name_station']}** (ID: {clicked_station['id_station']})")
                if st.button("🔄 Use this station in selection"):
                    st.session_state.selected_id_station = str(clicked_station['id_station'])
                    st.rerun()
            else:
                st.info("🖱️ Click a station marker on the map to select it to enable selection.")
        
        # Legend
        st.markdown("""
        **Map Legend:**
        - 🔴 **Red Star**: Currently selected station for detailed analysis  
        - 🔵 **Blue Markers**: Other monitoring station  
        - Click on markers for detailed Station information
        """)

        # Get selected site details
        selected_site = df[df['id_station'] == selected_id_station].iloc[0]
        phone_raw = selected_site.get("hp_petugas", "")
        phone_str = str(phone_raw).strip() if pd.notna(phone_raw) else "N/A"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            ### 📍 {selected_site['name_station']}
            
            **Basic Information:**
            - **Station ID:** {selected_site['id_station']}
            - **Station Name:** {selected_site['name_station']}
            - **Data Transport Type:** {selected_site['data_transport'] if pd.notna(selected_site['data_transport']) else 'N/A'} 
            
            **Location:**
            - **Province:** {selected_site['nama_propinsi']}
            - **District:** {selected_site['nama_kota']}
            - **Sub-district:** {selected_site['kecamatan']}
            - **Village:** {selected_site['kelurahan']}
            """)
        
        with col2:         
            st.markdown(f"""
            ### 🌍 Geographic Details
            
            **Coordinates:**
            - **Latitude:** {selected_site['latt_station']:}°
            - **Longitude:** {selected_site['long_station']:}°
            - **Elevation:** {selected_site['elv_station']} m
            
            **Administrative:**
            - **Agency:** {selected_site['instansi'] if pd.notna(selected_site['instansi']) else ''}
            - **Procurement Date:** {selected_site['tgl_pasang'].strftime("%m/%d/%Y")}
            - **Vendor:** {selected_site['nama_vendor']}            
            - **Officer Phone Number:** {phone_str}   
            
            **Address:**
            {selected_site['addr_instansi'] if pd.notna(selected_site['addr_instansi']) else 'N/A'}
            """)

    with tab2:
        st.subheader(f"🎯 Static Map for Selected Station Type")
        
        # Show static image map after interaction
        with st.container():
            image_filename = f"Layout {selected_type}.png"
            if os.path.isfile(image_filename):
                img = Image.open(image_filename)
                img = img.resize((1200, int(img.height * 1200 / img.width)))
                st.image(img, caption="Static Map")

                with open(image_filename, "rb") as f:
                    image_data = f.read()
                st.download_button(
                    label="📥 Download Image (PNG)",
                    data=image_data,
                    file_name=f"{selected_type}_layout_{pd.Timestamp.now().strftime('%Y%m%d')}.png",
                    mime="image/png"
                )
            else:
                st.warning(f"Image not found: {image_filename}")

    with tab3:
        st.subheader("📈 Network Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_province_distribution_chart(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_equipment_distribution_chart(df), use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Network Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Geographic Coverage:**
            - **Northernmost:** {df['latt_station'].max():.3f}°
            - **Southernmost:** {df['latt_station'].min():.3f}°
            - **Easternmost:** {df['long_station'].max():.3f}°
            - **Westernmost:** {df['long_station'].min():.3f}°
            """)
        
        with col2:
            st.info(f"""
            **Administrative Coverage:**
            - **Provinces:** {df['nama_propinsi'].nunique()}
            - **Districts:** {df['nama_kota'].nunique()}
            - **Sub-districts:** {df['kecamatan'].nunique()}
            """)
    
    with tab4:
        st.subheader("📋 Complete Station Directory")
        
        # Search functionality
        search_term = st.text_input("🔍 Search station by name or location:")
        
        if search_term:
            search_df = df[
                df['name_station'].str.contains(search_term, case=False, na=False) |
                df['nama_propinsi'].str.contains(search_term, case=False, na=False) |
                df['nama_kota'].str.contains(search_term, case=False, na=False)
            ]
        else:
            search_df = df

        # Format date to only show DD/MM/YYYY
        search_df['tgl_pasang_str'] = search_df['tgl_pasang'].dt.strftime('%d/%m/%Y')
        
        # Display sites table
        display_columns = ['id_station', 'name_station', 'nama_propinsi', 'nama_kota', 'latt_station', 'long_station', 'elv_station', 'tgl_pasang_str', 'nama_vendor']
        st.dataframe(
            search_df[display_columns].sort_values('id_station'),
            use_container_width=True,
            column_config={
                'id_station': 'Site ID',
                'name_station': 'Site Name',
                'nama_propinsi': 'Province',
                'nama_kota': 'District',
                'latt_station': st.column_config.NumberColumn('Latitude', format="%.3f"),
                'long_station': st.column_config.NumberColumn('Longitude', format="%.3f"),
                'elv_station': 'Ketinggian (m)',
                'tgl_pasang_str': 'Installation Year',
                'nama_vendor': 'Equipment Brand'
            }
        )
        
        # Export functionality
        with st.expander("📥 Download Site Data"):
            st.write("Choose your preferred export format:")

            col1, col2 = st.columns(2)

            with col1:
                # CSV export
                csv = search_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"Observation_Station_Data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            with col2:
                # XLSX export: Directly use the original file
                try:
                    with open("Metadata ALL - Sheet.xlsx", "rb") as f:
                        xlsx_data = f.read()

                    st.download_button(
                        label="⬇️ Download XLSX",
                        data=xlsx_data,
                        file_name=f"Observation_Station_Data_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except FileNotFoundError:
                    st.error("❌ The original Excel file was not found.")

main()
