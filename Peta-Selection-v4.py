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

def create_indonesia_map(df, selected_id_station=str('AWS Ujung Kulon')):
    """Create interactive map of Indonesia with all observation sites"""
    
    # Center map on Indonesia
    center_lat = -2.5
    center_lon = 129.0
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4.2,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers
    folium.TileLayer(
        tiles='CartoDB positron',
        attr='© OpenStreetMap contributors © CARTO'
    ).add_to(m)
    
    # Add markers for each site
    for idx, site in df.iterrows():
        # Determine marker color and size
        if (site['id_station']) == selected_id_station:
            color = 'red'
            icon = 'star'
            size = 15
        else:
            color = 'blue'
            icon = 'info-sign'
            size = 10
        
        # Create popup content
        popup_content = f"""
        <div style="width: 300px;">
            <h4>{site['name_station']}</h4>
            <hr>
            <b>Site ID:</b> {site['id_station']}<br>
            <b>Province:</b> {site['nama_propinsi']}<br>
            <b>District:</b> {site['nama_kota']}<br>
            <b>Coordinates:</b> {site['latt_station']:.3f}, {site['long_station']:.3f}<br>
            <b>Installation Year:</b> {site['tgl_pasang'] if pd.notna(site['tgl_pasang']) else 'N/A'}<br>
            <b>Equipment:</b> {site['nama_vendor'] if pd.notna(site['nama_vendor']) else 'N/A'}<br>
            <b>Address:</b> {site['addr_instansi'] if pd.notna(site['addr_instansi']) else 'N/A'}
        </div>
        """

        # Add marker
        folium.Marker(
            location=[site['latt_station'], site['long_station']],
            popup=folium.Popup(popup_content, max_width=350),
            tooltip=f"{site['name_station']} (ID: {site['id_station']})",
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
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
    """Create bar chart showing site distribution by province"""
    province_counts = df['nama_propinsi'].value_counts()
    
    fig = px.bar(
        x=province_counts.values,
        y=province_counts.index,
        orientation='h',
        title="Microclimate Sites Distribution by Province",
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

    
    # --- Sidebar: Filter Controls ---
    st.sidebar.header("🧭 Station Filtering")

    # ✅ Use radio for selecting station type
    station_types = df["JENIS"].unique().tolist()
    selected_type = st.sidebar.radio("Select Station Type", station_types)

    # Filter sites based on selected station type
    filtered_df = df[df["JENIS"] == selected_type].copy()

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
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Statistics", "📋 Site Directory", "🎯 Selected Site"])
    
    with tab1:
        st.subheader("🌍 Indonesia Observation Sites Map")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sites", len(df))
        with col2:
            st.metric("Provinces Covered", df['nama_propinsi'].nunique())
        with col3:
            st.metric("Active Since", str(df['tgl_pasang'].min()) if df['tgl_pasang'].notna().any() else "N/A")
        
        selected_type = selected_type  # from your radio button
        selected_id = st.session_state.selected_id_station  # from session

        # Use cached map
        map_obj = create_selective_map(
        df=df,
        selected_station_type=selected_type,
        selected_id_station=selected_id_station
        )
        
        map_data = st_folium(map_obj, width=1200, height=600)
        
        # --- Initialize clicked variables ---
        clicked_station = None
        clicked_lat = None
        clicked_lon = None

        # --- Handle map click ---
        if map_data:
            last_clicked = map_data.get("last_object_clicked")

            if last_clicked and "lat" in last_clicked and "lng" in last_clicked:
                clicked_lat = last_clicked["lat"]
                clicked_lon = last_clicked["lng"]

                # ✅ Find the nearest station only when click is valid
                clicked_station = df.loc[
                    ((df['latt_station'] - clicked_lat) ** 2 + (df['long_station'] - clicked_lon) ** 2).idxmin()
                ]

        # --- Display info if station was clicked ---
        if clicked_station is not None:
            st.info(f"📍 You clicked: **{clicked_station['name_station']}** (ID: {clicked_station['id_station']})")

            if st.button("🔄 Use this station in selection"):
                st.session_state.selected_id_station = str(clicked_station['id_station'])
                st.rerun()
        else:
            st.info("🖱️ Click a station marker on the map to select it.")

        # Build filename dynamically
        image_filename = f"Layout {selected_type}.png"

        # Check if the file exists before trying to load it
        if os.path.isfile(image_filename):
            st.image(image_filename, caption=f"Peta Statik Jaringan Peralatan: {image_filename}", width=1200)
        else:
            st.warning(f"Image not found: {image_filename}")
        
        # Map legend
        st.markdown("""
        **Map Legend:**
        - 🔴 **Red Star**: Currently selected station for detailed analysis
        - 🔵 **Blue Markers**: Other monitoring station
        - Click on markers for detailed Station information
        """)

        image_path = f"Layout {selected_type} Tes.png"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = f.read()
            st.download_button(
                label="📥 Download Image (PNG)",
                data=image_data,
                file_name=f"{selected_type}_layout_{pd.Timestamp.now().strftime('%Y%m%d')}.png",
                mime="image/png"
            )
    
    with tab2:
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
    
    with tab3:
        st.subheader("📋 Complete Site Directory")
        
        # Search functionality
        search_term = st.text_input("🔍 Search sites by name or location:")
        
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
        display_columns = ['id_station', 'name_station', 'nama_propinsi', 'nama_kota', 'latt_station', 'long_station', 'tgl_pasang_str', 'nama_vendor']
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
                'tgl_pasang_str': 'Installation Year',
                'nama_vendor': 'Equipment Brand'
            }
        )
        
        # Export functionality

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Site Data"):
                csv = search_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"microclimate_sites_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.subheader(f"🎯 Selected Site Details")
        
        # Get selected site details
        selected_site = df[df['id_station'] == selected_id_station].iloc[0]
        
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
            - **Officer Phone Number:** {selected_site['hp_petugas']}   
            
            **Address:**
            {selected_site['addr_instansi'] if pd.notna(selected_site['addr_instansi']) else 'N/A'}
            """)

main()