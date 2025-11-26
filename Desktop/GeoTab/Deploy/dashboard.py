import os
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta, date
import altair as alt

# --- Load environment variables ---
load_dotenv()

# --- Configuration ---
# Set page configuration with a custom title/subtitle via markdown for the header
# Note: The browser tab title will be "Speeding Insights Dashboard"
st.set_page_config(layout="wide", page_title="Speeding Insights Dashboard")

# --- Date & Timezone Logic ---
WEEKDAYS = ["FRI", "SAT", "SUN", "MON", "TUE", "WED", "THU"]

# Use the user's requested dates as the initial default date range.
INITIAL_START_DATE = date(2025, 11, 25)
INITIAL_END_DATE = date(2025, 12, 4)

# Initialize Session State
if 'start_date' not in st.session_state:
    st.session_state.start_date = INITIAL_START_DATE
if 'end_date' not in st.session_state:
    st.session_state.end_date = INITIAL_END_DATE
if 'active_day_filter' not in st.session_state:
    st.session_state.active_day_filter = "ALL"
if 'is_dark_mode' not in st.session_state:
    st.session_state.is_dark_mode = False # Initialize Dark Mode state
if 'text_color' not in st.session_state:
    st.session_state.text_color = "#1c1c1c" # Initialize text color

# --- Custom CSS Injection for Dark Mode and Alignment ---

def inject_custom_css(is_dark_mode):
    """Injects custom CSS for dark mode styling and chart alignment."""
    
    # Define color scheme
    if is_dark_mode:
        # Dark Mode Colors (deep and clean)
        bg_color = "#0e1117" # Main background
        secondary_bg_color = "#1f2532" # Sidebar, Alert backgrounds
        text_color = "#e6e6e6" # Light text
    else:
        # Light Mode Colors (crisp and bright)
        bg_color = "white" # Main background
        secondary_bg_color = "#f0f2f6" # Sidebar, Alert backgrounds
        text_color = "#1c1c1c" # Dark text
    
    # Store text color in session state for Altair chart labels
    st.session_state.text_color = text_color 

    # Use f-strings to inject dynamic variables
    st.markdown(f"""
        <style>
        /* 1. Global Background and Text Color Toggle */
        .main .block-container {{
            background-color: {bg_color};
            color: {text_color};
            transition: background-color 0.3s, color 0.3s;
        }}
        
        /* Apply text color to headings for consistency */
        h1, h2, h3, h4, .markdown-text-container {{
            color: {text_color} !important;
        }}

        /* Adjust the sidebar and sticky header background */
        .stSidebar {{
            background-color: {secondary_bg_color} !important;
        }}
        #dashboard-header {{
            background-color: {bg_color} !important;
        }}
        
        /* 2. Footer Style */
        .footer {{
            text-align: center;
            padding: 20px;
            margin-top: 50px;
            font-size: small;
            color: gray;
        }}

        /* Style the st.info box in both modes */
        .stAlert {{
            background-color: {secondary_bg_color};
            border-left-color: #4CAF50; /* Green for info */
            color: {text_color};
        }}
        .stAlert p, .stAlert h4 {{
            color: {text_color} !important; /* Ensure text inside alert is visible */
        }}
        
        /* Force buttons to respect the color scheme */
        .stButton>button {{
            color: {text_color};
            border: 1px solid {text_color};
            background-color: {bg_color};
        }}
        
        /* 4. Custom Flex Layout for Severity Distribution (Chart + Legend) */
        /* Targets the st.columns wrapper to apply flex properties */
        /* This is a common selector for Streamlit containers */
        .st-emotion-cache-p2sy0v > div:has(> .st-emotion-cache-10qik03) {{ 
            display: flex;
            justify-content: center;
            align-items: center; /* Vertical alignment for all content inside the columns */
            gap: 40px;
        }}
        
        /* Custom styling for the metric definitions expander in the sidebar */
        .stExpander {{
            border: 1px solid {text_color}44;
            border-radius: 8px;
            padding: 0;
            background-color: {secondary_bg_color};
        }}

        </style>
        """, unsafe_allow_html=True)


# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    """Creates and returns a SQLAlchemy engine instance using environment variables."""
    PG_HOST = os.getenv('PGHOST')
    PG_PORT = os.getenv('PGPORT', '5432')
    PG_DB = os.getenv('PGDB')
    PG_USER = os.getenv('PGUSER')
    PG_PASS = os.getenv('PGPASS')
    
    if not all([PG_HOST, PG_DB, PG_USER, PG_PASS]):
        st.error("Missing one or more required PostgreSQL environment variables.")
        st.stop()

    conn_string = f'postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}'
    return create_engine(conn_string)

# --- Data Fetching ---
@st.cache_data(ttl=300) # Cache data for 5 minutes
def load_speeding_data(start_date: date, end_date: date):
    """
    Loads speeding events and aggregates total distance driven by vehicle/driver 
    for the specified date window.
    """
    engine = get_db_connection()
    
    query = text(f"""
        WITH EventData AS (
            SELECT
                se.event_ts,
                se.driver_id,
                se.vehicle_id,
                se.shift_number,
                se.actual_speed_kmh,
                se.speed_limit_kmh,
                se.severity,
                se.location_text,
                se.duration_seconds,
                se.distance_km, 
                se.latitude,
                se.longitude,
                se.needs_verification,
                d.full_name AS driver_name,
                v.asset_number AS vehicle_asset
            FROM 
                ops.speeding_events se
            JOIN 
                ops.vehicles v ON se.vehicle_id = v.vehicle_id
            LEFT JOIN 
                ops.drivers d ON se.driver_id = d.driver_id
            WHERE
                se.event_ts >= :start_date AND se.event_ts <= :end_date + interval '1 day'
        ),
        
        VehicleDistance AS (
            SELECT 
                vehicle_id, 
                COALESCE(SUM(distance_km), 1000.0) AS total_distance_km
            FROM 
                ops.speeding_events
            WHERE
                event_ts >= :start_date AND event_ts <= :end_date + interval '1 day'
            GROUP BY 1
        )
        
        SELECT
            ed.*,
            COALESCE(vd.total_distance_km, 1000.0) AS total_distance_km
        FROM
            EventData ed
        LEFT JOIN
            VehicleDistance vd ON ed.vehicle_id = vd.vehicle_id;
    """)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
            
            # --- FIX: Ensure event_ts is a datetime object immediately ---
            df['event_ts'] = pd.to_datetime(df['event_ts'])
            
            # Post-processing in Pandas
            df['event_date'] = df['event_ts'].dt.date
            # Map day names for our dashboard buttons (FRI, SAT, SUN, MON, TUE, WED, THU)
            df['day_name'] = df['event_ts'].dt.day_name().str[:3].str.upper() 
            
            # Calculate severity labels
            severity_map = {1: 'Low (0-5 km/h over)', 2: 'Medium (5-10 km/h over)', 3: 'High (>10 km/h over)'}
            df['severity_label'] = df['severity'].map(severity_map).fillna('N/A')
            
            # Calculate Risk Score: Weighted sum of event counts
            df['risk_weight'] = df['severity'].fillna(0) * 1.5 
            
            # Fill NaNs for display
            df['driver_name'] = df['driver_name'].fillna('Unassigned')
            df['shift_number'] = df['shift_number'].fillna('N/A')
            
            return df
            
    except Exception as e:
        # Catch and report any database connection or query errors
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()


# --- UI Components ---

def create_sticky_header(data_available_days):
    """
    Implements the title, subtitle, and day selector buttons.
    Uses data_available_days to grey out buttons with no data.
    """
    
    # 1. Title and Subtitle - Updated to INZU and using id for CSS targeting
    st.markdown("""
        <div id='dashboard-header' style='position: sticky; top: 0; padding-top: 10px; padding-bottom: 10px; z-index: 1000;'>
            <h1 style='margin-bottom: 0px;'>INZU Speeding Insights Dashboard</h1>
            <p style='margin-top: 0px; color: gray;'>Weekly overview of driver behaviour, risk, and operational impact</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Day Selector Buttons
    st.markdown("---")
    
    # Use all unique day names present in the data for the button set
    all_known_days = sorted(list(set(WEEKDAYS) | set(data_available_days.keys())))
    days_to_display = all_known_days + ["ALL"]
    
    col_days = st.columns(len(days_to_display))
    
    for i, day in enumerate(days_to_display):
        button_label = day
        is_active = st.session_state.active_day_filter == day
        
        # Check if data is available for this specific day
        if day != "ALL":
            is_data_available = data_available_days.get(day, False)
        else:
            is_data_available = any(data_available_days.values())
        
        # Conditional style based on data availability
        disabled_state = not is_data_available
        type_color = "primary" if is_active and is_data_available else "secondary"
        
        # Using a click handler that only reruns if not disabled
        if col_days[i].button(button_label, 
                              use_container_width=True, 
                              type=type_color, 
                              disabled=disabled_state, 
                              key=f"day_btn_{day}"):
            if not disabled_state:
                st.session_state.active_day_filter = day
                st.rerun()

    st.markdown("---")

def filter_data_by_day(df):
    """Filters the main DataFrame based on the active day filter."""
    day_filter = st.session_state.active_day_filter
    
    if day_filter == "ALL":
        return df
    
    df_filtered = df[df['day_name'] == day_filter]
    
    if df_filtered.empty:
        # Fallback: If no data is found for the selected day, display a warning and show the full week's data.
        st.info(f"No speeding events recorded for the selected day ({day_filter}). Displaying full date range (ALL).")
        st.session_state.active_day_filter = "ALL"
        return df
        
    return df_filtered

# --- Section 0: STAKEHOLDER HIGH-LEVEL SUMMARY ---
def section_0_stakeholder_summary(df_filtered):
    """
    Shows the top 5 vehicles based on the maximum speed recorded,
    along with the speed limit for that event, providing a high-level alert.
    This view respects the Date Range and Day Filter but ignores sidebar filters.
    """
    st.header("Stakeholder Attention: Max Speed Alerts")
    
    if df_filtered.empty:
        st.warning("No speeding events recorded in the selected period to generate max speed alerts.")
        st.markdown("---")
        return

    # 1. Find the single event with the highest speed for each vehicle
    # Use idxmax to find the index of the max speed for each vehicle_asset
    idx = df_filtered.groupby('vehicle_asset')['actual_speed_kmh'].idxmax()
    max_speed_events = df_filtered.loc[idx].copy()

    # 2. Calculate the 'speed_over_limit'
    max_speed_events['speed_over_limit'] = max_speed_events['actual_speed_kmh'] - max_speed_events['speed_limit_kmh']

    # 3. Sort by Max Speed and take the top 5
    top_vehicles = max_speed_events.sort_values(by='actual_speed_kmh', ascending=False).head(5)
    
    # 4. Prepare data for Altair chart display
    # Sort ascending for Altair bar chart to display largest bar at the top
    chart_data = top_vehicles[[
        'vehicle_asset', 
        'actual_speed_kmh', 
        'speed_limit_kmh', 
        'speed_over_limit'
    ]].sort_values(by='actual_speed_kmh', ascending=True) 
    
    # Ensure all columns are correctly typed for Altair and readable labels
    chart_data['Max Speed (km/h)'] = chart_data['actual_speed_kmh'].astype(float).round(0).astype(int)
    chart_data['Limit (km/h)'] = chart_data['speed_limit_kmh'].astype(float).round(0).astype(int)
    chart_data['Over Limit (km/h)'] = chart_data['speed_over_limit'].astype(float).round(0).astype(int)

    # 5. Create the Altair Bar Chart
    base = alt.Chart(chart_data).encode(
        y=alt.Y('vehicle_asset', title='Vehicle Asset', sort='-x'),
        tooltip=[
            alt.Tooltip('vehicle_asset', title='Vehicle'),
            alt.Tooltip('Max Speed (km/h)'),
            alt.Tooltip('Limit (km/h)'),
            alt.Tooltip('Over Limit (km/h)')
        ]
    ).properties(
        title="Top 5 Vehicles by Maximum Recorded Speed",
        height=250 # Ensure a compact, prominent look
    )

    # Main bar showing the Max Speed
    bars = base.mark_bar(opacity=0.8, color='#FF4B4B').encode( # Using red for high alert
        x=alt.X('Max Speed (km/h)', title='Max Speed (km/h)'),
    )
    
    # Text labels for the Max Speed value
    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3 # Nudges text to the right
    ).encode(
        text=alt.Text('Max Speed (km/h)'),
        # Use session state for dynamic text color (set in inject_custom_css)
        color=alt.value(st.session_state.text_color) 
    )

    # Combine chart components
    chart = (bars + text).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("---")


# --- Section A: THE BIG PICTURE ---

def section_a_big_picture(df_filtered, df_full_week):
    st.header("A. The Big Picture – Overview")
    st.subheader(f"High-Level Metrics ({st.session_state.start_date} to {st.session_state.end_date})")
    
    total_events = len(df_filtered)
    total_severe_events = len(df_filtered[df_filtered['severity'] == 3])
    unique_drivers = df_filtered['driver_name'].nunique()
    unique_vehicles = df_filtered['vehicle_asset'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Speeding Events", total_events)
    col2.metric("Severe Events (Severity 3)", total_severe_events)
    col3.metric("Drivers Involved", unique_drivers)
    col4.metric("Vehicles Involved", unique_vehicles)
    
    st.markdown("---")

    # ROW 1: Event Trend by Day and Events by Shift (2 equal-width columns)
    col_charts_top = st.columns(2)
    
    # 1. Trend Line (Date Range)
    with col_charts_top[0]:
        st.subheader("Event Trend by Day")
        trend_data = df_full_week.groupby('event_date').size().reset_index(name='Events')
        
        window_start = st.session_state.start_date
        window_end = st.session_state.end_date
        full_date_range = pd.date_range(start=window_start, end=window_end)
        
        trend_data['event_date'] = pd.to_datetime(trend_data['event_date'])
        
        trend_data = trend_data.set_index('event_date').reindex(full_date_range).fillna(0).rename_axis('Date').reset_index()
        trend_data['Day'] = trend_data['Date'].dt.strftime('%m-%d')
        
        # Use Altair to set explicit height (250) for symmetry with Shift chart
        trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
            x=alt.X('Day', title='Date'),
            y=alt.Y('Events', title='Total Events'),
            tooltip=['Day', 'Events']
        ).properties(
            height=250 # Match Shift Chart height
        ).interactive()
        
        st.altair_chart(trend_chart, use_container_width=True)

    # 2. Shift Comparison Bar Chart (Altair Chart - Height 250)
    with col_charts_top[1]:
        st.subheader("Events by Shift")
        shift_data = df_filtered.groupby('shift_number').size().reset_index(name='Events')
        
        # Use Altair to color bars by shift number, using a categorical color scheme
        shift_chart = alt.Chart(shift_data).mark_bar().encode(
            x=alt.X('shift_number', title='Shift'),
            y=alt.Y('Events', title='Total Events'),
            color=alt.Color('shift_number', title='Shift Number', scale=alt.Scale(scheme='category10')), 
            tooltip=['shift_number', 'Events']
        ).properties(
            height=250 
        ).interactive()

        st.altair_chart(shift_chart, use_container_width=True)
        
    st.markdown("---")
        
    # ROW 2: Severity Distribution (Single centered chart, now in a 60/40 layout)
    st.subheader("Severity Distribution")

    # Fetch data including numeric severity for sorting the manual legend
    severity_data = df_filtered.groupby(['severity', 'severity_label']).size().reset_index(name='Count')
    
    # Colors for the manual legend (matching standard Altair Category10 for consistency)
    COLOR_MAP = {
        'Low (0-5 km/h over)': '#1f77b4',      # Severity 1 (Blue)
        'Medium (5-10 km/h over)': '#ff7f0e',  # Severity 2 (Orange)
        'High (>10 km/h over)': '#2ca02c',     # Severity 3 (Green)
    }

    # Define chart height for visual balance
    chart_height = 350
    outer_radius_ratio = chart_height * 0.45 
    inner_radius_ratio = chart_height * 0.15 
    
    # 1. Chart Column (60%) and Manual Legend Column (40%)
    # Custom CSS targets the container of these columns for flex centering.
    chart_col, legend_col = st.columns([6, 4]) 

    with chart_col:
        # Create Altair Donut Chart (with no legend as we are creating a manual one)
        base = alt.Chart(severity_data).encode(
            theta=alt.Theta("Count", stack=True)
        )
        
        pie = base.mark_arc(outerRadius=outer_radius_ratio, innerRadius=inner_radius_ratio).encode( 
            color=alt.Color("severity_label", 
                            title=None, 
                            scale=alt.Scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values())),
                            legend=None), # CRITICAL: Disable legend in Altair
            order=alt.Order("Count", sort="descending"),
            tooltip=["severity_label", "Count", alt.Tooltip("Count", format=',')]
        ).properties(
            height=chart_height,
        ).interactive() 

        st.altair_chart(pie, use_container_width=True)


    with legend_col:
        # Calculate percentages
        total_count = severity_data['Count'].sum()
        severity_data['Percentage'] = (severity_data['Count'] / total_count * 100).round(1)

        # Sort data by numeric severity level (1, 2, 3) for consistent legend order
        sorted_severity = severity_data.sort_values(by='severity', ascending=True)

        # Create the custom legend HTML/Markdown using safer single-line concatenation
        legend_html = "<div style='width: 100%; display: flex; flex-direction: column; justify-content: center;'>"
        
        for _, row in sorted_severity.iterrows():
            label = row['severity_label']
            count = row['Count']
            percent = row['Percentage']
            color = COLOR_MAP.get(label, 'gray') # Fallback color
            
            # Construct the item in a single, well-formed f-string line for better stability
            item_html = (
                f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                f'<div style="width: 15px; height: 15px; background-color: {color}; border-radius: 4px; margin-right: 10px; flex-shrink: 0;"></div>'
                f'<div style="flex-grow: 1;">'
                f'<strong style="font-size: 16px;">{label}</strong><br>'
                f'<span style="font-size: 14px; color: gray;">{count} Events ({percent}%)</span>'
                f'</div>'
                f'</div>'
            )
            
            legend_html += item_html
            
        legend_html += "</div>"
        
        # st.markdown is used to render the concatenated HTML string
        st.markdown(legend_html, unsafe_allow_html=True)
        
# Placeholder functions for other sections (remaining logic is unchanged)
def section_b_driver_insights(df_filtered, df_full_week):
    st.header("B. Where is the Risk Coming From? – Driver Insights")
    driver_summary = df_filtered.groupby('driver_name').agg(
        total_events=('event_ts', 'size'),
        severe_events=('severity', lambda x: (x == 3).sum()),
        total_distance_km=('total_distance_km', 'first'),
        total_risk_weight=('risk_weight', 'sum')
    ).reset_index()
    driver_summary['normalized_rate'] = (driver_summary['total_events'] / (driver_summary['total_distance_km'] / 100)).round(2).replace([np.inf, -np.inf], 0)
    driver_summary['risk_score'] = driver_risk_score = driver_summary['total_risk_weight']
    
    shift_dom = df_full_week.groupby(['driver_name', 'shift_number']).size().reset_index(name='Count')
    shift_dom['rank'] = shift_dom.groupby('driver_name')['Count'].rank(method='first', ascending=False)
    dominant_shift = shift_dom[shift_dom['rank'] == 1][['driver_name', 'shift_number']].rename(columns={'shift_number': 'shift_dominance'})
    
    driver_summary = pd.merge(driver_summary, dominant_shift, on='driver_name', how='left')
    
    risk_table = driver_summary.sort_values('risk_score', ascending=False).rename(columns={
        'normalized_rate': 'Events / 100 km',
        'shift_dominance': 'Shift Dominance',
        'total_events': 'Total Events',
        'severe_events': 'Severe Events',
        'driver_name': 'Driver name'
    })
    
    st.subheader("Driver Risk Score Table")
    st.dataframe(
        risk_table[['Driver name', 'Total Events', 'Severe Events', 'Events / 100 km', 'Shift Dominance', 'risk_score']],
        use_container_width=True,
        hide_index=True
    )
    st.markdown("---")
    
    st.subheader("Top 5 High-Risk Drivers")
    # Using top_5 generated from risk_table.head(5) which was missing in this function previously.
    top_5 = risk_table.head(5).sort_values('risk_score', ascending=True) 
    st.bar_chart(top_5, x='Driver name', y='risk_score', color="#FF4B4B", use_container_width=True)

def section_c_vehicle_insights(df_filtered):
    st.header("C. Is it the Driver or the Vehicle? – Vehicle Analysis")
    vehicle_summary = df_filtered.groupby('vehicle_asset').agg(
        total_events=('event_ts', 'size'),
        severe_events=('severity', lambda x: (x == 3).sum()),
        total_distance_km=('total_distance_km', 'first'),
        unique_drivers=('driver_name', 'nunique')
    ).reset_index()
    vehicle_summary['normalized_rate'] = (vehicle_summary['total_events'] / (vehicle_summary['total_distance_km'] / 100)).round(2).replace([np.inf, -np.inf], 0)

    st.subheader("Events Per Vehicle (Normalized Rate)")
    rate_chart_data = vehicle_summary.sort_values('normalized_rate', ascending=False).head(10)
    st.bar_chart(rate_chart_data, x='vehicle_asset', y='normalized_rate', use_container_width=True)

    st.subheader("Detailed Vehicle Performance")
    vehicle_table = vehicle_summary.rename(columns={
        'vehicle_asset': 'Vehicle',
        'total_events': 'Total Events',
        'severe_events': 'Severe Events',
        'unique_drivers': 'Drivers Operating',
        'normalized_rate': 'Events / 100 km'
    })
    
    st.dataframe(
        vehicle_table[['Vehicle', 'Drivers Operating', 'Severe Events', 'Events / 100 km']],
        use_container_width=True,
        hide_index=True
    )

def section_d_location_insights(df_filtered):
    st.header("D. Where is it Happening? – Hotspot Map")
    st.subheader("Geographical Hotspots (Color-coded by Severity)")
    map_data = df_filtered.dropna(subset=['latitude', 'longitude']).copy()
    map_data = map_data.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    map_data['size'] = map_data['severity'] * 5 
    
    if not map_data.empty:
        st.map(map_data, latitude='lat', longitude='lon', size='size', use_container_width=True)
    else:
        st.warning("No location data available for the current filter.")
        
    st.markdown("---")

    # This table is now explicitly outside any st.columns() block to use full width.
    st.subheader("Top 10 Hotspot Locations")
    hotspot_summary = df_filtered.groupby('location_text').agg(
        Count=('event_ts', 'size'),
        Avg_Severity=('severity', 'mean'),
        Avg_Hour=('event_ts', lambda x: x.dt.hour.mean())
    ).reset_index()
    hotspot_summary = hotspot_summary.sort_values('Count', ascending=False).head(10)
    hotspot_summary['Avg Severity'] = hotspot_summary['Avg_Severity'].round(1)
    hotspot_summary['Avg Time of Day'] = (hotspot_summary['Avg_Hour'] % 24).astype(int).astype(str) + ":00"
    
    st.dataframe(
        hotspot_summary[['location_text', 'Count', 'Avg Severity', 'Avg Time of Day']],
        use_container_width=True,
        hide_index=True
    )

def section_e_time_analysis(df_filtered):
    st.header("E. When is it Happening? – Time Patterns")
    df_filtered['event_hour'] = df_filtered['event_ts'].dt.hour
    heatmap_data = df_filtered.groupby(['day_name', 'event_hour']).size().reset_index(name='Events')
    pivot = heatmap_data.pivot_table(index='event_hour', columns='day_name', values='Events').fillna(0)
    
    st.subheader("Events by Hour of Day Heatmap")
    st.dataframe(pivot)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cumulative Severity Line")
        cumulative_severity = df_filtered.sort_values('event_ts').copy()
        cumulative_severity['cumulative_risk'] = cumulative_severity['risk_weight'].cumsum()
        st.line_chart(cumulative_severity, x='event_ts', y='cumulative_risk', use_container_width=True)

    with col2:
        st.subheader("Peak Speeding Window")
        peak_hour = df_filtered['event_hour'].mode().iloc[0] if not df_filtered.empty else 0
        st.metric(f"Peak Hour (Events)", f"{peak_hour:02}:00 - {peak_hour+1:02}:00")

def section_f_event_details(df_filtered):
    st.header("F. Event Details – Drilldown")
    st.subheader(f"Event Log ({len(df_filtered)} Records)")
    display_cols = df_filtered[[
        'event_ts', 
        'driver_name', 
        'vehicle_asset', 
        'shift_number', 
        'severity_label', 
        'actual_speed_kmh', 
        'speed_limit_kmh', 
        'duration_seconds',
        'location_text', 
        'needs_verification'
    ]].rename(columns={
        'event_ts': 'Time',
        'driver_name': 'Driver',
        'vehicle_asset': 'Vehicle',
        'shift_number': 'Shift',
        'severity_label': 'Severity',
        'actual_speed_kmh': 'Actual Speed',
        'speed_limit_kmh': 'Limit',
        'duration_seconds': 'Duration (s)',
        'location_text': 'Location',
        'needs_verification': 'Verified?'
    })
    
    st.dataframe(display_cols, use_container_width=True, hide_index=True)

def section_g_management_summary(df_filtered, driver_summary):
    st.header("G. Management Summary")
    
    if df_filtered.empty:
        st.info("No data to generate summary for the selected period.")
        return

    top_drivers = driver_summary.sort_values('risk_score', ascending=False).head(3)
    top_driver_names = ", ".join(top_drivers['driver_name'].tolist()) 
    top_events = top_drivers['total_events'].sum()
    total_events = len(df_filtered)
    percent_contribution = (top_events / total_events * 100).round(1) if total_events > 0 else 0
    
    shift_counts = df_filtered.groupby('shift_number').size()
    risky_shift = shift_counts.idxmax() if not shift_counts.empty else "N/A"
    
    top_location = df_filtered['location_text'].mode().iloc[0] if not df_filtered.empty else "N/A"
    
    summary_text = f"""
    This period's analysis ({st.session_state.start_date} to {st.session_state.end_date}) highlights key areas for operational focus:

    - **Risk Concentration:** The top **3 high-risk drivers** ({top_driver_names}) contributed to **{percent_contribution}%** of the total speeding events recorded in the filtered period.
    - **Shift Pressure:** **Shift {risky_shift}** experienced the highest frequency of incidents. Targeted supervision or shift rebalancing may be required.
    - **Location Alert:** The most dangerous hotspot this period was identified as **{top_location}**, suggesting a review of site speed limits, terrain, or route design is necessary.
    - **Cost Impact:** While hard costs are variable, the cumulative **{len(df_filtered)}** events represent significant risk and potential fuel/maintenance overhead.
    """
    st.info(summary_text)

# --- Main Application Loop ---

def main():
    
    # --- Inject Custom CSS based on Dark Mode state ---
    # This call now also sets st.session_state.text_color
    inject_custom_css(st.session_state.is_dark_mode)
    
    # --- Date Range Selector and Global Filters ---
    
    st.sidebar.header("Date Range Selection")
    
    # 1. Date Range Selector
    selected_start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state.start_date,
        key='start_date_input'
    )
    selected_end_date = st.sidebar.date_input(
        "End Date",
        value=st.session_state.end_date,
        key='end_date_input'
    )
    
    # Check if dates were manually changed
    if selected_start_date != st.session_state.start_date or selected_end_date != st.session_state.end_date:
        st.session_state.start_date = selected_start_date
        st.session_state.end_date = selected_end_date
        st.session_state.active_day_filter = "ALL" # Reset day filter on new range
        st.rerun() # Rerun to load new data

    window_start = st.session_state.start_date
    window_end = st.session_state.end_date

    # --- Toggle Dark/Light Mode Button ---
    st.sidebar.markdown("---")
    
    mode_label = "Switch to Light Mode" if st.session_state.is_dark_mode else "Switch to Dark Mode"
    if st.sidebar.button(mode_label, use_container_width=True, key='dark_mode_toggle_button'):
        st.session_state.is_dark_mode = not st.session_state.is_dark_mode
        st.rerun()
    st.sidebar.markdown("---")
    
    # --- Metric Definitions Expander (Key that explains metrics) ---
    with st.sidebar.expander("🔑 Key Metrics Explained", expanded=False):
        st.markdown("**Risk Score**")
        st.markdown("Quantifies overall safety using a **weighted sum** of all risky events based on severity. ***Lower is better.***")
        
        st.markdown("**Events/100km**")
        st.markdown("The raw frequency of incidents, **normalized by distance**. This allows fair comparison between drivers/vehicles regardless of total mileage.")
        
        st.markdown("**Normalized Rate**")
        st.markdown("A general term for any metric adjusted by exposure (distance, time, context) to allow for *fair comparisons*.")
        
        st.markdown("**Shift Dominance**")
        st.markdown("Identifies the **driver** or **shift** responsible for the majority of vehicle usage (mileage/events) in a given period.")
        
    st.sidebar.markdown("---")


    # --- Data Loading (Full window_start to window_end) ---
    df_full_week = load_speeding_data(window_start, window_end)
    
    # --- Determine Data Availability Status ---
    if not df_full_week.empty:
        available_day_names = df_full_week['day_name'].unique().tolist()
        data_available_days = {day: day in available_day_names for day in available_day_names}
    else:
        data_available_days = {day: False for day in WEEKDAYS}
        st.error(f"Cannot load dashboard. Database connection failed or no data between {window_start} and {window_end}.")
        return

    # --- TOP HEADER AND DAY SELECTION ---
    create_sticky_header(data_available_days)

    # --- Apply Day Filter (This result is used for the Stakeholder summary) ---
    current_filter_has_data = data_available_days.get(st.session_state.active_day_filter, False)
    if st.session_state.active_day_filter != "ALL" and not current_filter_has_data:
        st.session_state.active_day_filter = "ALL"
        
    df_filtered_by_day = filter_data_by_day(df_full_week)
    
    # --- SECTION 0: STAKEHOLDER HIGH-LEVEL SUMMARY ---
    # This section uses df_filtered_by_day (Date + Day filter applied, ignores sidebar filters)
    section_0_stakeholder_summary(df_filtered_by_day)

    # --- SIDEBAR CONTENT (Filters applied to the day-filtered data) ---
    
    st.sidebar.header("Data Filters (Applied to Selected Day/Range)")
    
    # 1. Driver Dropdown
    all_drivers = sorted(df_filtered_by_day['driver_name'].unique().tolist())
    selected_drivers = st.sidebar.multiselect("Select Driver(s)", options=all_drivers, default=[d for d in all_drivers if d != 'Unassigned'])
    
    # 2. Vehicle Dropdown
    all_vehicles = sorted(df_filtered_by_day['vehicle_asset'].unique().tolist())
    selected_vehicles = st.sidebar.multiselect("Select Vehicle(s)", options=all_vehicles, default=all_vehicles)
    
    # 3. Severity Selector
    all_severity_labels = sorted(df_filtered_by_day['severity_label'].dropna().unique().tolist())
    selected_severity = st.sidebar.multiselect("Select Severity Level(s)", options=all_severity_labels, default=all_severity_labels)

    # 4. Shift Selector
    all_shifts = sorted(df_filtered_by_day['shift_number'].unique().tolist())
    selected_shifts = st.sidebar.multiselect("Select Shift(s)", options=all_shifts, default=all_shifts)
    
    # 5. Apply Sidebar Filters
    df_final_filtered = df_filtered_by_day[
        (df_filtered_by_day['driver_name'].isin(selected_drivers)) &
        (df_filtered_by_day['vehicle_asset'].isin(selected_vehicles)) &
        (df_filtered_by_day['severity_label'].isin(selected_severity)) &
        (df_filtered_by_day['shift_number'].isin(selected_shifts))
    ]

    # Check if the final filtered dataset is empty
    if df_final_filtered.empty and len(df_full_week) > 0:
        st.warning("No events match the selected day and filter criteria. Try broadening your sidebar selections.")
        return
    elif df_final_filtered.empty and len(df_full_week) == 0:
        st.warning(f"No events recorded in the reporting window ({window_start} to {window_end}).")
        return


    # --- Dashboard Sections (A-G) ---
    section_a_big_picture(df_final_filtered, df_full_week)
    
    # Must recalculate driver summary on the final filtered set for Sections B & G
    driver_summary = df_final_filtered.groupby('driver_name').agg(
        total_events=('event_ts', 'size'),
        severe_events=('severity', lambda x: (x == 3).sum()),
        total_distance_km=('total_distance_km', 'first'),
        total_risk_weight=('risk_weight', 'sum')
    ).reset_index()
    driver_summary['normalized_rate'] = (driver_summary['total_events'] / (driver_summary['total_distance_km'] / 100)).round(2).replace([np.inf, -np.inf], 0)
    driver_summary['risk_score'] = driver_summary['total_risk_weight']
    
    section_b_driver_insights(df_final_filtered, df_full_week)
    st.markdown("---")
    
    section_c_vehicle_insights(df_final_filtered)
    st.markdown("---")
    
    section_d_location_insights(df_final_filtered)
    st.markdown("---")

    section_e_time_analysis(df_final_filtered)
    st.markdown("---")
    
    section_f_event_details(df_final_filtered)
    st.markdown("---")
    
    section_g_management_summary(df_final_filtered, driver_summary)

    # --- Footer ---
    st.markdown("""
        <div class="footer">
            Made by Taizya Kasitu
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()