import streamlit as st
import pandas as pd
import math
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Vending Cabinet Optimizer", layout="wide")

# --- HELPER FUNCTIONS ---

def clean_code(series):
    """Removes spaces from a pandas series (column) for matching."""
    return series.astype(str).str.replace(" ", "").str.strip().str.upper()

def check_fit(item_l, item_w, item_h, bin_l, bin_w, bin_h):
    """
    Determines how many items fit in a bin considering rotation and stacking.
    Returns the maximum number of items per single bin.
    """
    # Check 1: Vertical Height (Stacking)
    if item_h > bin_h:
        return 0
    vertical_stack = math.floor(bin_h / item_h)
    
    # Check 2: Footprint (Rotation Allowed)
    # Option A: Normal Orientation
    fits_normal = (item_l <= bin_l and item_w <= bin_w)
    
    # Option B: Rotated Orientation
    fits_rotated = (item_w <= bin_l and item_l <= bin_w)
    
    if not (fits_normal or fits_rotated):
        return 0

    count_normal = 0
    if fits_normal:
        count_normal = math.floor(bin_l / item_l) * math.floor(bin_w / item_w)
        
    count_rotated = 0
    if fits_rotated:
        count_rotated = math.floor(bin_l / item_w) * math.floor(bin_w / item_l)
        
    base_count = max(count_normal, count_rotated)
    
    return base_count * vertical_stack

def optimize_packing(inventory_df, drawer_db_df):
    """
    Core Logic: Minimize TOTAL DRAWER HEIGHT required.
    """
    results = []
    
    # Ensure Drawer DB columns are numeric
    cols_to_num = ['BinWidth', 'BinLength', 'BinHeight', 'QtyBins']
    for col in cols_to_num:
        drawer_db_df[col] = pd.to_numeric(drawer_db_df[col], errors='coerce')

    for index, row in inventory_df.iterrows():
        item_code = row['Material ID/ Product Code']
        qty_needed = row['Quantity']
        
        # Item Dims
        i_l = float(row['Length (mm)'])
        i_w = float(row['Width (mm)'])
        i_h = float(row['Height (mm)'])
        
        best_drawer = None
        min_total_height_cost = float('inf')
        best_qty_per_drawer = 0
        best_drawer_qty_needed = 0
        best_drawer_height_inch = 0

        # Try every drawer type
        for _, drawer in drawer_db_df.iterrows():
            d_id = drawer['DrawerID']
            b_w = drawer['BinWidth']
            b_l = drawer['BinLength']
            b_h = drawer['BinHeight']
            b_qty = drawer['QtyBins']
            
            # Infer Drawer Height (3" vs 6")
            if b_h > 100:
                drawer_height_inch = 6
            else:
                drawer_height_inch = 3
                
            items_per_bin = check_fit(i_l, i_w, i_h, b_l, b_w, b_h)
            
            if items_per_bin > 0:
                total_capacity_per_drawer = items_per_bin * b_qty
                drawers_needed = math.ceil(qty_needed / total_capacity_per_drawer)
                
                # Cost function: Total vertical cabinet inches used
                total_height_cost = drawers_needed * drawer_height_inch
                
                if total_height_cost < min_total_height_cost:
                    min_total_height_cost = total_height_cost
                    best_drawer = d_id
                    best_qty_per_drawer = total_capacity_per_drawer
                    best_drawer_qty_needed = drawers_needed
                    best_drawer_height_inch = drawer_height_inch

        # Record Result
        if best_drawer:
            results.append({
                "Material ID/ Product Code": item_code,
                "Quantity": qty_needed,
                "Best Drawer Type": best_drawer,
                "Drawer Height (in)": best_drawer_height_inch,
                "Items Per Drawer": best_qty_per_drawer,
                "Drawers Required": best_drawer_qty_needed,
                "Total Height Required (in)": min_total_height_cost
            })
        else:
            results.append({
                "Material ID/ Product Code": item_code,
                "Quantity": qty_needed,
                "Best Drawer Type": "NO FIT",
                "Drawer Height (in)": 0,
                "Items Per Drawer": 0,
                "Drawers Required": 0,
                "Total Height Required (in)": 0
            })
            
    return pd.DataFrame(results)

# --- MAIN APP LAYOUT ---

st.title("📦 Vending Machine Cabinet Optimizer")

st.sidebar.header("Data Upload")
inv_file = st.sidebar.file_uploader("1. Inventory Input (Excel)", type=['xlsx'])
prod_file = st.sidebar.file_uploader("2. Product Database (Excel)", type=['xlsx'])
draw_file = st.sidebar.file_uploader("3. Drawer Database (Excel)", type=['xlsx'])

if inv_file and prod_file and draw_file:
    try:
        # Load Data
        inv_df = pd.read_excel(inv_file)
        prod_df = pd.read_excel(prod_file)
        draw_df = pd.read_excel(draw_file)

        # Standardize Columns
        inv_df.columns = inv_df.columns.str.strip()
        prod_df.columns = prod_df.columns.str.strip()
        draw_df.columns = draw_df.columns.str.strip()
        
        # --- STAGE 1: MATCHING ---
        st.subheader("Step 1: Data Matching")
        
        # Create match keys
        inv_df['Match_ID'] = clean_code(inv_df['Material ID/ Product Code'])
        prod_df['Match_ID_Mat'] = clean_code(prod_df['Material ID'])
        prod_df['Match_ID_Prod'] = clean_code(prod_df['Product Code'])
        
        # Lookup Logic
        def get_dim(row, col_name):
            if pd.notnull(row.get(col_name)) and row.get(col_name) != 0:
                return row.get(col_name)
            
            match_mat = prod_df[prod_df['Match_ID_Mat'] == row['Match_ID']]
            if not match_mat.empty:
                return match_mat.iloc[0][col_name]
            
            match_prod = prod_df[prod_df['Match_ID_Prod'] == row['Match_ID']]
            if not match_prod.empty:
                return match_prod.iloc[0][col_name]
            return None

        # Handling Typo in Height (Just in case)
        if 'Heigth (mm)' in prod_df.columns:
            prod_df.rename(columns={'Heigth (mm)': 'Height (mm)'}, inplace=True)
        if 'Heigth (mm)' in inv_df.columns:
            inv_df.rename(columns={'Heigth (mm)': 'Height (mm)'}, inplace=True)

        inv_df['Length (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Length (mm)'), axis=1)
        inv_df['Width (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Width (mm)'), axis=1)
        inv_df['Height (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Height (mm)'), axis=1)

        # Separate Valid and Missing
        missing_mask = (inv_df['Length (mm)'].isna()) | (inv_df['Width (mm)'].isna()) | (inv_df['Height (mm)'].isna())
        missing_df = inv_df[missing_mask].copy()
        valid_df = inv_df[~missing_mask].copy()

        # --- DISPLAY STATUS ---
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.success(f"✅ {len(valid_df)} items matched successfully.")
        
        final_df_to_process = pd.DataFrame()
        ready_to_calculate = False

        if not missing_df.empty:
            col_stat2.error(f"⚠️ {len(missing_df)} items missing dimensions.")
            
            st.markdown("### 🛠️ Resolve Missing Items")
            st.info("You can either fix the missing items by uploading a corrected file, or skip them.")
            
            col_act1, col_act2 = st.columns([1, 1])
            
            with col_act1:
                st.markdown("**Option A: Fix Missing Data**")
                # 1. Download
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    missing_df.to_excel(writer, index=False)
                st.download_button(
                    label="1. Download Missing Items File",
                    data=buffer,
                    file_name="items_missing_dimensions.xlsx",
                    mime="application/vnd.ms-excel"
                )
                # 2. Upload Fix
                fixed_file = st.file_uploader("2. Upload Corrected File", type=['xlsx'], key="fix_upload")
                
            with col_act2:
                st.markdown("**Option B: Ignore Missing**")
                skip_missing = st.checkbox("Skip missing items and calculate only valid ones")

            # Logic to Determine Final Dataset
            if fixed_file:
                try:
                    fixed_df = pd.read_excel(fixed_file)
                    # We assume user filled in the columns. Combine with valid_df
                    final_df_to_process = pd.concat([valid_df, fixed_df], ignore_index=True)
                    st.success(f"Fixed file uploaded! Total items to process: {len(final_df_to_process)}")
                    ready_to_calculate = True
                except Exception as e:
                    st.error("Error reading fixed file. Please ensure columns match.")
            
            elif skip_missing:
                final_df_to_process = valid_df
                st.warning(f"Skipping missing items. Processing {len(final_df_to_process)} valid items.")
                ready_to_calculate = True
            
            else:
                st.stop() # Stop execution until user takes action
                
        else:
            # No missing items
            final_df_to_process = valid_df
            ready_to_calculate = True

        # --- STAGE 2: OPTIMIZATION ---
        if ready_to_calculate:
            st.divider()
            if st.button("🚀 Calculate Drawer Configuration", type="primary"):
                with st.spinner("Optimizing bin packing..."):
                    result_df = optimize_packing(final_df_to_process, draw_df)
                    
                    # --- STAGE 3: RESULTS ---
                    st.subheader("Results")
                    
                    # Metrics
                    total_drawers = result_df['Drawers Required'].sum()
                    total_height_inches = result_df['Total Height Required (in)'].sum()
                    cabinets_needed = math.ceil(total_height_inches / 33)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Drawers", f"{int(total_drawers)}")
                    m2.metric("Total Vertical Height", f"{int(total_height_inches)}\"")
                    m3.metric("Cabinets Needed (Max 33\")", f"{cabinets_needed}")
                    
                    # Breakdowns
                    t1, t2 = st.tabs(["Summary", "Detailed Pick List"])
                    
                    with t1:
                        summary = result_df.groupby(['Best Drawer Type', 'Drawer Height (in)']).agg(
                            Total_Drawers=('Drawers Required', 'sum'),
                            Items_Stored=('Quantity', 'count')
                        ).reset_index()
                        st.dataframe(summary, use_container_width=True)
                    
                    with t2:
                        st.dataframe(result_df, use_container_width=True)
                    
                    # Download
                    buffer_res = io.BytesIO()
                    with pd.ExcelWriter(buffer_res, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, sheet_name="Pick List", index=False)
                        summary.to_excel(writer, sheet_name="Summary", index=False)
                        
                    st.download_button(
                        label="📥 Download Final Layout Report",
                        data=buffer_res,
                        file_name="Vending_Layout_Plan.xlsx",
                        mime="application/vnd.ms-excel"
                    )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Waiting for file uploads...")
