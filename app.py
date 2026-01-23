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

    # Calculate base fit (assuming simple grid packing on floor)
    # Note: Vending bins are usually tight, but we calculate max floor capacity just in case
    # We take the best of Normal vs Rotated for the base layer
    
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
    Core Logic:
    1. Iterate through every item in inventory.
    2. Test against every drawer type.
    3. Select drawer type that minimizes TOTAL DRAWER HEIGHT required.
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

        # Try every drawer type
        for _, drawer in drawer_db_df.iterrows():
            d_id = drawer['DrawerID']
            b_w = drawer['BinWidth']
            b_l = drawer['BinLength']
            b_h = drawer['BinHeight']
            b_qty = drawer['QtyBins']
            
            # Determine Drawer Physical Height (Inches) based on Bin Height (mm) approximation
            # Logic: If bin is > ~80mm it's likely a 6" drawer, else 3". 
            # Or we can rely on user naming. Let's infer from mm height.
            # 3 inches ~ 76mm. 6 inches ~ 152mm.
            if b_h > 100:
                drawer_height_inch = 6
            else:
                drawer_height_inch = 3
                
            items_per_bin = check_fit(i_l, i_w, i_h, b_l, b_w, b_h)
            
            if items_per_bin > 0:
                total_capacity_per_drawer = items_per_bin * b_qty
                drawers_needed = math.ceil(qty_needed / total_capacity_per_drawer)
                
                # "Cost" is the total vertical cabinet space used
                total_height_cost = drawers_needed * drawer_height_inch
                
                # We want to minimize height cost (Primary) and Maximize Density (Secondary)
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
st.markdown("""
This app calculates the optimal drawer configuration for your vending machine.
""")

st.sidebar.header("Data Upload")

# 1. File Uploads
inv_file = st.sidebar.file_uploader("1. Inventory Input (Excel)", type=['xlsx'])
prod_file = st.sidebar.file_uploader("2. Product Database (Excel)", type=['xlsx'])
draw_file = st.sidebar.file_uploader("3. Drawer Database (Excel)", type=['xlsx'])

if inv_file and prod_file and draw_file:
    try:
        # Load Data
        inv_df = pd.read_excel(inv_file)
        prod_df = pd.read_excel(prod_file)
        draw_df = pd.read_excel(draw_file)

        # Standardize Column Names (Strip whitespace from headers)
        inv_df.columns = inv_df.columns.str.strip()
        prod_df.columns = prod_df.columns.str.strip()
        draw_df.columns = draw_df.columns.str.strip()
        
        # --- STAGE 1: DATA ENRICHMENT ---
        st.subheader("Step 1: Data Validation")
        
        # Create match keys
        inv_df['Match_ID'] = clean_code(inv_df['Material ID/ Product Code'])
        prod_df['Match_ID_Mat'] = clean_code(prod_df['Material ID'])
        prod_df['Match_ID_Prod'] = clean_code(prod_df['Product Code'])
        
        # We need to fill missing dimensions in Inventory
        # Check if Dims exist in Input, if not, try to find them
        
        def get_dim(row, col_name):
            # If input has value, use it
            if pd.notnull(row.get(col_name)) and row.get(col_name) != 0:
                return row.get(col_name)
            
            # Look up by Material ID
            match_mat = prod_df[prod_df['Match_ID_Mat'] == row['Match_ID']]
            if not match_mat.empty:
                return match_mat.iloc[0][col_name]
            
            # Look up by Product Code
            match_prod = prod_df[prod_df['Match_ID_Prod'] == row['Match_ID']]
            if not match_prod.empty:
                return match_prod.iloc[0][col_name]
                
            return None

        # Apply lookup
        # Map DB column names to Input names for the lookup function
        # Input: Length (mm) | DB: Length (mm)
        inv_df['Length (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Length (mm)'), axis=1)
        inv_df['Width (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Width (mm)'), axis=1)
        inv_df['Height (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Height (mm)'), axis=1) # Corrected typo "Heigth" to "Height" if standard, but user wrote "Heigth"
        
        # Note: User prompt had "Heigth (mm)". I will handle both spellings just in case.
        if 'Heigth (mm)' in prod_df.columns:
            # Rename for consistency
            prod_df.rename(columns={'Heigth (mm)': 'Height (mm)'}, inplace=True)
            
        # Re-run lookup for height if the column name was tricky
        inv_df['Height (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Height (mm)'), axis=1)

        # Identify Missing Items
        missing_mask = (inv_df['Length (mm)'].isna()) | (inv_df['Width (mm)'].isna()) | (inv_df['Height (mm)'].isna())
        missing_df = inv_df[missing_mask]
        valid_df = inv_df[~missing_mask].copy()

        if not missing_df.empty:
            st.error(f"⚠️ Found {len(missing_df)} items with missing dimensions!")
            st.write("Please download the file below, fill in the dimensions (L, W, H), and re-upload as your Inventory Input.")
            
            # Create download for missing
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                missing_df.to_excel(writer, index=False)
            
            st.download_button(
                label="Download Missing Items File",
                data=buffer,
                file_name="items_missing_dimensions.xlsx",
                mime="application/vnd.ms-excel"
            )
            
            st.warning("Calculation stopped until dimensions are provided.")
            
        else:
            st.success(f"All {len(valid_df)} items matched successfully! Proceeding to optimization.")
            
            # --- STAGE 2: OPTIMIZATION ---
            if st.button("Calculate Drawer Configuration"):
                with st.spinner("Calculating optimal bin packing..."):
                    result_df = optimize_packing(valid_df, draw_df)
                    
                    # --- STAGE 3: AGGREGATION & RESULTS ---
                    st.divider()
                    st.subheader("Results")
                    
                    # Summary Metrics
                    total_drawers = result_df['Drawers Required'].sum()
                    total_height_inches = result_df['Total Height Required (in)'].sum()
                    cabinets_needed = math.ceil(total_height_inches / 33)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Drawers", f"{int(total_drawers)}")
                    col2.metric("Total Vertical Height", f"{int(total_height_inches)}\"")
                    col3.metric("Cabinets Needed (Max 33\")", f"{cabinets_needed}")
                    
                    # Grouped Layout View
                    st.write("### Drawer Breakdown")
                    summary = result_df.groupby(['Best Drawer Type', 'Drawer Height (in)']).agg(
                        Total_Drawers=('Drawers Required', 'sum'),
                        Items_Stored=('Quantity', 'count')
                    ).reset_index()
                    st.dataframe(summary)

                    # Detail View
                    st.write("### Pick List Detail")
                    st.dataframe(result_df)
                    
                    # Download Final Result
                    buffer_res = io.BytesIO()
                    with pd.ExcelWriter(buffer_res, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, sheet_name="Pick List", index=False)
                        summary.to_excel(writer, sheet_name="Summary", index=False)
                        
                    st.download_button(
                        label="Download Final Layout",
                        data=buffer_res,
                        file_name="Vending_Layout_Plan.xlsx",
                        mime="application/vnd.ms-excel"
                    )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check your column headers match the requirements exactly.")

else:
    st.info("Please upload all 3 files to begin.")
    
    with st.expander("See Required Column Headers"):
        st.markdown("""
        **1. Inventory Input:**
        `Material ID/ Product Code`, `Quantity`, `Length (mm)`, `Width (mm)`, `Height (mm)` (optional dims)
        
        **2. Product DB:**
        `Material ID`, `Product Code`, `Length (mm)`, `Width (mm)`, `Height (mm)`
        
        **3. Drawer DB:**
        `DrawerID`, `BinWidth`, `BinLength`, `BinHeight`, `QtyBins`
        """)
