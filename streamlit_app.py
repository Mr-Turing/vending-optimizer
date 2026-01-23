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
    Determines how many items fit in a SINGLE bin.
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

def consolidate_drawers(item_results, drawer_db_drawers):
    """
    Advanced Logic:
    Tries to move items from 'Overflow' drawers (new drawers with very few items)
    into the 'Empty Space' of other existing drawers.
    """
    # 1. Setup Drawer Lookups
    drawer_map = drawer_db_drawers.set_index('DrawerID').to_dict('index')
    
    # 2. Calculate Initial State (Usage per Drawer Type)
    # Group items by assigned drawer
    df = pd.DataFrame(item_results)
    if df.empty or 'Type of drawer' not in df.columns:
        return item_results # No optimization possible
        
    # Helper to recalc state
    def get_drawer_state(current_results):
        state = {}
        for d_id, props in drawer_map.items():
            # Find items assigned to this drawer
            items = [r for r in current_results if r['Type of drawer'] == d_id]
            total_bins = sum(r['quantity of bins needed'] for r in items)
            capacity = props['QtyBins']
            
            drawers_needed = math.ceil(total_bins / capacity)
            bins_available_total = drawers_needed * capacity
            free_slots = bins_available_total - total_bins
            
            # Identify the "Overflow": How many bins are in the last drawer?
            remainder = total_bins % capacity
            if remainder == 0 and total_bins > 0: remainder = capacity # Full
            if total_bins == 0: remainder = 0
            
            state[d_id] = {
                'items': items,
                'total_bins': total_bins,
                'drawers_needed': drawers_needed,
                'free_slots': free_slots,
                'overflow_bins': remainder,
                'is_overflow': (remainder > 0 and remainder < capacity) # It's not perfectly full
            }
        return state

    # 3. Iterative Consolidation Loop
    # We try to empty the smallest overflows first.
    max_passes = 3 # Avoid infinite loops
    for _ in range(max_passes):
        state = get_drawer_state(item_results)
        changes_made = False
        
        # Sort drawer types by overflow size (ascending) -> Try to eliminate small overflows first
        # Only look at types that actually have an overflow to eliminate
        candidates = sorted(
            [d for d, s in state.items() if s['is_overflow'] and s['total_bins'] > 0],
            key=lambda k: state[k]['overflow_bins']
        )
        
        for source_id in candidates:
            source_info = state[source_id]
            # We want to move items from this source to ANY destination that has free slots
            
            # Sort items in this drawer to find small ones (easier to move)
            # We effectively want to move the "overflow" amount. 
            # Since items are just a list, we try to move ANY item that helps reduce the count.
            movable_items = sorted(source_info['items'], key=lambda x: x['quantity of bins needed'])
            
            for item in movable_items:
                # Can we move this item?
                # Try all potential destinations
                best_dest = None
                
                # Look for destinations with enough free slots
                possible_dests = [d for d, s in state.items() if s['free_slots'] > 0]
                
                for dest_id in possible_dests:
                    if dest_id == source_id: continue # Don't move to self
                    
                    dest_props = drawer_map[dest_id]
                    dest_state = state[dest_id]
                    
                    # 1. Check Physical Fit in new drawer type
                    # We need original dimensions which are not in item_results directly.
                    # We stored them or need to access them? 
                    # Ah, we need to pass the raw dimensions through.
                    # Let's assume item_results has a hidden dict or we rely on re-lookup?
                    # Better: Add hidden dims to item_results in main loop.
                    
                    if '_raw_dims' not in item: continue
                    i_l, i_w, i_h = item['_raw_dims']
                    
                    new_fit = check_fit(i_l, i_w, i_h, 
                                      dest_props['BinLength'], 
                                      dest_props['BinWidth'], 
                                      dest_props['BinHeight'])
                    
                    if new_fit > 0:
                        # 2. Calculate Bins Needed in new drawer
                        qty_req = item['Quantity Requested']
                        new_bins_needed = math.ceil(qty_req / new_fit)
                        
                        # 3. Check if it fits in FREE slots (Don't create new drawers)
                        if new_bins_needed <= dest_state['free_slots']:
                            # Found a valid move!
                            best_dest = dest_id
                            break # Take the first valid destination
                
                if best_dest:
                    # EXECUTE MOVE
                    dest_props = drawer_map[best_dest]
                    
                    # Recalc bins for this item
                    i_l, i_w, i_h = item['_raw_dims']
                    new_fit = check_fit(i_l, i_w, i_h, dest_props['BinLength'], dest_props['BinWidth'], dest_props['BinHeight'])
                    new_bins = math.ceil(item['Quantity Requested'] / new_fit)
                    
                    # Update Item Record
                    item['Type of drawer'] = best_dest
                    item['quantity per bin'] = new_fit
                    item['quantity of bins needed'] = new_bins
                    item['_drawer_height'] = 6 if dest_props['BinHeight'] > 100 else 3
                    
                    # Update State manually to reflect move (so we can move more items in this pass)
                    state[source_id]['total_bins'] -= item['quantity of bins needed'] # Old bins (wait, this is tricky, we overwrote it)
                    # Correct approach: We must update state carefully or just break and re-loop
                    # To be safe, let's mark change and Break to re-calculate state fresh
                    changes_made = True
                    break 
            
            if changes_made: break
        
        if not changes_made:
            break # No more optimizations possible
            
    return item_results

def optimize_packing(inventory_df, drawer_db_full, enable_consolidation=True):
    """
    Logic: Find best drawer share for each item, then aggregate.
    """
    item_results = []
    
    # 1. Separate Costs from Drawer Definitions
    cost_rows = drawer_db_full[drawer_db_full['BinWidth'].isna()]
    drawer_db_drawers = drawer_db_full[drawer_db_full['BinWidth'].notna()].copy()
    
    # Extract Base Costs
    base_cabinet_cost = 0
    shipping_cost = 0
    
    for _, row in cost_rows.iterrows():
        name = str(row['DrawerID']).lower()
        price = float(row['Price']) if pd.notnull(row['Price']) else 0
        if "base" in name:
            base_cabinet_cost = price
        elif "shipping" in name:
            shipping_cost = price
            
    # Ensure Drawer DB columns are numeric
    cols_to_num = ['BinWidth', 'BinLength', 'BinHeight', 'QtyBins', 'Price']
    for col in cols_to_num:
        if col == 'Price':
             drawer_db_drawers[col] = pd.to_numeric(drawer_db_drawers[col], errors='coerce').fillna(0)
        else:
             drawer_db_drawers[col] = pd.to_numeric(drawer_db_drawers[col], errors='coerce')

    # Optimization Loop
    for index, row in inventory_df.iterrows():
        item_code = row['Material ID/ Product Code']
        
        # Round Quantity UP
        raw_qty = float(row['Quantity'])
        qty_needed = math.ceil(raw_qty) 
        
        # Item Dims
        i_l = float(row['Length (mm)'])
        i_w = float(row['Width (mm)'])
        i_h = float(row['Height (mm)'])
        
        best_drawer_id = None
        min_cost_share = float('inf')
        
        selected_bins_needed = 0
        selected_items_per_bin = 0
        selected_drawer_height = 0

        # Try every drawer type
        for _, drawer in drawer_db_drawers.iterrows():
            d_id = drawer['DrawerID']
            b_w = drawer['BinWidth']
            b_l = drawer['BinLength']
            b_h = drawer['BinHeight']
            b_qty_slots = drawer['QtyBins'] 
            
            # Infer Drawer Height
            if b_h > 100:
                drawer_height_inch = 6
            else:
                drawer_height_inch = 3
                
            items_per_bin = check_fit(i_l, i_w, i_h, b_l, b_w, b_h)
            
            if items_per_bin > 0:
                bins_needed = math.ceil(qty_needed / items_per_bin)
                
                # Cost share calculation (Minimize Vertical Space Share)
                drawer_usage_fraction = bins_needed / b_qty_slots
                height_cost_share = drawer_usage_fraction * drawer_height_inch
                
                if height_cost_share < min_cost_share:
                    min_cost_share = height_cost_share
                    best_drawer_id = d_id
                    selected_bins_needed = bins_needed
                    selected_items_per_bin = items_per_bin
                    selected_drawer_height = drawer_height_inch

        # Record Result
        if best_drawer_id:
            item_results.append({
                "Material ID/ Product Code": item_code,
                "Quantity Requested": qty_needed,
                "Type of drawer": best_drawer_id,
                "quantity per bin": selected_items_per_bin,
                "quantity of bins needed": selected_bins_needed,
                "_drawer_height": selected_drawer_height,
                "_raw_dims": (i_l, i_w, i_h) # Store for consolidation
            })
        else:
            item_results.append({
                "Material ID/ Product Code": item_code,
                "Quantity Requested": qty_needed,
                "Type of drawer": "NO FIT",
                "quantity per bin": 0,
                "quantity of bins needed": 0,
                "_drawer_height": 0,
                "_raw_dims": (0,0,0)
            })

    # --- POST-PROCESSING: CONSOLIDATION ---
    if enable_consolidation:
        item_results = consolidate_drawers(item_results, drawer_db_drawers)
            
    # --- AGGREGATION STEP ---
    results_df = pd.DataFrame(item_results)
    
    # Filter valid
    valid_results = results_df[results_df['Type of drawer'] != "NO FIT"]
    
    # Lookups
    drawer_caps = drawer_db_drawers.set_index('DrawerID')['QtyBins'].to_dict()
    drawer_prices = drawer_db_drawers.set_index('DrawerID')['Price'].to_dict()

    # Updated Groupby
    grouped = valid_results.groupby(['Type of drawer', '_drawer_height'])
    
    summary_list = []
    total_cabinet_height = 0
    total_drawer_cost = 0
    
    for (d_type, d_height), group in grouped:
        total_bins_needed = group['quantity of bins needed'].sum()
        bins_per_drawer = drawer_caps.get(d_type, 1)
        price_per_drawer = drawer_prices.get(d_type, 0)
        
        drawers_count = math.ceil(total_bins_needed / bins_per_drawer)
        type_height_total = drawers_count * d_height
        type_cost_total = drawers_count * price_per_drawer
        
        total_cabinet_height += type_height_total
        total_drawer_cost += type_cost_total
        
        summary_list.append({
            "Drawer Type": d_type,
            "Drawer Height": d_height,
            "Total Bins Used": total_bins_needed,
            "Drawers Required": drawers_count,
            "Unit Price": f"${price_per_drawer:,.2f}",
            "Total Price": type_cost_total, 
            "Vertical Space (in)": type_height_total
        })
        
    summary_df = pd.DataFrame(summary_list)
    
    # Calculate Cabinet Globals
    cabinets_needed = math.ceil(total_cabinet_height / 33) if total_cabinet_height > 0 else 0
    total_base_cost = cabinets_needed * base_cabinet_cost
    total_shipping_cost = cabinets_needed * shipping_cost
    grand_total = total_drawer_cost + total_base_cost + total_shipping_cost
    
    cost_summary = {
        "cabinets_needed": cabinets_needed,
        "total_height": total_cabinet_height,
        "drawer_subtotal": total_drawer_cost,
        "base_subtotal": total_base_cost,
        "shipping_subtotal": total_shipping_cost,
        "grand_total": grand_total,
        "unit_base": base_cabinet_cost,
        "unit_shipping": shipping_cost
    }
    
    # Clean output columns
    display_df = results_df.drop(columns=['_drawer_height', '_raw_dims'])
    
    return display_df, summary_df, cost_summary

# --- MAIN APP LAYOUT ---

st.title("📦 Vending Machine Cabinet Optimizer")

st.sidebar.header("Data Upload")
inv_file = st.sidebar.file_uploader("1. Inventory Input (Excel)", type=['xlsx'])
prod_file = st.sidebar.file_uploader("2. Product Database (Excel)", type=['xlsx'])
draw_file = st.sidebar.file_uploader("3. Drawer Database (Excel/CSV)", type=['xlsx', 'csv'])

st.sidebar.divider()
st.sidebar.header("Settings")
use_topoff = st.sidebar.checkbox("Enable Cabinet Top-Off", value=True, help="Tries to fit overflow items into empty spaces of other drawers to save cabinets.")

if inv_file and prod_file and draw_file:
    try:
        # Load Data
        inv_df = pd.read_excel(inv_file)
        prod_df = pd.read_excel(prod_file)
        
        if draw_file.name.endswith('.csv'):
            draw_df = pd.read_csv(draw_file)
        else:
            draw_df = pd.read_excel(draw_file)

        # Standardize Columns
        inv_df.columns = inv_df.columns.str.strip()
        prod_df.columns = prod_df.columns.str.strip()
        draw_df.columns = draw_df.columns.str.strip()
        
        # --- STAGE 1: MATCHING ---
        st.subheader("Step 1: Data Matching")
        
        inv_df['Match_ID'] = clean_code(inv_df['Material ID/ Product Code'])
        prod_df['Match_ID_Mat'] = clean_code(prod_df['Material ID'])
        prod_df['Match_ID_Prod'] = clean_code(prod_df['Product Code'])
        
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

        if 'Heigth (mm)' in prod_df.columns:
            prod_df.rename(columns={'Heigth (mm)': 'Height (mm)'}, inplace=True)
        if 'Heigth (mm)' in inv_df.columns:
            inv_df.rename(columns={'Heigth (mm)': 'Height (mm)'}, inplace=True)

        inv_df['Length (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Length (mm)'), axis=1)
        inv_df['Width (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Width (mm)'), axis=1)
        inv_df['Height (mm)'] = inv_df.apply(lambda x: get_dim(x, 'Height (mm)'), axis=1)

        missing_mask = (inv_df['Length (mm)'].isna()) | (inv_df['Width (mm)'].isna()) | (inv_df['Height (mm)'].isna())
        missing_df = inv_df[missing_mask].copy()
        valid_df = inv_df[~missing_mask].copy()

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
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    missing_df.to_excel(writer, index=False)
                st.download_button("1. Download Missing Items File", buffer, "items_missing_dimensions.xlsx")
                fixed_file = st.file_uploader("2. Upload Corrected File", type=['xlsx'], key="fix_upload")
                
            with col_act2:
                st.markdown("**Option B: Ignore Missing**")
                skip_missing = st.checkbox("Skip missing items and calculate only valid ones")

            if fixed_file:
                try:
                    fixed_df = pd.read_excel(fixed_file)
                    final_df_to_process = pd.concat([valid_df, fixed_df], ignore_index=True)
                    st.success(f"Fixed file uploaded! Total items to process: {len(final_df_to_process)}")
                    ready_to_calculate = True
                except Exception as e:
                    st.error("Error reading fixed file.")
            elif skip_missing:
                final_df_to_process = valid_df
                st.warning("Skipping missing items.")
                ready_to_calculate = True
            else:
                st.stop()
        else:
            final_df_to_process = valid_df
            ready_to_calculate = True

        # --- STAGE 2: OPTIMIZATION ---
        if ready_to_calculate:
            st.divider()
            if st.button("🚀 Calculate Drawer Configuration", type="primary"):
                with st.spinner("Optimizing bin packing..."):
                    
                    detail_df, summary_df, costs = optimize_packing(final_df_to_process, draw_df, enable_consolidation=use_topoff)
                    
                    # --- DISPLAY RESULTS ---
                    st.subheader("Results")
                    
                    # 1. Top Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Drawers", f"{int(summary_df['Drawers Required'].sum())}")
                    m2.metric("Total Vertical Height", f"{int(costs['total_height'])}\"")
                    m3.metric("Cabinets Needed", f"{costs['cabinets_needed']}")
                    
                    st.divider()
                    
                    # 2. Financials
                    st.subheader("💰 Financial Breakdown")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    c1.metric("Drawer Subtotal", f"${costs['drawer_subtotal']:,.2f}")
                    c2.metric("Base Cabinets", f"${costs['base_subtotal']:,.2f}", 
                              help=f"{costs['cabinets_needed']} x ${costs['unit_base']:,.2f}")
                    c3.metric("Shipping", f"${costs['shipping_subtotal']:,.2f}",
                              help=f"{costs['cabinets_needed']} x ${costs['unit_shipping']:,.2f}")
                    c4.metric("GRAND TOTAL", f"${costs['grand_total']:,.2f}", delta_color="inverse")
                    
                    st.divider()

                    # 3. Tabs
                    t1, t2 = st.tabs(["Summary (Order This)", "Detailed Pick List"])
                    with t1:
                        # Format the price column for display
                        display_summary = summary_df.copy()
                        display_summary['Total Price'] = display_summary['Total Price'].apply(lambda x: f"${x:,.2f}")
                        
                        st.markdown("#### Drawers to Order")
                        st.dataframe(display_summary, use_container_width=True)
                    with t2:
                        st.markdown("#### Bin Assignments")
                        st.dataframe(detail_df, use_container_width=True)
                    
                    # 4. Download
                    buffer_res = io.BytesIO()
                    with pd.ExcelWriter(buffer_res, engine='xlsxwriter') as writer:
                        summary_df.to_excel(writer, sheet_name="Summary Order", index=False)
                        detail_df.to_excel(writer, sheet_name="Pick List", index=False)
                        
                        cost_df = pd.DataFrame([
                            {"Item": "Drawer Subtotal", "Cost": costs['drawer_subtotal']},
                            {"Item": f"Base Cabinets ({costs['cabinets_needed']} @ ${costs['unit_base']})", "Cost": costs['base_subtotal']},
                            {"Item": f"Shipping ({costs['cabinets_needed']} @ ${costs['unit_shipping']})", "Cost": costs['shipping_subtotal']},
                            {"Item": "TOTAL PROJECT COST", "Cost": costs['grand_total']}
                        ])
                        cost_df.to_excel(writer, sheet_name="Financials", index=False)
                        
                    st.download_button("📥 Download Final Layout Report", buffer_res, "Vending_Layout_Plan.xlsx")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Waiting for file uploads...")
