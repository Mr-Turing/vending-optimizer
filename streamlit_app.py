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
    Returns: count (int)
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

def get_failure_reason(i_l, i_w, i_h, drawer_db_drawers):
    """
    Diagnose why an item didn't fit any drawer.
    Returns a string explanation.
    """
    # Calculate absolute max dimensions available across ALL drawers
    max_h = drawer_db_drawers['BinHeight'].max()
    max_l = drawer_db_drawers['BinLength'].max()
    max_w = drawer_db_drawers['BinWidth'].max()
    
    # Note: L and W are rotatable, so we compare sorted pairs
    # Bin Footprint Max: The largest bin available
    # We assume the "Largest" bin has max_l and max_w (usually L6)
    # But strictly, we should find the bin with the largest min_dim and largest max_dim
    # For simplification with standard vending bins:
    max_dim_1 = max(max_l, max_w)
    max_dim_2 = min(max_l, max_w)
    
    reasons = []
    
    # 1. Check Height
    if i_h > max_h:
        reasons.append(f"Height {i_h}mm > Max {max_h}mm")
        
    # 2. Check Footprint
    i_dim_1 = max(i_l, i_w)
    i_dim_2 = min(i_l, i_w)
    
    if i_dim_2 > max_dim_2:
        reasons.append(f"Width {i_dim_2}mm > Max {max_dim_2}mm")
    elif i_dim_1 > max_dim_1:
        reasons.append(f"Length {i_dim_1}mm > Max {max_dim_1}mm")
        
    if not reasons:
        return "Complex Fit Issue (Volume/Shape)"
        
    return "; ".join(reasons)

def consolidate_drawers(item_results, drawer_db_drawers):
    """
    Advanced Logic:
    Tries to move items from 'Overflow' drawers into 'Empty Space' of others.
    """
    drawer_map = drawer_db_drawers.set_index('DrawerID').to_dict('index')
    
    # Helper to recalc state
    def get_drawer_state(current_results):
        state = {}
        for d_id, props in drawer_map.items():
            items = [r for r in current_results if r['Type of drawer'] == d_id]
            total_bins = sum(r['quantity of bins needed'] for r in items)
            capacity = props['QtyBins']
            
            drawers_needed = math.ceil(total_bins / capacity)
            bins_available_total = drawers_needed * capacity
            free_slots = bins_available_total - total_bins
            
            remainder = total_bins % capacity
            if remainder == 0 and total_bins > 0: remainder = capacity
            if total_bins == 0: remainder = 0
            
            state[d_id] = {
                'items': items,
                'total_bins': total_bins,
                'drawers_needed': drawers_needed,
                'free_slots': free_slots,
                'overflow_bins': remainder,
                'is_overflow': (remainder > 0 and remainder < capacity)
            }
        return state

    max_passes = 3
    for _ in range(max_passes):
        state = get_drawer_state(item_results)
        changes_made = False
        
        candidates = sorted(
            [d for d, s in state.items() if s['is_overflow'] and s['total_bins'] > 0],
            key=lambda k: state[k]['overflow_bins']
        )
        
        for source_id in candidates:
            source_info = state[source_id]
            movable_items = sorted(source_info['items'], key=lambda x: x['quantity of bins needed'])
            
            for item in movable_items:
                best_dest = None
                possible_dests = [d for d, s in state.items() if s['free_slots'] > 0]
                
                for dest_id in possible_dests:
                    if dest_id == source_id: continue
                    dest_props = drawer_map[dest_id]
                    dest_state = state[dest_id]
                    
                    if '_raw_dims' not in item: continue
                    i_l, i_w, i_h = item['_raw_dims']
                    
                    new_fit = check_fit(i_l, i_w, i_h, 
                                      dest_props['BinLength'], 
                                      dest_props['BinWidth'], 
                                      dest_props['BinHeight'])
                    
                    if new_fit > 0:
                        qty_req = item['Quantity Requested']
                        new_bins_needed = math.ceil(qty_req / new_fit)
                        if new_bins_needed <= dest_state['free_slots']:
                            best_dest = dest_id
                            break 
                
                if best_dest:
                    dest_props = drawer_map[best_dest]
                    i_l, i_w, i_h = item['_raw_dims']
                    new_fit = check_fit(i_l, i_w, i_h, dest_props['BinLength'], dest_props['BinWidth'], dest_props['BinHeight'])
                    new_bins = math.ceil(item['Quantity Requested'] / new_fit)
                    
                    item['Type of drawer'] = best_dest
                    item['quantity per bin'] = new_fit
                    item['quantity of bins needed'] = new_bins
                    item['_drawer_height'] = 6 if dest_props['BinHeight'] > 100 else 3
                    
                    changes_made = True
                    break 
            
            if changes_made: break
        if not changes_made: break
            
    return item_results

def optimize_packing(inventory_df, drawer_db_full, enable_consolidation=True):
    """
    Main Optimization Loop.
    Returns: display_df, summary_df, costs, no_fit_df
    """
    item_results = []
    no_fit_results = []
    
    # 1. Parse Drawer DB
    cost_rows = drawer_db_full[drawer_db_full['BinWidth'].isna()]
    drawer_db_drawers = drawer_db_full[drawer_db_full['BinWidth'].notna()].copy()
    
    base_cabinet_cost = 0
    shipping_cost = 0
    
    for _, row in cost_rows.iterrows():
        name = str(row['DrawerID']).lower()
        price = float(row['Price']) if pd.notnull(row['Price']) else 0
        if "base" in name:
            base_cabinet_cost = price
        elif "shipping" in name:
            shipping_cost = price
            
    cols_to_num = ['BinWidth', 'BinLength', 'BinHeight', 'QtyBins', 'Price']
    for col in cols_to_num:
        if col == 'Price':
             drawer_db_drawers[col] = pd.to_numeric(drawer_db_drawers[col], errors='coerce').fillna(0)
        else:
             drawer_db_drawers[col] = pd.to_numeric(drawer_db_drawers[col], errors='coerce')

    # 2. Packing Loop
    for index, row in inventory_df.iterrows():
        item_code = row['Material ID/ Product Code']
        raw_qty = float(row['Quantity'])
        qty_needed = math.ceil(raw_qty) 
        
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
            
            if b_h > 100:
                drawer_height_inch = 6
            else:
                drawer_height_inch = 3
                
            items_per_bin = check_fit(i_l, i_w, i_h, b_l, b_w, b_h)
            
            if items_per_bin > 0:
                bins_needed = math.ceil(qty_needed / items_per_bin)
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
                "_raw_dims": (i_l, i_w, i_h)
            })
        else:
            # IT DOES NOT FIT ANYWHERE
            reason = get_failure_reason(i_l, i_w, i_h, drawer_db_drawers)
            no_fit_results.append({
                "Material ID/ Product Code": item_code,
                "Quantity": qty_needed,
                "Length (mm)": i_l,
                "Width (mm)": i_w,
                "Height (mm)": i_h,
                "Failure Reason": reason
            })

    # 3. Consolidation (Only on fitted items)
    if enable_consolidation and item_results:
        item_results = consolidate_drawers(item_results, drawer_db_drawers)
            
    # 4. Aggregation
    results_df = pd.DataFrame(item_results)
    no_fit_df = pd.DataFrame(no_fit_results)
    
    if results_df.empty:
        # Handle case where NOTHING fits
        return results_df, pd.DataFrame(), {'grand_total':0}, no_fit_df

    # Filter valid just in case
    valid_results = results_df[results_df['Type of drawer'] != "NO FIT"]
    
    drawer_caps = drawer_db_drawers.set_index('DrawerID')['QtyBins'].to_dict()
    drawer_prices = drawer_db_drawers.set_index('DrawerID')['Price'].to_dict()

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
    
    # Financials
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
    
    display_df = results_df.drop(columns=['_drawer_height', '_raw_dims'])
    
    return display_df, summary_df, cost_summary, no_fit_df

# --- MAIN APP LAYOUT ---

st.title("📦 Vending Machine Cabinet Optimizer")

st.sidebar.header("Data Upload")
inv_file = st.sidebar.file_uploader("1. Inventory Input (Excel)", type=['xlsx'])
prod_file = st.sidebar.file_uploader("2. Product Database (Excel)", type=['xlsx'])
draw_file = st.sidebar.file_uploader("3. Drawer Database (Excel/CSV)", type=['xlsx', 'csv'])

st.sidebar.divider()
st.sidebar.header("Settings")
use_topoff = st.sidebar.checkbox("Enable Cabinet Top-Off", value=True)

if inv_file and prod_file and draw_file:
    try:
        inv_df = pd.read_excel(inv_file)
        prod_df = pd.read_excel(prod_file)
        if draw_file.name.endswith('.csv'):
            draw_df = pd.read_csv(draw_file)
        else:
            draw_df = pd.read_excel(draw_file)

        inv_df.columns = inv_df.columns.str.strip()
        prod_df.columns = prod_df.columns.str.strip()
        draw_df.columns = draw_df.columns.str.strip()
        
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
        col_stat1.success(f"✅ {len(valid_df)} items matched.")
        
        final_df_to_process = pd.DataFrame()
        ready_to_calculate = False

        if not missing_df.empty:
            col_stat2.error(f"⚠️ {len(missing_df)} items missing dimensions.")
            st.info("Fix or Skip to proceed.")
            
            c1, c2 = st.columns(2)
            with c1:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    missing_df.to_excel(writer, index=False)
                st.download_button("Download Missing Report", buffer, "missing_dims.xlsx")
                fixed_file = st.file_uploader("Upload Fixed File", type=['xlsx'], key="fix")
                
            with c2:
                skip_missing = st.checkbox("Skip missing items")

            if fixed_file:
                fixed_df = pd.read_excel(fixed_file)
                final_df_to_process = pd.concat([valid_df, fixed_df], ignore_index=True)
                ready_to_calculate = True
            elif skip_missing:
                final_df_to_process = valid_df
                ready_to_calculate = True
            else:
                st.stop()
        else:
            final_df_to_process = valid_df
            ready_to_calculate = True

        if ready_to_calculate:
            st.divider()
            if st.button("🚀 Calculate", type="primary"):
                with st.spinner("Optimizing..."):
                    
                    detail_df, summary_df, costs, no_fit_df = optimize_packing(final_df_to_process, draw_df, enable_consolidation=use_topoff)
                    
                    st.subheader("Results")
                    
                    # --- NO FIT WARNING ---
                    if not no_fit_df.empty:
                        st.error(f"⚠️ {len(no_fit_df)} items could not fit in ANY drawer!")
                        with st.expander("See Items That Didn't Fit"):
                            st.dataframe(no_fit_df)
                            
                        # Download No Fit
                        buff_nf = io.BytesIO()
                        with pd.ExcelWriter(buff_nf, engine='xlsxwriter') as writer:
                            no_fit_df.to_excel(writer, index=False)
                        st.download_button("📥 Download 'No Fit' Report", buff_nf, "items_not_packed.xlsx", mime="application/vnd.ms-excel")
                        st.divider()
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Drawers", f"{int(summary_df['Drawers Required'].sum())}")
                    m2.metric("Total Vertical Height", f"{int(costs['total_height'])}\"")
                    m3.metric("Cabinets Needed", f"{costs['cabinets_needed']}")
                    
                    # Financials
                    st.subheader("💰 Financials")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drawer Cost", f"${costs['drawer_subtotal']:,.2f}")
                    c2.metric("Base Cabinets", f"${costs['base_subtotal']:,.2f}")
                    c3.metric("Shipping", f"${costs['shipping_subtotal']:,.2f}")
                    c4.metric("GRAND TOTAL", f"${costs['grand_total']:,.2f}", delta_color="inverse")
                    
                    # Data Tabs
                    t1, t2 = st.tabs(["Summary Order", "Pick List"])
                    with t1:
                        st.dataframe(summary_df, use_container_width=True)
                    with t2:
                        st.dataframe(detail_df, use_container_width=True)
                    
                    # Final Download
                    buffer_res = io.BytesIO()
                    with pd.ExcelWriter(buffer_res, engine='xlsxwriter') as writer:
                        summary_df.to_excel(writer, sheet_name="Summary Order", index=False)
                        detail_df.to_excel(writer, sheet_name="Pick List", index=False)
                        if not no_fit_df.empty:
                            no_fit_df.to_excel(writer, sheet_name="Not Packed", index=False)
                        
                        cost_df = pd.DataFrame([
                            {"Item": "Drawer Subtotal", "Cost": costs['drawer_subtotal']},
                            {"Item": "Base Cabinets", "Cost": costs['base_subtotal']},
                            {"Item": "Shipping", "Cost": costs['shipping_subtotal']},
                            {"Item": "TOTAL", "Cost": costs['grand_total']}
                        ])
                        cost_df.to_excel(writer, sheet_name="Financials", index=False)
                        
                    st.download_button("📥 Download Full Report", buffer_res, "Vending_Layout_Plan.xlsx")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload files to start.")
