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

def get_effective_bin_dims(bin_l, bin_w, clearance_mm):
    """
    Calculates usable bin dimensions based on clearance rules.
    Rule: 
    - Largest dimension gets User Input clearance subtracted.
    - Smallest dimension gets FIXED 3mm clearance subtracted.
    """
    dims = [bin_l, bin_w]
    dims.sort() # [small, large]
    
    small_dim = dims[0]
    large_dim = dims[1]
    
    # Calculate clearances
    clear_large = clearance_mm
    clear_small = 3.0 # FIXED as per new requirement
    
    # Subtract from BIN dimensions
    eff_small = small_dim - clear_small
    eff_large = large_dim - clear_large
    
    return eff_small, eff_large

def check_fit(item_l, item_w, item_h, bin_l, bin_w, bin_h, clearance_mm=10):
    """
    Determines how many items fit in a SINGLE bin using effective bin dimensions.
    """
    # 1. Get Effective Bin Dimensions (Usable Space)
    eff_bin_small, eff_bin_large = get_effective_bin_dims(bin_l, bin_w, clearance_mm)
    
    # If clearance makes bin negative, it fits nothing
    if eff_bin_small <= 0 or eff_bin_large <= 0:
        return 0

    # 2. Check Vertical Height (Strict)
    if item_h > bin_h:
        return 0
    vertical_stack = math.floor(bin_h / item_h)
    
    # 3. Check Footprint
    # Sort item dimensions
    item_dims = [item_l, item_w]
    item_dims.sort() # [small, large]
    
    # Check if Item fits in Effective Bin (Small <= Small AND Large <= Large)
    if item_dims[0] <= eff_bin_small and item_dims[1] <= eff_bin_large:
        # IT FITS!
        # Now calculate count based on orientation
        
        # Orient A: Align Item Long side with Bin Long side
        count_a = math.floor(eff_bin_large / item_dims[1]) * math.floor(eff_bin_small / item_dims[0])
        
        # Orient B: Align Item Long side with Bin Short side (Only possible if Item Large <= Bin Small)
        count_b = 0
        if item_dims[1] <= eff_bin_small:
            count_b = math.floor(eff_bin_large / item_dims[0]) * math.floor(eff_bin_small / item_dims[1])
            
        base_count = max(count_a, count_b)
        return base_count * vertical_stack
    
    return 0

def get_failure_reason(i_l, i_w, i_h, drawer_db_drawers, clearance_mm):
    """
    Diagnose why an item didn't fit.
    """
    max_h = drawer_db_drawers['BinHeight'].max()
    
    # Find the largest physical bin to check against
    max_l = drawer_db_drawers['BinLength'].max()
    max_w = drawer_db_drawers['BinWidth'].max()
    
    # Get effective dims of that largest bin
    eff_max_small, eff_max_large = get_effective_bin_dims(max_l, max_w, clearance_mm)
    
    reasons = []
    
    if i_h > max_h:
        reasons.append(f"Height {i_h}mm > Max Available {max_h}mm")
        
    # Check footprint
    item_dims = [i_l, i_w]
    item_dims.sort()
    
    if item_dims[0] > eff_max_small:
        reasons.append(f"Min Width {item_dims[0]}mm > Max Usable Width {eff_max_small:.1f}mm")
    elif item_dims[1] > eff_max_large:
        reasons.append(f"Max Length {item_dims[1]}mm > Max Usable Length {eff_max_large:.1f}mm")
        
    if not reasons:
        return "Complex Fit Issue (Shape/Volume)"
        
    return "; ".join(reasons)

def consolidate_drawers(item_results, drawer_db_drawers, clearance_mm):
    """
    Moves overflow items to empty spaces.
    """
    drawer_map = drawer_db_drawers.set_index('DrawerID').to_dict('index')
    
    def get_drawer_state(current_results):
        state = {}
        for d_id, props in drawer_map.items():
            items = [r for r in current_results if r['Type of drawer'] == d_id]
            total_bins = sum(r['quantity of bins needed'] for r in items)
            capacity = props['QtyBins']
            
            if capacity == 0: 
                drawers_needed = 0
                free_slots = 0
                remainder = 0
            else:
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

    for _ in range(3):
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
                    if dest_props['QtyBins'] == 0: continue 

                    if '_raw_dims' not in item: continue
                    i_l, i_w, i_h = item['_raw_dims']
                    
                    new_fit = check_fit(i_l, i_w, i_h, 
                                      dest_props['BinLength'], 
                                      dest_props['BinWidth'], 
                                      dest_props['BinHeight'],
                                      clearance_mm)
                    
                    if new_fit > 0:
                        new_bins_needed = math.ceil(item['Quantity Requested'] / new_fit)
                        if new_bins_needed <= dest_state['free_slots']:
                            best_dest = dest_id
                            break 
                
                if best_dest:
                    dest_props = drawer_map[best_dest]
                    i_l, i_w, i_h = item['_raw_dims']
                    new_fit = check_fit(i_l, i_w, i_h, dest_props['BinLength'], dest_props['BinWidth'], dest_props['BinHeight'], clearance_mm)
                    
                    item['Type of drawer'] = best_dest
                    item['quantity per bin'] = new_fit
                    item['quantity of bins needed'] = math.ceil(item['Quantity Requested'] / new_fit)
                    item['_drawer_height'] = 6 if dest_props['BinHeight'] > 100 else 3
                    
                    changes_made = True
                    break 
            if changes_made: break
        if not changes_made: break
            
    return item_results

def fill_cabinet_gaps(summary_df, total_height_current, strategy, drawer_db_full):
    """
    Ensures total height is a multiple of 33" by filling the gap.
    """
    cabinets_needed = math.ceil(total_height_current / 33)
    if cabinets_needed == 0: cabinets_needed = 1
    target_height = cabinets_needed * 33
    gap = target_height - total_height_current
    
    if gap <= 0:
        return summary_df, total_height_current, []

    df = summary_df.copy()
    changes_log = []
    
    # Get Prices for Empties
    try:
        p_e3 = drawer_db_full.loc[drawer_db_full['DrawerID']=='Empty3', 'Price'].values[0]
    except: p_e3 = 0
    try:
        p_e6 = drawer_db_full.loc[drawer_db_full['DrawerID']=='Empty6', 'Price'].values[0]
    except: p_e6 = 0

    # STRATEGY 1: EXPAND EXISTING 3" -> 6"
    if strategy == 'expand':
        new_rows = []
        price_map = drawer_db_full.set_index('DrawerID')['Price'].to_dict()
        
        for idx, row in df.iterrows():
            d_type = row['Drawer Type']
            if d_type.endswith('3') and 'Empty' not in d_type:
                target_type = d_type.replace('3', '6')
                if target_type in price_map:
                    upgrades_possible = row['Drawers Required']
                    upgrades_needed = math.ceil(gap / 3)
                    upgrades_to_perform = min(upgrades_possible, upgrades_needed)
                    
                    if upgrades_to_perform > 0:
                        gap_filled = upgrades_to_perform * 3
                        
                        # Add upgraded part (AS DICT)
                        new_rows.append({
                            "Drawer Type": target_type,
                            "Drawer Height": 6,
                            "Total Bins Used": 0, # Virtual move
                            "Drawers Required": upgrades_to_perform,
                            "Unit Price": f"${price_map.get(target_type,0):,.2f}",
                            "Total Price": upgrades_to_perform * price_map.get(target_type,0),
                            "Vertical Space (in)": upgrades_to_perform * 6,
                            "Notes": "Expanded from 3\""
                        })
                        
                        # Add remaining part (AS DICT)
                        remaining = upgrades_possible - upgrades_to_perform
                        if remaining > 0:
                            row['Drawers Required'] = remaining
                            row['Vertical Space (in)'] = remaining * 3
                            unit_p = float(str(row['Unit Price']).replace('$','').replace(',',''))
                            row['Total Price'] = remaining * unit_p
                            new_rows.append(row.to_dict()) # Convert Series to Dict
                            
                        changes_log.append(f"Expanded {int(upgrades_to_perform)}x {d_type} to {target_type}")
                        gap -= gap_filled
                        continue 
            
            # Unchanged row (AS DICT)
            new_rows.append(row.to_dict()) # Convert Series to Dict
            
        df = pd.DataFrame(new_rows)
    
    # STRATEGY 2: FILL REMAINING GAP
    if gap > 0:
        num_e6 = math.floor(gap / 6)
        gap -= num_e6 * 6
        num_e3 = math.ceil(gap / 3) 
        gap -= num_e3 * 3 
        
        if num_e6 > 0:
            df = pd.concat([df, pd.DataFrame([{
                "Drawer Type": "Empty6",
                "Drawer Height": 6,
                "Total Bins Used": 0,
                "Drawers Required": num_e6,
                "Unit Price": f"${p_e6:,.2f}",
                "Total Price": num_e6 * p_e6,
                "Vertical Space (in)": num_e6 * 6,
                "Notes": "Gap Filler"
            }])], ignore_index=True)
            changes_log.append(f"Added {num_e6}x Empty6")
            
        if num_e3 > 0:
            df = pd.concat([df, pd.DataFrame([{
                "Drawer Type": "Empty3",
                "Drawer Height": 3,
                "Total Bins Used": 0,
                "Drawers Required": num_e3,
                "Unit Price": f"${p_e3:,.2f}",
                "Total Price": num_e3 * p_e3,
                "Vertical Space (in)": num_e3 * 3,
                "Notes": "Gap Filler"
            }])], ignore_index=True)
            changes_log.append(f"Added {num_e3}x Empty3")

    new_total_height = df['Vertical Space (in)'].sum()
    return df, new_total_height, changes_log

def optimize_packing(inventory_df, drawer_db_full, enable_consolidation, clearance_mm, fill_strategy):
    item_results = []
    no_fit_results = []
    
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

        for _, drawer in drawer_db_drawers.iterrows():
            d_id = drawer['DrawerID']
            b_w = drawer['BinWidth']
            b_l = drawer['BinLength']
            b_h = drawer['BinHeight']
            b_qty_slots = drawer['QtyBins'] 
            
            if b_qty_slots <= 0: continue
            
            if b_h > 100:
                drawer_height_inch = 6
            else:
                drawer_height_inch = 3
            
            items_per_bin = check_fit(i_l, i_w, i_h, b_l, b_w, b_h, clearance_mm)
            
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
            reason = get_failure_reason(i_l, i_w, i_h, drawer_db_drawers, clearance_mm)
            no_fit_results.append({
                "Material ID/ Product Code": item_code,
                "Quantity": qty_needed,
                "Length (mm)": i_l,
                "Width (mm)": i_w,
                "Height (mm)": i_h,
                "Failure Reason": reason
            })

    if enable_consolidation and item_results:
        item_results = consolidate_drawers(item_results, drawer_db_drawers, clearance_mm)
            
    results_df = pd.DataFrame(item_results)
    no_fit_df = pd.DataFrame(no_fit_results)
    
    if results_df.empty:
        return results_df, pd.DataFrame(), {'grand_total':0}, no_fit_df, []

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
            "Vertical Space (in)": type_height_total,
            "Notes": ""
        })
        
    summary_df = pd.DataFrame(summary_list)
    
    summary_df, total_cabinet_height, fill_logs = fill_cabinet_gaps(
        summary_df, total_cabinet_height, fill_strategy, drawer_db_full
    )
    
    total_drawer_cost = summary_df['Total Price'].sum()
    
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
    
    return display_df, summary_df, cost_summary, no_fit_df, fill_logs

# --- MAIN APP LAYOUT ---

st.title("📦 Vending Machine Cabinet Optimizer")

st.sidebar.header("Data Upload")

with st.sidebar.expander("📄 Download Templates", expanded=False):
    st.write("Get empty files with correct headers:")
    
    df_inv_temp = pd.DataFrame(columns=["Material ID/ Product Code", "Quantity", "Length (mm)", "Width (mm)", "Height (mm)"])
    buffer_inv = io.BytesIO()
    with pd.ExcelWriter(buffer_inv, engine='xlsxwriter') as writer:
        df_inv_temp.to_excel(writer, index=False)
    st.download_button("1. Inventory Template", buffer_inv, "template_inventory.xlsx")
    
    df_prod_temp = pd.DataFrame(columns=["Material ID", "Product Code", "Length (mm)", "Width (mm)", "Height (mm)"])
    buffer_prod = io.BytesIO()
    with pd.ExcelWriter(buffer_prod, engine='xlsxwriter') as writer:
        df_prod_temp.to_excel(writer, index=False)
    st.download_button("2. Product DB Template", buffer_prod, "template_product_db.xlsx")

    df_draw_temp = pd.DataFrame(columns=["DrawerID", "BinWidth", "BinLength", "BinHeight", "QtyBins", "Price"])
    buffer_draw = io.BytesIO()
    with pd.ExcelWriter(buffer_draw, engine='xlsxwriter') as writer:
        df_draw_temp.to_excel(writer, index=False)
    st.download_button("3. Drawer DB Template", buffer_draw, "template_drawer_db.xlsx")

st.sidebar.divider()

inv_file = st.sidebar.file_uploader("1. Inventory Input (Excel)", type=['xlsx'])
prod_file = st.sidebar.file_uploader("2. Product Database (Excel)", type=['xlsx'])
draw_file = st.sidebar.file_uploader("3. Drawer Database (Excel/CSV)", type=['xlsx', 'csv'])

st.sidebar.divider()
st.sidebar.header("Settings")
use_topoff = st.sidebar.checkbox("Enable Drawer Consolidation", value=True, help="Moves overflow items to empty space in other drawers.")

skip_missing_setting = st.sidebar.checkbox("Skip missing items (ignore errors)", value=False, help="If checked, items without dimensions will be ignored instead of stopping calculation.")

clearance = st.sidebar.number_input("Large Dimension Clearance (mm)", min_value=0.0, value=10.0, step=1.0, help="Subtracts this amount from the bin's largest dimension. The smallest dimension gets a fixed 3mm subtraction.")

fill_strat = st.sidebar.selectbox(
    "Cabinet Gap Filling Strategy", 
    ["expand", "empty"],
    help="Strategies to reach 33 inches:\n'expand': Upgrade 3\" drawers to 6\" to use up space.\n'empty': Just add empty drawers."
)

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
            
            if skip_missing_setting:
                st.warning(f"Skipping {len(missing_df)} items as per Settings.")
                final_df_to_process = valid_df
                ready_to_calculate = True
            else:
                st.info("To proceed, upload fixed data OR enable 'Skip missing items' in Settings.")
                
                c1, c2 = st.columns(2)
                with c1:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        missing_df.to_excel(writer, index=False)
                    st.download_button("Download Missing Report", buffer, "missing_dims.xlsx")
                with c2:
                    fixed_file = st.file_uploader("Upload Fixed File", type=['xlsx'], key="fix")

                if fixed_file:
                    fixed_df = pd.read_excel(fixed_file)
                    final_df_to_process = pd.concat([valid_df, fixed_df], ignore_index=True)
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
                    
                    detail_df, summary_df, costs, no_fit_df, logs = optimize_packing(
                        final_df_to_process, 
                        draw_df, 
                        enable_consolidation=use_topoff, 
                        clearance_mm=clearance,
                        fill_strategy=fill_strat
                    )
                    
                    st.subheader("Results")
                    
                    if not no_fit_df.empty:
                        st.error(f"⚠️ {len(no_fit_df)} items could not fit in ANY drawer!")
                        with st.expander("See Items That Didn't Fit"):
                            st.dataframe(no_fit_df)
                            
                        buff_nf = io.BytesIO()
                        with pd.ExcelWriter(buff_nf, engine='xlsxwriter') as writer:
                            no_fit_df.to_excel(writer, index=False)
                        st.download_button("📥 Download 'No Fit' Report", buff_nf, "items_not_packed.xlsx", mime="application/vnd.ms-excel")
                        st.divider()
                    
                    if logs:
                        with st.expander("ℹ️ Cabinet Gap Adjustments (33\" Compliance)"):
                            for log in logs:
                                st.text(f"- {log}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Drawers", f"{int(summary_df['Drawers Required'].sum())}")
                    m2.metric("Total Vertical Height", f"{int(costs['total_height'])}\"")
                    m3.metric("Cabinets Needed", f"{costs['cabinets_needed']}")
                    
                    st.subheader("💰 Financials")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drawer Cost", f"${costs['drawer_subtotal']:,.2f}")
                    c2.metric("Base Cabinets", f"${costs['base_subtotal']:,.2f}")
                    c3.metric("Shipping", f"${costs['shipping_subtotal']:,.2f}")
                    c4.metric("GRAND TOTAL", f"${costs['grand_total']:,.2f}", delta_color="inverse")
                    
                    t1, t2 = st.tabs(["Summary Order", "Pick List"])
                    with t1:
                        st.dataframe(summary_df, use_container_width=True)
                    with t2:
                        st.dataframe(detail_df, use_container_width=True)
                    
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
