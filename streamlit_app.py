import streamlit as st
import pandas as pd
import math
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Vending Cabinet Optimizer", layout="wide")

# --- HELPER FUNCTIONS ---

def clean_code_str(val):
    """Helper to clean a single string value."""
    if pd.isna(val) or val == "":
        return None
    return str(val).replace(" ", "").strip().upper()

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
    eff_bin_small, eff_bin_large = get_effective_bin_dims(bin_l, bin_w, clearance_mm)
    
    if eff_bin_small <= 0 or eff_bin_large <= 0:
        return 0

    if item_h > bin_h:
        return 0
    vertical_stack = math.floor(bin_h / item_h)
    
    item_dims = [item_l, item_w]
    item_dims.sort()
    
    if item_dims[0] <= eff_bin_small and item_dims[1] <= eff_bin_large:
        # Orient A
        count_a = math.floor(eff_bin_large / item_dims[1]) * math.floor(eff_bin_small / item_dims[0])
        # Orient B
        count_b = 0
        if item_dims[1] <= eff_bin_small:
            count_b = math.floor(eff_bin_large / item_dims[0]) * math.floor(eff_bin_small / item_dims[1])
            
        base_count = max(count_a, count_b)
        return base_count * vertical_stack
    
    return 0

def get_failure_reason(i_l, i_w, i_h, drawer_db_drawers, clearance_mm):
    """Diagnose why an item didn't fit."""
    max_h = drawer_db_drawers['BinHeight'].max()
    max_l = drawer_db_drawers['BinLength'].max()
    max_w = drawer_db_drawers['BinWidth'].max()
    eff_max_small, eff_max_large = get_effective_bin_dims(max_l, max_w, clearance_mm)
    
    reasons = []
    if i_h > max_h:
        reasons.append(f"Height {i_h}mm > Max Available {max_h}mm")
        
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
    """Moves overflow items to empty spaces."""
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
                        new_bins_needed = math.ceil(item['Packages to Store'] / new_fit)
                        if new_bins_needed <= dest_state['free_slots']:
                            best_dest = dest_id
                            break 
                
                if best_dest:
                    dest_props = drawer_map[best_dest]
                    i_l, i_w, i_h = item['_raw_dims']
                    new_fit = check_fit(i_l, i_w, i_h, dest_props['BinLength'], dest_props['BinWidth'], dest_props['BinHeight'], clearance_mm)
                    
                    item['Type of drawer'] = best_dest
                    item['quantity per bin'] = new_fit
                    item['quantity of bins needed'] = math.ceil(item['Packages to Store'] / new_fit)
                    item['_drawer_height'] = 6 if dest_props['BinHeight'] > 100 else 3
                    changes_made = True
                    break 
            if changes_made: break
        if not changes_made: break
            
    return item_results

def fill_cabinet_gaps(summary_df, total_height_current, strategy, drawer_db_full):
    """Ensures total height is a multiple of 33"."""
    cabinets_needed = math.ceil(total_height_current / 33)
    if cabinets_needed == 0: cabinets_needed = 1
    target_height = cabinets_needed * 33
    gap = target_height - total_height_current
    
    if gap <= 0:
        return summary_df, total_height_current, []

    df = summary_df.copy()
    changes_log = []
    
    try: p_e3 = drawer_db_full.loc[drawer_db_full['DrawerID']=='Empty3', 'Price'].values[0]
    except: p_e3 = 0
    try: p_e6 = drawer_db_full.loc[drawer_db_full['DrawerID']=='Empty6', 'Price'].values[0]
    except: p_e6 = 0

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
                        
                        # Calculate proportional bin usage
                        # Assumption: bins are distributed evenly among drawers
                        avg_bins_per_drawer = row['Total Bins Used'] / row['Drawers Required']
                        bins_moving = avg_bins_per_drawer * upgrades_to_perform
                        
                        new_rows.append({
                            "Drawer Type": target_type,
                            "Drawer Height": 6,
                            "Total Bins Used": bins_moving, # Carry over the usage count
                            "Drawers Required": upgrades_to_perform,
                            "Unit Price": f"${price_map.get(target_type,0):,.2f}",
                            "Total Price": upgrades_to_perform * price_map.get(target_type,0),
                            "Vertical Space (in)": upgrades_to_perform * 6,
                            "Notes": "Expanded from 3\""
                        })
                        
                        remaining = upgrades_possible - upgrades_to_perform
                        if remaining > 0:
                            row['Drawers Required'] = remaining
                            row['Vertical Space (in)'] = remaining * 3
                            row['Total Bins Used'] = row['Total Bins Used'] - bins_moving # Adjust remaining
                            unit_p = float(str(row['Unit Price']).replace('$','').replace(',',''))
                            row['Total Price'] = remaining * unit_p
                            new_rows.append(row.to_dict())
                            
                        changes_log.append(f"Expanded {int(upgrades_to_perform)}x {d_type} to {target_type}")
                        gap -= gap_filled
                        continue 
            new_rows.append(row.to_dict())
        df = pd.DataFrame(new_rows)
    
    if gap > 0:
        num_e6 = math.floor(gap / 6)
        gap -= num_e6 * 6
        num_e3 = math.ceil(gap / 3) 
        gap -= num_e3 * 3 
        if num_e6 > 0:
            df = pd.concat([df, pd.DataFrame([{
                "Drawer Type": "Empty6", "Drawer Height": 6, "Total Bins Used": 0, "Drawers Required": num_e6,
                "Unit Price": f"${p_e6:,.2f}", "Total Price": num_e6 * p_e6, "Vertical Space (in)": num_e6 * 6, "Notes": "Gap Filler"
            }])], ignore_index=True)
            changes_log.append(f"Added {num_e6}x Empty6")
        if num_e3 > 0:
            df = pd.concat([df, pd.DataFrame([{
                "Drawer Type": "Empty3", "Drawer Height": 3, "Total Bins Used": 0, "Drawers Required": num_e3,
                "Unit Price": f"${p_e3:,.2f}", "Total Price": num_e3 * p_e3, "Vertical Space (in)": num_e3 * 3, "Notes": "Gap Filler"
            }])], ignore_index=True)
            changes_log.append(f"Added {num_e3}x Empty3")

    new_total_height = df['Vertical Space (in)'].sum()
    return df, new_total_height, changes_log

@st.cache_data
def build_product_map(prod_df):
    """
    Creates a master dictionary for O(1) matching.
    Cached to run only once per file upload.
    """
    product_map = {}
    for idx, row in prod_df.iterrows():
        l = row.get('Length Packed Product Metric', 0)
        w = row.get('Width Packed Product Metric', 0)
        h = row.get('Height Packed Product Metric', 0)
        pkg_qty = row.get('Package Quantity', 1)
        
        if pd.isna(pkg_qty) or pkg_qty <= 0: pkg_qty = 1
        
        data_packet = {
            'L': float(l),
            'W': float(w),
            'H': float(h),
            'PkgQty': float(pkg_qty),
            'Desc': str(row.get('Item Description', ''))
        }
        
        keys_to_map = [
            row.get('Material ID'),
            row.get('Ordering Code'),
            row.get('Ordering Code Packed'),
            row.get('Ordering Code ANSI')
        ]
        
        for k in keys_to_map:
            clean_k = clean_code_str(k)
            if clean_k:
                product_map[clean_k] = data_packet
                
    return product_map

def optimize_packing(inventory_df, drawer_db_full, product_map, enable_consolidation, clearance_mm, fill_strategy):
    item_results = []
    no_fit_results = []
    
    # Prepare Drawer DB
    drawer_db_drawers = drawer_db_full[drawer_db_full['BinWidth'].notna()].copy()
    cost_rows = drawer_db_full[drawer_db_full['BinWidth'].isna()]
    
    base_cabinet_cost = 0
    shipping_cost = 0
    for _, row in cost_rows.iterrows():
        name = str(row['DrawerID']).lower()
        price = float(row['Price']) if pd.notnull(row['Price']) else 0
        if "base" in name: base_cabinet_cost = price
        elif "shipping" in name: shipping_cost = price
            
    cols_to_num = ['BinWidth', 'BinLength', 'BinHeight', 'QtyBins', 'Price']
    for col in cols_to_num:
        if col == 'Price': drawer_db_drawers[col] = pd.to_numeric(drawer_db_drawers[col], errors='coerce').fillna(0)
        else: drawer_db_drawers[col] = pd.to_numeric(drawer_db_drawers[col], errors='coerce')

    # Main Processing Loop
    for index, row in inventory_df.iterrows():
        input_code_raw = row['Material ID/ Product Code']
        match_id = clean_code_str(input_code_raw)
        req_qty_pieces = float(row['Quantity'])
        
        # 1. Lookup Product Data
        prod_data = product_map.get(match_id)
        
        if prod_data:
            i_l, i_w, i_h = prod_data['L'], prod_data['W'], prod_data['H']
            pkg_qty = prod_data['PkgQty']
            packages_needed = math.ceil(req_qty_pieces / pkg_qty)
            is_alien = False
        else:
            i_l = float(row.get('Length (mm)', 0)) if pd.notnull(row.get('Length (mm)')) else 0
            i_w = float(row.get('Width (mm)', 0)) if pd.notnull(row.get('Width (mm)')) else 0
            i_h = float(row.get('Height (mm)', 0)) if pd.notnull(row.get('Height (mm)')) else 0
            
            if i_l > 0 and i_w > 0 and i_h > 0:
                packages_needed = math.ceil(req_qty_pieces)
                pkg_qty = 1
                is_alien = True
            else:
                continue

        # 2. Optimization Logic
        best_drawer_id = None
        min_cost_share = float('inf')
        selected_bins_needed = 0
        selected_items_per_bin = 0
        selected_drawer_height = 0

        for _, drawer in drawer_db_drawers.iterrows():
            d_id = drawer['DrawerID']
            b_qty_slots = drawer['QtyBins'] 
            if b_qty_slots <= 0: continue
            
            b_w, b_l, b_h = drawer['BinWidth'], drawer['BinLength'], drawer['BinHeight']
            drawer_height_inch = 6 if b_h > 100 else 3
            
            items_per_bin = check_fit(i_l, i_w, i_h, b_l, b_w, b_h, clearance_mm)
            
            if items_per_bin > 0:
                bins_needed = math.ceil(packages_needed / items_per_bin)
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
                "Material ID/ Product Code": input_code_raw,
                "Qty Requested (Pieces)": req_qty_pieces,
                "Package Size": pkg_qty,
                "Packages to Store": packages_needed,
                "Type of drawer": best_drawer_id,
                "quantity per bin": selected_items_per_bin,
                "quantity of bins needed": selected_bins_needed,
                "_drawer_height": selected_drawer_height,
                "_raw_dims": (i_l, i_w, i_h)
            })
        else:
            reason = get_failure_reason(i_l, i_w, i_h, drawer_db_drawers, clearance_mm)
            no_fit_results.append({
                "Material ID/ Product Code": input_code_raw,
                "Qty (Pieces)": req_qty_pieces,
                "Package Size": pkg_qty if not is_alien else "N/A",
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
    
    for (d_type, d_height), group in grouped:
        total_bins_needed = group['quantity of bins needed'].sum()
        bins_per_drawer = drawer_caps.get(d_type, 1)
        price_per_drawer = drawer_prices.get(d_type, 0)
        drawers_count = math.ceil(total_bins_needed / bins_per_drawer)
        type_height_total = drawers_count * d_height
        type_cost_total = drawers_count * price_per_drawer
        total_cabinet_height += type_height_total
        
        # Kept as float for calculation, will format later
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
    summary_df, total_cabinet_height, fill_logs = fill_cabinet_gaps(summary_df, total_cabinet_height, fill_strategy, drawer_db_full)
    
    # --- FINAL FORMATTING PASS (X of Y) ---
    formatted_rows = []
    for idx, row in summary_df.iterrows():
        d_type = row['Drawer Type']
        bins_used = row['Total Bins Used']
        drawers = row['Drawers Required']
        
        # Look up capacity per drawer
        cap_per_drawer = drawer_caps.get(d_type, 0)
        total_cap = drawers * cap_per_drawer
        
        # Format "X of Y"
        if total_cap > 0:
            row['Total Bins Used'] = f"{int(bins_used)} of {int(total_cap)}"
        else:
            row['Total Bins Used'] = "0 of 0" # Empty drawers
            
        formatted_rows.append(row)
    
    summary_df = pd.DataFrame(formatted_rows)
    # --------------------------------------

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
    
    df_prod_temp = pd.DataFrame(columns=[
        "Material ID", "Ordering Code", "Ordering Code Packed", "Ordering Code ANSI", 
        "Package Quantity", "Length Packed Product Metric", "Width Packed Product Metric", "Height Packed Product Metric", "Item Description"
    ])
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
prod_file = st.sidebar.file_uploader("2. Product Database (Excel)", type=['xlsx', 'csv'])
draw_file = st.sidebar.file_uploader("3. Drawer Database (Excel/CSV)", type=['xlsx', 'csv'])

st.sidebar.divider()
st.sidebar.header("Settings")
use_topoff = st.sidebar.checkbox("Enable Drawer Consolidation", value=True)
skip_missing_setting = st.sidebar.checkbox("Skip missing items (ignore errors)", value=False)
clearance = st.sidebar.number_input("Large Dimension Clearance (mm)", min_value=0.0, value=10.0, step=1.0)
fill_strat = st.sidebar.selectbox("Cabinet Gap Filling Strategy", ["expand", "empty"])

if inv_file and prod_file and draw_file:
    try:
        # Load Files
        inv_df = pd.read_excel(inv_file)
        
        if prod_file.name.endswith('.csv'):
            prod_df = pd.read_csv(prod_file)
        else:
            prod_df = pd.read_excel(prod_file)
            
        if draw_file.name.endswith('.csv'):
            draw_df = pd.read_csv(draw_file)
        else:
            draw_df = pd.read_excel(draw_file)

        # Standardize headers
        inv_df.columns = inv_df.columns.str.strip()
        prod_df.columns = prod_df.columns.str.strip()
        draw_df.columns = draw_df.columns.str.strip()
        
        # --- BUILD INDEX (CACHED) ---
        with st.spinner("Indexing Product DB..."):
            product_map = build_product_map(prod_df)

        st.subheader("Step 1: Data Matching")
        
        # Check matches using the cached map
        inv_df['Match_ID'] = inv_df['Material ID/ Product Code'].apply(clean_code_str)
        
        def check_status(row):
            if row['Match_ID'] in product_map:
                return "OK"
            l = row.get('Length (mm)')
            w = row.get('Width (mm)')
            h = row.get('Height (mm)')
            if pd.notnull(l) and pd.notnull(w) and pd.notnull(h) and l>0 and w>0 and h>0:
                return "Alien (Manual Dims)"
            return "Missing"
            
        inv_df['Status'] = inv_df.apply(check_status, axis=1)
        
        missing_df = inv_df[inv_df['Status'] == "Missing"].copy()
        valid_df = inv_df[inv_df['Status'] != "Missing"].copy()

        col_stat1, col_stat2 = st.columns(2)
        col_stat1.success(f"✅ {len(valid_df)} items matched or valid.")
        
        final_df_to_process = pd.DataFrame()
        ready_to_calculate = False

        if not missing_df.empty:
            col_stat2.error(f"⚠️ {len(missing_df)} items missing data.")
            if skip_missing_setting:
                st.warning("Skipping missing items per settings.")
                final_df_to_process = valid_df
                ready_to_calculate = True
            else:
                st.info("Fix or enable Skip.")
                c1, c2 = st.columns(2)
                with c1:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        missing_df.to_excel(writer, index=False)
                    st.download_button("Download Missing Report", buffer, "missing_data.xlsx")
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
                        product_map,
                        enable_consolidation=use_topoff, 
                        clearance_mm=clearance,
                        fill_strategy=fill_strat
                    )
                    
                    st.subheader("Results")
                    
                    if not no_fit_df.empty:
                        st.error(f"⚠️ {len(no_fit_df)} items could not fit!")
                        with st.expander("View No Fit Items"):
                            st.dataframe(no_fit_df)
                        b_nf = io.BytesIO()
                        with pd.ExcelWriter(b_nf, engine='xlsxwriter') as writer:
                            no_fit_df.to_excel(writer, index=False)
                        st.download_button("Download No Fit Report", b_nf, "no_fit.xlsx")
                    
                    if logs:
                        with st.expander("Gap Filling Logs"):
                            for l in logs: st.text(l)
                            
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Drawers", f"{int(summary_df['Drawers Required'].sum())}")
                    m2.metric("Total Height", f"{int(costs['total_height'])}\"")
                    m3.metric("Cabinets", f"{costs['cabinets_needed']}")
                    
                    st.subheader("Financials")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Drawer Cost", f"${costs['drawer_subtotal']:,.2f}")
                    c2.metric("Base Cost", f"${costs['base_subtotal']:,.2f}")
                    c3.metric("Shipping", f"${costs['shipping_subtotal']:,.2f}")
                    c4.metric("Total", f"${costs['grand_total']:,.2f}")
                    
                    t1, t2 = st.tabs(["Summary", "Details"])
                    with t1: st.dataframe(summary_df)
                    with t2: st.dataframe(detail_df)
                    
                    buffer_res = io.BytesIO()
                    with pd.ExcelWriter(buffer_res, engine='xlsxwriter') as writer:
                        summary_df.to_excel(writer, sheet_name="Summary", index=False)
                        detail_df.to_excel(writer, sheet_name="Details", index=False)
                        cost_df = pd.DataFrame([{"Item":"Total", "Cost":costs['grand_total']}])
                        cost_df.to_excel(writer, sheet_name="Cost", index=False)
                    st.download_button("Download Report", buffer_res, "Final_Report.xlsx")

    except Exception as e:
        st.error(f"Error: {e}")
