"""
src/geospatial.py  —  Collapse Signature Project
Clean rewrite with all fixes:
  - Year-range aware mapping (no empire bleeds past collapse)
  - Antimeridian polygon splitting
  - Robust Folium map builder
  - Plotly animation builder with era-based chunking
"""

import os, json, math
import geopandas as gpd
import pandas as pd
import numpy as np
import folium

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASEMAP_DIR  = os.path.join(BASE_DIR, "data", "geospatial",
                             "historical-basemaps", "geojson")
MAPPING_FILE = os.path.join(BASE_DIR, "data", "geospatial",
                             "civilisation_mapping.json")
CIVS_FILE    = os.path.join(BASE_DIR, "data", "processed",
                             "civilisations_clean.csv")

# ── COLOUR PALETTE ─────────────────────────────────────────────────────────────
_PALETTE = [
    "#C9A84C","#B8860B","#DAA520","#CD853F","#D2691E","#A0522D","#8B4513","#F4A460",
    "#4E79A7","#2E86AB","#1B4F72","#5DADE2","#85C1E9","#3498DB","#2980B9","#1F618D",
    "#154360","#76D7C4","#48C9B0","#1ABC9C","#17A589","#148F77","#0E6655",
    "#E15759","#C0392B","#922B21","#7B241C","#641E16","#E74C3C","#CB4335","#B03A2E",
    "#F28E2B","#E67E22","#D35400","#BA4A00","#A04000","#F39C12",
    "#76B7B2","#59A14F","#27AE60","#1E8449","#196F3D","#52BE80","#82E0AA","#2ECC71",
    "#B07AA1","#8E44AD","#7D3C98","#6C3483","#5B2C6F","#9B59B6","#A569BD","#BB8FCE",
    "#EDC948","#F1C40F","#D4AC0D","#B7950B","#9A7D0A",
    "#2C3E50","#1A252F","#273746","#212F3D","#566573","#717D7E","#808B96",
    "#AEB6BF","#BDC3C7","#D5D8DC",
]

def _build_colour_map(civs_df):
    names = sorted(set(civs_df['name'].tolist()))
    return {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}


# ── SNAPSHOT UTILITIES ─────────────────────────────────────────────────────────
def get_available_years():
    years = []
    for fname in os.listdir(BASEMAP_DIR):
        if not (fname.startswith("world_") and fname.endswith(".geojson")):
            continue
        stem = fname.replace("world_","").replace(".geojson","")
        try:
            y = -int(stem.replace("bc","")) if stem.startswith("bc") else int(stem)
            if -3100 <= y <= 2010:
                years.append(y)
        except ValueError:
            continue
    return sorted(years)

def nearest_snapshot_year(target, available):
    return min(available, key=lambda y: abs(y - target))

def year_to_filename(year):
    return f"world_bc{abs(year)}.geojson" if year < 0 else f"world_{year}.geojson"

def load_snapshot(year):
    fpath = os.path.join(BASEMAP_DIR, year_to_filename(year))
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    return gpd.read_file(fpath)


# ── MAPPING HELPERS ───────────────────────────────────────────────────────────
def _get_year_aware_reverse(mapping, year):
    """
    Build {repo_name: civ_name} for a specific year.
    Mapping values are lists of [repo_name, from_year, to_year].
    Only includes repo names whose year range contains `year`.
    """
    reverse = {}
    for civ_name, entries in mapping.items():
        if not entries:
            continue
        for entry in entries:
            if isinstance(entry, list) and len(entry) == 3:
                rn, from_y, to_y = entry
                if from_y <= year <= to_y and rn not in reverse:
                    reverse[rn] = civ_name
            elif isinstance(entry, str):
                # Legacy string format fallback
                if entry not in reverse:
                    reverse[entry] = civ_name
    return reverse


# ── CIVILISATION METADATA HELPERS ─────────────────────────────────────────────
def _clean_civs_df(civs_df):
    df = civs_df.copy()
    for col in df.columns:
        if 'year' in col.lower():
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def _is_active(civ_row, year):
    try:
        return int(civ_row['founded_year']) <= year <= int(civ_row['collapse_end_year'])
    except Exception:
        return False

def _get_opacity(civ_row, year):
    try:
        founded = int(civ_row.get('founded_year', year))
        peak    = int(civ_row.get('peak_year', year))
        c_start = int(civ_row.get('collapse_start_year', year))
        c_end   = int(civ_row.get('collapse_end_year', year))
    except Exception:
        return 0.6

    if year > c_end or year < founded:
        return 0.0
    if year <= peak:
        frac = (year - founded) / (peak - founded) if peak != founded else 1.0
        return max(0.3, min(0.85, 0.3 + frac * 0.55))
    elif year <= c_start:
        return 0.85
    else:
        frac = (year - c_start) / max(1, c_end - c_start)
        return max(0.08, 0.85 - frac * 0.77)


# ── ANTIMERIDIAN SPLIT ────────────────────────────────────────────────────────
def _split_antimeridian(lons, lats):
    segs, cur_lo, cur_la = [], [lons[0]], [lats[0]]
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i-1]) > 180:
            if len(cur_lo) >= 3:
                segs.append((cur_lo[:], cur_la[:]))
            cur_lo, cur_la = [lons[i]], [lats[i]]
        else:
            cur_lo.append(lons[i]); cur_la.append(lats[i])
    if len(cur_lo) >= 3:
        segs.append((cur_lo, cur_la))
    return segs or [(lons, lats)]

def _polygon_segments(geom):
    geom = geom.simplify(0.3, preserve_topology=True)
    if geom is None or geom.is_empty:
        return
    polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
    for poly in polys:
        if poly.is_empty or len(poly.exterior.coords) < 3:
            continue
        coords = list(poly.exterior.coords)
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        if lons[0] != lons[-1] or lats[0] != lats[-1]:
            lons.append(lons[0]); lats.append(lats[0])
        if max(lons) - min(lons) > 180:
            yield from _split_antimeridian(lons, lats)
        else:
            yield lons, lats


# ── FOLIUM EXPLORER MAP ────────────────────────────────────────────────────────
def build_empire_map(target_year, civs_df, mapping, colour_map,
                     available_years, show_stressors=True,
                     stressors_df=None, filter_regions=None,
                     filter_civs=None):

    civs_df    = _clean_civs_df(civs_df)
    snap_year  = nearest_snapshot_year(target_year, available_years)
    year_label = f"{abs(target_year)} {'BCE' if target_year < 0 else 'CE'}"

    plot_civs = civs_df.copy()
    if filter_regions:
        plot_civs = plot_civs[plot_civs['region'].isin(filter_regions)]
    if filter_civs:
        plot_civs = plot_civs[plot_civs['name'].isin(filter_civs)]

    m = folium.Map(location=[25, 20], zoom_start=2,
                   tiles='CartoDB dark_matter', prefer_canvas=True)

    # Year-aware reverse mapping
    year_reverse_raw = _get_year_aware_reverse(mapping, target_year)

    # Further filter: only include civs that are active at target_year
    # AND exist in our dataset AND pass region/civ filters
    plot_civ_names = set(plot_civs['name'].tolist())
    civ_meta = {row['name']: row.to_dict()
                for _, row in plot_civs.iterrows()
                if _is_active(row, target_year)}

    reverse = {}
    for rn, civ_name in year_reverse_raw.items():
        if civ_name in civ_meta:
            reverse[rn] = civ_meta[civ_name]

    try:
        gdf = load_snapshot(snap_year)
        empire_gdf = gdf[gdf['NAME'].isin(reverse.keys())].copy()
    except Exception:
        empire_gdf = gpd.GeoDataFrame()

    for _, row in empire_gdf.iterrows():
        civ_data = reverse.get(row['NAME'])
        if not civ_data:
            continue
        civ_name = civ_data['name']
        colour   = colour_map.get(civ_name, '#888888')
        opacity  = _get_opacity(civ_data, target_year)
        if opacity < 0.05:
            continue

        def fy(y):
            try: return f"{abs(int(y))} {'BCE' if int(y)<0 else 'CE'}"
            except: return '?'

        popup_html = (
            f"<div style='font-family:sans-serif;min-width:200px;max-width:260px;'>"
            f"<h4 style='margin:0 0 6px;color:{colour};font-size:14px;'>{civ_name}</h4>"
            f"<hr style='margin:4px 0;border-color:#444;'>"
            f"<b>Region:</b> {civ_data.get('region','?')}<br>"
            f"<b>Founded:</b> {fy(civ_data.get('founded_year'))}<br>"
            f"<b>Peak:</b> {fy(civ_data.get('peak_year'))}<br>"
            f"<b>Collapsed:</b> {fy(civ_data.get('collapse_end_year'))}<br>"
            f"<b>Peak pop:</b> {civ_data.get('peak_population_m','?')}M<br>"
            f"<b>Trigger:</b> {civ_data.get('primary_collapse_trigger','?')}<br>"
            f"<div style='margin-top:6px;font-size:10px;color:#777;'>"
            f"Viewing: {year_label}</div></div>"
        )
        try:
            folium.GeoJson(
                row['geometry'].__geo_interface__,
                style_function=lambda x, c=colour, o=opacity: {
                    'fillColor': c, 'color': c,
                    'weight': 0.8, 'fillOpacity': o,
                    'opacity': min(1.0, o + 0.2),
                },
                tooltip=folium.Tooltip(civ_name,
                    style="font-family:sans-serif;font-size:12px;"),
                popup=folium.Popup(popup_html, max_width=280)
            ).add_to(m)
        except Exception:
            continue

    # Lydian Kingdom bubble fallback
    lyd = civs_df[civs_df['name'] == 'Lydian Kingdom']
    for _, row in lyd.iterrows():
        if _is_active(row, target_year):
            colour  = colour_map.get(row['name'], '#888')
            opacity = _get_opacity(row, target_year)
            if opacity > 0.05:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=15, color=colour, fill=True,
                    fill_color=colour, fill_opacity=opacity * 0.7,
                    tooltip=f"{row['name']} (approximate location)"
                ).add_to(m)

    # Stressor labels
    if show_stressors and stressors_df is not None:
        try:
            active_s = stressors_df[
                (stressors_df['start_year'] <= target_year) &
                (stressors_df['end_year']   >= target_year)
            ]
            sev_col  = {'catastrophic':'#E15759','severe':'#F28E2B',
                        'moderate':'#EDC948','beneficial':'#59A14F'}
            type_ico = {'climate':'🌡','pandemic':'☠','migration':'⚔',
                        'war':'⚔','systemic_collapse':'💥'}
            html_labels = ""
            for i, (_, s) in enumerate(active_s.iterrows()):
                sc  = sev_col.get(s['severity'], '#999')
                ico = type_ico.get(s['event_type'], '⚠')
                top = 10 + i * 32
                html_labels += (
                    f"<div style='position:fixed;top:{top}px;right:10px;"
                    f"background:rgba(14,11,7,0.88);border:1px solid {sc};"
                    f"border-radius:3px;padding:3px 8px;font-family:sans-serif;"
                    f"font-size:11px;color:{sc};white-space:nowrap;z-index:9999;'>"
                    f"{ico} {s['event_name']}</div>"
                )
            if html_labels:
                m.get_root().html.add_child(folium.Element(html_labels))
        except Exception:
            pass

    # Legend — only truly active empires
    try:
        active_now = [
            row['name'] for _, row in plot_civs.iterrows()
            if _is_active(row, target_year)
        ]
        if active_now:
            items = "".join(
                f'<div style="display:flex;align-items:center;margin:2px 0;font-size:10px;">'
                f'<div style="width:12px;height:12px;border-radius:2px;'
                f'background:{colour_map.get(n,"#888")};margin-right:6px;flex-shrink:0;"></div>'
                f'<span style="color:#e8d5b0;">{n}</span></div>'
                for n in sorted(active_now)
            )
            m.get_root().html.add_child(folium.Element(f"""
            <div style="position:fixed;bottom:20px;left:20px;
                        background:rgba(14,11,7,0.92);border:1px solid #3a2e1a;
                        border-radius:4px;padding:10px 14px;font-family:sans-serif;
                        max-height:280px;overflow-y:auto;z-index:1000;">
                <div style="font-size:11px;font-weight:bold;color:#c9a84c;
                            margin-bottom:6px;letter-spacing:0.1em;">
                    ACTIVE EMPIRES — {year_label}
                </div>{items}</div>"""))
    except Exception:
        pass

    return m


# ── PLOTLY ANIMATION BUILDER ──────────────────────────────────────────────────
def build_plotly_animation(civs_df, mapping, colour_map, years_to_animate):
    import plotly.graph_objects as go

    civs_df = _clean_civs_df(civs_df)
    meta    = {row['name']: row.to_dict() for _, row in civs_df.iterrows()}

    plot_years   = [y for y in years_to_animate if -3100 <= y <= 2010]
    frames       = []
    frame_labels = []

    for year in plot_years:
        fpath = os.path.join(BASEMAP_DIR, year_to_filename(year))
        if not os.path.exists(fpath):
            continue

        # Year-aware reverse mapping for this specific year
        year_reverse = _get_year_aware_reverse(mapping, year)

        gdf        = gpd.read_file(fpath)
        empire_gdf = gdf[gdf['NAME'].isin(year_reverse.keys())].copy()

        year_label = f"{abs(year)} {'BCE' if year < 0 else 'CE'}"
        frame_labels.append(year_label)

        traces      = []
        legend_seen = set()

        for _, row in empire_gdf.iterrows():
            repo_name = row['NAME']
            civ_name  = year_reverse.get(repo_name)
            if not civ_name:
                continue

            # Must exist in dataset
            civ_data = meta.get(civ_name)
            if civ_data is None:
                continue

            # Strict lifespan check
            if not _is_active(civ_data, year):
                continue

            opacity = _get_opacity(civ_data, year)
            if opacity < 0.05:
                continue

            colour = colour_map.get(civ_name, '#888888')

            def fy(y):
                try: return f"{abs(int(y))} {'BCE' if int(y)<0 else 'CE'}"
                except: return '?'

            hover = (
                f"<b>{civ_name}</b><br>"
                f"Region: {civ_data.get('region','?')}<br>"
                f"Founded: {fy(civ_data.get('founded_year'))}<br>"
                f"Peak: {fy(civ_data.get('peak_year'))}<br>"
                f"Collapsed: {fy(civ_data.get('collapse_end_year'))}<br>"
                f"Peak pop: {civ_data.get('peak_population_m','?')}M<br>"
                f"Trigger: {civ_data.get('primary_collapse_trigger','?')}"
            )

            try:
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                for seg_lons, seg_lats in _polygon_segments(geom):
                    if len(seg_lons) < 3:
                        continue
                    show_leg = civ_name not in legend_seen
                    if show_leg:
                        legend_seen.add(civ_name)
                    traces.append(go.Scattergeo(
                        lon=seg_lons, lat=seg_lats,
                        mode='lines',
                        fill='toself',
                        fillcolor=colour,
                        opacity=opacity,
                        line=dict(color=colour, width=0.6),
                        name=civ_name,
                        text=hover,
                        hoverinfo='text',
                        showlegend=show_leg,
                    ))
            except Exception:
                continue

        frames.append(go.Frame(
            data=traces,
            name=year_label,
            layout=go.Layout(annotations=[dict(
                x=0.5, y=1.03, xref='paper', yref='paper',
                text=year_label, showarrow=False,
                font=dict(size=22, color='#c9a84c', family='Cinzel, serif'),
                bgcolor='rgba(14,11,7,0.85)',
                bordercolor='#3a2e1a', borderwidth=1, borderpad=6
            )])
        ))

    if not frames:
        return go.Figure()

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            annotations=frames[0].layout.annotations,
            geo=dict(
                showland=True,       landcolor='#1c1510',
                showocean=True,      oceancolor='#0a0e14',
                showlakes=True,      lakecolor='#0a0e14',
                showcoastlines=True, coastlinecolor='#2a2218',
                coastlinewidth=0.6,
                showcountries=False, showframe=False,
                bgcolor='#0e0b07',
                projection_type='natural earth',
                lataxis_range=[-60, 80],
                lonaxis_range=[-170, 180],
            ),
            paper_bgcolor='#0e0b07',
            font_color='#e8d5b0',
            height=620,
            margin=dict(t=50, b=90, l=0, r=0),
            legend=dict(
                bgcolor='rgba(14,11,7,0.85)',
                bordercolor='#3a2e1a', borderwidth=1,
                font=dict(color='#e8d5b0', size=9),
                x=0.01, y=0.99,
                title=dict(text='Empire', font=dict(color='#c9a84c'))
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'bgcolor': '#161009',
                'bordercolor': '#3a2e1a',
                'font': {'color':'#c9a84c','size':13,'family':'Cinzel, serif'},
                'x': 0.5, 'xanchor': 'center',
                'y': -0.10, 'yanchor': 'top',
                'buttons': [
                    {'label': '  ▶  Play', 'method': 'animate',
                     'args': [None, {
                         'frame': {'duration':800,'redraw':True},
                         'fromcurrent': True,
                         'transition': {'duration':400,'easing':'quadratic-in-out'}
                     }]},
                    {'label': '  ⏸  Pause', 'method': 'animate',
                     'args': [[None], {
                         'frame': {'duration':0,'redraw':False},
                         'mode': 'immediate',
                         'transition': {'duration':0}
                     }]}
                ]
            }],
            sliders=[{
                'active': 0,
                'bgcolor': '#161009',
                'bordercolor': '#3a2e1a',
                'tickcolor': '#3a2e1a',
                'font': {'color':'#9a8060','size':9},
                'currentvalue': {
                    'prefix': '',
                    'font': {'color':'#c9a84c','size':15,'family':'Cinzel, serif'},
                    'xanchor': 'center',
                    'visible': True,
                    'offset': 20
                },
                'pad': {'t':65,'b':10},
                'len': 0.9, 'x': 0.05, 'y': -0.02,
                'steps': [
                    {'method':'animate','label':fl,
                     'args':[[fl],{
                         'frame':{'duration':800,'redraw':True},
                         'mode':'immediate',
                         'transition':{'duration':400}
                     }]}
                    for fl in frame_labels
                ],
                'transition': {'duration':400,'easing':'cubic-in-out'}
            }]
        )
    )
    return fig


# ── ASSET LOADER ──────────────────────────────────────────────────────────────
def load_all_assets():
    civs_df         = _clean_civs_df(pd.read_csv(CIVS_FILE))
    with open(MAPPING_FILE) as f:
        mapping     = json.load(f)
    colour_map      = _build_colour_map(civs_df)
    available_years = get_available_years()
    sp = os.path.join(BASE_DIR, "data", "processed", "external_stressors.csv")
    stressors_df = pd.read_csv(sp) if os.path.exists(sp) else None
    return civs_df, mapping, colour_map, available_years, stressors_df


if __name__ == "__main__":
    print("Testing geospatial pipeline...")
    civs_df, mapping, colour_map, available_years, stressors_df = load_all_assets()
    print(f"Civilisations: {len(civs_df)}")
    print(f"Snapshots:     {len(available_years)}")
    for ty in [-200, 600, 1938]:
        rev   = _get_year_aware_reverse(mapping, ty)
        snap  = nearest_snapshot_year(ty, available_years)
        gdf   = load_snapshot(snap)
        found = []
        for name in gdf['NAME'].dropna().unique():
            if name in rev:
                civ = rev[name]
                civ_data = civs_df[civs_df['name'] == civ]
                if not civ_data.empty and _is_active(civ_data.iloc[0], ty):
                    found.append(civ)
        label = f"{abs(ty)} {'BCE' if ty<0 else 'CE'}"
        print(f"\nYear {label}: {sorted(set(found))}")
    print("\nDone.")
