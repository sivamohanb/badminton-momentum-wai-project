import pandas as pd
import zipfile, shutil, os
import xml.etree.ElementTree as ET

D = ''
HYPER_FILENAME = 'BadmintonMomentum.hyper'  # will sit in Data/ inside the twbx

tables = {
    'games_by_disc': pd.read_pickle(D + 'tableau_games_by_disc.pkl'),
    'games_by_year': pd.read_pickle(D + 'tableau_games_by_year.pkl'),
    'cond_prob': pd.read_pickle(D + 'tableau_cond_prob.pkl'),
    'run_len': pd.read_pickle(D + 'tableau_run_len.pkl'),
    'odds_ratio': pd.read_pickle(D + 'tableau_odds_ratio.pkl'),
    'model_auc': pd.read_pickle(D + 'tableau_model_auc.pkl'),
    'margin_dist': pd.read_pickle(D + 'tableau_margin_dist.pkl'),
    'game_level': pd.read_pickle(D + 'tableau_game_level.pkl'),
}

def local_type(series):
    if pd.api.types.is_integer_dtype(series):
        return 'integer'
    if pd.api.types.is_float_dtype(series):
        return 'real'
    return 'string'

def default_agg(local_t):
    return 'Sum' if local_t in ('integer', 'real') else 'Count'

# Numeric fields that are used as discrete/categorical axes in the worksheets below
# (e.g. Year, Length, Margin) must be declared role='dimension' consistently in BOTH
# the <datasource> column list and every worksheet's <datasource-dependencies> that
# uses them - a role/type mismatch between the two is a likely cause of blank views.
FORCE_DIMENSION = {
    ('games_by_year', 'Year'),
    ('run_len', 'Length'),
    ('margin_dist', 'Margin'),
}

# Single source of truth for every (datasource, field) -> (datatype, role, vtype)
FIELD_INFO = {}

def field_info(ds_name, field, lt):
    forced = (ds_name, field) in FORCE_DIMENSION
    dtype = 'integer' if lt == 'integer' else ('real' if lt == 'real' else 'string')
    role = 'dimension' if (forced or lt == 'string') else 'measure'
    vtype = 'nominal' if (forced or lt == 'string') else 'quantitative'
    return dtype, role, vtype

# ---------------- datasource XML ----------------
def make_datasource(ds_name, table_name, df):
    conn_name = f"hyper.{table_name}"
    relation = (f"<relation connection='{conn_name}' name='{table_name}' "
                f"table='[Extract].[{table_name}]' type='table' />")
    metadata_records = []
    columns_xml = []
    for i, c in enumerate(df.columns):
        lt = local_type(df[c])
        agg = default_agg(lt)
        remote_type = {'integer': '20', 'real': '5', 'string': '129'}[lt]
        metadata_records.append(f"""
      <metadata-record class='column'>
        <remote-name>{c}</remote-name>
        <remote-type>{remote_type}</remote-type>
        <local-name>[{c}]</local-name>
        <parent-name>[{table_name}]</parent-name>
        <remote-alias>{c}</remote-alias>
        <ordinal>{i}</ordinal>
        <local-type>{lt}</local-type>
        <aggregation>{agg}</aggregation>
        <contains-null>true</contains-null>
      </metadata-record>""")
        dtype, role, vtype = field_info(ds_name, c, lt)
        FIELD_INFO[(ds_name, c)] = (dtype, role, vtype)
        columns_xml.append(f"  <column datatype='{dtype}' name='[{c}]' role='{role}' type='{vtype}' />")

    return f"""
  <datasource caption='{ds_name}' inline='true' name='{ds_name}' version='18.1'>
    <connection class='federated'>
      <named-connections>
        <named-connection caption='{HYPER_FILENAME}' name='{conn_name}'>
          <connection class='hyper' dbname='Data/{HYPER_FILENAME}' schema='Extract' server='' />
        </named-connection>
      </named-connections>
      {relation}
      <metadata-records>{''.join(metadata_records)}
      </metadata-records>
    </connection>
{chr(10).join(columns_xml)}
  </datasource>"""

datasources_xml = "\n".join(make_datasource(name, name, df) for name, df in tables.items())

# ---------------- worksheet XML ----------------
def make_bar_worksheet(ws_name, ds_name, dim_field, meas_field, meas_agg='Sum', color_field=None):
    dim_dtype, _, dim_vtype = FIELD_INFO[(ds_name, dim_field)]
    meas_dtype, _, meas_vtype = FIELD_INFO[(ds_name, meas_field)]
    dim_shelf_name = f"none:{dim_field}:nk"
    meas_shelf_name = f"{meas_agg.lower()}:{meas_field}:qk"
    color_enc = ""
    color_instance = ""
    color_col_decl = ""
    if color_field:
        color_dtype, _, color_vtype = FIELD_INFO[(ds_name, color_field)]
        color_shelf_name = f"none:{color_field}:nk"
        color_enc = f"<color column='[{ds_name}].[{color_shelf_name}]' />"
        color_instance = (f"\n      <column-instance column='[{color_field}]' derivation='None' "
                           f"name='[{color_shelf_name}]' pivot='key' type='{color_vtype}' />")
        color_col_decl = f"\n      <column datatype='{color_dtype}' name='[{color_field}]' role='dimension' type='{color_vtype}' />"
    dep_cols = f"""
      <column datatype='{dim_dtype}' name='[{dim_field}]' role='dimension' type='{dim_vtype}' />
      <column datatype='{meas_dtype}' name='[{meas_field}]' role='measure' type='{meas_vtype}' />{color_col_decl}"""
    # column-instance elements bind the shelf identifiers used in <rows>/<cols>/<encodings>
    # back to the real columns with an explicit aggregation ("derivation") - without these,
    # Tableau accepts the file but has nothing to resolve the shelf references to, and the
    # view renders blank.
    dep_cols += f"""
      <column-instance column='[{dim_field}]' derivation='None' name='[{dim_shelf_name}]' pivot='key' type='{dim_vtype}' />
      <column-instance column='[{meas_field}]' derivation='{meas_agg}' name='[{meas_shelf_name}]' pivot='key' type='{meas_vtype}' />{color_instance}"""
    rows_shelf = f"[{ds_name}].[{meas_shelf_name}]"
    cols_shelf = f"[{ds_name}].[{dim_shelf_name}]"
    return f"""
  <worksheet name='{ws_name}'>
    <table>
      <view>
        <datasources>
          <datasource name='{ds_name}' />
        </datasources>
        <datasource-dependencies datasource='{ds_name}'>{dep_cols}
        </datasource-dependencies>
        <aggregation value='true' />
      </view>
      <style />
      <panes>
        <pane>
          <view>
            <breakdown value='auto' />
          </view>
          <mark class='Automatic' />
          <encodings>
            {color_enc}
          </encodings>
        </pane>
      </panes>
      <rows>{rows_shelf}</rows>
      <cols>{cols_shelf}</cols>
    </table>
  </worksheet>"""

worksheets = []
worksheets.append(make_bar_worksheet('Games by Discipline', 'games_by_disc', 'Type', 'Games'))
worksheets.append(make_bar_worksheet('Games by Year', 'games_by_year', 'Year', 'Games'))
worksheets.append(make_bar_worksheet('Conditional Win Probability', 'cond_prob', 'Type', 'Probability', 'Avg', color_field='Condition'))
worksheets.append(make_bar_worksheet('Run Length - Observed', 'run_len', 'Length', 'Observed', 'Sum'))
worksheets.append(make_bar_worksheet('Run Length - IID Benchmark', 'run_len', 'Length', 'IIDBenchmark', 'Sum'))
worksheets.append(make_bar_worksheet('Odds Ratio by Discipline', 'odds_ratio', 'Type', 'OddsRatio', 'Sum'))
worksheets.append(make_bar_worksheet('Model AUC Comparison', 'model_auc', 'Model', 'TestAUC', 'Avg', color_field='Task'))
worksheets.append(make_bar_worksheet('Margin Distribution', 'margin_dist', 'Margin', 'Games', 'Sum'))
worksheets_xml = "\n".join(worksheets)

WORKSHEET_NAMES = ['Games by Discipline', 'Games by Year', 'Conditional Win Probability',
                    'Run Length - Observed', 'Run Length - IID Benchmark',
                    'Odds Ratio by Discipline', 'Model AUC Comparison', 'Margin Distribution']

# ---------------- dashboard XML ----------------
def dash_zone(idx, ws_name, x, y, w, h):
    return f"<zone h='{h}' id='{idx}' name='{ws_name}' w='{w}' x='{x}' y='{y}' />"

dashboard1_zones = "\n".join([
    dash_zone(1, 'Conditional Win Probability', 0, 0, 50, 50),
    dash_zone(2, 'Odds Ratio by Discipline', 50, 0, 50, 50),
    dash_zone(3, 'Run Length - Observed', 0, 50, 50, 50),
    dash_zone(4, 'Run Length - IID Benchmark', 50, 50, 50, 50),
])

dashboard2_zones = "\n".join([
    dash_zone(1, 'Model AUC Comparison', 0, 0, 60, 100),
    dash_zone(2, 'Margin Distribution', 60, 0, 40, 100),
])

dashboards_xml = f"""
  <dashboard name='Momentum Findings'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' sizing-mode='automatic' />
    <zones>
      <zone h='100000' id='0' type-v2='layout-basic' w='100000' x='0' y='0'>
        {dashboard1_zones}
      </zone>
    </zones>
  </dashboard>
  <dashboard name='Predictive Models'>
    <style />
    <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' sizing-mode='automatic' />
    <zones>
      <zone h='100000' id='0' type-v2='layout-basic' w='100000' x='0' y='0'>
        {dashboard2_zones}
      </zone>
    </zones>
  </dashboard>"""

# ---------------- story XML ----------------
def story_point(idx, caption, ws_name):
    return f"""
      <storypoint caption='{caption}' id='{idx}'>
        <storyworksheet name='{ws_name}' />
      </storypoint>"""

story_points = "\n".join([
    story_point(1, '11,871 professional games, 2015-2017', 'Games by Discipline'),
    story_point(2, 'No hot hand: probability drops after a win', 'Conditional Win Probability'),
    story_point(3, 'The effect is strongest in doubles', 'Odds Ratio by Discipline'),
    story_point(4, 'Fewer long streaks than chance predicts', 'Run Length - Observed'),
    story_point(5, 'AI models: weak point-level signal, strong win-probability model', 'Model AUC Comparison'),
])

# ---------------- story: NOTE ----------------
# Tableau's workbook schema does not accept a top-level <storyboards> element
# (confirmed by validation against real Tableau Desktop). Stories are left out
# of the generated XML; see the README for a 30-second manual step to add one.
story_xml = ""

# ---------------- windows XML (required section listing open views) ----------------
windows = "\n".join([f"<window name='{n}'><cards /></window>" for n in WORKSHEET_NAMES])
windows += "\n<window name='Momentum Findings'><cards /></window>"
windows += "\n<window name='Predictive Models'><cards /></window>"

TWB = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook source-build='2023.3.0' source-platform='linux' version='18.1' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <preferences />
  <datasources>{datasources_xml}
  </datasources>
  <worksheets>{worksheets_xml}
  </worksheets>
  <dashboards>{dashboards_xml}
  </dashboards>
  <windows>
    {windows}
  </windows>
</workbook>
"""

twb_path = D + 'Badminton_Momentum_Story.twb'
with open(twb_path, 'w', encoding='utf-8') as f:
    f.write(TWB)

# validate well-formed XML
try:
    ET.parse(twb_path)
    print("XML is well-formed.")
except ET.ParseError as e:
    print("XML PARSE ERROR:", e)
    raise

print("TWB written:", twb_path, os.path.getsize(twb_path), "bytes")
