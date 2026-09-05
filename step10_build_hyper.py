import pandas as pd
from tableauhyperapi import (HyperProcess, Telemetry, Connection, CreateMode,
                              SqlType, TableDefinition, TableName, Inserter, NOT_NULLABLE, NULLABLE)

D = ''
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

PY_TO_SQL = {}

def sql_type_for(series):
    if pd.api.types.is_integer_dtype(series):
        return SqlType.big_int()
    if pd.api.types.is_float_dtype(series):
        return SqlType.double()
    return SqlType.text()

hyper_path = D + 'BadmintonMomentum.hyper'

# Target the oldest on-disk Hyper format this engine can produce (database version 1),
# for maximum backward compatibility with the reader's installed Tableau version - the
# freshly pip-installed Hyper API otherwise defaults to its newest format (version 4),
# which a somewhat older Tableau Desktop/Public install may not be able to open at all
# (manifesting as a generic, unhelpful "Internal Error").
HYPER_PARAMS = {'default_database_version': '1'}

with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, parameters=HYPER_PARAMS) as hyper:
    with Connection(endpoint=hyper.endpoint, database=hyper_path, create_mode=CreateMode.CREATE_AND_REPLACE) as conn:
        conn.catalog.create_schema('Extract')
        for name, df in tables.items():
            cols = []
            for c in df.columns:
                cols.append(TableDefinition.Column(c, sql_type_for(df[c]), NULLABLE))
            tdef = TableDefinition(table_name=TableName('Extract', name), columns=cols)
            conn.catalog.create_table(tdef)
            with Inserter(conn, tdef) as inserter:
                for row in df.itertuples(index=False):
                    vals = []
                    for v, c in zip(row, df.columns):
                        if pd.api.types.is_integer_dtype(df[c]):
                            vals.append(int(v))
                        elif pd.api.types.is_float_dtype(df[c]):
                            vals.append(float(v))
                        else:
                            vals.append(str(v))
                    inserter.add_row(vals)
                inserter.execute()
            print(f"Created table {name}: {len(df)} rows, columns={list(df.columns)}")

print("Hyper extract built:", hyper_path)
