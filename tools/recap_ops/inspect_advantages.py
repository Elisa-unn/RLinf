import pandas as pd, glob
for p in sorted(glob.glob('/workspace/datasets/RECAP-Libero10-Task0-48succ-Data/*/meta/advantages_*.parquet')):
    df = pd.read_parquet(p)
    print(p.replace('/workspace/datasets/RECAP-Libero10-Task0-48succ-Data/',''))
    print('  rows', len(df), '| cols', list(df.columns))
    for c in df.columns:
        if df[c].dtype == bool or set(df[c].dropna().unique()[:4]) <= {0,1,True,False}:
            print(f'  {c}: dtype={df[c].dtype} true_rate={df[c].mean():.4f}')
    print('  head:'); print(df.head(3).to_string(max_colwidth=28).replace('\n','\n    '))
