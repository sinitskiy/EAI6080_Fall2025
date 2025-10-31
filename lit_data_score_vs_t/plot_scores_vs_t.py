import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob

benchmarks = [
    'GPQA', 
    'MMLU',
    'MMLU-Pro',
    'MMMU'
    ]

# TODO
# add metadata for missing top models

try:
    df_models = pd.read_csv('metadata_on_models.csv')
    print(f"Total models with metadata loaded: {len(df_models)}")
    df_models['release_date'] = pd.to_datetime(df_models['release_date'], errors='coerce')
except FileNotFoundError:
    # merge metadata from both files
    df1 = pd.read_csv('Models_Metadata.csv')
    # Parse Dates
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%B %Y')
        except:
            try:
                return pd.to_datetime(date_str, format='%B %d %Y')
            except:
                try:
                    return pd.to_datetime(date_str)
                except:
                    return None

    df1['release_date'] = df1['time_released'].apply(parse_date)
    df_failed = df1[df1['release_date'].isna()]
    if len(df_failed) > 0:
        print(f"Parsing dates failed for {df_failed[['model', 'time_released']]}")

    df2 = pd.read_csv('Model_info.csv')
    df2['release_date'] = df2['time_released'].apply(parse_date)
    df_failed = df2[df2['release_date'].isna()]
    if len(df_failed) > 0:
        print(f"Parsing dates failed for {df_failed[['model', 'time_released']]}")
        
    # df1 contains dates, so it has priority
    df2_missing_in_1 = df2[~df2['model'].isin(set(df1['model']))]
    df_models = pd.concat([df1, df2_missing_in_1], ignore_index=True)
    df_models.to_csv('metadata_on_models.csv', index=False)
    print(f"Total models with metadata found in two files: {len(df_models)}")

missing_metadata_overall = None
for benchmark in benchmarks:
    print(f"Starting with benchmark: {benchmark}")
    dfs = []
    for name in glob.glob(f'{benchmark}_*.csv'):
        df = pd.read_csv(name)
        source = name[len(f"{benchmark}_"):-4] if name.lower().endswith('.csv') else name
        df['source'] = source
        dfs.append(df)

    df_benchmark = pd.concat(dfs, ignore_index=True)
    df_benchmark = df_benchmark[df_benchmark['score'] > 0.01]

    # Find and remove duplicates
    duplicates = df_benchmark[df_benchmark.duplicated(subset=['model'], keep=False)]
    print(f"Duplicate models:\n{duplicates}\nTaking occurrences with largest scores.")
    df_benchmark = df_benchmark.sort_values('score', ascending=False).drop_duplicates(subset=['model'], keep='first')
    print(f"Total models with scores: {len(df_benchmark)}")


    # Merge with model metadata
    df_merged = df_benchmark.merge(df_models[['model', 'release_date', 'open_vs_proprietary']], on='model', how='inner')
    print(f"Models with metadata: {len(df_merged)}")
    missing_metadata = df_benchmark[~df_benchmark['model'].isin(set(df_merged['model']))]
    missing_metadata = missing_metadata.sort_values('score', ascending=False)
    missing_metadata_overall = missing_metadata if missing_metadata_overall is None else pd.concat([missing_metadata_overall, missing_metadata], ignore_index=True)
    print(f"Models missing metadata:\n{', '.join(missing_metadata['model'])}")


    df_merged = df_merged.dropna(subset=['release_date', 'score'])
    df_merged = df_merged.sort_values('release_date').reset_index(drop=True)
    print(f"Models with valid dates: {len(df_merged)}")
    df_merged.to_csv(f'data_with_dates_and_categories_{benchmark.lower()}.csv', index=False)
    min_date, max_date = df_merged['release_date'].min(), df_merged['release_date'].max()
    df_merged['date_numeric'] = (df_merged['release_date'] - min_date).dt.days

    # Separate by category
    proprietary_data = df_merged[df_merged['open_vs_proprietary'] == 'proprietary'].copy()
    opensource_data = df_merged[df_merged['open_vs_proprietary'] == 'open-source'].copy()

    # Calculate cumulative best score over time
    proprietary_data['cumulative_best'] = proprietary_data['score'].cummax()
    opensource_data['cumulative_best'] = opensource_data['score'].cummax()

    # Group by date to handle duplicates
    proprietary_grouped = proprietary_data.groupby('date_numeric').agg({
        'cumulative_best': 'max'
    }).reset_index()

    opensource_grouped = opensource_data.groupby('date_numeric').agg({
        'cumulative_best': 'max'
    }).reset_index()

    # Prepare curves
    x_prop = proprietary_grouped['date_numeric'].values
    y_prop = proprietary_grouped['cumulative_best'].values * 100
    x_open = opensource_grouped['date_numeric'].values
    y_open = opensource_grouped['cumulative_best'].values * 100

    # Calculate what's at the end
    latest_prop_score = y_prop[-1]
    latest_open_score = y_open[-1]
    gap = latest_prop_score - latest_open_score
    latest_date = max(x_prop[-1], x_open[-1])
    x_prop = np.append(x_prop, latest_date)
    y_prop = np.append(y_prop, latest_prop_score)
    x_open = np.append(x_open, latest_date)
    y_open = np.append(y_open, latest_open_score)
    print(f"Latest gap: {gap:.1f}%")

    # plot
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.step(x_prop, y_prop, where='post', color='blue', linewidth=5, zorder=10)
    ax.step(x_open, y_open, where='post', color='green', linewidth=5, zorder=10)
    # add original datapoints
    ax.plot(proprietary_data['date_numeric'], proprietary_data['score'] * 100, 'o', color='blue', markersize=16, alpha=0.6, label='Proprietary Data Points', zorder=5)
    ax.plot(opensource_data['date_numeric'], opensource_data['score'] * 100, 'o', color='green', markersize=16, alpha=0.6, label='Open-source Data Points', zorder=5)
    ax.set_xlabel('Time', fontsize=32, fontweight='bold')
    ax.set_ylabel('Current best score', fontsize=32, fontweight='bold', rotation=90, ha='right', va='top')
    ax.tick_params(axis='y', which='major', length=12, width=2, direction='out', pad=10)
    ax.yaxis.set_label_coords(-0.1, 0.9)
    yticks = [0, 25, 50, 75, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, ha='right', fontsize=20)
    ax.set_ylim(0, 100)
    date_range = (max_date - min_date).days
    ax.set_xlim(-date_range*0.02, date_range*1.02)
    # X-axis: label Jan 1 of each year within the data range
    years = range(min_date.year - 1, max_date.year + 1)
    jan1_dates = [datetime(y, 1, 1) for y in years if datetime(y, 1, 1) >= min_date and datetime(y, 1, 1) <= max_date]
    if jan1_dates:
        xticks = [(d - min_date).days for d in jan1_dates]
        ax.set_xticks(xticks)
        ax.set_xticklabels([d.strftime('Jan 1 %Y') for d in jan1_dates], ha='center', fontsize=20)
        ax.tick_params(axis='x', which='major', length=12, width=2, direction='out', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(f'{benchmark}_curves_over_time.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# print out missing metadata
if missing_metadata_overall is not None and len(missing_metadata_overall) > 0:
    missing_metadata_overall = missing_metadata_overall.sort_values('score', ascending=False)
    missing_metadata_overall = missing_metadata_overall.drop_duplicates(subset=['model'], keep='first')
    print(f"\nOverall models missing metadata ({len(missing_metadata_overall)}):")
    for idx, row in missing_metadata_overall.iterrows():
        print(f"{row['model']}, taken from {row['source']}")
    missing_metadata_overall[['model', 'source']].to_csv('models_missing_metadata_overall.csv', index=False)