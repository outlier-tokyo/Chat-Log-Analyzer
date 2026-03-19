import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional

# レイアウトの共通設定
THEME_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Roboto, sans-serif", size=14),
    margin=dict(l=40, r=40, t=60, b=40),
)

def plot_user_demographics(df: pd.DataFrame, title: str = "User Demographics"):
    """
    ユーザー属性（年代・性別）の分布を可視化する
    """
    if 'attribute_age' not in df.columns and 'attribute_gender' not in df.columns:
        print("[WARN] No demographic columns found in DataFrame")
        return None

    # 性別分布
    if 'attribute_gender' in df.columns:
        gender_counts = df.groupby('attribute_gender').size().reset_index(name='count')
        fig = px.pie(
            gender_counts, 
            values='count', 
            names='attribute_gender', 
            title=f"{title} - Gender",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig.update_layout(**THEME_LAYOUT)
        return fig
    
    return None

def plot_topic_distribution(df: pd.DataFrame, title: str = "Topic Distribution"):
    """
    クラスタごとの話題の割合を可視化する
    """
    if 'cluster' not in df.columns:
        print("[WARN] 'cluster' column not found in DataFrame")
        return None
    
    cluster_counts = df.groupby('cluster').size().reset_index(name='count')
    # ノイズ (-1) のラベルを見やすく変換
    cluster_counts['cluster_label'] = cluster_counts['cluster'].apply(
        lambda x: f"Cluster {x}" if x != -1 else "Noise"
    )
    
    fig = px.bar(
        cluster_counts.sort_values('count', ascending=False),
        x='cluster_label',
        y='count',
        color='count',
        text='count',
        title=title,
        labels={'cluster_label': 'Topic Cluster', 'count': 'Message Volume'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(textposition='outside')
    
    return fig

def plot_daily_volume(df: pd.DataFrame, timestamp_col: str = 'timestamp'):
    """
    メッセージ量の時系列推移を可視化する
    """
    if timestamp_col not in df.columns:
        print(f"[WARN] '{timestamp_col}' column not found")
        return None
    
    # 日付単位で集計
    df_time = df.copy()
    df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
    volume_series = df_time.resample('D', on=timestamp_col).size().reset_index(name='count')
    
    fig = px.line(
        volume_series,
        x=timestamp_col,
        y='count',
        title="Message Volume over Time",
        labels={'count': 'Number of Messages', timestamp_col: 'Date'},
        render_mode='svg'
    )
    fig.update_layout(**THEME_LAYOUT)
    fig.update_traces(line=dict(width=3, color='#636EFA'), mode='lines+markers')
    
    return fig

def plot_word_frequency(word_freq_df: pd.DataFrame, top_n: int = 20):
    """
    単語出現頻度の棒グラフ
    """
    fig = px.bar(
        word_freq_df.head(top_n),
        x='frequency',
        y='word',
        orientation='h',
        title=f"Top {top_n} Keywords",
        color='frequency',
        color_continuous_scale='Blues'
    )
    fig.update_layout(**THEME_LAYOUT)
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig