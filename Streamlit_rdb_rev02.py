import os
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import sqlite3
import googlemaps
from datetime import datetime
import urllib.request
from bs4 import BeautifulSoup
import urllib.parse  # URLエンコード、デコード


# 環境変数の読み込み
load_dotenv()

GOOGLEMAPS_API_KEY = os.getenv("GOOGLEMAPS_API_KEY")


# セッション状態の初期化
if 'show_all' not in st.session_state:
    st.session_state['show_all'] = False  # 初期状態は地図上の物件のみを表示

# 地図上以外の物件も表示するボタンの状態を切り替える関数
def toggle_show_all():
    st.session_state['show_all'] = not st.session_state['show_all']

# DBからデータを読み込む関数
def load_data_from_DB():
    conn = sqlite3.connect("Step3-1_database.db")

    query = "SELECT * FROM tech0_01"
    df = pd.read_sql_query(query, conn)

    return df

def load_data_from_DB2():
    conn = sqlite3.connect("Step3-1_database.db")

    query = "SELECT * FROM favorite_location"
    df = pd.read_sql_query(query, conn)

    return df


# データフレームの前処理を行う関数
def preprocess_dataframe(df):
    # '家賃' 列を浮動小数点数に変換し、NaN値を取り除く
    df['家賃'] = pd.to_numeric(df['家賃'], errors='coerce')
    df = df.dropna(subset=['家賃'])
    return df

def make_clickable(url, name):
    return f'<a target="_blank" href="{url}">{name}</a>'

# 地図を作成し、マーカーを追加する関数
def create_map(filtered_df):
    # 地図の初期設定
    map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # マーカーを追加
    for idx, row in filtered_df.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            # ポップアップに表示するHTMLコンテンツを作成
            popup_html = f"""
            <b>名称:</b> {row['名称']}<br>
            <b>アドレス:</b> {row['アドレス']}<br>
            <b>家賃:</b> {row['家賃']}万円<br>
            <b>間取り:</b> {row['間取り']}<br>
            <a href="{row['物件詳細URL']}" target="_blank">物件詳細</a>
            """
            # HTMLをポップアップに設定
            popup = folium.Popup(popup_html, max_width=400)
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup
            ).add_to(m)

    return m

# 検索結果を表示する関数
def display_search_results(filtered_df):
    # 物件番号を含む新しい列を作成
    filtered_df['物件番号'] = range(1, len(filtered_df) + 1)
    filtered_df['物件詳細URL'] = filtered_df['物件詳細URL'].apply(lambda x: make_clickable(x, "リンク"))
    display_columns = ['物件番号', '名称', 'アドレス', '階数', '家賃', '間取り', '物件詳細URL']
    filtered_df_display = filtered_df[display_columns]
    st.markdown(filtered_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

#周辺施設の検索
def search_nearby_facilities(latitude, longitude):


    # 環境変数から認証情報を取得
    gmaps = googlemaps.Client(key=GOOGLEMAPS_API_KEY)

    # 緯度と経度を用いた位置情報の設定
    home_location = f"{latitude},{longitude}"

    # 各施設を検索し、徒歩での移動時間を計算
    places = ['convenience_store', 'supermarket', 'gym']
    st.sidebar.write("■物件周辺施設")
    for place in places:
        # 最寄りの施設を検索
        result = gmaps.places_nearby(location=home_location, radius=1000, type=place)
        if result['results']:
            nearest_place = result['results'][0]
            place_location = nearest_place['geometry']['location']
            place_lat_lng = f"{place_location['lat']},{place_location['lng']}"

            # 徒歩での移動時間を計算
            directions_result = gmaps.directions(home_location,
                                                 place_lat_lng,
                                                 mode="walking",
                                                 departure_time=datetime.now())
            
            if directions_result and directions_result[0]['legs']:
                walk_duration = directions_result[0]['legs'][0]['duration']['text']
                
                st.sidebar.write(f" {place}: {nearest_place['name']} - 徒歩: {walk_duration}")
            else:
                st.sidebar.write(f" {place}: {nearest_place['name']} - Walking time not available")
        else:
            st.sidebar.writes(f"No {place} found within 1000 meters.")

#電車移動時間のスクレイピング
def search_transit_times(home_stations, destinations):
    # Yahoo乗換の基本URL
    url0 = 'https://transit.yahoo.co.jp/search/result?from='
    url1 = '&flatlon=&to='
    url2 = '&viacode=&viacode=&viacode=&shin=&ex=&hb=&al=&lb=&sr=&type=1&ws=3&s=&ei=&fl=1&tl=3&expkind=1&ticket=ic&mtf=1&userpass=0&detour_id=&fromgid=&togid=&kw='

    # 結果を保存するためのリスト
    results_list = []

    # 各出発駅と目的地駅の組み合わせで移動時間を検索
    for home_index, home_row in home_stations.iterrows():
        home_station_encoded = urllib.parse.quote(home_row['home_station'])
        for dest_index, dest_row in destinations.iterrows():
            for station_key in ['closest_station1', 'closest_station2']:
                if pd.isna(dest_row[station_key]):
                    continue
                dest_station = dest_row[station_key]
                dest_station_encoded = urllib.parse.quote(dest_station)
                url = url0 + home_station_encoded + url1 + dest_station_encoded + url2
                
                # URLにアクセスしてHTMLを取得
                req = urllib.request.urlopen(url)
                html = req.read().decode('utf-8')
                soup = BeautifulSoup(html, 'html.parser')
                time_elements = soup.select("li.time")
                
                if time_elements:
                    time_required = time_elements[1].select_one('span.small').text.strip()
                else:
                    time_required = '時間未確定'
                
                # 結果をリストに追加
                results_list.append({
                    '出発駅': home_row['home_station'],
                    '目的地': dest_row['location_name'],
                    '最寄り駅': dest_station,
                    '所要時間': time_required
                })

    # 結果のリストからデータフレームを作成
    results_df = pd.DataFrame(results_list)
    results_df = results_df.pivot_table(index='出発駅', columns=['目的地', '最寄り駅'], values='所要時間', aggfunc='first')
    return results_df


# メインのアプリケーション
def main():

    df = load_data_from_DB()
    df = preprocess_dataframe(df)

    # StreamlitのUI要素（スライダー、ボタンなど）の各表示設定
    st.title('賃貸物件情報の可視化')

    # エリアと家賃フィルタバーを1:2の割合で分割
    col1, col2 = st.columns([1, 2])

    with col1:
        # エリア選択
        area = st.radio('■ エリア選択', df['区'].unique())


    with col2:
        # 家賃範囲選択のスライダーをfloat型で設定し、小数点第一位まで表示
        price_min, price_max = st.slider(
            '■ 家賃範囲 (万円)', 
            min_value=float(1), 
            max_value=float(df['家賃'].max()),
            value=(float(df['家賃'].min()), float(df['家賃'].max())),
            step=0.1,  # ステップサイズを0.1に設定
            format='%.1f'
        )

    with col2:
    # 間取り選択のデフォルト値をすべてに設定
        type_options = st.multiselect('■ 間取り選択', df['間取り'].unique(), default=df['間取り'].unique())


    # フィルタリング/ フィルタリングされたデータフレームの件数を取得
    filtered_df = df[(df['区'].isin([area])) & (df['間取り'].isin(type_options))]
    filtered_df = filtered_df[(filtered_df['家賃'] >= price_min) & (filtered_df['家賃'] <= price_max)]
    filtered_count = len(filtered_df)

    # 'latitude' と 'longitude' 列を数値型に変換し、NaN値を含む行を削除
    filtered_df['latitude'] = pd.to_numeric(filtered_df['latitude'], errors='coerce')
    filtered_df['longitude'] = pd.to_numeric(filtered_df['longitude'], errors='coerce')
    filtered_df2 = filtered_df.dropna(subset=['latitude', 'longitude'])


    # 検索ボタン / # フィルタリングされたデータフレームの件数を表示
    col2_1, col2_2 = st.columns([1, 2])

    with col2_2:
        st.write(f"物件検索数: {filtered_count}件 / 全{len(df)}件")

    # 検索ボタン
    if col2_1.button('検索＆更新', key='search_button'):
        # 検索ボタンが押された場合、セッションステートに結果を保存
        st.session_state['filtered_df'] = filtered_df
        st.session_state['filtered_df2'] = filtered_df2
        st.session_state['search_clicked'] = True

    # Streamlitに地図を表示
    if st.session_state.get('search_clicked', False):
        m = create_map(st.session_state.get('filtered_df2', filtered_df2))
        folium_static(m)

    # 地図の下にラジオボタンを配置し、選択したオプションに応じて表示を切り替える
    show_all_option = st.radio(
        "表示オプションを選択してください:",
        ('地図上の検索物件のみ', 'すべての検索物件'),
        index=0 if not st.session_state.get('show_all', False) else 1,
        key='show_all_option'
    )

    # ラジオボタンの選択に応じてセッションステートを更新
    st.session_state['show_all'] = (show_all_option == 'すべての検索物件')

    # 検索結果の表示
    if st.session_state.get('search_clicked', False):
        if st.session_state['show_all']:
            display_search_results(st.session_state.get('filtered_df', filtered_df))  # 全データ
        else:
            display_search_results(st.session_state.get('filtered_df2', filtered_df2))  # 地図上の物件のみ

    

    property_list = filtered_df['名称'].tolist()
    selected_property = st.sidebar.selectbox("候補物件を選択してください", property_list)
        
    if selected_property:
    # 選択された物件に一致する行をDataFrameから取得
        property_info = filtered_df[filtered_df['名称'] == selected_property].iloc[0]
        st.write(f"選択された物件: {selected_property}")
        #st.dataframe(property_info)
        #home_address=property_info['アドレス']
        #home_station = property_info['アクセス1駅名']
        #home_station = home_station.replace('駅', '')
        home_latitude=property_info['latitude']
        home_longitude=property_info['longitude']

        search_nearby_facilities(home_latitude, home_longitude) 

    # Streamlit サイドバーにボタンを設置
    if st.sidebar.button('移動時間を検索'):
        access_columns = [col for col in property_info.index if col.startswith('アクセス') and col.endswith('駅名')]
        home_stations_list = [{'home_station': property_info[col].replace('駅', '')} for col in access_columns if pd.notna(property_info[col])]
        home_stations = pd.DataFrame(home_stations_list)

        #最寄り駅の表示
        #for key, value in home_stations.iterrows():
        #    st.write(f"{key}: {value['home_station']}")
        # 関数呼び出し
        transit_results = search_transit_times(home_stations, destinations)
        st.write("各出発駅から目的地までの移動時間", transit_results)
        
    else:
        st.sidebar.write("検索ボタンを押してください")

#サイドバー
df2 = load_data_from_DB2()

st.sidebar.write("お気に入りの場所")
#登録ボタンを入れてdf2に追加登録できるようにする

# ユーザーが複数の場所を選択できるようにチェックボックスを生成
selected_places = []
for place in df2['location_name']:
    # チェックボックスで場所を選択
    if st.sidebar.checkbox(place, key=place):
        selected_places.append(place)

# 選択された場所に基づいて新しいDataFrameを作成
destinations = df2[df2['location_name'].isin(selected_places)]

# 選択されたDataFrameを表示
# st.write("選択された場所:")
# st.dataframe(selected_df2)


# アプリケーションの実行
if __name__ == "__main__":
    if 'search_clicked' not in st.session_state:
        st.session_state['search_clicked'] = False
    if 'show_all' not in st.session_state:
        st.session_state['show_all'] = False
    main()