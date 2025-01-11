import holidays
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from script import dataloader
import geopandas as gpd
from tqdm import tqdm
from main import get_parameters

def validate_and_create_pivot_table(combined_data):
    # 중복 데이터 확인
    duplicates = combined_data.duplicated(subset=['Date', 'Time', 'Link_ID'], keep=False)
    
    # 중복이 있는 경우 경고 메시지 출력
    if duplicates.any():
        duplicate_entries = combined_data[duplicates]
        print(f"Warning: Duplicate entries found! {len(duplicate_entries)} duplicate rows detected.")
        print(duplicate_entries.head())  # 중복된 일부 데이터 확인
    
    # Pivot 테이블 생성
    pivot_data = combined_data.pivot_table(
        index=['Date', 'Time'], 
        columns='Link_ID', 
        values='Avg_Speed', 
        aggfunc='mean', 
        fill_value=0  # 누락된 값은 0으로 채움
    )
    
    # 날짜와 시간 강제 정렬
    pivot_data = pivot_data.sort_index()
    
    return pivot_data
def get_files_list(number,option='non-holidays',start=None, data_dir='/home/ssy/extract_its_data'):
    # 파일 리스트 가져오기
    all_files = [f for f in os.listdir(data_dir) if f.endswith('_5Min.csv')]
    if option == 'all':
        print(f"Available files (all): {len(all_files)}")
        output=random.sample(all_files, number)
        print(output)
    elif option == 'non-holidays':
        # 공휴일 데이터 정의
        kr_holidays = holidays.KR(years=range(2024, 2025))
        # 공휴일 제외
        files = [
            f for f in all_files
            if datetime.strptime(f[:8], '%Y%m%d').date() not in kr_holidays
            and datetime.strptime(f[:8], '%Y%m%d').weekday() < 5  # 월~금 (0~4)
        ]
        print(f"Available files (non-holidays): {len(files)}")
        output=random.sample(files, number)
        print(output)
    elif option == 'sequential':
        # 시작 날짜가 없는 경우 에러 반환
        if start is None:
            raise ValueError("For 'sequential' option, 'start' must be provided in YYYYMMDD format.")
        
        # 시작 날짜 처리
        try:
            start_date = datetime.strptime(start, '%Y%m%d').date()
        except ValueError:
            raise ValueError("Invalid 'start' date format. Expected 'YYYYMMDD'.")
        
        # 파일 이름에서 날짜 추출 및 정렬
        sorted_files = sorted(all_files, key=lambda f: datetime.strptime(f[:8], '%Y%m%d').date())

        # 연속된 날짜의 파일 선택
        selected_files = []
        current_date = start_date

        while len(selected_files) < number:
            current_date_str = current_date.strftime('%Y%m%d')
            matching_files = [f for f in sorted_files if f.startswith(current_date_str)]

            if matching_files:
                selected_files.extend(matching_files)

            current_date += timedelta(days=1)

            # 파일 개수 초과 방지
            if len(selected_files) >= number:
                selected_files = selected_files[:number]
                break

        print(f"Available files (sequential): {len(selected_files)}")
        output = selected_files
    else:
        raise ValueError(f"Invalid option '{option}'. Choose from 'all', 'non-holidays', 'sequential'.")
    
    return output
def process_and_save_speed_matrix(data_dir, file, dataset_path, map_file_name='filtered_nodes_filtered_links_table.csv'):
    # 맵핑 테이블 경로
    mapping_file_path = os.path.join(dataset_path, map_file_name)
    
    # 맵핑 테이블 로드
    if not os.path.exists(mapping_file_path):
        print(f"Mapping file not found: {mapping_file_path}.")
        args, device, blocks = get_parameters()
        print(f"Loaded parameters: {args}")
        nodes_name = "filtered_nodes.shp"
        links_name = "filtered_links.shp"
        nodes_gdf = gpd.read_file(os.path.join(dataset_path, nodes_name))
        links_gdf = gpd.read_file(os.path.join(dataset_path, links_name))
        save_option, dataset_path_new = dataloader.check_table_files(dataset_path, nodes_name, links_name)
        dense_matrix, n_links = dataloader.create_adjacency_matrix(links_gdf, nodes_gdf, save_option, dataset_path_new, args.k_threshold)
        print(f"Link-Index map saved to {dataset_path_new}")
    
    mapping_table = pd.read_csv(mapping_file_path)
    print(f"Loaded mapping table with {len(mapping_table)} entries.")
    
    # 맵핑 테이블에서 순서대로 링크 ID 가져오기
    link_order = mapping_table.sort_values('Matrix_Index')['Link_ID'].tolist()
    
    # 시간순 데이터를 저장할 리스트
    all_data = []
    processed_files = []
    
    # 파일 리스트 순회
    for f in tqdm(file, desc=f'Loding data from {data_dir}'):
        file_path = os.path.join(data_dir, f)
        
        # 파일 로드
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping.")
            continue
        
        data = pd.read_csv(file_path, header=None)
        data.columns = ['Date', 'Time', 'Link_ID', 'Some_Column', 'Avg_Speed', 'Other']
        
        # 필요한 열만 유지
        data = data[['Link_ID', 'Date', 'Time', 'Avg_Speed']]
        
        # 시간순 정렬
        data = data.sort_values(by=['Date', 'Time'])
        
        # 데이터 추가
        all_data.append(data)
        processed_files.append(f)

    # 전체 데이터를 하나의 DataFrame으로 병합
    if not all_data:
        print("No valid data found. Exiting.")
        return
    
    combined_data = pd.concat(all_data, ignore_index=True)
    pivot_data = validate_and_create_pivot_table(combined_data)

    # 쌍둥이 피벗 테이블 생성
    combined_data['Date'] = pd.to_datetime(combined_data['Date'].astype(str), format='%Y%m%d')
    combined_data['Weight'] = calculate_weight_vectorized(combined_data['Date'])
    weight_pivot = combined_data.pivot_table(
        index=['Date', 'Time'], columns='Link_ID', values='Weight', aggfunc='mean', fill_value=0
    )

    # 누락된 링크 ID를 추가하고 0으로 채우기
    for link_id in tqdm(link_order, desc='Remove missing links'):
        if link_id not in pivot_data.columns:
            pivot_data[link_id] = 0  # 없는 링크 ID는 0으로 채움
        if link_id not in weight_pivot.columns:
            weight_pivot[link_id] = 0  # 없는 링크 ID는 0으로 채움

    # 링크 ID 순서 재정렬
    pivot_data = pivot_data[link_order]
    weight_pivot = weight_pivot[link_order]

    print(f"Number of rows in pivot_data: {len(pivot_data)}")
    
    # 속도 값만 남기고 저장 (인덱스 제거)
    output_data = pivot_data.to_numpy()  # numpy 배열로 변환하여 순수 값만 유지
    weight_data = weight_pivot.to_numpy()
    speed_output_path = os.path.join(dataset_path, 'vel.csv')
    weight_output_path = os.path.join(dataset_path, 'weights.csv')
    
    pd.DataFrame(output_data, columns=link_order).to_csv(speed_output_path, index=False, header=False)
    pd.DataFrame(weight_data, columns=link_order).to_csv(weight_output_path, index=False, header=False)
    
    print(f"Speed matrix saved to {speed_output_path}")
    print(f"Weight matrix saved to {weight_output_path}")

    # 처리된 파일 이름 저장
    file_list_path = os.path.join(dataset_path, 'processed_files.txt')
    with open(file_list_path, 'w') as file_list:
        file_list.write("\n".join(processed_files))

    print(f"Processed file list saved to {file_list_path}")

def process_and_save_speed_matrix_chunk(data_dir, file_list, dataset_path, map_file_name='filtered_nodes_filtered_links_table.csv'):
    # 맵핑 테이블 로드
    mapping_file_path = os.path.join(dataset_path, map_file_name)
    if not os.path.exists(mapping_file_path):
        print(f"Mapping file not found: {mapping_file_path}.")
        args, device, blocks = get_parameters()
        print(f"Loaded parameters: {args}")
        nodes_name = "filtered_nodes.shp"
        links_name = "filtered_links.shp"
        nodes_gdf = gpd.read_file(os.path.join(dataset_path, nodes_name))
        links_gdf = gpd.read_file(os.path.join(dataset_path, links_name))
        save_option, dataset_path_new = dataloader.check_table_files(dataset_path, nodes_name, links_name)
        dense_matrix, n_links = dataloader.create_adjacency_matrix(links_gdf, nodes_gdf, save_option, dataset_path_new, args.k_threshold)
        print(f"Link-Index map saved to {dataset_path_new}")
    
    mapping_table = pd.read_csv(mapping_file_path)
    print(f"Loaded mapping table with {len(mapping_table)} entries.")
    link_order = mapping_table.sort_values('Matrix_Index')['Link_ID'].tolist()
    
    # 결과 파일 경로
    speed_output_path = os.path.join(dataset_path, 'vel.csv')
    weight_output_path = os.path.join(dataset_path, 'weights.csv')
    
    # 파일 초기화 (빈 파일 생성)
    pd.DataFrame(columns=link_order).to_csv(speed_output_path, index=False, header=False)
    pd.DataFrame(columns=link_order).to_csv(weight_output_path, index=False, header=False)
    
    # 처리된 파일 목록
    processed_files = []
    
    # 배치 크기 설정
    BATCH_SIZE = 10  # 한 번에 처리할 파일 수
    
    # 파일을 배치로 나누어 처리
    for i in tqdm(range(0, len(file_list), BATCH_SIZE), desc=f'Loading data from {data_dir}'):
        batch_files = file_list[i:i + BATCH_SIZE]
        batch_data = []
        
        for f in batch_files:
            file_path = os.path.join(data_dir, f)
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}, skipping.")
                continue
            
            try:
                # 파일 로드
                data = pd.read_csv(file_path, header=None)
                data.columns = ['Date', 'Time', 'Link_ID', 'Some_Column', 'Avg_Speed', 'Other']
                data = data[['Link_ID', 'Date', 'Time', 'Avg_Speed']]
                data = data.sort_values(by=['Date', 'Time'])
                batch_data.append(data)
                processed_files.append(f)
            except Exception as e:
                print(f"Error processing file {f}: {str(e)}")
                continue
        
        if not batch_data:
            continue
            
        # 배치 데이터 통합 및 처리
        combined_batch = pd.concat(batch_data, ignore_index=True)
        combined_batch['Date'] = pd.to_datetime(combined_batch['Date'].astype(str), format='%Y%m%d')
        
        # 가중치 계산
        combined_batch['Weight'] = calculate_weight_vectorized(combined_batch['Date'])
        
        # 피벗 테이블 생성
        pivot_speed = combined_batch.pivot_table(
            index=['Date', 'Time'], 
            columns='Link_ID', 
            values='Avg_Speed', 
            aggfunc='mean',
            fill_value=0
        ).sort_index()  # 시간순 보장
        
        pivot_weight = combined_batch.pivot_table(
            index=['Date', 'Time'],
            columns='Link_ID',
            values='Weight',
            aggfunc='mean',
            fill_value=0
        ).sort_index()  # 시간순 보장
        
        # 누락된 링크 처리
        for link_id in link_order:
            if link_id not in pivot_speed.columns:
                pivot_speed[link_id] = 0
            if link_id not in pivot_weight.columns:
                pivot_weight[link_id] = 0
        
        # 링크 순서 정렬
        pivot_speed = pivot_speed[link_order]
        pivot_weight = pivot_weight[link_order]
        
        # numpy 배열로 변환
        speed_array = pivot_speed.to_numpy()
        weight_array = pivot_weight.to_numpy()
        
        # 배치 결과를 파일에 추가
        with open(speed_output_path, 'a') as f:
            pd.DataFrame(speed_array, columns=link_order).to_csv(f, 
                                                               index=False, 
                                                               header=False)
        
        with open(weight_output_path, 'a') as f:
            pd.DataFrame(weight_array, columns=link_order).to_csv(f, 
                                                                index=False, 
                                                                header=False)
        
        # 메모리 정리
        del batch_data, combined_batch, pivot_speed, pivot_weight
        del speed_array, weight_array
    
    # 처리된 파일 목록 저장
    file_list_path = os.path.join(dataset_path, 'processed_files.txt')
    with open(file_list_path, 'w') as file_list:
        file_list.write("\n".join(processed_files))

    print(f"Speed matrix saved to {speed_output_path}")
    print(f"Weight matrix saved to {weight_output_path}")
    print(f"Processed file list saved to {file_list_path}")

def calculate_weight_vectorized(dates):
    """
    날짜에 따른 가중치를 벡터화 방식으로 처리.
    주말(토,일)과 공휴일은 1, 월요일은 0, 금요일은 0.5, 그 외 평일은 0.1
    
    Args:
        dates: pandas datetime series
    Returns:
        Series: 각 날짜별 가중치
    """
    # 공휴일 설정
    kr_holidays = holidays.KR(years=range(2024, 2025))
    
    # 주말 여부
    weekdays = dates.dt.weekday
    
    # 공휴일 여부 (True/False 시리즈)
    is_holiday = dates.dt.date.map(lambda x: x in kr_holidays)
    
    # 가중치 계산 (주말이거나 공휴일이면 1)
    weights = pd.Series(index=dates.index)
    
    # 기본 가중치 설정
    weights.loc[weekdays == 0] = 0  # 월요일
    weights.loc[weekdays == 4] = 0.5  # 금요일
    weights.loc[(weekdays > 0) & (weekdays < 4)] = 0.1  # 화-목요일
    weights.loc[weekdays >= 5] = 1  # 주말
    
    # 공휴일은 1로 덮어쓰기
    weights.loc[is_holiday] = 1

    return weights


file=get_files_list(50,option='sequential',start='20240101')
data_dir='/home/ssy/extract_its_data'
path=os.path.join(data_dir,file[0])
data=pd.read_csv(path,header=None)
dataset_path_new='./data/seoul'
print(f"Loaded data from {data}")
process_and_save_speed_matrix_chunk(data_dir, file, dataset_path_new)