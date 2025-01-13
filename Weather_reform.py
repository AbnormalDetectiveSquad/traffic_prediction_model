import pandas as pd
import h5py
import os
import numpy as np
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator
import time
import pandas as pd
import numpy as np
from pyproj import Transformer
import math
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class WeatherMapper:
    def __init__(self, weather_processor, chunk_size=10000):
        """
        날씨 데이터 매핑을 위한 클래스
        
        Parameters:
        -----------
        weather_processor : WeatherDataProcessor
            초기화된 WeatherDataProcessor 인스턴스
        chunk_size : int
            한 번에 처리할 데이터 크기
        """
        self.weather_processor = weather_processor
        self.chunk_size = chunk_size
        self.transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
        self.link_grid_coords = {}
    
    def check_weather_distribution(self, data):
        """날씨 데이터의 분포를 확인하는 함수"""
        print("\n=== 날씨 데이터 분포 ===")
        print("\nPTY 값 분포:")
        print(data['PTY'].value_counts())
        
        print("\nRN1 기초 통계:")
        print(data['RN1'].describe())
        
        # 0이 아닌 값들만 확인
        non_zero_rn1 = data[data['RN1'] > 0]
        if len(non_zero_rn1) > 0:
            print("\n0이 아닌 RN1 값들:")
            print(non_zero_rn1['RN1'].describe())

    def convert_to_grid(self, x, y):
        """좌표계 변환 (EPSG:5179 → 기상청 격자)"""
        lon, lat = self.transformer.transform(x, y)
        
        v1 = 0 if lat > 38.0 else (6.0 if lat > 32.0 else 12.0)
        v2 = 0.0 if lon > 126.0 else 6.0
        
        nx = int(math.floor(((lon - 120.0) * 1.5) + 1 - v2))
        ny = int(math.floor(((lat - 24.0) * 1.5) + 1 - v1))
        
        return nx, ny
    def normalize_weather_data(self,data):
        """날씨 데이터 정규화"""
        # RN1 정규화 (0~68.5 → 0~1)
        data['RN1'] = data['RN1'] / 68.5
        
        # PTY 인코딩
        pty_mapping = {
            0: 0.0,      # 없음
            1: 0.8,      # 비
            2: 0.6,      # 비/눈
            3: 0.4,      # 눈
            5: -0.4,     # 빗방울
            6: -0.6,     # 빗방울눈날림
            7: -0.8      # 눈날림
        }
        data['PTY'] = data['PTY'].map(pty_mapping).fillna(0.0)
        
        return data
    def prepare_link_coords(self, links_gdf):
        """링크별 격자 좌표 미리 계산"""
        print("링크별 격자 좌표 계산 중...")
        
        # 중심점 계산
        if not all(col in links_gdf.columns for col in ['center_x', 'center_y']):
            links_gdf['center_x'] = links_gdf.geometry.centroid.x
            links_gdf['center_y'] = links_gdf.geometry.centroid.y
        
        # 각 링크별 격자 좌표 계산
        for _, row in tqdm(links_gdf.iterrows(), total=len(links_gdf)):
            nx, ny = self.convert_to_grid(row['center_x'], row['center_y'])
            self.link_grid_coords[row['LINK_ID']] = (nx, ny)
            
    def format_datetime(self, date, time):
        """날짜/시간 문자열 변환"""
        date_str = str(date)
        time_str = str(time).zfill(4)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:]}:00"
    
    def process_chunk(self, chunk):
        """데이터 청크 처리"""
        chunk = chunk.copy()
        chunk['PTY'] = 0.0
        chunk['RN1'] = 0.0
        
        # 날짜/시간별 그룹화
        chunk['datetime_str'] = chunk.apply(
            lambda x: self.format_datetime(x['Date'], x['Time']), axis=1
        )
        
        # 각 고유한 시간에 대해
        for datetime_str in chunk['datetime_str'].unique():
            time_mask = chunk['datetime_str'] == datetime_str
            
            # 각 고유한 링크에 대해
            for link_id in chunk.loc[time_mask, 'Link_ID'].unique():
                if link_id in self.link_grid_coords:
                    nx, ny = self.link_grid_coords[link_id]
                    try:
                        weather_data = self.weather_processor.get_weather_data(nx, ny, datetime_str)
                        if isinstance(weather_data, dict):
                            mask = time_mask & (chunk['Link_ID'] == link_id)
                            chunk.loc[mask, 'PTY'] = weather_data['PTY']
                            chunk.loc[mask, 'RN1'] = weather_data['RN1']
                    except Exception as e:
                        continue
        chunk = self.normalize_weather_data(chunk)
        return chunk.drop('datetime_str', axis=1)
    
    def add_weather_info(self, data, links_gdf, output_path=None, save_interval=None):
        """
        전체 데이터에 날씨 정보 추가
        
        Parameters:
        -----------
        data : pandas.DataFrame
            처리할 교통 데이터
        links_gdf : GeoDataFrame
            링크 정보가 있는 GeoDataFrame
        output_path : str, optional
            중간 결과를 저장할 경로
        save_interval : int, optional
            몇 개의 청크마다 저장할지 지정
            
        Returns:
        --------
        pandas.DataFrame
            날씨 정보가 추가된 데이터프레임
        """
        # 링크 좌표 미리 계산
        if not self.link_grid_coords:
            self.prepare_link_coords(links_gdf)
        
        total_chunks = len(data) // self.chunk_size + (1 if len(data) % self.chunk_size else 0)
        processed_chunks = []
        
        print(f"\n전체 데이터 크기: {len(data):,}행")
        print(f"청크 크기: {self.chunk_size:,}행")
        print(f"총 청크 수: {total_chunks:,}개")
        
        try:
            for i in tqdm(range(total_chunks), desc="청크 처리 중"):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, len(data))
                chunk = data.iloc[start_idx:end_idx]
                
                processed_chunk = self.process_chunk(chunk)
                processed_chunks.append(processed_chunk)
                
                # 중간 저장
                if output_path and save_interval and (i + 1) % save_interval == 0:
                    temp_df = pd.concat(processed_chunks, ignore_index=True)
                    temp_path = f"{output_path}_temp_{i+1}.parquet"
                    temp_df.to_parquet(temp_path)
                    processed_chunks = []  # 메모리 해제
                    print(f"\n중간 결과 저장됨: {temp_path}")
        
        except KeyboardInterrupt:
            print("\n처리가 중단되었습니다. 지금까지 처리된 데이터를 반환합니다.")
        
        if processed_chunks:
            final_df = pd.concat(processed_chunks, ignore_index=True)
            
            if output_path:
                final_df.to_parquet(f"{output_path}_final.parquet")
                print(f"\n최종 결과 저장됨: {output_path}_final.parquet")
            
            return final_df
        
        return None










class WeatherDataProcessor:
    def __init__(self):
        self.interpolators = {}
        
    def load_and_combine_csv(self, file_paths):
        """CSV 파일들을 불러와서 결합하고 정렬"""
        dfs = [pd.read_csv(path) for path in file_paths]
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # 시간 포맷 통일
        combined_df['TM'] = pd.to_datetime(combined_df['TM'])
        
        # nx, ny 범위 확인 및 정렬 (일반적인 기상청 격자 범위: nx=1~149, ny=1~253)
        combined_df = combined_df[
            (combined_df['nx'] >= 1) & (combined_df['nx'] <= 149) &
            (combined_df['ny'] >= 1) & (combined_df['ny'] <= 253)
        ]
        
        return combined_df.sort_values(['nx', 'ny', 'TM'])

    def convert_and_save_h5(self, df, output_path):
        """정렬된 데이터를 HDF5 파일로 변환하여 저장"""
        # 고유값 추출 및 정렬
        unique_nx = sorted(df['nx'].unique())
        unique_ny = sorted(df['ny'].unique())
        unique_time = sorted(df['TM'].unique())
        
        nx_size = len(unique_nx)
        ny_size = len(unique_ny)
        time_size = len(unique_time)
        
        # 5차원 배열 초기화
        data_array = np.zeros((nx_size, ny_size, time_size, 2), dtype=float)
        
        # DataFrame을 피벗하여 빠른 데이터 채우기
        for var_idx, var_name in enumerate(['PTY', 'RN1']):
            for time_idx, time in enumerate(unique_time):
                time_data = df[df['TM'] == time]
                for _, row in time_data.iterrows():
                    nx_idx = unique_nx.index(row['nx'])
                    ny_idx = unique_ny.index(row['ny'])
                    data_array[nx_idx, ny_idx, time_idx, var_idx] = row[var_name]
        
        # HDF5 파일로 저장
        with h5py.File(output_path, 'w') as h5_file:
            h5_file.create_dataset('weather_data', data=data_array)
            h5_file.create_dataset('nx', data=np.array(unique_nx))
            h5_file.create_dataset('ny', data=np.array(unique_ny))
            # 시간을 유닉스 타임스탬프로 저장
            timestamps = np.array([pd.Timestamp(t).timestamp() for t in unique_time])
            h5_file.create_dataset('time', data=timestamps)
    
    def load_h5_and_create_interpolator(self, h5_path):
        """HDF5 파일을 로드하고 interpolator 생성"""
        with h5py.File(h5_path, 'r') as h5_file:
            self.weather_data = h5_file['weather_data'][:]
            self.nx_coords = h5_file['nx'][:]
            self.ny_coords = h5_file['ny'][:]
            self.timestamps = h5_file['time'][:]
            
            # PTY와 RN1에 대한 interpolator 생성
            for i, var_name in enumerate(['PTY', 'RN1']):
                self.interpolators[var_name] = RegularGridInterpolator(
                    (self.nx_coords, self.ny_coords, self.timestamps),
                    self.weather_data[:, :, :, i],
                    method='linear',
                    bounds_error=False,
                    fill_value=None
                )
    
    def get_weather_data(self, nx, ny, datetime_str):
        """주어진 좌표와 시간에 대한 기상 데이터 보간값 반환"""
        try:
            # 시간 문자열을 타임스탬프로 변환
            timestamp = pd.Timestamp(datetime_str).timestamp()
            
            # 입력값이 범위 내에 있는지 확인
            if (nx < min(self.nx_coords) or nx > max(self.nx_coords) or
                ny < min(self.ny_coords) or ny > max(self.ny_coords) or
                timestamp < min(self.timestamps) or timestamp > max(self.timestamps)):
                raise ValueError("입력된 좌표 또는 시간이 데이터 범위를 벗어났습니다.")
            
            # 보간 수행
            point = np.array([nx, ny, timestamp])
            pty = float(self.interpolators['PTY'](point))
            rn1 = float(self.interpolators['RN1'](point))
            
            return {
                'PTY': round(pty, 1),
                'RN1': round(rn1, 1)
            }
            
        except Exception as e:
            return f"에러 발생: {str(e)}"
        
    def get_max_rn1(self):
        """전체 기간의 RN1 최대값 반환"""
        return np.max(self.weather_data[:, :, :, 1])  # RN1은 인덱스 1

    def analyze_rn1(self):
        """전체 기간의 RN1 분석 정보 반환"""
        rn1_data = self.weather_data[:, :, :, 1]  # RN1은 인덱스 1
        max_val = np.max(rn1_data)
        
        # 최대값 위치 찾기
        max_indices = np.where(rn1_data == max_val)
        nx_idx, ny_idx, time_idx = max_indices[0][0], max_indices[1][0], max_indices[2][0]
        
        # 실제 좌표값과 시간 가져오기
        max_nx = self.nx_coords[nx_idx]
        max_ny = self.ny_coords[ny_idx]
        max_time = datetime.fromtimestamp(self.timestamps[time_idx])
        
        return {
            'max_value': float(max_val),
            'location': (float(max_nx), float(max_ny)),
            'datetime': max_time.strftime('%Y-%m-%d %H:%M:%S')
        }



def run_performance_test():
    """성능 테스트를 위한 함수"""
    import time
    from datetime import datetime, timedelta
    import random
    
    # 테스트 설정
    n_queries = 1000  # 테스트할 쿼리 수
    
    # 테스트 데이터 생성
    nx_range = (57, 63)
    ny_range = (125, 127)
    start_date = datetime(2023, 9, 1)
    end_date = datetime(2024, 12, 31)
    
    # 랜덤 쿼리 생성
    test_queries = []
    for _ in range(n_queries):
        random_nx = random.uniform(nx_range[0], nx_range[1])
        random_ny = random.uniform(ny_range[0], ny_range[1])
        random_days = random.randint(0, (end_date - start_date).days)
        random_minutes = random.randint(0, 24*60-1)
        random_time = start_date + timedelta(days=random_days, minutes=random_minutes)
        test_queries.append((random_nx, random_ny, random_time.strftime('%Y-%m-%d %H:%M:%S')))
    
    return test_queries



def main():
    # 파일 경로 설정
    file_paths = ["./Weather/2023_pty_rn1.csv", "./Weather/2024_pty_rn1.csv"]
    output_path = "./Weather/weather_data.h5"
    
    # WeatherDataProcessor 인스턴스 생성
    processor = WeatherDataProcessor()
    
    # 데이터 처리 및 저장 시간 측정
    print("\n=== 데이터 처리 성능 테스트 ===")
    
    start_time = time.time()
    print("\n1. 데이터 로딩 및 결합 중...")
    combined_df = processor.load_and_combine_csv(file_paths)
    load_time = time.time() - start_time
    print(f"   소요 시간: {load_time:.2f}초")
    
    start_time = time.time()
    print("\n2. HDF5 파일 생성 중...")
    processor.convert_and_save_h5(combined_df, output_path)
    convert_time = time.time() - start_time
    print(f"   소요 시간: {convert_time:.2f}초")
    
    start_time = time.time()
    print("\n3. 보간기 초기화 중...")
    processor.load_h5_and_create_interpolator(output_path)
    init_time = time.time() - start_time
    print(f"   소요 시간: {init_time:.2f}초")
    
    print(f"\n전처리 완료. HDF5 파일 저장됨: {output_path}")
    print(f"총 전처리 시간: {load_time + convert_time + init_time:.2f}초")
    
    # 쿼리 성능 테스트
    print("\n=== 쿼리 성능 테스트 ===")
    test_queries = run_performance_test()
    
    # 단일 쿼리 테스트
    test_nx, test_ny, test_time = test_queries[0]
    start_time = time.time()
    result = processor.get_weather_data(test_nx, test_ny, test_time)
    single_query_time = time.time() - start_time
    
    print(f"\n단일 쿼리 테스트:")
    print(f"좌표 ({test_nx:.1f}, {test_ny:.1f})와 시간 {test_time}의 보간 결과:")
    print(f"결과: {result}")
    print(f"소요 시간: {single_query_time*1000:.2f}ms")
    
    # 벌크 쿼리 테스트
    print(f"\n벌크 쿼리 테스트 (총 {len(test_queries)}개 쿼리):")
    start_time = time.time()
    for nx, ny, time_str in test_queries:
        processor.get_weather_data(nx, ny, time_str)
    bulk_query_time = time.time() - start_time
    
    print(f"총 소요 시간: {bulk_query_time:.2f}초")
    print(f"쿼리당 평균 시간: {bulk_query_time*1000/len(test_queries):.2f}ms")

if __name__ == "__main__":
    main()


# 사용 예시
"""
from weather_reform import WeatherDataProcessor

# WeatherDataProcessor 초기화
processor = WeatherDataProcessor()
processor.load_h5_and_create_interpolator("./Weather/weather_data.h5")

# WeatherMapper 초기화
mapper = WeatherMapper(processor, chunk_size=10000)

# 날씨 정보 추가 (중간 저장 활성화)
result_df = mapper.add_weather_info(
    data=traffic_data,
    links_gdf=links_gdf,
    output_path="./weather_mapping_results",
    save_interval=10
)
"""