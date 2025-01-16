import pandas as pd
import h5py
import os
import numpy as np
from datetime import datetime
#from scipy.interpolate import RegularGridInterpolator
import time
import numpy as np
from pyproj import Transformer
import math
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime, timedelta
import os
import warnings
import hashlib


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
        self.weather_grid_points = None
        self.cache_dir = "./Weather/grid_cache"
        self.mapping_file = None
        self.link_ids = None
        self.grid_coords = None
        os.makedirs(self.cache_dir, exist_ok=True)
        self.initialize_weather_grid()
        
    def initialize_weather_grid(self):
        """기상청 격자점 초기화"""
        nx_range = range(57, 64)  # 57에서 63까지
        ny_range = range(125, 128)  # 125에서 127까지
        grid_points = []
        for nx in nx_range:
            for ny in ny_range:
                grid_points.append((nx, ny))
                
        self.weather_grid_points = np.array(grid_points)
    
    def generate_cache_filename(self, links_gdf):
        """링크 데이터를 기반으로 캐시 파일 이름 생성"""
        # 링크 데이터의 특성을 기반으로 해시 생성
        link_ids = sorted(links_gdf['LINK_ID'].unique())
        hash_input = '_'.join(map(str, link_ids))
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:10]
        return f"grid_mapping_{hash_value}.h5"
    
    def save_grid_mapping(self, links_gdf):
        """격자 매핑을 H5 파일로 저장"""
        filename = self.generate_cache_filename(links_gdf)
        filepath = os.path.join(self.cache_dir, filename)
        
        # link_id와 grid 좌표를 numpy array로 변환
        link_ids = np.array(list(self.link_grid_coords.keys()))
        grid_coords = np.array(list(self.link_grid_coords.values()))
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('link_ids', data=link_ids.astype(np.float32))
            f.create_dataset('grid_coords', data=grid_coords)
            f.attrs['creation_date'] = datetime.now().isoformat()
        
        print(f"격자 매핑이 저장됨: {filepath}")
    
    def open_grid_mapping(self, links_gdf):
        """저장된 격자 매핑 파일 열기"""
        filename = self.generate_cache_filename(links_gdf)
        filepath = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(filepath):
            try:
                self.mapping_file = h5py.File(filepath, 'r')
                self.link_ids = self.mapping_file['link_ids'][:]
                self.grid_coords = self.mapping_file['grid_coords'][:]
                self.link_grid_coords = dict(zip(self.link_ids, map(tuple, self.grid_coords)))
                print(f"격자 매핑 파일을 열음: {filepath}")
                return True
            except Exception as e:
                print(f"격자 매핑 파일 열기 실패: {e}")
                self.close_grid_mapping()
                return False
        return False

    def close_grid_mapping(self):
        """격자 매핑 파일 닫기 및 리소스 정리"""
        if self.mapping_file is not None:
            try:
                self.mapping_file.close()
            except Exception as e:
                print(f"격자 매핑 파일 닫기 실패: {e}")
            finally:
                self.mapping_file = None
                self.link_ids = None
                self.grid_coords = None
    
    def find_nearest_grid_point(self, nx, ny):
        """가장 가까운 기상청 격자점 찾기"""
        point = np.array([nx, ny])
        distances = np.sqrt(np.sum((self.weather_grid_points - point) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        return tuple(self.weather_grid_points[nearest_idx])
    
    def convert_to_grid(self, x, y):
        """좌표계 변환 (EPSG:5179 → 기상청 격자)"""
        lon, lat = self.transformer.transform(x, y)
        
        v1 = 0 if lat > 38.0 else (6.0 if lat > 32.0 else 12.0)
        v2 = 0.0 if lon > 126.0 else 6.0
        
        nx = ((lon - 120.0) * 1.5) + 1 - v2
        ny = ((lat - 24.0) * 1.5) + 1 - v1
        
        return nx, ny
    
    def prepare_link_coords(self, links_gdf):
        """링크별 격자 좌표와 가장 가까운 기상청 격자점 계산"""
        # 먼저 저장된 매핑 파일 열기 시도
        if self.open_grid_mapping(links_gdf):
            return
        
        print("저장된 격자 매핑을 찾을 수 없음. 새로운 매핑 계산 중...")
        
        if not all(col in links_gdf.columns for col in ['center_x', 'center_y']):
            links_gdf['center_x'] = links_gdf.geometry.centroid.x
            links_gdf['center_y'] = links_gdf.geometry.centroid.y
        
        for _, row in tqdm(links_gdf.iterrows(), total=len(links_gdf)):
            nx, ny = self.convert_to_grid(row['center_x'], row['center_y'])
            nearest_nx, nearest_ny = self.find_nearest_grid_point(nx, ny)
            self.link_grid_coords[row['LINK_ID']] = (nearest_nx, nearest_ny)
        
        # 계산된 매핑 저장
        self.save_grid_mapping(links_gdf)
    
    def normalize_weather_data(self, data):
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
    
    def process_chunk(self, chunk):
        """데이터 청크 처리"""
        chunk = chunk.copy()
        chunk['PTY'] = 0.0
        chunk['RN1'] = 0.0
        
        # 날짜/시간별 그룹화
        chunk['datetime_str'] = chunk.apply(
            lambda x: f"{str(x['Date'])[:4]}-{str(x['Date'])[4:6]}-{str(x['Date'])[6:]} {str(x['Time']).zfill(4)[:2]}:{str(x['Time']).zfill(4)[2:]}:00",
            axis=1
        )
        
        # 각 고유한 시간에 대해
        for datetime_str in chunk['datetime_str'].unique():
            time_mask = chunk['datetime_str'] == datetime_str
            
            # 각 고유한 링크에 대해
            for link_id in chunk.loc[time_mask, 'Link_ID'].unique():
                if link_id in self.link_grid_coords:
                    nx, ny = self.link_grid_coords[link_id]
                    
                    try:
                        # 시간에 대해 전후 데이터 가져오기
                        before_data = self.weather_processor.get_nearest_before_data(nx, ny, datetime_str)
                        after_data = self.weather_processor.get_nearest_after_data(nx, ny, datetime_str)
                        
                        if before_data or after_data:
                            # 결과 저장 (가장 가까운 시간의 데이터 사용)
                            weather_data = before_data if before_data else after_data
                            mask = time_mask & (chunk['Link_ID'] == link_id)
                            chunk.loc[mask, 'PTY'] = weather_data['PTY']
                            chunk.loc[mask, 'RN1'] = weather_data['RN1']
                            
                    except Exception as e:
                        continue
        
        chunk = self.normalize_weather_data(chunk)
        return chunk.drop('datetime_str', axis=1)
    
    def add_weather_info(self, data, links_gdf, output_path=None, save_interval=None):
        """전체 데이터에 날씨 정보 추가"""
        if not self.link_grid_coords:
            self.prepare_link_coords(links_gdf)
        
        total_chunks = len(data) // self.chunk_size + (1 if len(data) % self.chunk_size else 0)
        processed_chunks = []
        
        try:
            for i in range(total_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, len(data))
                chunk = data.iloc[start_idx:end_idx]
                
                processed_chunk = self.process_chunk(chunk)
                processed_chunks.append(processed_chunk)
                
                # 중간 저장
                if output_path and save_interval and (i + 1) % save_interval == 0:
                    temp_df = pd.concat(processed_chunks, ignore_index=True)
                    temp_path = f"{output_path}_temp_{i+1}.h5"
                    
                    with h5py.File(temp_path, 'w') as f:
                        for col in temp_df.columns:
                            f.create_dataset(col, data=temp_df[col].values)
                    
                    processed_chunks = []
        
        except KeyboardInterrupt:
            print("\n처리가 중단되었습니다.")
        
        if processed_chunks:
            final_df = pd.concat(processed_chunks, ignore_index=True)
            
            if output_path:
                final_path = f"{output_path}_final.h5"
                with h5py.File(final_path, 'w') as f:
                    for col in final_df.columns:
                        f.create_dataset(col, data=final_df[col].values)
            
            return final_df
        
        return None

    def __del__(self):
        """소멸자: 열린 파일 핸들러 정리"""
        self.close_grid_mapping()









class WeatherDataProcessor:
    def __init__(self):
        """날씨 데이터 처리를 위한 클래스 초기화"""
        self.weather_data = None
        self.nx_coords = None
        self.ny_coords = None
        self.timestamps = None
        self.h5_file = None
    
    def load_h5_and_create_interpolator(self, h5_path):
        """HDF5 파일 로드 및 데이터 준비
        
        Args:
            h5_path (str): 날씨 데이터가 저장된 H5 파일 경로
        """
        try:
            self.h5_file = h5py.File(h5_path, 'r')
            self.weather_data = self.h5_file['weather_data']
            self.nx_coords = self.h5_file['nx'][:]
            self.ny_coords = self.h5_file['ny'][:]
            self.timestamps = self.h5_file['time'][:]
            print(f"날씨 데이터 로드 완료: nx 범위 [{min(self.nx_coords)}-{max(self.nx_coords)}], "
                  f"ny 범위 [{min(self.ny_coords)}-{max(self.ny_coords)}]")
        except Exception as e:
            raise RuntimeError(f"날씨 데이터 로드 실패: {str(e)}")
    
    def __del__(self):
        """소멸자: H5 파일 핸들러 정리"""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception as e:
                print(f"H5 파일 닫기 실패: {str(e)}")
    
    def find_grid_indices(self, nx, ny):
        """격자점의 인덱스 찾기"""
        if self.weather_data is None:
            raise RuntimeError("날씨 데이터가 로드되지 않았습니다.")
        
        try:
            nx_idx = np.where(self.nx_coords == int(round(nx)))[0]
            ny_idx = np.where(self.ny_coords == int(round(ny)))[0]
            
            if len(nx_idx) == 0 or len(ny_idx) == 0:
                return None, None
            
            return nx_idx[0], ny_idx[0]
        except Exception:
            return None, None
    
    def get_nearest_before_data(self, nx, ny, datetime_str):
        """주어진 시간 이전의 가장 가까운 데이터 찾기"""
        if self.weather_data is None:
            raise RuntimeError("날씨 데이터가 로드되지 않았습니다.")
        
        nx_idx, ny_idx = self.find_grid_indices(nx, ny)
        if nx_idx is None or ny_idx is None:
            return None
        
        try:
            target_timestamp = pd.Timestamp(datetime_str).timestamp()
            before_idx = np.searchsorted(self.timestamps, target_timestamp) - 1
            
            if before_idx < 0:
                return None
            
            return {
                'time': datetime.fromtimestamp(self.timestamps[before_idx]).strftime('%Y-%m-%d %H:%M:%S'),
                'PTY': float(self.weather_data[nx_idx, ny_idx, before_idx, 0]),
                'RN1': float(self.weather_data[nx_idx, ny_idx, before_idx, 1])
            }
        except Exception:
            return None
    
    def get_nearest_after_data(self, nx, ny, datetime_str):
        """주어진 시간 이후의 가장 가까운 데이터 찾기"""
        if self.weather_data is None:
            raise RuntimeError("날씨 데이터가 로드되지 않았습니다.")
        
        nx_idx, ny_idx = self.find_grid_indices(nx, ny)
        if nx_idx is None or ny_idx is None:
            return None
        
        try:
            target_timestamp = pd.Timestamp(datetime_str).timestamp()
            after_idx = np.searchsorted(self.timestamps, target_timestamp)
            
            if after_idx >= len(self.timestamps):
                return None
            
            return {
                'time': datetime.fromtimestamp(self.timestamps[after_idx]).strftime('%Y-%m-%d %H:%M:%S'),
                'PTY': float(self.weather_data[nx_idx, ny_idx, after_idx, 0]),
                'RN1': float(self.weather_data[nx_idx, ny_idx, after_idx, 1])
            }
        except Exception:
            return None
    
    def get_data_range_info(self):
        """데이터의 시간 및 좌표 범위 정보 반환"""
        if self.weather_data is None:
            raise RuntimeError("날씨 데이터가 로드되지 않았습니다.")
        
        start_time = datetime.fromtimestamp(self.timestamps[0])
        end_time = datetime.fromtimestamp(self.timestamps[-1])
        
        return {
            'time_range': (start_time, end_time),
            'nx_range': (int(min(self.nx_coords)), int(max(self.nx_coords))),
            'ny_range': (int(min(self.ny_coords)), int(max(self.ny_coords))),
            'pty_range': (float(np.min(self.weather_data[:,:,:,0])), 
                         float(np.max(self.weather_data[:,:,:,0]))),
            'rn1_range': (float(np.min(self.weather_data[:,:,:,1])), 
                         float(np.max(self.weather_data[:,:,:,1])))
        }



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

    
    # 단일 쿼리 테스트

    start_time = time.time()

    single_query_time = time.time() - start_time
    
    print(f"\n단일 쿼리 테스트:")

    print(f"소요 시간: {single_query_time*1000:.2f}ms")
    

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