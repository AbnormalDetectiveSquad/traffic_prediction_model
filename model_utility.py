from Weather_reform import WeatherDataProcessor, WeatherMapper
from datetime import datetime, timedelta
import pandas as pd
# 프로세서 인스턴스 생성
processor = WeatherDataProcessor()

# h5 파일 로드
processor.load_h5_and_create_interpolator("./Weather/weather_data.h5")

# RN1 분석
rn1_info = processor.analyze_rn1()
print("\nRN1 분석 결과:")
print(f"최대 강수량: {rn1_info['max_value']}mm")
print(f"발생 위치: nx={rn1_info['location'][0]}, ny={rn1_info['location'][1]}")
print(f"발생 시각: {rn1_info['datetime']}")

#PTY와 RN1의 전반적인 분포를 확인하기 위한 테스트 코드를 작성해드리겠습니다:
# 특정 시점과 위치에서의 날씨 데이터 샘플링 체크
print("\n=== 날씨 데이터 샘플링 테스트 ===")

# 테스트할 위치와 시간들
test_locations = [
    (60, 126),  # 중심부
    (57, 125),  # 좌측
    (63, 127)   # 우측
]

test_times = [
    "2023-09-15 12:00:00",  # 한낮
    "2023-09-15 00:00:00",  # 자정
    rn1_info['datetime']     # 최대 강수량 발생 시점
]

print("\n1. 최대 강수량 지점 주변 날씨 변화:")
max_rain_location = rn1_info['location']
max_rain_time = datetime.strptime(rn1_info['datetime'], "%Y-%m-%d %H:%M:%S")

# 최대 강수량 발생 전후 1시간 확인
for hour_offset in range(-1, 2):
    check_time = max_rain_time + timedelta(hours=hour_offset)
    weather = processor.get_weather_data(
        max_rain_location[0], 
        max_rain_location[1], 
        check_time.strftime("%Y-%m-%d %H:%M:%S")
    )
    print(f"\n시간: {check_time}")
    print(f"PTY: {weather['PTY']}")
    print(f"RN1: {weather['RN1']}mm")

print("\n2. 다양한 위치에서의 날씨 샘플링:")
for nx, ny in test_locations:
    for test_time in test_times:
        weather = processor.get_weather_data(nx, ny, test_time)
        print(f"\n위치 (nx={nx}, ny={ny}), 시간: {test_time}")
        print(f"PTY: {weather['PTY']}")
        print(f"RN1: {weather['RN1']}mm")

print("\n3. 비가 온 시점 찾기")
# 최대 강수 지점에서 전체 기간 중 비가 온 시점들 찾기
nx, ny = rn1_info['location']
rain_times = []

# 샘플링 간격을 6시간으로 설정
start_time = datetime(2023, 9, 1)
end_time = datetime(2024, 12, 31)
current_time = start_time

while current_time <= end_time:
    try:
        weather = processor.get_weather_data(
            nx, ny, 
            current_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        if weather['RN1'] > 0 or weather['PTY'] > 0:
            rain_times.append((current_time, weather['PTY'], weather['RN1']))
    except:
        pass
    current_time += timedelta(hours=6)

print(f"\n비가 온 시점들 (6시간 간격 샘플링):")
for time, pty, rn1 in rain_times:
    print(f"시간: {time}, PTY: {pty}, RN1: {rn1}mm")

print("\n4. PTY 값 분포 확인")
pty_values = set()
for nx, ny in test_locations:
    current_time = start_time
    while current_time <= start_time + timedelta(days=7):  # 일주일간 샘플링
        try:
            weather = processor.get_weather_data(
                nx, ny,
                current_time.strftime("%Y-%m-%d %H:%M:%S")
            )
            if weather['PTY'] > 0:
                pty_values.add(weather['PTY'])
        except:
            pass
        current_time += timedelta(hours=1)

print(f"\n발견된 PTY 값들: {sorted(list(pty_values))}")
print("\n3. 비가 온 시점 찾기")
# PTY와 RN1 정규화를 위한 매핑
pty_mapping = {
    0: 0.0,      # 없음
    1: 0.8,      # 비
    2: 0.6,      # 비/눈
    3: 0.4,      # 눈
    5: -0.4,     # 빗방울
    6: -0.6,     # 빗방울눈날림
    7: -0.8      # 눈날림
}

nx, ny = rn1_info['location']
rain_times = []

start_time = datetime(2023, 9, 1)
end_time = datetime(2024, 12, 31)
current_time = start_time

while current_time <= end_time:
    try:
        weather = processor.get_weather_data(
            nx, ny, 
            current_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        if weather['RN1'] > 0 or weather['PTY'] > 0:
            # 정규화 적용
            normalized_pty = pty_mapping.get(weather['PTY'], 0.0)
            normalized_rn1 = weather['RN1'] / 68.5
            rain_times.append((current_time, normalized_pty, normalized_rn1))
    except:
        pass
    current_time += timedelta(hours=6)

print(f"\n비가 온 시점들 (6시간 간격 샘플링, 정규화된 값):")
for time, pty, rn1 in rain_times:
    print(f"시간: {time}, PTY: {pty:.3f}, RN1: {rn1:.3f}")

 #WeatherMapper 인스턴스 생성
processor = WeatherDataProcessor()
processor.load_h5_and_create_interpolator("./Weather/weather_data.h5")
mapper = WeatherMapper(processor, chunk_size=500000)

# 테스트용 데이터프레임 생성
test_data = pd.DataFrame({
    'PTY': [0, 1, 2, 3, 5, 6, 7],
    'RN1': [0, 14.5, 0.5, 10.0, 20.0, 68.5, 30.0]
})
test_data_cp=test_data.copy()
# 정규화 함수 테스트
normalized_data = mapper.normalize_weather_data(test_data)

print("\n정규화 전/후 비교:")
for i in range(len(test_data)):
    print(f"\n원본  - PTY: {test_data_cp['PTY'].iloc[i]}, RN1: {test_data_cp['RN1'].iloc[i]}mm")
    print(f"정규화 - PTY: {normalized_data['PTY'].iloc[i]:.3f}, RN1: {normalized_data['RN1'].iloc[i]:.3f}")