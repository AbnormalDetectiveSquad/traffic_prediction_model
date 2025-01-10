import os
import pandas as pd

# 병합할 파일 디렉토리 경로 설정
directory = "./"  # 현재 디렉토리

# 파일 그룹별로 머지할 파일 패턴 설정
patterns = ["baddebugX", "baddebugY", "gooddebugX", "gooddebugY", "baddebugS", "gooddebugS"]

# 파일 머지 함수 정의 (열 단위로 결합)
def merge_files_as_columns(pattern, output_filename):
    files = [f for f in os.listdir(directory) if f.startswith(pattern) and f.endswith(".csv")]

    # 숫자가 포함된 파일만 필터링
    files_with_numbers = [f for f in files if any(char.isdigit() for char in f)]
    
    # 파일 이름에서 숫자를 추출하여 정렬
    files_with_numbers.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  
    
    # 병합 작업
    merged_data = pd.DataFrame()
    for file in files_with_numbers:
        file_path = os.path.join(directory, file)
        data = pd.read_csv(file_path, header=None)  # 각 파일 읽기
        
        # 데이터를 병합: 기존 데이터 옆에 열로 추가
        if merged_data.empty:
            merged_data = data  # 첫 번째 파일은 그대로 사용
        else:
            merged_data = pd.concat([merged_data, data.reset_index(drop=True)], axis=1)  # 열 추가

    # 결과 저장
    merged_data.to_csv(os.path.join(directory, output_filename), index=False, header=False)
    print(f"Merged {pattern} files into {output_filename}")
    for file in files_with_numbers:
        os.remove(os.path.join(directory, file))
        print(f"Deleted file: {file}")
# 각 패턴별로 파일 병합 수행
for pattern in patterns:
    merge_files_as_columns(pattern, f"{pattern}_merged_columns.csv")
