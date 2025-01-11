import threading
import queue
import h5py
import os
import warnings


class DataLoaderContext:
    def __init__(
        self,
        custom_loader,
        filepath,
        dataset_name_x,
        dataset_name_y,
        batch_size,
        buffer_size=300,
        shuffle=False,
        split_ratio=[0.7, 0.15, 0.15],
        device="cpu",
    ):
        """
        HDF5 파일에서 데이터를 로드하는 컨텍스트 관리자
        Args:
            custom_loader (callable): 사용자 정의 데이터 로더 함수
            filepath (str): HDF5 파일 경로
            dataset_name_x (str): 입력 데이터셋 이름
            dataset_name_y (str): 라벨 데이터셋 이름
            batch_size (int): 배치 크기
            buffer_size (int): 큐에 유지할 최대 배치 개수
            shuffle (bool): 셔플 여부
            split_ratio (list): [training, validation, test] 비율
            device (str): 'cpu' 또는 'cuda'
        """
                # 기본 파라미터 검증
        if not os.path.exists(filepath):
            raise IOError(f"File not found: {filepath}")
        if not callable(custom_loader):
            raise ValueError("custom_loader must be callable")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        if buffer_size > 1000:  # 메모리 보호를 위한 상한선
            warnings.warn("Large buffer_size may cause memory issues")
        if device not in ['cpu', 'cuda']:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        if not all(0 < r < 1 for r in split_ratio) or abs(sum(split_ratio) - 1.0) > 1e-6:
            raise ValueError("split_ratio must be positive and sum to 1")
        self.custom_loader = custom_loader
        self.filepath = filepath
        self.dataset_name_x = dataset_name_x
        self.dataset_name_y = dataset_name_y
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.device = device
        self.split_ratio = [r / sum(split_ratio) for r in split_ratio]  # 비율 정규화
        self.data_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.producer_thread = None
        try:
            self.total_size = self._get_dataset_length()
        except Exception as e:
            raise IOError(f"Failed to read dataset: {str(e)}")

        self.train_range, self.val_range, self.test_range = self._calculate_splits()
        self.current_range = self.train_range

    def get_sizes(self):
        """각 split의 크기를 반환"""
        train_size = self.train_range[1] - self.train_range[0]
        val_size = self.val_range[1] - self.val_range[0]
        test_size = self.test_range[1] - self.test_range[0]
        return {'train': train_size, 'val': val_size, 'test': test_size}
    def get_queue_status(self):
        """큐의 현재 상태 반환"""
        return {
            'size': self.data_queue.qsize(),
            'maxsize': self.data_queue.maxsize,
            'empty': self.data_queue.empty(),
            'full': self.data_queue.full()
        }
    def get_current_mode(self):
        """현재 모드 반환"""
        if self.current_range == self.train_range:
            return 'train'
        elif self.current_range == self.val_range:
            return 'val'
        else:
            return 'test'
    def _get_dataset_length(self):
        """HDF5 파일에서 데이터셋 길이를 읽어옴"""
        with h5py.File(self.filepath, "r") as f:
            dataset = f[self.dataset_name_x]
            return len(dataset)

    def _calculate_splits(self):
        """[training, validation, test] 비율로 데이터셋 범위를 계산"""
        train_ratio, val_ratio, test_ratio = self.split_ratio
        train_end = int(self.total_size * train_ratio)
        val_end = train_end + int(self.total_size * val_ratio)

        train_range = (0, train_end)
        val_range = (train_end, val_end)
        test_range = (val_end, self.total_size)

        return train_range, val_range, test_range

    def set_mode(self, mode):
        """훈련, 검증, 테스트 모드 설정"""
        if mode == "train":
            self.current_range = self.train_range
        elif mode == "validation":
            self.current_range = self.val_range
        elif mode == "test":
            self.current_range = self.test_range
        else:
            raise ValueError("Invalid mode. Choose from ['train', 'validation', 'test']")

    def _producer(self):
        """custom_loader를 호출하여 데이터를 읽어 큐에 배치를 넣음"""
        start_idx, end_idx = self.current_range
        data_generator = self.custom_loader(
            filepath=self.filepath,
            dataset_name_x=self.dataset_name_x,
            dataset_name_y=self.dataset_name_y,
            batch_size=self.batch_size,
            start_idx=start_idx,
            end_idx=end_idx,
            shuffle=self.shuffle,
            device=self.device,
        )

        for batch in data_generator:
            if self.stop_event.is_set():
                break
            self.data_queue.put(batch)

        self.data_queue.put(None)  # 에포크 종료 신호

    def __enter__(self):
        """스레드 시작"""
        self.producer_thread = threading.Thread(target=self._producer, daemon=True)
        self.producer_thread.start()
        return self.data_queue

    def __exit__(self, exc_type, exc_value, traceback):
        """스레드 종료 및 리소스 정리"""
        self.stop_event.set()
        self.producer_thread.join()
        while not self.data_queue.empty():
            self.data_queue.get()

class BatchFetcher:
    def __init__(self, data_queue):
        """
        데이터 큐에서 배치를 가져오는 반복자 클래스
        Args:
            data_queue (queue.Queue): 데이터 큐
        """
        self.data_queue = data_queue

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.data_queue.get()
        if batch is None:  # 종료 신호
            raise StopIteration
        return batch
