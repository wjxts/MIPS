import time 

class Timer:
    def __init__(self):
        self.start = time.time()
        self.last_time = self.start
        self.forward_time = 0 # 模型forward运算的时间
        self.steps = 0
    
    def step(self) -> None:
        self.steps += 1
    
    def elapsed(self) -> str:
        current_time = time.time()
        t = int(current_time - self.start)
        return self.convert_seconds(t)
    
    def start_train_step(self):
        self.last_time = time.time()
    
    def end_train_step(self):
        current_time = time.time()
        self.forward_time += current_time - self.last_time
    
    @property
    def train_time_ratio(self) -> float:
        total_time = time.time() - self.start
        loda_data_time = total_time - self.forward_time
        ratio = loda_data_time / self.forward_time
        s = f"{loda_data_time:.2f}s/{self.forward_time:.2f}s (ratio:{ratio:.2f})"
        return s 
    
    def eta(self, remaining_steps) -> str:
        current_time = time.time()
        t = current_time - self.start
        remaining_time = (t / self.steps) * remaining_steps
        return self.convert_seconds(int(remaining_time))
    
    def convert_seconds(self, seconds: int) -> str:
        hours, remainder = divmod(seconds, 3600)  # 计算小时数和剩余秒数
        minutes, secs = divmod(remainder, 60)    # 计算分钟数和剩余秒数
        if hours == 0 and minutes == 0:
            return f"{secs:02}s"
        elif hours == 0:
            return f"{minutes:02}min:{secs:02}s"
        else:
            return f"{hours:02}hour:{minutes:02}min:{secs:02}s"