"""
Task Manager - مدیریت صف وظایف و وضعیت‌ها
"""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
from queue import PriorityQueue


class TaskStatus(Enum):
    """وضعیت‌های task"""
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TaskResult:
    """نتیجه اجرای task"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    generated_files: List[str] = field(default_factory=list)
    commit_hash: Optional[str] = None


@dataclass
class TaskExecution:
    """اطلاعات اجرای task"""
    task_name: str
    feature_name: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[TaskResult] = None
    retry_count: int = 0
    max_retries: int = 3


class TaskQueue:
    """صف اولویت‌دار وظایف"""
    
    def __init__(self):
        self.queue = PriorityQueue()
        self.tasks: Dict[str, TaskExecution] = {}
        self.running_tasks: List[str] = []
    
    def add_task(self, feature_name: str, task_name: str, priority: int):
        """اضافه کردن task به صف"""
        task_id = f"{feature_name}.{task_name}"
        
        if task_id not in self.tasks:
            execution = TaskExecution(
                task_name=task_name,
                feature_name=feature_name,
                status=TaskStatus.PENDING
            )
            self.tasks[task_id] = execution
            self.queue.put((priority, task_id))
    
    def get_next_task(self) -> Optional[TaskExecution]:
        """دریافت task بعدی از صف"""
        if self.queue.empty():
            return None
        
        _, task_id = self.queue.get()
        return self.tasks.get(task_id)
    
    def mark_running(self, task_id: str):
        """علامت‌گذاری task به عنوان در حال اجرا"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.RUNNING
            self.tasks[task_id].start_time = datetime.now()
            self.running_tasks.append(task_id)
    
    def mark_completed(self, task_id: str, result: TaskResult):
        """علامت‌گذاری task به عنوان تکمیل شده"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].end_time = datetime.now()
            self.tasks[task_id].result = result
            if task_id in self.running_tasks:
                self.running_tasks.remove(task_id)
    
    def mark_failed(self, task_id: str, result: TaskResult):
        """علامت‌گذاری task به عنوان ناموفق"""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.tasks[task_id].end_time = datetime.now()
            self.tasks[task_id].result = result
            if task_id in self.running_tasks:
                self.running_tasks.remove(task_id)
    
    def get_running_count(self) -> int:
        """تعداد task های در حال اجرا"""
        return len(self.running_tasks)
    
    def is_empty(self) -> bool:
        """بررسی خالی بودن صف"""
        return self.queue.empty()


class TaskManager:
    """مدیر اصلی وظایف"""
    
    def __init__(self, state_file: str = "./task_state.json"):
        self.queue = TaskQueue()
        self.state_file = Path(state_file)
        self.max_concurrent_tasks = 2
        
        # بازیابی وضعیت قبلی در صورت وجود
        self._load_state()
    
    def add_feature_tasks(self, feature_name: str, tasks: List[Any], priority: int):
        """اضافه کردن تمام task های یک feature"""
        for task in tasks:
            self.queue.add_task(feature_name, task.name, priority)
    
    def can_start_new_task(self) -> bool:
        """بررسی امکان شروع task جدید"""
        return self.queue.get_running_count() < self.max_concurrent_tasks
    
    def get_next_pending_task(self) -> Optional[TaskExecution]:
        """دریافت task بعدی برای اجرا"""
        if not self.can_start_new_task():
            return None
        
        return self.queue.get_next_task()
    
    def start_task(self, task_exec: TaskExecution) -> str:
        """شروع اجرای task"""
        task_id = f"{task_exec.feature_name}.{task_exec.task_name}"
        self.queue.mark_running(task_id)
        self._save_state()
        return task_id
    
    def complete_task(self, task_id: str, result: TaskResult):
        """تکمیل موفق task"""
        self.queue.mark_completed(task_id, result)
        self._save_state()
    
    def fail_task(self, task_id: str, result: TaskResult, retry: bool = True):
        """شکست task"""
        task_exec = self.queue.tasks.get(task_id)
        
        if task_exec and retry and task_exec.retry_count < task_exec.max_retries:
            # تلاش مجدد
            task_exec.retry_count += 1
            task_exec.status = TaskStatus.PENDING
            priority = 0  # اولویت بالا برای retry
            self.queue.queue.put((priority, task_id))
            if task_id in self.queue.running_tasks:
                self.queue.running_tasks.remove(task_id)
        else:
            # شکست نهایی
            self.queue.mark_failed(task_id, result)
        
        self._save_state()
    
    def get_task_status(self, feature_name: str, task_name: str) -> Optional[TaskStatus]:
        """دریافت وضعیت task"""
        task_id = f"{feature_name}.{task_name}"
        task_exec = self.queue.tasks.get(task_id)
        return task_exec.status if task_exec else None
    
    def get_all_tasks(self) -> List[TaskExecution]:
        """دریافت تمام task ها"""
        return list(self.queue.tasks.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskExecution]:
        """دریافت task ها بر اساس وضعیت"""
        return [t for t in self.queue.tasks.values() if t.status == status]
    
    def get_feature_progress(self, feature_name: str) -> Dict[str, int]:
        """دریافت پیشرفت یک feature"""
        feature_tasks = [
            t for t in self.queue.tasks.values() 
            if t.feature_name == feature_name
        ]
        
        total = len(feature_tasks)
        completed = len([t for t in feature_tasks if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in feature_tasks if t.status == TaskStatus.FAILED])
        running = len([t for t in feature_tasks if t.status == TaskStatus.RUNNING])
        pending = total - completed - failed - running
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'running': running,
            'pending': pending,
            'progress_percent': (completed / total * 100) if total > 0 else 0
        }
    
    def _save_state(self):
        """ذخیره وضعیت task ها"""
        state = {}
        for task_id, task_exec in self.queue.tasks.items():
            state[task_id] = {
                'task_name': task_exec.task_name,
                'feature_name': task_exec.feature_name,
                'status': task_exec.status.value,
                'start_time': task_exec.start_time.isoformat() if task_exec.start_time else None,
                'end_time': task_exec.end_time.isoformat() if task_exec.end_time else None,
                'retry_count': task_exec.retry_count,
                'result': {
                    'success': task_exec.result.success,
                    'output': task_exec.result.output,
                    'error': task_exec.result.error,
                    'duration': task_exec.result.duration,
                    'generated_files': task_exec.result.generated_files,
                    'commit_hash': task_exec.result.commit_hash
                } if task_exec.result else None
            }
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def _load_state(self):
        """بازیابی وضعیت task ها"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            for task_id, task_data in state.items():
                result = None
                if task_data['result']:
                    result = TaskResult(
                        success=task_data['result']['success'],
                        output=task_data['result']['output'],
                        error=task_data['result']['error'],
                        duration=task_data['result']['duration'],
                        generated_files=task_data['result']['generated_files'],
                        commit_hash=task_data['result']['commit_hash']
                    )
                
                task_exec = TaskExecution(
                    task_name=task_data['task_name'],
                    feature_name=task_data['feature_name'],
                    status=TaskStatus(task_data['status']),
                    start_time=datetime.fromisoformat(task_data['start_time']) if task_data['start_time'] else None,
                    end_time=datetime.fromisoformat(task_data['end_time']) if task_data['end_time'] else None,
                    result=result,
                    retry_count=task_data['retry_count']
                )
                
                self.queue.tasks[task_id] = task_exec
                
                # بازگردانی running tasks
                if task_exec.status == TaskStatus.RUNNING:
                    self.queue.running_tasks.append(task_id)
        
        except Exception as e:
            print(f"⚠️  خطا در بازیابی وضعیت: {e}")
    
    def clear_completed_tasks(self):
        """پاک کردن task های تکمیل شده"""
        completed_ids = [
            task_id for task_id, task_exec in self.queue.tasks.items()
            if task_exec.status == TaskStatus.COMPLETED
        ]
        
        for task_id in completed_ids:
            del self.queue.tasks[task_id]
        
        self._save_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """دریافت آمار کلی"""
        all_tasks = self.get_all_tasks()
        
        return {
            'total_tasks': len(all_tasks),
            'completed': len([t for t in all_tasks if t.status == TaskStatus.COMPLETED]),
            'failed': len([t for t in all_tasks if t.status == TaskStatus.FAILED]),
            'running': len([t for t in all_tasks if t.status == TaskStatus.RUNNING]),
            'pending': len([t for t in all_tasks if t.status == TaskStatus.PENDING]),
            'average_duration': sum(
                t.result.duration for t in all_tasks 
                if t.result and t.status == TaskStatus.COMPLETED
            ) / max(len([t for t in all_tasks if t.status == TaskStatus.COMPLETED]), 1)
        }


# تست سریع
if __name__ == "__main__":
    from config import Task
    
    manager = TaskManager()
    
    # ایجاد task های تست
    tasks = [
        Task("task1", "توضیحات task 1", ["file1.py"], ["test1.py"]),
        Task("task2", "توضیحات task 2", ["file2.py"], ["test2.py"]),
        Task("task3", "توضیحات task 3", ["file3.py"], ["test3.py"])
    ]
    
    manager.add_feature_tasks("feature-test", tasks, priority=1)
    
    # شروع task
    task_exec = manager.get_next_pending_task()
    if task_exec:
        task_id = manager.start_task(task_exec)
        print(f"✅ Task شروع شد: {task_id}")
        
        # تکمیل task
        result = TaskResult(success=True, output="کد تولید شد", duration=2.5)
        manager.complete_task(task_id, result)
        print(f"✅ Task تکمیل شد: {task_id}")
    
    # نمایش آمار
    stats = manager.get_statistics()
    print(f"\n📊 آمار: {json.dumps(stats, indent=2, ensure_ascii=False)}")