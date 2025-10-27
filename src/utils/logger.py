"""
Advanced Logging System - سیستم logging پیشرفته با قابلیت rotation
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json


class ColoredFormatter(logging.Formatter):
    """فرمت‌کننده رنگی برای console"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # فرمت: [زمان] [سطح] [ماژول] پیام
        record.levelname = f"{log_color}{record.levelname}{reset}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """فرمت‌کننده JSON برای لاگ‌های ساختاری"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # اضافه کردن exception در صورت وجود
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # اضافه کردن extra fields
        if hasattr(record, 'feature_name'):
            log_data['feature_name'] = record.feature_name
        if hasattr(record, 'task_name'):
            log_data['task_name'] = record.task_name
        
        return json.dumps(log_data, ensure_ascii=False)


class AutoDevLogger:
    """کلاس اصلی Logging"""
    
    def __init__(
        self,
        name: str = "auto-dev-llm",
        log_path: str = "./logs",
        level: str = "INFO",
        per_feature_log: bool = True,
        rotation: str = "1 day",
        retention: str = "30 days",
        json_logs: bool = True
    ):
        self.name = name
        self.log_path = Path(log_path)
        self.level = getattr(logging, level.upper())
        self.per_feature_log = per_feature_log
        self.rotation = rotation
        self.retention = retention
        self.json_logs = json_logs
        
        # ایجاد پوشه logs
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # تنظیم logger اصلی
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        
        # پاک کردن handlers قبلی
        self.logger.handlers.clear()
        
        # اضافه کردن handlers
        self._setup_console_handler()
        self._setup_file_handler()
    
    def _setup_console_handler(self):
        """تنظیم handler کنسول (رنگی)"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        formatter = ColoredFormatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """تنظیم handler فایل (با rotation)"""
        log_file = self.log_path / f"{self.name}.log"
        
        # استفاده از TimedRotatingFileHandler برای rotation بر اساس زمان
        if "day" in self.rotation.lower():
            file_handler = TimedRotatingFileHandler(
                filename=log_file,
                when='midnight',
                interval=1,
                backupCount=30,  # نگهداری 30 روز
                encoding='utf-8'
            )
        else:
            # استفاده از RotatingFileHandler برای rotation بر اساس حجم
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=10,
                encoding='utf-8'
            )
        
        file_handler.setLevel(self.level)
        
        # استفاده از JSON formatter اگر فعال باشد
        if self.json_logs:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt='[%(asctime)s] [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def create_feature_logger(self, feature_name: str) -> logging.Logger:
        """ایجاد logger جداگانه برای هر feature"""
        if not self.per_feature_log:
            return self.logger
        
        feature_logger_name = f"{self.name}.{feature_name}"
        feature_logger = logging.getLogger(feature_logger_name)
        feature_logger.setLevel(self.level)
        feature_logger.propagate = False
        
        # پاک کردن handlers قبلی
        feature_logger.handlers.clear()
        
        # ایجاد فایل لاگ جداگانه
        feature_log_file = self.log_path / f"{feature_name}.log"
        file_handler = RotatingFileHandler(
            filename=feature_log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding='utf-8'
        )
        
        if self.json_logs:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt='[%(asctime)s] [%(levelname)s] [%(task_name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        feature_logger.addHandler(file_handler)
        
        # اضافه کردن console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(
            fmt='[%(asctime)s] [%(levelname)s] [%(task_name)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        feature_logger.addHandler(console_handler)
        
        return feature_logger
    
    def debug(self, msg: str, **kwargs):
        """لاگ سطح DEBUG"""
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        """لاگ سطح INFO"""
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        """لاگ سطح WARNING"""
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, **kwargs):
        """لاگ سطح ERROR"""
        self.logger.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs):
        """لاگ سطح CRITICAL"""
        self.logger.critical(msg, extra=kwargs)
    
    def log_task_start(self, feature_name: str, task_name: str):
        """لاگ شروع task"""
        self.info(
            f"🚀 شروع task: {task_name}",
            feature_name=feature_name,
            task_name=task_name
        )
    
    def log_task_complete(self, feature_name: str, task_name: str, duration: float):
        """لاگ اتمام موفق task"""
        self.info(
            f"✅ اتمام task: {task_name} (مدت: {duration:.2f}s)",
            feature_name=feature_name,
            task_name=task_name
        )
    
    def log_task_error(self, feature_name: str, task_name: str, error: Exception):
        """لاگ خطای task"""
        self.error(
            f"❌ خطا در task: {task_name} - {str(error)}",
            feature_name=feature_name,
            task_name=task_name,
            exc_info=True
        )
    
    def log_llm_request(self, prompt: str, model: str, tokens: int):
        """لاگ درخواست به LLM"""
        self.debug(
            f"🤖 درخواست LLM: model={model}, tokens={tokens}",
            prompt_preview=prompt[:100] + "..." if len(prompt) > 100 else prompt
        )
    
    def log_llm_response(self, response: str, tokens: int, duration: float):
        """لاگ پاسخ LLM"""
        self.debug(
            f"📥 پاسخ LLM: tokens={tokens}, duration={duration:.2f}s",
            response_preview=response[:100] + "..." if len(response) > 100 else response
        )
    
    def log_git_commit(self, commit_hash: str, message: str):
        """لاگ commit گیت"""
        self.info(f"📦 Git commit: {commit_hash[:7]} - {message}")
    
    def log_rollback(self, feature_name: str, reason: str):
        """لاگ rollback"""
        self.warning(
            f"🔄 Rollback: {feature_name} - دلیل: {reason}",
            feature_name=feature_name
        )
    
    def log_deploy_stage(self, stage: str, traffic_percent: int):
        """لاگ مراحل deploy"""
        self.info(f"🚢 Deploy {stage}: {traffic_percent}% traffic")
    
    def log_version_bump(self, old_version: str, new_version: str, bump_type: str):
        """لاگ تغییر نسخه"""
        self.info(f"📌 Version bump ({bump_type}): {old_version} → {new_version}")


# تست سریع
if __name__ == "__main__":
    logger = AutoDevLogger(
        name="test-logger",
        log_path="./logs",
        level="DEBUG"
    )
    
    logger.info("🎉 سیستم logging راه‌اندازی شد")
    logger.debug("این یک پیام debug است")
    logger.warning("این یک هشدار است")
    logger.error("این یک خطا است")
    
    # تست feature logger
    feature_logger = logger.create_feature_logger("test-feature")
    feature_logger.info("لاگ مربوط به feature", task_name="test-task")
    
    print("\n✅ لاگ‌ها در پوشه ./logs ذخیره شدند")