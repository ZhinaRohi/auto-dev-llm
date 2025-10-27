"""
Config Loader - خواندن و اعتبارسنجی تنظیمات پروژه
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class LLMMode(Enum):
    """حالت‌های LLM"""
    OFFLINE = "offline"
    ONLINE = "online"
    MCP = "mcp"


class DeployType(Enum):
    """نوع استقرار"""
    CANARY = "canary"
    BLUE_GREEN = "blue-green"
    ROLLING = "rolling"


@dataclass
class LLMConfig:
    """تنظیمات LLM"""
    mode: LLMMode
    offline_model: Dict[str, str] = field(default_factory=dict)
    mcp: Dict[str, Any] = field(default_factory=dict)
    online: Dict[str, str] = field(default_factory=dict)
    fallback_online: bool = True


@dataclass
class SchedulerConfig:
    """تنظیمات Scheduler"""
    active_hours: Dict[str, int]
    max_concurrent_tasks: int = 2
    check_interval: int = 60
    cpu_threshold: int = 80


@dataclass
class GitConfig:
    """تنظیمات Git"""
    auto_commit: bool = True
    commit_message_template: str = "Auto: {feature_name}"
    branch_pattern: str = "auto-dev/{feature_name}"
    auto_push: bool = False


@dataclass
class VersioningConfig:
    """تنظیمات Versioning"""
    semantic_versioning: bool = True
    auto_bump: bool = True
    version_file: str = "VERSION"
    rules: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """تنظیمات Logging"""
    log_path: str = "./logs"
    level: str = "INFO"
    per_feature_log: bool = True
    rotation: str = "1 day"
    retention: str = "30 days"


@dataclass
class RollbackConfig:
    """تنظیمات Rollback"""
    enabled: bool = True
    backup_path: str = "./backups"
    max_backups: int = 10
    auto_rollback_on_error: bool = True


@dataclass
class DeployConfig:
    """تنظیمات Deploy"""
    enabled: bool = True
    type: DeployType = DeployType.CANARY
    canary: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """وظیفه"""
    name: str
    description: str
    files: List[str]
    tests: List[str]
    status: str = "pending"  # pending, running, done, failed


@dataclass
class Feature:
    """ویژگی"""
    name: str
    priority: int
    description: str
    tasks: List[Task]
    approved: bool = False


@dataclass
class ProjectConfig:
    """تنظیمات کلی پروژه"""
    project_name: str
    description: str
    version: str
    llm: LLMConfig
    scheduler: SchedulerConfig
    git: GitConfig
    versioning: VersioningConfig
    logging: LoggingConfig
    rollback: RollbackConfig
    deploy: DeployConfig
    features: List[Feature]


class ConfigLoader:
    """خواندن و اعتبارسنجی فایل تنظیمات"""
    
    def __init__(self, spec_path: str = "./specs/project_spec.yaml"):
        self.spec_path = Path(spec_path)
        self.config: Optional[ProjectConfig] = None
    
    def load(self) -> ProjectConfig:
        """خواندن فایل YAML"""
        if not self.spec_path.exists():
            raise FileNotFoundError(f"فایل {self.spec_path} یافت نشد!")
        
        with open(self.spec_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self.config = self._parse_config(data)
        self._validate_config()
        return self.config
    
    def _parse_config(self, data: Dict[str, Any]) -> ProjectConfig:
        """تبدیل dict به dataclass"""
        
        # LLM Config
        llm_data = data.get('llm', {})
        llm_config = LLMConfig(
            mode=LLMMode(llm_data.get('mode', 'mcp')),
            offline_model=llm_data.get('offline_model', {}),
            mcp=llm_data.get('mcp', {}),
            online=llm_data.get('online', {}),
            fallback_online=llm_data.get('fallback_online', True)
        )
        
        # Scheduler Config
        scheduler_data = data.get('scheduler', {})
        scheduler_config = SchedulerConfig(
            active_hours=scheduler_data.get('active_hours', {}),
            max_concurrent_tasks=scheduler_data.get('max_concurrent_tasks', 2),
            check_interval=scheduler_data.get('check_interval', 60),
            cpu_threshold=scheduler_data.get('cpu_threshold', 80)
        )
        
        # Git Config
        git_data = data.get('git', {})
        git_config = GitConfig(
            auto_commit=git_data.get('auto_commit', True),
            commit_message_template=git_data.get('commit_message_template', ''),
            branch_pattern=git_data.get('branch_pattern', ''),
            auto_push=git_data.get('auto_push', False)
        )
        
        # Versioning Config
        version_data = data.get('versioning', {})
        versioning_config = VersioningConfig(
            semantic_versioning=version_data.get('semantic_versioning', True),
            auto_bump=version_data.get('auto_bump', True),
            version_file=version_data.get('version_file', 'VERSION'),
            rules=version_data.get('rules', {})
        )
        
        # Logging Config
        log_data = data.get('logging', {})
        logging_config = LoggingConfig(
            log_path=log_data.get('log_path', './logs'),
            level=log_data.get('level', 'INFO'),
            per_feature_log=log_data.get('per_feature_log', True),
            rotation=log_data.get('rotation', '1 day'),
            retention=log_data.get('retention', '30 days')
        )
        
        # Rollback Config
        rollback_data = data.get('rollback', {})
        rollback_config = RollbackConfig(
            enabled=rollback_data.get('enabled', True),
            backup_path=rollback_data.get('backup_path', './backups'),
            max_backups=rollback_data.get('max_backups', 10),
            auto_rollback_on_error=rollback_data.get('auto_rollback_on_error', True)
        )
        
        # Deploy Config
        deploy_data = data.get('deploy', {})
        deploy_config = DeployConfig(
            enabled=deploy_data.get('enabled', True),
            type=DeployType(deploy_data.get('type', 'canary')),
            canary=deploy_data.get('canary', {}),
            health_check=deploy_data.get('health_check', {})
        )
        
        # Features
        features = []
        for feat_data in data.get('features', []):
            tasks = []
            for task_data in feat_data.get('tasks', []):
                task = Task(
                    name=task_data['name'],
                    description=task_data['description'],
                    files=task_data.get('files', []),
                    tests=task_data.get('tests', [])
                )
                tasks.append(task)
            
            feature = Feature(
                name=feat_data['name'],
                priority=feat_data.get('priority', 999),
                description=feat_data['description'],
                tasks=tasks
            )
            features.append(feature)
        
        # مرتب‌سازی features بر اساس اولویت
        features.sort(key=lambda f: f.priority)
        
        return ProjectConfig(
            project_name=data['project_name'],
            description=data['description'],
            version=data.get('version', '0.1.0'),
            llm=llm_config,
            scheduler=scheduler_config,
            git=git_config,
            versioning=versioning_config,
            logging=logging_config,
            rollback=rollback_config,
            deploy=deploy_config,
            features=features
        )
    
    def _validate_config(self) -> None:
        """اعتبارسنجی تنظیمات"""
        if not self.config:
            raise ValueError("تنظیمات بارگذاری نشده است!")
        
        # بررسی LLM
        if self.config.llm.mode == LLMMode.MCP:
            if not self.config.llm.mcp.get('api_url'):
                raise ValueError("api_url برای MCP تنظیم نشده است!")
        
        # بررسی Scheduler
        if self.config.scheduler.active_hours:
            start = self.config.scheduler.active_hours.get('start', 0)
            end = self.config.scheduler.active_hours.get('end', 24)
            if not (0 <= start < 24 and 0 <= end <= 24 and start < end):
                raise ValueError("ساعات کاری نامعتبر است!")
        
        # بررسی Features
        if not self.config.features:
            raise ValueError("هیچ feature تعریف نشده است!")
        
        for feature in self.config.features:
            if not feature.tasks:
                raise ValueError(f"Feature '{feature.name}' هیچ task ندارد!")
    
    def get_pending_features(self) -> List[Feature]:
        """دریافت features تایید نشده"""
        return [f for f in self.config.features if not f.approved]
    
    def approve_feature(self, feature_name: str) -> None:
        """تایید یک feature"""
        for feature in self.config.features:
            if feature.name == feature_name:
                feature.approved = True
                return
        raise ValueError(f"Feature '{feature_name}' یافت نشد!")
    
    def get_approved_features(self) -> List[Feature]:
        """دریافت features تایید شده"""
        return [f for f in self.config.features if f.approved]


# تست سریع
if __name__ == "__main__":
    loader = ConfigLoader("./specs/project_spec.yaml")
    config = loader.load()
    print(f"✅ پروژه: {config.project_name}")
    print(f"✅ نسخه: {config.version}")
    print(f"✅ حالت LLM: {config.llm.mode.value}")
    print(f"✅ تعداد Features: {len(config.features)}")
    
    for feature in config.features:
        print(f"\n📌 {feature.name} (اولویت: {feature.priority})")
        for task in feature.tasks:
            print(f"   - {task.name}: {task.description}")