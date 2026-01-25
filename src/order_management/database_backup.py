"""
数据库备份模块
每天自动备份数据库，保留一个月
"""
import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """数据库备份管理器"""
    
    def __init__(
        self,
        db_path: str,
        backup_dir: Optional[str] = None,
        retention_days: int = 30,
    ):
        """
        初始化备份管理器
        
        Args:
            db_path: 数据库文件路径
            backup_dir: 备份目录（默认：数据库文件同目录下的 backups 子目录）
            retention_days: 保留天数（默认：30天）
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir) if backup_dir else self.db_path.parent / "backups"
        self.retention_days = retention_days
        self._backup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    async def start(self) -> None:
        """启动备份任务"""
        if self._running:
            return
        self._running = True
        
        # 立即执行一次备份
        await self.backup_now()
        
        # 启动定时备份任务
        self._backup_task = asyncio.create_task(self._backup_loop())
        logger.info(f"✅ 数据库备份任务已启动: {self.db_path}")
    
    async def stop(self) -> None:
        """停止备份任务"""
        self._running = False
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        logger.info("数据库备份任务已停止")
    
    async def backup_now(self) -> bool:
        """立即执行备份"""
        try:
            if not self.db_path.exists():
                logger.warning(f"数据库文件不存在: {self.db_path}")
                return False
            
            # 生成备份文件名
            today = datetime.now().strftime("%Y%m%d")
            backup_filename = f"{self.db_path.stem}_{today}.db.backup"
            backup_path = self.backup_dir / backup_filename
            
            # 如果今天的备份已存在，跳过
            if backup_path.exists():
                logger.debug(f"今天的备份已存在: {backup_path}")
                return True
            
            # 执行备份（使用 shutil.copy2 保留元数据）
            shutil.copy2(self.db_path, backup_path)
            
            # 如果启用了 WAL 模式，也需要备份 WAL 和 SHM 文件
            wal_path = Path(str(self.db_path) + "-wal")
            shm_path = Path(str(self.db_path) + "-shm")
            
            if wal_path.exists():
                shutil.copy2(wal_path, self.backup_dir / f"{backup_filename}-wal")
            if shm_path.exists():
                shutil.copy2(shm_path, self.backup_dir / f"{backup_filename}-shm")
            
            logger.info(f"✅ 数据库备份完成: {backup_path}")
            
            # 清理旧备份
            await self._cleanup_old_backups()
            
            return True
        except Exception as e:
            logger.error(f"❌ 数据库备份失败: {e}", exc_info=True)
            return False
    
    async def _backup_loop(self) -> None:
        """备份循环：每天执行一次备份"""
        while self._running:
            try:
                # 等待到明天 00:00:00
                now = datetime.now()
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                wait_seconds = (tomorrow - now).total_seconds()
                
                logger.info(f"下次备份将在 {wait_seconds / 3600:.1f} 小时后执行")
                await asyncio.sleep(wait_seconds)
                
                # 执行备份
                await self.backup_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"备份循环异常: {e}", exc_info=True)
                # 出错后等待1小时再重试
                await asyncio.sleep(3600)
    
    async def _cleanup_old_backups(self) -> None:
        """清理超过保留期的旧备份"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_str = cutoff_date.strftime("%Y%m%d")
            
            deleted_count = 0
            for backup_file in self.backup_dir.glob("*.db.backup*"):
                # 从文件名提取日期（格式：{db_name}_{YYYYMMDD}.db.backup）
                try:
                    # 处理文件名，可能是 .db.backup 或 .db.backup-wal 或 .db.backup-shm
                    stem = backup_file.stem
                    if stem.endswith("-wal") or stem.endswith("-shm"):
                        stem = stem.rsplit("-", 1)[0]
                    
                    # 提取日期部分（最后一个下划线后的部分）
                    parts = stem.split("_")
                    if len(parts) >= 2:
                        date_str = parts[-1]
                        if date_str < cutoff_str:
                            backup_file.unlink()
                            deleted_count += 1
                except (ValueError, IndexError, OSError) as e:
                    # 如果无法解析日期或删除失败，跳过
                    logger.debug(f"清理备份文件时跳过: {backup_file}, 错误: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个旧备份文件")
        except Exception as e:
            logger.warning(f"清理旧备份失败: {e}")
