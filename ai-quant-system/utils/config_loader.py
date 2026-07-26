"""
AI量化交易系统 - 配置加载器
从YAML文件加载配置，支持环境变量覆盖
"""
import os
import yaml
from pathlib import Path
from copy import deepcopy


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "settings.yaml"
            )
        self.config_path = Path(config_path)
        self._config = {}
        self._load()

    def _load(self) -> None:
        """加载YAML配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # 环境变量覆盖
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖"""
        env_prefix = "QUANT_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace("_", ".")
                self._deep_update(self._config, self._parse_path(config_key), value)

    def _parse_path(self, path: str) -> list:
        """解析点分路径为列表"""
        return path.split(".")

    def _deep_update(self, d: dict, keys: list, value: any) -> None:
        """深层更新字典"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = self._convert_type(value, d.get(keys[-1], ""))

    def _convert_type(self, value: str, default: any) -> any:
        """转换类型"""
        if default:
            return type(default)(value) if not isinstance(default, (list, dict)) else default
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        return value

    def get(self, key: str, default=None):
        """获取配置值（支持点分路径）"""
        keys = key.split(".")
        obj = self._config
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k, default)
                if obj is None and k != keys[-1]:
                    return default
            else:
                return default
        return obj

    def get_section(self, section: str) -> dict:
        """获取配置段落"""
        return deepcopy(self._config.get(section, {}))

    @property
    def config(self) -> dict:
        """获取完整配置"""
        return deepcopy(self._config)

    def validate(self) -> list:
        """验证配置完整性"""
        errors = []
        required_sections = ["system", "screening", "research", "execution", "risk"]
        for section in required_sections:
            if section not in self._config:
                errors.append(f"缺少必需配置段落: {section}")

        # 验证初筛层配置
        screening = self._config.get("screening", {})
        if screening.get("lightgbm", {}).get("learning_rate", 0) <= 0:
            errors.append("LightGBM learning_rate必须大于0")

        # 验证风控配置
        risk = self._config.get("risk", {})
        metrics = risk.get("metrics", {})
        if metrics.get("max_drawdown", 0) <= 0:
            errors.append("max_drawdown必须大于0")

        return errors


def load_config(config_path: str = None) -> ConfigLoader:
    """便捷函数：加载配置"""
    return ConfigLoader(config_path)
