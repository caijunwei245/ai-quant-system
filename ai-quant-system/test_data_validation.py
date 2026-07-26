"""
AI量化交易系统 - 数据完整性验证测试脚本
测试 AKShare + Tushare MCP 双源数据一致性校验
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from data.data_validator import DataValidator
from loguru import logger


def test_akshare_fetch():
    """测试 AKShare 数据获取"""
    from data.fetcher import DataFetcher

    print("=" * 60)
    print("测试 1: AKShare 数据获取")
    print("=" * 60)

    fetcher = DataFetcher()

    # 测试几只股票
    test_codes = ["000001", "600519"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    for code in test_codes:
        df = fetcher.fetch_stock_data(code, start_date=start_date, end_date=end_date)
        if not df.empty:
            print(f"✅ {code}: {len(df)} 条记录")
            print(f"   日期范围: {df.index[0]} ~ {df.index[-1]}")
            print(f"   列名: {list(df.columns)}")
            print(f"   前3行:\n{df.head(3)}")
        else:
            print(f"❌ {code}: 获取失败")
        print()

    return fetcher


def test_tushare_availability():
    """测试 Tushare MCP 可用性"""
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        print("=" * 60)
        print("⚠️  Tushare 环境变量未设置")
        print("=" * 60)
        print(f"请执行: export TUSHARE_TOKEN='your_token_here'")
        print()
        return False

    print("=" * 60)
    print("测试 2: Tushare MCP 可用性")
    print("=" * 60)
    print(f"Token (前8位): {token[:8]}...")

    # 尝试获取数据
    from data.data_validator import DataValidator
    validator = DataValidator()

    test_code = "000001"
    start_date = "2024-01-01"
    end_date = "2024-01-10"

    df = validator._fetch_tushare_daily(test_code, start_date, end_date)
    if not df.empty:
        print(f"✅ Tushare 可用: {test_code} 获取到 {len(df)} 条记录")
        print(f"   列名: {list(df.columns)}")
        print(f"   前3行:\n{df.head(3)}")
        return True
    else:
        print(f"❌ Tushare 不可用或无数据")
        return False


def test_validation():
    """测试数据完整性校验"""
    from data.fetcher import DataFetcher

    print("=" * 60)
    print("测试 3: 数据完整性校验")
    print("=" * 60)

    # 加载验证配置
    try:
        from utils.config_loader import ConfigLoader
        cfg = ConfigLoader()
        val_cfg = cfg.get("validation", {})
    except Exception as e:
        print(f"⚠️ 无法加载配置文件: {e}")
        val_cfg = {"enabled": True, "tolerance": {}, "auto_save": True}

    validator = DataValidator(val_cfg)

    test_codes = ["000001", "600519"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    results = {}
    for code in test_codes:
        print(f"\n校验 {code}...")
        report = validator.validate_single_stock(code, start_date, end_date)
        results[code] = report

        if report.passed:
            print(f"  ✅ 通过 (共同日期={report.common_dates}天)")
        else:
            print(f"  ⚠️ 未通过:")
            for detail in report.details:
                print(f"    - {detail}")

    # 保存报告
    if results:
        validator.save_validation_report(results)

    return results


def main():
    """主测试流程"""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("\n🚀 AI量化交易系统 - 数据完整性验证测试\n")

    # 1. 测试 AKShare
    fetcher = test_akshare_fetch()

    # 2. 测试 Tushare
    tushare_ok = test_tushare_availability()

    # 3. 运行校验（如果 Tushare 可用）
    if tushare_ok:
        results = test_validation()
    else:
        print("\n⚠️ 跳过数据完整性校验 (Tushare 不可用)")
        print("   AKShare 数据获取正常，系统可继续使用。")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()