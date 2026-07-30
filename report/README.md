# report — 评测进度邮件汇报模块

实时监控 evalscope 评测进度，通过 SMTP 邮件定期发送正确率报告。

## 适用场景

- 长时间运行的 evalscope 评测（GSM8K、HumanEval 等），需要**实时了解进度和正确率**
- 支持 **pass@k**（repeats 多采样）的正确率计算
- 评测跑完或停滞时自动发送最终报告

## 架构

```
evalscope reviews/*.jsonl
        ↓ 读取
  EvalReporter.check()
        ↓ 计算
  EvalSnapshot (pass@1 / pass@k)
        ↓ 格式化
  SmtpMailer.send() → 163/QQ SMTP → 邮箱
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `eval_reporter.py` | 评测进度汇报主模块，含 CLI 入口 |
| `smtp_mailer.py` | SMTP 邮件发送工具类 |
| `__init__.py` | 包标识 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r ../proxy/requirements.txt
```

### 2. 设置邮箱环境变量

```bash
export MAIL_USER="你的邮箱@163.com"
export MAIL_PASS="你的163SMTP授权码"
```

### 3. 启动汇报

**pass@1 单次评测（如 GSM8K temp0.4）：**

```bash
python -m report.eval_reporter \
    --reviews /workspace/outputs/reviews/*/gsm8k_main.jsonl \
    --total 1319 \
    --baseline 85.90
```

**pass@3 评测（repeats=3）：**

```bash
python -m report.eval_reporter \
    --work-dir /workspace/outputs/gsm8k_think_pass3 \
    --total 1319 \
    --repeats 3 \
    --baseline 85.90
```

**自定义间隔和收件人：**

```bash
python -m report.eval_reporter \
    --reviews /path/to/reviews/*.jsonl \
    --total 1319 \
    --interval 120 \
    --to admin@163.com
```

### 4. 后台运行

```bash
nohup python -m report.eval_reporter \
    --reviews /workspace/outputs/reviews/*/gsm8k_main.jsonl \
    --total 1319 \
    --interval 120 \
    --baseline 85.90 \
    > /tmp/eval_reporter.log 2>&1 &
```

## 作为模块调用

```python
from report.smtp_mailer import SmtpMailer
from report.eval_reporter import EvalReporter, EvalConfig

mailer = SmtpMailer("smtp.163.com", 465, "user@163.com", "authcode")
config = EvalConfig(
    reviews_path="/workspace/outputs/reviews/*/gsm8k_main.jsonl",
    total=1319,
    repeats=1,
    baseline=85.90,
    interval=120,
)

reporter = EvalReporter(mailer, config)
reporter.run()  # 持续监控
```

## 配置项

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--reviews` | (必填或 --work-dir) | reviews 文件路径（glob 模式） |
| `--work-dir` | (必填或 --reviews) | evalscope 输出目录 |
| `--total` | (必填) | 评测总题数 |
| `--repeats` | 1 | repeats 数（1=单pass，3=pass@3） |
| `--baseline` | 0 | 基线正确率（用于对比） |
| `--interval` | 120 | 检查间隔（秒） |
| `--to` | =发件人 | 收件邮箱 |
| `--smtp-host` | smtp.163.com | SMTP 服务器 |
| `--smtp-port` | 465 | SMTP 端口 |

## 正确率计算逻辑

### pass@1（repeats=1）
每个 index 一条记录，直接统计 `acc=1.0` 的比例。

### pass@k（repeats≥2）
reviews 中 `index` 是全局 sample 序号（0,1,2,...），同一题的 k 次采样占 k 个连续 index。
- 按 `index // repeats` 分组为同一题
- **pass@1**：每组取第 1 次采样的正确率
- **pass@k**：每组取 max（任一对即对）
- **捞回数** = pass@k 正确数 - pass@1 正确数