#!/usr/bin/env python3
"""
evalscope 评测进度实时监控与邮件汇报模块。

核心功能：
  - 读取 evalscope reviews/*.jsonl 文件，实时计算正确率
  - 支持 pass@1（单次采样）和 pass@k（repeats 多采样，按 index//k 分组取 max）
  - 通过 SMTP 邮件定期发送进度报告
  - 可作为 CLI 工具直接运行，也可作为库调用

用法：
  # 作为 CLI 工具（需先 export MAIL_USER / MAIL_PASS）
  python -m report.eval_reporter \\
      --reviews /path/to/reviews/*.jsonl \\
      --interval 120 \\
      --total 1319 \\
      --repeats 3 \\
      --baseline 85.90

  # 作为模块调用
  from report.eval_reporter import EvalReporter
  reporter = EvalReporter(reviews_path="/path/to/reviews/*.jsonl", total=1319)
  reporter.run()
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from report.smtp_mailer import SmtpMailer


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EvalSnapshot:
    """一次评测检查的快照结果。"""
    num_questions: int = 0        # 已判分题数
    total: int = 0                # 总题数
    pass1_correct: int = 0        # pass@1 正确数
    pass1_acc: float = 0.0       # pass@1 正确率
    passk_correct: int = 0        # pass@k 正确数
    passk_acc: float = 0.0       # pass@k 正确率
    recovered: int = 0            # pass@k 比 pass@1 多捞回题数
    reviews_rows: int = 0         # reviews 文件总行数
    finished: bool = False        # 是否完成
    error: str = ""               # 错误信息


@dataclass
class EvalConfig:
    """评测配置。"""
    reviews_path: str = ""        # reviews 文件路径（支持 glob）
    total: int = 0                # 总题数
    repeats: int = 1              # repeats 数（1=单pass）
    baseline: float = 0.0         # 基线正确率
    interval: int = 120           # 检查间隔（秒）
    stale_limit: int = 5          # 连续多少次进度不变判定完成
    stale_threshold: int = 0      # 允许停滞判定的最低进度题数（默认 90% total）
    work_dir: str = ""            # evalscope 输出目录（替代 reviews_path）
    predictions_path: str = ""    # predictions 文件路径（可选，用于进度显示）

    def __post_init__(self):
        if not self.stale_threshold:
            self.stale_threshold = int(self.total * 0.9)


# ============================================================
# 核心计算
# ============================================================

def parse_reviews(filepath: str) -> List[dict]:
    """从 jsonl 文件加载 reviews 数据。"""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_snapshot(
    records: List[dict],
    total: int,
    repeats: int = 1,
) -> EvalSnapshot:
    """从 reviews 记录计算评测快照。

    Args:
        records: reviews 的 json 对象列表
        total: 评测总题数
        repeats: repeats 数（1=单pass，3=pass@3）

    Returns:
        EvalSnapshot 包含 pass@1、pass@k 等指标
    """
    if not records:
        return EvalSnapshot(total=total, reviews_rows=0)

    def acc_of(r):
        return r.get("sample_score", {}).get("score", {}).get("value", {}).get("acc", 0)

    n_rows = len(records)

    if repeats > 1:
        # pass@k: 按 index // repeats 分组，取 max
        groups = defaultdict(list)
        for r in records:
            idx = r.get("index", 0)
            groups[idx // repeats].append(acc_of(r))

        num_q = len(groups)
        p1_correct = sum(1 for v in groups.values() if sorted(v)[0] == 1.0)
        pk_correct = sum(1 for v in groups.values() if max(v) == 1.0)
        recovered = pk_correct - p1_correct
    else:
        # pass@1: 每个 index 一条记录
        by_idx = {}
        for r in records:
            idx = r.get("index")
            if idx not in by_idx:
                by_idx[idx] = acc_of(r)
        num_q = len(by_idx)
        p1_correct = sum(1 for a in by_idx.values() if a == 1.0)
        pk_correct = p1_correct
        recovered = 0

    p1_acc = (p1_correct / num_q * 100) if num_q else 0.0
    pk_acc = (pk_correct / num_q * 100) if num_q else 0.0

    finished = num_q >= total

    return EvalSnapshot(
        num_questions=num_q,
        total=total,
        pass1_correct=p1_correct,
        pass1_acc=p1_acc,
        passk_correct=pk_correct,
        passk_acc=pk_acc,
        recovered=recovered,
        reviews_rows=n_rows,
        finished=finished,
    )


def format_snapshot(
    snap: EvalSnapshot,
    config: EvalConfig,
    round_n: int = 0,
) -> str:
    """将快照格式化为邮件正文。"""
    prog_pct = snap.num_questions / snap.total * 100 if snap.total else 0
    diff = snap.passk_acc - config.baseline
    tag = f"pass@{config.repeats}" if config.repeats > 1 else "pass@1"

    lines = [
        f"评测进度汇报 (第 {round_n} 次)",
        f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"【进度】 {snap.num_questions}/{snap.total} 题 ({prog_pct:.1f}%)",
        f"  reviews 行数: {snap.reviews_rows}",
        "",
        f"【pass@1 实时】 {snap.pass1_acc:.2f}%  ({snap.pass1_correct}/{snap.num_questions} 对)",
        f"  = 每题首次采样正确率",
    ]

    if config.repeats > 1:
        lines += [
            "",
            f"【{tag} 实时】 {snap.passk_acc:.2f}%  ({snap.passk_correct}/{snap.num_questions} 对)",
            f"  = 每题 {config.repeats} 次采样任一对即对",
            f"  = 比 pass@1 多捞回 {snap.recovered} 题",
        ]

    if config.baseline:
        lines += [
            "",
            f"【对比基线】 {config.baseline}%",
            f"  {tag} vs 基线: {diff:+.2f}pp",
        ]

    lines += [
        "",
        f"【状态】 {'✅ 已完成' if snap.finished else '⏳ 进行中'}",
    ]

    return "\n".join(lines)


# ============================================================
# 汇报器
# ============================================================

class EvalReporter:
    """评测进度汇报器。

    定期检查 evalscope reviews 文件，计算正确率，通过邮件发送报告。

    Usage:
        reporter = EvalReporter(
            reviews_path="/workspace/outputs/reviews/*/gsm8k_main.jsonl",
            total=1319,
            repeats=1,
            baseline=85.90,
            mailer=mailer,
        )
        reporter.run()
    """

    def __init__(
        self,
        mailer: SmtpMailer,
        config: EvalConfig,
        to: Optional[str] = None,
    ):
        self.mailer = mailer
        self.config = config
        self.to = to or mailer.user
        self._last_total = 0
        self._stale = 0
        self._round = 0

    def _resolve_reviews_path(self) -> str:
        """找到 reviews 文件路径。"""
        import glob
        # 如果指定了 work_dir，从 work_dir 找 reviews
        if self.config.work_dir:
            pattern = os.path.join(self.config.work_dir, "reviews", "*", "*.jsonl")
            files = sorted(glob.glob(pattern))
            if files:
                return files[-1]
        # 直接用 reviews_path
        pattern = self.config.reviews_path
        files = sorted(glob.glob(pattern))
        if files:
            return files[-1]
        return ""

    def _resolve_predictions_path(self) -> str:
        """找到 predictions 文件路径（辅助判断进度）。"""
        import glob
        if self.config.predictions_path:
            files = sorted(glob.glob(self.config.predictions_path))
            if files:
                return files[-1]
        if self.config.work_dir:
            pattern = os.path.join(self.config.work_dir, "predictions", "*", "*.jsonl")
            files = sorted(glob.glob(pattern))
            if files:
                return files[-1]
        return ""

    def check(self) -> EvalSnapshot:
        """执行一次检查。"""
        rev_path = self._resolve_reviews_path()
        if not rev_path or not os.path.exists(rev_path):
            return EvalSnapshot(total=self.config.total, error="reviews 文件未找到")

        try:
            records = parse_reviews(rev_path)
        except Exception as e:
            return EvalSnapshot(total=self.config.total, error=str(e))

        return compute_snapshot(records, self.config.total, self.config.repeats)

    def run_once(self) -> EvalSnapshot:
        """执行一次检查并发送邮件（如有数据变化）。"""
        self._round += 1
        snap = self.check()

        if snap.error:
            print(f"[Round {self._round}] 错误: {snap.error}")
            return snap

        tag = f"pass@{self.config.repeats}" if self.config.repeats > 1 else "pass@1"
        prog_pct = snap.num_questions / snap.total * 100 if snap.total else 0
        diff = snap.passk_acc - self.config.baseline

        # 日志
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{snap.num_questions}/{snap.total} ({prog_pct:.1f}%) "
            f"P1={snap.pass1_acc:.2f}% "
            f"{tag}={snap.passk_acc:.2f}% "
            f"捞回{snap.recovered}题 "
            f"diff={diff:+.2f}pp"
        )

        # 构造邮件
        body = format_snapshot(snap, self.config, self._round)
        subject = (
            f"[评测] {tag}: {snap.num_questions}/{snap.total} "
            f"({prog_pct:.1f}%) {snap.passk_acc:.2f}%"
        )

        self.mailer.send(subject, body, to=self.to)
        return snap

    def run(self):
        """持续监控，定期检查并发送邮件。

        自动处理完成判定（进度到总数 / 停滞超限）。
        """
        print(f"EvalReporter 启动: reviews={self.config.reviews_path}")
        print(f"  total={self.config.total} repeats={self.config.repeats}")
        print(f"  间隔={self.config.interval}s 发给={self.to}")
        print(f"  Ctrl+C 停止")
        print()

        def _sigint(sig, frame):
            print("\n用户中断，退出")
            sys.exit(0)

        signal.signal(signal.SIGINT, _sigint)

        while True:
            snap = self.run_once()

            # 完成判定
            if snap.finished:
                final_body = format_snapshot(snap, self.config, self._round)
                self.mailer.send(
                    f"[评测完成] {snap.passk_acc:.2f}% ({snap.num_questions}/{snap.total})",
                    final_body,
                    to=self.to,
                )
                print(f"评测完成，退出")
                break

            # 停滞判定
            if snap.num_questions == self._last_total and snap.num_questions > 0:
                self._stale += 1
                if (
                    self._stale >= self.config.stale_limit
                    and snap.num_questions >= self.config.stale_threshold
                ):
                    final_body = format_snapshot(snap, self.config, self._round)
                    self.mailer.send(
                        f"[评测完成,停滞] {snap.passk_acc:.2f}%",
                        final_body,
                        to=self.to,
                    )
                    print(f"进度停滞 {self.config.stale_limit} 次，判定完成，退出")
                    break
            else:
                self._stale = 0
            self._last_total = snap.num_questions

            time.sleep(self.config.interval)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="evalscope 评测进度邮件汇报工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # pass@1 单次评测
  python -m report.eval_reporter \\
      --reviews /workspace/outputs/reviews/*/gsm8k_main.jsonl \\
      --total 1319 --baseline 85.90

  # pass@3 评测（repeats=3）
  python -m report.eval_reporter \\
      --work-dir /workspace/outputs/gsm8k_think_pass3 \\
      --total 1319 --repeats 3 --baseline 85.90

  # 指定邮箱配置
  MAIL_USER="user@163.com" MAIL_PASS="authcode" \\
  python -m report.eval_reporter --reviews ... --to user@163.com
        """,
    )

    parser.add_argument("--reviews", help="reviews 文件路径（glob 模式）")
    parser.add_argument("--work-dir", help="evalscope 输出目录（替代 --reviews）")
    parser.add_argument("--total", type=int, required=True, help="评测总题数")
    parser.add_argument("--repeats", type=int, default=1, help="repeats 数")
    parser.add_argument("--baseline", type=float, default=0, help="基线正确率")
    parser.add_argument(
        "--interval", type=int, default=120, help="检查间隔（秒，默认 120）"
    )
    parser.add_argument(
        "--to", default="", help="收件邮箱（默认等于发件邮箱）"
    )
    parser.add_argument(
        "--stale-limit", type=int, default=5, help="停滞判定次数（默认 5）"
    )

    # SMTP 配置（优先从环境变量读取）
    parser.add_argument("--smtp-host", default="smtp.163.com", help="SMTP 服务器")
    parser.add_argument("--smtp-port", type=int, default=465, help="SMTP 端口")
    parser.add_argument("--mail-user", default=os.environ.get("MAIL_USER", ""))
    parser.add_argument("--mail-pass", default=os.environ.get("MAIL_PASS", ""))

    args = parser.parse_args()

    if not args.mail_user or not args.mail_pass:
        print(
            "错误: 必须设置 MAIL_USER 和 MAIL_PASS 环境变量，"
            "或通过 --mail-user / --mail-pass 指定"
        )
        sys.exit(1)

    if not args.reviews and not args.work_dir:
        print("错误: 必须指定 --reviews 或 --work-dir")
        sys.exit(1)

    config = EvalConfig(
        reviews_path=args.reviews or "",
        total=args.total,
        repeats=args.repeats,
        baseline=args.baseline,
        interval=args.interval,
        stale_limit=args.stale_limit,
        work_dir=args.work_dir or "",
    )

    mailer = SmtpMailer(args.smtp_host, args.smtp_port, args.mail_user, args.mail_pass)
    reporter = EvalReporter(mailer, config, to=args.to or None)
    reporter.run()


if __name__ == "__main__":
    main()