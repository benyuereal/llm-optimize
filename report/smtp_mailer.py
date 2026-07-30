#!/usr/bin/env python3
"""
SMTP 邮件发送工具模块。

封装 163/QQ 邮箱等 SSL 发信能力，支持纯文本和 HTML 格式。
"""

import smtplib, ssl
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from typing import Optional


class SmtpMailer:
    """SMTP 邮件发送器。

    Args:
        host: SMTP 服务器地址，如 smtp.163.com / smtp.qq.com
        port: SSL 端口，通常 465
        user: 邮箱账号，如 user@163.com
        password: 邮箱授权码（非登录密码）

    Usage:
        mailer = SmtpMailer("smtp.163.com", 465, "user@163.com", "auth_code")
        mailer.send("主题", "正文", "收件人@163.com")
    """

    def __init__(self, host: str, port: int, user: str, password: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self._display_name = user.split("@")[0]

    def send(
        self,
        subject: str,
        body: str,
        to: Optional[str] = None,
        html: bool = False,
    ) -> bool:
        """发送一封邮件。

        Args:
            subject: 邮件主题
            body: 邮件正文
            to: 收件地址，默认发给自己
            html: 是否 HTML 格式

        Returns:
            bool: 发送成功返回 True
        """
        to = to or self.user
        msg = MIMEText(body, "html" if html else "plain", "utf-8")
        msg["From"] = formataddr((self._display_name, self.user))
        msg["To"] = to
        msg["Subject"] = Header(subject, "utf-8")

        try:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.host, self.port, timeout=30, context=ctx) as s:
                s.login(self.user, self.password)
                s.sendmail(self.user, [to], msg.as_string())
            return True
        except smtplib.SMTPException as exc:
            print(f"[SmtpMailer] 发送失败: {exc}")
            return False

    def send_report(
        self,
        title: str,
        lines: list[tuple[str, str]],
        footer: str = "",
        to: Optional[str] = None,
    ) -> bool:
        """发送格式化报告邮件（Key-Value 列表）。

        Args:
            title: 邮件标题
            lines: (key, value) 列表，如 [("进度", "100/1319"), ("正确率", "86.81%")]
            footer: 邮件末尾附加文字
            to: 收件地址
        """
        body_parts = [f"【{title}】"]
        for k, v in lines:
            body_parts.append(f"{k}: {v}")
        if footer:
            body_parts.append("")
            body_parts.append(footer)
        body = "\n".join(body_parts)
        return self.send(f"[评测] {title}", body, to=to)


if __name__ == "__main__":
    # 连通性测试
    import os
    user = os.environ.get("MAIL_USER", "")
    pwd = os.environ.get("MAIL_PASS", "")
    if user and pwd:
        m = SmtpMailer("smtp.163.com", 465, user, pwd)
        ok = m.send("[测试] SMTP 邮件模块连通性测试", "收到此邮件说明 SmtpMailer 模块正常工作。")
        print(f"测试邮件 {'发送成功 ✓' if ok else '发送失败 ✗'}")
    else:
        print("请设置 MAIL_USER 和 MAIL_PASS 环境变量后运行测试")