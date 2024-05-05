import psutil
import os
import signal
import smtplib
from email.mime.text import MIMEText
from time import sleep
import argparse
def check_process(pid):
    """Check whether pid exists in the current process table."""
    return psutil.pid_exists(pid)


def send_email(receiver_email,pid):
    """Send an email to the specified receiver."""
    sender_email = "13929107765@163.com"
    sender_password = "BUCEHICQHRMFZOUE"

    message = MIMEText(f"你的程序 PID {pid} 停了，这可能是因为程序已经运行成功或者出现了bug，请及时检查！.")
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = f"4090上的程序{pid}已经停止！"

    # Setting up the SMTP server
    server = smtplib.SMTP('smtp.163.com', 25)
    server.starttls()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, receiver_email, message.as_string())
    server.quit()

def main():
    parser = argparse.ArgumentParser(description="Monitor a process and send an email if it stops.")
    parser.add_argument("pid", type=int, help="The PID of the process to monitor")
    args = parser.parse_args()
    
    TARGET_PID = args.pid
    EMAIL = "779027322@qq.com"
    CHECK_INTERVAL = 100  # in seconds

    while True:
        if not check_process(TARGET_PID):
            send_email(EMAIL, TARGET_PID)
            print('程序结束')
            break  # End the loop if process stops
        sleep(CHECK_INTERVAL)
    
if __name__ == "__main__":
    main()
    # send_email("779027322@qq.com",352491)