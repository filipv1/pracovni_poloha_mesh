"""
Email Service for Notifications
Handles sending email notifications for job completion/failure
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import logging
from typing import List, Optional
import threading
import queue
import time

logger = logging.getLogger(__name__)


class EmailService:
    """Email service with async sending and retry logic"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email or username
        
        # Email queue for async sending
        self.email_queue = queue.Queue()
        self.sending = False
        self.worker_thread = None
        
        # Start email worker if credentials are provided
        if username and password:
            self.start_worker()
        else:
            logger.warning("Email service not configured - missing credentials")
    
    def start_worker(self):
        """Start the background email worker"""
        if self.sending:
            return
            
        self.sending = True
        self.worker_thread = threading.Thread(target=self._email_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Email worker started")
    
    def stop_worker(self):
        """Stop the email worker"""
        self.sending = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Email worker stopped")
    
    def _email_worker(self):
        """Background worker for sending emails"""
        while self.sending:
            try:
                # Get email from queue
                try:
                    email_data = self.email_queue.get(timeout=5)
                except queue.Empty:
                    continue
                
                # Send with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self._send_email_smtp(
                            email_data['to'],
                            email_data['subject'],
                            email_data['html_body'],
                            email_data.get('text_body')
                        )
                        logger.info(f"Email sent to {email_data['to']}")
                        break
                    except Exception as e:
                        logger.error(f"Email send attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries - 1:
                            # Exponential backoff
                            time.sleep(2 ** attempt)
                        else:
                            logger.error(f"Failed to send email after {max_retries} attempts")
                            
            except Exception as e:
                logger.error(f"Email worker error: {e}")
                time.sleep(5)
    
    def send_email(self, to_email: str, subject: str, html_body: str, text_body: str = None):
        """Queue an email for sending"""
        if not self.username or not self.password:
            logger.warning(f"Cannot send email - service not configured")
            return False
            
        email_data = {
            'to': to_email,
            'subject': subject,
            'html_body': html_body,
            'text_body': text_body
        }
        
        self.email_queue.put(email_data)
        logger.info(f"Email queued for {to_email}")
        return True
    
    def _send_email_smtp(self, to_email: str, subject: str, html_body: str, text_body: str = None):
        """Send email via SMTP"""
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = self.from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add text and HTML parts
        if text_body:
            text_part = MIMEText(text_body, 'plain')
            msg.attach(text_part)
        
        html_part = MIMEText(html_body, 'html')
        msg.attach(html_part)
        
        # Send email
        context = ssl.create_default_context()
        
        with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
            server.ehlo()
            if self.smtp_port == 587:  # TLS
                server.starttls(context=context)
                server.ehlo()
            server.login(self.username, self.password)
            server.send_message(msg)
    
    def send_completion_email(self, job, files: List):
        """Send job completion email"""
        user = job.user
        
        # Generate download links HTML
        download_links = ""
        for f in files:
            file_icon = "📊" if f.file_type == 'xlsx' else "📦"
            download_links += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">
                    {file_icon} {f.filename}
                </td>
                <td style="padding: 10px; border: 1px solid #ddd;">
                    {f.size_mb:.1f} MB
                </td>
                <td style="padding: 10px; border: 1px solid #ddd;">
                    <a href="{f.r2_url}" style="color: #4CAF50; text-decoration: none; font-weight: bold;">
                        Download
                    </a>
                </td>
            </tr>
            """
        
        # Calculate processing time
        processing_time = "N/A"
        if job.time_elapsed_seconds:
            minutes = int(job.time_elapsed_seconds // 60)
            seconds = int(job.time_elapsed_seconds % 60)
            processing_time = f"{minutes}m {seconds}s"
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>✅ Your Analysis is Ready!</h1>
                </div>
                
                <div class="content">
                    <h2>Hello {user.username},</h2>
                    <p>Your pose analysis has been completed successfully.</p>
                    
                    <h3>📹 Video Details:</h3>
                    <ul>
                        <li><strong>Filename:</strong> {job.video_filename}</li>
                        <li><strong>Size:</strong> {job.video_size_mb:.1f} MB</li>
                        <li><strong>Processing Time:</strong> {processing_time}</li>
                    </ul>
                    
                    <h3>📥 Download Your Results:</h3>
                    <table>
                        <thead>
                            <tr style="background-color: #4CAF50; color: white;">
                                <th style="padding: 10px; text-align: left;">File</th>
                                <th style="padding: 10px; text-align: left;">Size</th>
                                <th style="padding: 10px; text-align: left;">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {download_links}
                        </tbody>
                    </table>
                    
                    <p style="margin-top: 20px; padding: 10px; background-color: #fff3cd; border: 1px solid #ffc107;">
                        ⚠️ <strong>Important:</strong> Files will be available for download for 7 days.
                    </p>
                </div>
                
                <div class="footer">
                    <p>Processed on {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
                    <p>Job ID: {job.id}</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_body = f"""
        Your Analysis is Ready!
        
        Hello {user.username},
        
        Your pose analysis has been completed successfully.
        
        Video Details:
        - Filename: {job.video_filename}
        - Size: {job.video_size_mb:.1f} MB
        - Processing Time: {processing_time}
        
        Download your results:
        {''.join([f"- {f.filename} ({f.size_mb:.1f} MB): {f.r2_url}" for f in files])}
        
        Important: Files will be available for download for 7 days.
        
        Processed on {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        Job ID: {job.id}
        """
        
        self.send_email(user.email, "✅ Pose Analysis Completed", html_body, text_body)
    
    def send_failure_email(self, job, error_message: str):
        """Send job failure notification"""
        user = job.user
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; text-align: center; }}
                .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; }}
                .error-box {{ background-color: #ffebee; border: 1px solid #f44336; padding: 15px; margin: 20px 0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>❌ Processing Failed</h1>
                </div>
                
                <div class="content">
                    <h2>Hello {user.username},</h2>
                    <p>Unfortunately, we encountered an error while processing your video.</p>
                    
                    <h3>📹 Video Details:</h3>
                    <ul>
                        <li><strong>Filename:</strong> {job.video_filename}</li>
                        <li><strong>Size:</strong> {job.video_size_mb:.1f} MB</li>
                        <li><strong>Attempts:</strong> {job.retry_count}</li>
                    </ul>
                    
                    <div class="error-box">
                        <h3>🔍 Error Details:</h3>
                        <pre style="white-space: pre-wrap; word-wrap: break-word;">
{error_message}
                        </pre>
                    </div>
                    
                    <h3>💡 What to do next:</h3>
                    <ol>
                        <li>Check if your video meets the requirements (MP4 format, max 30 minutes)</li>
                        <li>Try uploading the video again</li>
                        <li>If the problem persists, contact support with Job ID: <strong>{job.id}</strong></li>
                    </ol>
                </div>
                
                <div class="footer">
                    <p>Failed on {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
                    <p>Job ID: {job.id}</p>
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_body = f"""
        Processing Failed
        
        Hello {user.username},
        
        Unfortunately, we encountered an error while processing your video.
        
        Video Details:
        - Filename: {job.video_filename}
        - Size: {job.video_size_mb:.1f} MB
        - Attempts: {job.retry_count}
        
        Error Details:
        {error_message}
        
        What to do next:
        1. Check if your video meets the requirements (MP4 format, max 30 minutes)
        2. Try uploading the video again
        3. If the problem persists, contact support with Job ID: {job.id}
        
        Failed on {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        Job ID: {job.id}
        """
        
        self.send_email(user.email, "❌ Pose Analysis Failed", html_body, text_body)
    
    def send_test_email(self, to_email: str) -> bool:
        """Send a test email to verify configuration"""
        html_body = """
        <html>
            <body>
                <h2>Test Email</h2>
                <p>This is a test email from the Flask RunPod Application.</p>
                <p>If you received this, your email configuration is working correctly!</p>
                <p>Sent at: {}</p>
            </body>
        </html>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        text_body = f"""
        Test Email
        
        This is a test email from the Flask RunPod Application.
        If you received this, your email configuration is working correctly!
        
        Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_email(to_email, "Test Email - Flask RunPod App", html_body, text_body)