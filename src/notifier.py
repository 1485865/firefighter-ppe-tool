from plyer import notification

def send_alert(title, message):
    try:
        notification.notify(
            title=title,
            message=message,
            app_name='Firefighter Tool V5.0',
            timeout=5
        )
    except:
        pass