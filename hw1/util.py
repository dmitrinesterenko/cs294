from datetime import datetime

def epoch():
    return int((datetime.utcnow() - datetime(1970, 1, 1)).total_seconds())
