import os
import logging
import json
import time
import base64
import hmac
import hashlib

logger = logging.getLogger(__name__)

def generate_livekit_token(room_name, participant_id):
    try:
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        
        if not api_key or not api_secret:
            logger.error("LiveKit credentials not set")
            return None
        
        # Token expiration (1 hour)
        exp = int(time.time()) + 3600
        
        # Create payload
        payload = {
            "sub": participant_id,
            "iss": api_key,
            "nbf": int(time.time()),
            "exp": exp,
            "video": {
                "canPublish": True,
                "canPublishData": True,
                "canSubscribe": True,
                "room": room_name,
                "roomJoin": True
            }
        }
        
        # Encode header
        header = {"typ": "JWT", "alg": "HS256"}
        
        header_encoded = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip('=')
        
        payload_encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip('=')
        
        # Create signature
        message = f"{header_encoded}.{payload_encoded}"
        signature = base64.urlsafe_b64encode(
            hmac.new(
                api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
        ).decode().rstrip('=')
        
        token = f"{message}.{signature}"
        logger.info(f"LiveKit token generated")
        return token
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return None