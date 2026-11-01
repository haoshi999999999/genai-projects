from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import json
import secrets

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

PASSWORD_HASH_ITERATIONS = 600_000


def _encode_segment(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_segment(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

'''def hash_password(password: str):
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PASSWORD_HASH_ITERATIONS,
        _encode_segment(salt),
        _encode_segment(password_hash),
    )'''

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str):
    return hashlib.sha256(password.encode()).hexdigest() == hashed


'''def verify_password(password: str, hashed: str):
    try:
        algorithm, iterations, encoded_salt, encoded_hash = hashed.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations)
        if iterations <= 0:
            return False
        salt = _decode_segment(encoded_salt)
        expected_hash = _decode_segment(encoded_hash)
        actual_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual_hash, expected_hash)
    except (AttributeError, TypeError, ValueError):
        return False'''

def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = int(expire.timestamp())
    header = _encode_segment(json.dumps({"alg": ALGORITHM, "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _encode_segment(json.dumps(to_encode, separators=(",", ":")).encode())
    signing_input = f"{header}.{payload}"
    signature = hmac.new(SECRET_KEY.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_encode_segment(signature)}"

def decode_token(token: str):
    try:
        header, payload, signature = token.split(".")
        signing_input = f"{header}.{payload}"
        expected = hmac.new(SECRET_KEY.encode(), signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_decode_segment(signature), expected):
            raise ValueError("Invalid token signature")
        if json.loads(_decode_segment(header)).get("alg") != ALGORITHM:
            raise ValueError("Unsupported token algorithm")
        claims = json.loads(_decode_segment(payload))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid token") from exc
    if "exp" in claims and datetime.utcnow().timestamp() >= claims["exp"]:
        raise ValueError("Token has expired")
    return claims
