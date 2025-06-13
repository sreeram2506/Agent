from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ChallengeRequired, BadPassword, TwoFactorRequired
import os
from dotenv import load_dotenv

load_dotenv()

def create_instagram_session():
    client = Client()
    username = os.getenv('INSTAGRAM_USERNAME')
    password = os.getenv('INSTAGRAM_PASSWORD')
    session_path = os.path.join('assets', 'instagram_session.json')

    if not username or not password:
        print("Error: Please set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD in .env file")
        return False

    # Check for existing session
    if os.path.exists(session_path):
        try:
            print(f"Loading existing session from {os.path.abspath(session_path)}...")
            client.load_settings(session_path)
            client.login(username, password)
            print("✅ Loaded existing session successfully")
            return True
        except LoginRequired:
            print("Existing session is invalid, attempting new login...")
        except Exception as e:
            print(f"Error loading session: {e}")

    # Attempt new login
    try:
        print(f"Attempting to log in as {username}...")
        client.login(username, password)
    except TwoFactorRequired:
        print("2FA is enabled. Please enter the verification code sent to your device:")
        code = input("Enter 2FA code: ").strip()
        try:
            client.login(username, password, verification_code=code)
        except Exception as e:
            print(f"❌ 2FA login failed: {e}")
            return False
    except BadPassword:
        print("❌ Error: Incorrect password")
        return False
    except ChallengeRequired:
        print("❌ Error: Instagram requires a login challenge (e.g., email/SMS verification)")
        # Handle challenge manually or configure challenge resolver
        return False
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return False

    # Save session
    try:
        os.makedirs('assets', exist_ok=True)
        client.dump_settings(session_path)
        print(f"✅ Session file created at: {os.path.abspath(session_path)}")
        return True
    except Exception as e:
        print(f"❌ Error saving session: {e}")
        return False

if __name__ == "__main__":
    if create_instagram_session():
        print("Session creation successful")
    else:
        print("Session creation failed")