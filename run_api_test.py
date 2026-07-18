import sys
# Add virtual environment site-packages if running globally
sys.path.append(".venv/lib/python3.12/site-packages")

from environments.speaker_listener import SpeakerListenerEnv
from pettingzoo.test import api_test

try:
    print("Starting API test...")
    env = SpeakerListenerEnv()
    api_test(env)
    print("API test passed successfully!")
except Exception as e:
    print(f"\nAPI test failed with error:")
    import traceback
    traceback.print_exc()
