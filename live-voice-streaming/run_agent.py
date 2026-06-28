import traceback
import sys

print("Starting run_agent.py", flush=True)
try:
    import agent
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
except BaseException as e:
    print(f"Caught BaseException: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
print("Finished agent import successfully", flush=True)
