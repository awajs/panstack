"""Pipeline orchestrator.

Goal:
- load burst RAWs
- pick base frame (best face sharpness)
- align frames on background
- stack frames (weighted) to create motion blur background
- freeze face from base frame
"""

def make_panstack(*args, **kwargs):
    raise NotImplementedError("Implement make_panstack in pipeline.py")
