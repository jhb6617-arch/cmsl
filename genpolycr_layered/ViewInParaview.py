from paraview.simple import *
import math

view = GetActiveView()

# Keep Z up
view.CameraViewUp = [0, 0, 1]

# Distance of camera from the origin
r = 100   # adjust as needed

# 45 degree azimuth rotation about Z
angle = math.radians(45)

# Give the camera some height in Z so we see the top face
x = r * math.cos(angle)
y = r * math.sin(angle)
z = r * 0.5   # <-- add some elevation (0.5 * r = 45° elevation)

view.CameraPosition = [x, y, z]
view.CameraFocalPoint = [0, 0, 0]

ResetCamera()
Render()
