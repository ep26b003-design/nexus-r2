import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("model/robot.xml")
data = mujoco.MjData(model)

# Open the 3D viewer window
mujoco.viewer.launch(model, data)