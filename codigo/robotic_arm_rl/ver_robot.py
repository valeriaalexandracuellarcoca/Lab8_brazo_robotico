import mujoco
import mujoco.viewer

# Cargar el modelo desde el archivo XML
model = mujoco.MjModel.from_xml_path("E:/1-2025/IA/Labs/Lab8/Lab8_brazo_robotico/codigo/robotic_arm_rl/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)

# Lanzar el visor interactivo
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Avanzar la simulación
        mujoco.mj_step(model, data)
        # Sincronizar el visor
        viewer.sync()