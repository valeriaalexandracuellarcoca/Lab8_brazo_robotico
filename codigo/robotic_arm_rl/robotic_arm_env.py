import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class RoboticArmEnv(gym.Env):
    def __init__(self, xml_path="E:/1-2025/IA/Labs/Lab8/Lab8_brazo_robotico/codigo/robotic_arm_rl/universal_robots_ur5e/ur5e.xml"):
        super(RoboticArmEnv, self).__init__()

        # Cargar el modelo MuJoCo desde el archivo XML
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Definir el espacio de acciones (6 articulaciones del brazo)
        self.action_space = spaces.Box(
            low=np.array([-6.2831, -6.2831, -3.1415, -6.2831, -6.2831, -6.2831]),
            high=np.array([6.2831, 6.2831, 3.1415, 6.2831, 6.2831, 6.2831]),
            dtype=np.float32
        )

        # Definir el espacio de estados con límites específicos
        self.observation_space = spaces.Box(
            low=np.array([-6.28319, -6.28319, -3.1415, -6.28319, -6.28319, -6.28319,  # Articulaciones
                          -0.5, -0.1, 0.05,  # Posición vaso (x, y, z)
                          -1, -1, -1, -1]),  # Cuaternión
            high=np.array([6.28319, 6.28319, 3.1415, 6.28319, 6.28319, 6.28319,  # Articulaciones
                           0.5, 0.1, 0.5,    # Posición vaso
                           1, 1, 1, 1]),     # Cuaternión
            dtype=np.float32
        )

        # Parámetros del entorno
        self.max_steps = 1000  # Máximo número de pasos por episodio
        self.step_count = 0
        self.table_height = 0.05  # Altura de la superficie de la mesa
        self.glass_height = 0.05  # Altura del vaso (ajustada según el XML)
        self.time_limit = 50.0    # Tiempo límite del episodio en segundos simulados
        self.dt = self.model.opt.timestep  # Paso de tiempo de la simulación

        # Índices para acceder a los datos del vaso
        self.glass_body_id = self.model.body("glass").id
        self.glass_qpos_start = 6  # Índice donde comienzan las posiciones del vaso en qpos

        # Contexto de renderización off-screen
        self._offscreen_renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reiniciar el estado de la simulación
        mujoco.mj_resetData(self.model, self.data)

        # Establecer la posición inicial del brazo (keyframe 'home')
        mujoco.mj_forward(self.model, self.data)

        # Reposicionar el vaso aleatoriamente en el eje X (-0.5 a 0.5 metros)
        glass_x = self.np_random.uniform(-0.5, 0.5)
        self.data.qpos[self.glass_qpos_start:self.glass_qpos_start+3] = [glass_x, 0, self.table_height + self.glass_height]
        self.data.qpos[self.glass_qpos_start+3:self.glass_qpos_start+7] = [1, 0, 0, 0]  # Cuaternión nulo

        # Propagar los cambios
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Estado: [ángulos de las articulaciones, posición del vaso, orientación del vaso]
        return np.concatenate([
            self.data.qpos[:6],  # Ángulos de las 6 articulaciones
            self.data.qpos[self.glass_qpos_start:self.glass_qpos_start+7]  # Posición (3) y orientación (4) del vaso
        ])

    def step(self, action):
        # Aplicar las acciones (control de las articulaciones)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        done = False
        truncated = False

        # Calcular la recompensa
        reward = self._compute_reward()

        # Verificar condiciones de terminación
        if self._is_glass_fallen():
            done = True  # Terminar cuando el vaso cae
        elif self.step_count >= self.max_steps or self.data.time >= self.time_limit:
            truncated = True  # Terminar si se excede el límite de pasos o tiempo

        obs = self._get_obs()
        info = {}

        return obs, reward, done, truncated, info

    def _compute_reward(self):
        # Recompensa basada en la distancia del efector final al vaso
        glass_pos = self.data.qpos[self.glass_qpos_start:self.glass_qpos_start+3]
        ee_pos = self.data.site("attachment_site").xpos  # Posición del efector final
        distance = np.linalg.norm(glass_pos - ee_pos)

        # Recompensa negativa por paso y distancia, alta recompensa si el vaso cae
        reward = -0.1 - distance
        if self._is_glass_fallen():
            reward += 100.0  # Recompensa alta por hacer caer el vaso
        return reward

    def _is_glass_flipped(self):
        # No necesitamos esta función para este objetivo, pero la dejamos por compatibilidad
        quat = self.data.qpos[self.glass_qpos_start+3:self.glass_qpos_start+7]
        rotmat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotmat, quat)
        rotmat = rotmat.reshape(3, 3)
        z_axis = rotmat[:, 2]
        return abs(np.dot(z_axis, [0, 0, 1])) < 0.2

    def _is_glass_fallen(self):
        # Verificar si el vaso ha caído de la mesa (z < altura de la mesa)
        glass_z = self.data.qpos[self.glass_qpos_start+2]
        return glass_z < self.table_height

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            if self._offscreen_renderer is None:
                self._offscreen_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._offscreen_renderer.update_scene(self.data)
            return self._offscreen_renderer.render()
        else:
            raise NotImplementedError(f"Modo de renderización {mode} no soportado")

    def close(self):
        if self._offscreen_renderer is not None:
            self._offscreen_renderer = None