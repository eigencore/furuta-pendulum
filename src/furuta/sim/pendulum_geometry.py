"""Constantes geométricas del eslabón del péndulo, leídas del URDF.

Fuente única de verdad para la masa / COM / inercia del cuerpo del péndulo:
- Onshape (CAD) es la herramienta de diseño.
- `model/model.urdf` es la descripción canónica exportada del robot.
- Este módulo lee el URDF al importarse y expone tres constantes:
    PENDULUM_MASS_KG, PENDULUM_COM_M, PENDULUM_I_COM_SWING_KG_M2.

Son propiedades *geométricas*: las fija la forma de la pieza, no el montaje
de cada robot. No cambian entre reconstrucciones, así que no se aleatorizan.
Lo que sí varía (fricción, rodamientos, etc.) sigue pasando por el pipeline
de sysid.

Usa solo la biblioteca estándar a propósito, para que el flujo de sysid pueda
importarlo sin arrastrar MuJoCo / gymnasium.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


# Disposición del repositorio:
#   <repo>/src/furuta/sim/pendulum_geometry.py   <- este archivo
#   <repo>/model/model.urdf
# parents[0] = sim, [1] = furuta, [2] = src, [3] = <repo>
URDF_PATH = Path(__file__).resolve().parents[3] / "model" / "model.urdf"


def _load_pendulum_geometry(urdf_path: Path) -> tuple[float, float, float]:
    """Lee del URDF la masa del péndulo, la distancia del COM al eje de giro e I_com.

    Devuelve (mass_kg, com_m, I_com_swing_kg_m2), donde:
    - mass_kg: masa total del péndulo.
    - com_m: distancia perpendicular desde el eje de rotación (el eje x del
      eslabón, que coincide con el eje de la articulación `arm_to_pendulum`)
      hasta el COM. Es sqrt(y² + z²) del origen inercial: la componente x va
      *a lo largo* del eje de rotación y no afecta a la dinámica de oscilación
      (ver la nota del URDF en `model/model.urdf`).
    - I_com_swing_kg_m2: momento de inercia del péndulo respecto a su propio
      COM, a lo largo del eje de oscilación (ixx del tensor de inercia en el COM).
    """
    # XPath: dentro del <link name="pendulum"> busca su bloque <inertial>.
    inertial = ET.parse(urdf_path).getroot().find(
        "./link[@name='pendulum']/inertial"
    )
    if inertial is None:
        raise RuntimeError(
            f"Could not find <link name='pendulum'>/<inertial> in {urdf_path}"
        )
    mass = float(inertial.find("mass").get("value"))
    # <origin xyz="x y z"> es la posición del COM respecto al eslabón.
    xyz = [float(v) for v in inertial.find("origin").get("xyz").split()]
    # Distancia perpendicular al eje x (Pitágoras): se ignora x porque va a
    # lo largo del eje de giro.
    com_m = (xyz[1] ** 2 + xyz[2] ** 2) ** 0.5
    ixx = float(inertial.find("inertia").get("ixx"))
    return mass, com_m, ixx


# Se evalúa una sola vez, al importar el módulo.
PENDULUM_MASS_KG, PENDULUM_COM_M, PENDULUM_I_COM_SWING_KG_M2 = (
    _load_pendulum_geometry(URDF_PATH)
)
