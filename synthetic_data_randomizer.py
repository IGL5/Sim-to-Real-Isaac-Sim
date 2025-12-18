# standalone_drone_sdg_iter1_fixed.py

from omni.isaac.kit import SimulationApp
import os
import argparse

# --- CONFIGURACIÓN DE LANZAMIENTO ---
parser = argparse.ArgumentParser("Dataset Generator - Drone Context")
parser.add_argument("--headless", type=bool, default=False, help="Ejecutar sin GUI")
parser.add_argument("--height", type=int, default=544, help="Altura imagen")
parser.add_argument("--width", type=int, default=960, help="Anchura imagen")
parser.add_argument("--num_frames", type=int, default=100, help="Frames a generar de prueba")
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_drone_data_iter1", help="Salida")

args, unknown_args = parser.parse_known_args()

CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless,
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

simulation_app = SimulationApp(launch_config=CONFIG)

# --- IMPORTS DE REPLICATOR E ISAAC ---
import omni.replicator.core as rep
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.usd


# -----------------------------------------------------------------------------
# 1. ASSETS Y UTILIDADES
# -----------------------------------------------------------------------------

def get_server_path():
    """Obtiene la ruta base del servidor Nucleus (localhost o remoto)"""
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("ERROR: No se encontró el servidor Nucleus. Revisa tu instalación de Isaac Sim.")
        return ""
    return assets_root_path


SERVER_PATH = get_server_path()

# LISTA DE COCHES (Solo turismos, excluyendo furgonetas/camiones)
CAR_ASSETS = [
    SERVER_PATH + "/Isaac/Vehicles/Audi/A8/audi_a8.usd",
    SERVER_PATH + "/Isaac/Vehicles/Tesla/ModelS/model_s.usd",
    SERVER_PATH + "/Isaac/Vehicles/Porsche/911/911.usd",
]

# TEXTURA DEL SUELO (Hierba o Asfalto básico)
GROUND_TEXTURE = SERVER_PATH + "/Isaac/Materials/Textures/Patterns/nv_artificialgrass_green.jpg"


# -----------------------------------------------------------------------------
# 2. FUNCIONES DE ESCENA
# -----------------------------------------------------------------------------

def setup_environment():
    """Crea el suelo base (Iteración 1: Plano simple con textura)"""
    # Creamos un plano muy grande (scale=100 equivale a mucho terreno)
    plane = rep.create.plane(scale=100, visible=True)

    # CORRECCIÓN: Eliminado el argumento 'scale' que causaba el crash.
    # La textura se aplicará con mapeado por defecto.
    with plane:
        rep.randomizer.texture(textures=[GROUND_TEXTURE])

    return plane


def add_cars():
    """Añade los coches a la escena con la etiqueta semántica correcta"""
    if not SERVER_PATH: return None

    # semantics=[('class', 'car')] fuerza a que todo el objeto sea detectado como 'car'.
    cars = [rep.create.from_usd(path, semantics=[('class', 'car')]) for path in CAR_ASSETS]

    return rep.create.group(cars)


def setup_camera():
    """Configura la cámara del dron"""
    cam = rep.create.camera(clipping_range=(0.1, 100000))
    return cam


def setup_sun():
    """Luz básica para exteriores"""
    # SphereLight muy lejana o DistantLight simulan el sol
    light = rep.create.light(light_type="Distant", intensity=3000, rotation=(300, 0, 0))
    return light


# -----------------------------------------------------------------------------
# 3. ORQUESTADOR (Lógica por Frame)
# -----------------------------------------------------------------------------

def main():
    # 3.1 Preparar elementos estáticos
    setup_environment()
    setup_sun()

    # 3.2 Preparar elementos dinámicos
    rep_car_group = add_cars()
    cam = setup_camera()

    # Render product
    render_product = rep.create.render_product(cam, (args.width, args.height))

    # 3.3 Trigger: Qué pasa en cada frame
    with rep.trigger.on_frame(num_frames=args.num_frames):

        # A. Movimiento del Dron (Cámara)
        with cam:
            # Posición: Aleatoria en X e Y (-20m a 20m).
            # Altura (Z): De 10m a 30m (Típico vuelo de dron ligero)
            rep.modify.pose(
                position=rep.distribution.uniform((-20, -20, 10), (20, 20, 30)),
                look_at=(0, 0, 0)
            )

        # B. Randomización de los Coches
        if rep_car_group:
            with rep_car_group:
                # Dispersar los coches en el suelo (Z=0)
                rep.modify.pose(
                    position=rep.distribution.uniform((-10, -10, 0), (10, 10, 0)),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
                )

    # -------------------------------------------------------------------------
    # 4. WRITER (Guardar datos)
    # -------------------------------------------------------------------------

    writer = rep.WriterRegistry.get("KittiWriter")
    print(f"--> Guardando datos en: {args.data_dir}")

    writer.initialize(
        output_dir=args.data_dir,
        omit_semantic_type=True
    )

    writer.attach(render_product)

    # -------------------------------------------------------------------------
    # 5. EJECUCIÓN
    # -------------------------------------------------------------------------
    print("Iniciando generación...")
    rep.orchestrator.run()

    # Bucle para mantener la app viva mientras genera
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()
    simulation_app.close()
    print("Generación finalizada.")


if __name__ == "__main__":
    main()