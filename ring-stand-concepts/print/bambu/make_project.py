"""Package the ring stand as a Bambu Studio *project* 3MF with print settings embedded.

Usage: python3 make_project.py <BambuStudio resources/profiles/BBL dir>

Bambu Studio only loads settings from a 3MF whose model metadata says it was written by
BambuStudio and that carries Metadata/project_settings.config. On load it starts from the
named system presets (printer / process / filament) and keeps our values only for the keys
listed in `different_settings_to_system`, so those lists must match OVERRIDES below.
"""
import json, os, sys, zipfile
import numpy as np
import trimesh
from resolve import index, resolve

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "taipei101_tray_ring_stand.stl")
OUT = os.path.join(HERE, "..", "taipei101_tray_ring_stand_A1mini_project.3mf")
APP_VERSION = "02.08.04.57"

PRINTER = "Bambu Lab A1 mini 0.4 nozzle"
PROCESS = "0.08mm Extra Fine @BBL A1M"
FILAMENT = "Bambu PLA Matte @BBL A1M"
COLOUR = "#9B9EA0"          # matte grey

PROCESS_OVERRIDES = {
    "wall_loops": "3",
    "wall_generator": "arachne",
    "precise_outer_wall": "1",
    "elefant_foot_compensation": "0.15",
    "top_shell_layers": "10",
    "bottom_shell_layers": "6",
    "sparse_infill_density": "20%",
    "sparse_infill_pattern": "gyroid",
    "outer_wall_speed": ["60"],
    "inner_wall_speed": ["120"],
    "seam_position": "aligned",
    "top_surface_pattern": "monotonicline",
    "ironing_type": "no ironing",
    "enable_support": "0",
    "brim_type": "no_brim",
}
FILAMENT_OVERRIDES = {"slow_down_layer_time": ["8"]}
META_KEYS = {"name", "inherits", "from", "setting_id", "instantiation", "type", "version",
             "compatible_printers", "compatible_prints", "filament_id", "description", "renamed_from"}


def known_keys(profiles):
    """Every option key Bambu Studio defines (parsed from its PrintConfig.cpp)."""
    import re
    src = os.path.join(profiles, "..", "..", "..", "src", "libslic3r", "PrintConfig.cpp")
    return set(re.findall(r'this->add(?:_nullable)?\("([a-z0-9_]+)"', open(src).read()))


def project_config(profiles):
    idx = index(profiles)
    m, p, f = (resolve(idx, "machine", PRINTER), resolve(idx, "process", PROCESS),
               resolve(idx, "filament", FILAMENT))
    known = known_keys(profiles)
    for k in list(PROCESS_OVERRIDES) + list(FILAMENT_OVERRIDES):
        assert k in known, f"Bambu Studio has no setting named {k}"
    cfg = {}
    for d in (m, f, p):
        cfg.update({k: v for k, v in d.items() if k not in META_KEYS})
    cfg.update(PROCESS_OVERRIDES)
    cfg.update(FILAMENT_OVERRIDES)
    cfg.update({
        "name": "project_settings", "from": "project", "version": APP_VERSION,
        "printer_settings_id": PRINTER, "print_settings_id": PROCESS, "filament_settings_id": [FILAMENT],
        "inherits_group": ["", "", ""],
        "different_settings_to_system": [";".join(sorted(PROCESS_OVERRIDES)),
                                         ";".join(sorted(FILAMENT_OVERRIDES)), ""],
        "filament_ids": [f.get("filament_id", "")],
        "print_compatible_printers": p.get("compatible_printers", []),
        "filament_colour": [COLOUR],
        "curr_bed_type": "Textured PEI Plate",
        "printer_model": m.get("printer_model", "Bambu Lab A1 mini"),
    })
    return cfg


def model_xml(mesh):
    v = "\n".join(f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in mesh.vertices)
    t = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    # place on the A1 mini bed (180 x 180) centre
    tf = "1 0 0 0 1 0 0 0 1 90 90 0"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">
 <metadata name="Application">BambuStudio-{APP_VERSION}</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="Title">Taipei 101 ring stand</metadata>
 <resources>
  <object id="1" type="model">
   <mesh>
    <vertices>
{v}
    </vertices>
    <triangles>
{t}
    </triangles>
   </mesh>
  </object>
 </resources>
 <build>
  <item objectid="1" transform="{tf}" printable="1"/>
 </build>
</model>
'''


MODEL_SETTINGS = '''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="1">
    <metadata key="name" value="Taipei 101 ring stand"/>
    <metadata key="extruder" value="1"/>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <model_instance>
      <metadata key="object_id" value="1"/>
      <metadata key="instance_id" value="0"/>
    </model_instance>
  </plate>
</config>
'''

CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Default Extension="config" ContentType="text/xml"/>
</Types>
'''
RELS = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
'''

if __name__ == "__main__":
    mesh = trimesh.load(SRC, force="mesh")
    assert mesh.is_watertight and mesh.is_volume
    cfg = project_config(sys.argv[1])
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", model_xml(mesh))
        z.writestr("Metadata/project_settings.config", json.dumps(cfg, indent=4, sort_keys=True))
        z.writestr("Metadata/model_settings.config", MODEL_SETTINGS)
    # Backup: user presets for File > Import > Import Configs
    for kind, base, name, ov in (("process", PROCESS, "Taipei101 0.08mm @BBL A1M", PROCESS_OVERRIDES),
                                 ("filament", FILAMENT, "Taipei101 PLA Matte @BBL A1M", FILAMENT_OVERRIDES)):
        preset = {"type": kind, "name": name, "inherits": base, "from": "User", "instantiation": "true",
                  "version": APP_VERSION, ("print_settings_id" if kind == "process" else "filament_settings_id"):
                  name if kind == "process" else [name]}
        preset.update(ov)
        with open(os.path.join(HERE, "..", f"preset_{kind}_taipei101.json"), "w") as fh:
            json.dump(preset, fh, indent=4)
    print("wrote", os.path.relpath(OUT, HERE), f"({os.path.getsize(OUT) / 1e6:.1f} MB), {len(cfg)} settings")
    for k in sorted(PROCESS_OVERRIDES) + sorted(FILAMENT_OVERRIDES):
        print(f"  {k} = {cfg[k]}")
