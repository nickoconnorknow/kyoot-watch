"""Resolve a Bambu Studio system preset (follow `inherits`) from the BambuStudio profiles dir."""
import json, os, sys, glob

def index(profiles_dir):
    idx = {}
    for kind in ("machine", "process", "filament"):
        for f in glob.glob(os.path.join(profiles_dir, kind, "*.json")):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if "name" in d:
                idx[(kind, d["name"])] = d
    return idx

def resolve(idx, kind, name):
    d = idx[(kind, name)]
    base = resolve(idx, kind, d["inherits"]) if d.get("inherits") else {}
    out = dict(base); out.update(d)
    return out

if __name__ == "__main__":
    idx = index(sys.argv[1])
    for kind, name in [("machine", "Bambu Lab A1 mini 0.4 nozzle"), ("process", "0.08mm Extra Fine @BBL A1M"),
                       ("filament", "Bambu PLA Matte @BBL A1M")]:
        r = resolve(idx, kind, name)
        print(kind, name, len(r), "keys; chain inherits:", idx[(kind, name)].get("inherits"))
        for k in ["extruder_variant_list", "filament_extruder_variant", "print_extruder_variant", "print_extruder_id",
                  "filament_id", "setting_id", "layer_height", "wall_loops", "seam_position", "nozzle_diameter",
                  "filament_colour", "printer_variant", "printer_model", "default_filament_colour", "curr_bed_type",
                  "slow_down_layer_time", "precise_outer_wall", "outer_wall_speed", "inner_wall_speed",
                  "top_shell_layers", "bottom_shell_layers", "sparse_infill_density", "sparse_infill_pattern",
                  "elefant_foot_compensation", "wall_generator", "ironing_type", "enable_support", "brim_type",
                  "top_surface_pattern", "print_settings_id", "version", "instantiation", "compatible_printers"]:
            if k in r: print("   ", k, "=", json.dumps(r[k])[:100])
