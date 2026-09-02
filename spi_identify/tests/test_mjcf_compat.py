"""Static MJCF compatibility guards for the vendored SPI model copy.

The gradmotion training image (py3.8) resolves pip 'mujoco' to 2.3.6, whose
XML schema rejects MuJoCo-3.x-only elements (e.g. <jointactuatorfrc>).
spi_identify/resources/mjcf/ therefore vendors a copy of X1_infer's model with
those passive sensor lines stripped (dynamics unchanged). These tests pin
that invariant so a future refresh of the vendored copy cannot silently
reintroduce incompatible elements or drift from the X1_infer source.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_SERIAL = REPO / "X1_infer/module/sim_module/model/mjcf/robot/xyber_x1/xyber_x1_serial.xml"
VENDORED_SERIAL = REPO / "spi_identify/resources/mjcf/robot/xyber_x1/xyber_x1_serial.xml"
VENDORED_FLAT = REPO / "spi_identify/resources/mjcf/xyber_x1_flat.xml"

# elements rejected by mujoco 2.3.6's schema (extend if new ones appear)
MUJOCO3_ONLY_ELEMENTS = ("jointactuatorfrc",)


class TestMjcfCompat(unittest.TestCase):
    def test_vendored_exists(self):
        self.assertTrue(VENDORED_SERIAL.is_file(), VENDORED_SERIAL)
        self.assertTrue(VENDORED_FLAT.is_file(), VENDORED_FLAT)

    def test_no_mujoco3_only_elements(self):
        for path in (VENDORED_SERIAL, VENDORED_FLAT):
            text = path.read_text()
            for el in MUJOCO3_ONLY_ELEMENTS:
                self.assertNotIn(el, text, f"{el} present in {path}")

    def test_vendored_equals_source_minus_3x_sensors(self):
        """Drift guard: vendored copy must track the X1_infer source exactly,
        except for the stripped sensor lines and the meshdir re-point."""
        self.assertTrue(SRC_SERIAL.is_file(), "X1_infer source serial.xml missing")
        src = SRC_SERIAL.read_text().splitlines()
        src = [ln for ln in src if not any(el in ln for el in MUJOCO3_ONLY_ELEMENTS)]
        src = [re.sub(r'meshdir="[^"]*"', 'meshdir="X"', ln) for ln in src]
        ven = VENDORED_SERIAL.read_text().splitlines()
        ven = [re.sub(r'meshdir="[^"]*"', 'meshdir="X"', ln) for ln in ven]
        self.assertEqual(src, ven,
                         "vendored MJCF drifted from X1_infer source — re-vendor "
                         "(cp + strip jointactuatorfrc + fix meshdir)")

    def test_meshdir_resolves_to_x1_infer_meshes(self):
        m = re.search(r'meshdir="([^"]+)"', VENDORED_SERIAL.read_text())
        self.assertIsNotNone(m, "meshdir attribute missing")
        target = (VENDORED_SERIAL.parent / m.group(1)).resolve()
        # MuJoCo resolves asset paths relative to the MAIN (flat) xml file
        main_rel = re.search(r'meshdir="([^"]+)"',
                             VENDORED_SERIAL.read_text()).group(1)
        target = (VENDORED_FLAT.parent / main_rel).resolve()
        self.assertTrue(target.is_dir(),
                        f"meshdir target missing: {target}")
        self.assertGreater(len(list(target.glob('*.STL'))), 30,
                           "mesh dir unexpectedly empty (<30 STLs)")


if __name__ == "__main__":
    unittest.main()
