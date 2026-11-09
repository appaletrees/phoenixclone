"""Physical CAN identity tests; all sysfs devices are temporary fixtures."""
import copy
from pathlib import Path
import tempfile
import unittest

from wam.inference.piper_can import discover_can_interfaces, resolve_can_config


class PiperCanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.net = self.root / 'net'
        self.net.mkdir()
        self.config = {'left_can': 'can_arm1', 'right_can': 'can_arm2',
                       'can_usb_serials': {'left': 'LEFT_USB', 'right': 'RIGHT_USB'},
                       'feedback_id_offsets': {'can_arm1': 0, 'can_arm2': 255}}

    def device(self, name, serial, *, up=True):
        path = self.net / name
        path.mkdir()
        usb = self.root / name
        (usb / 'interface').mkdir(parents=True)
        (usb / 'serial').write_text(serial)
        (path / 'device').symlink_to(usb / 'interface')
        (path / 'type').write_text('280')
        (path / 'flags').write_text('0x81' if up else '0x80')
        (path / 'operstate').write_text('unknown')

    def resolve(self, config=None, **kwargs):
        return resolve_can_config(self.config if config is None else config,
                                  sysfs_root=self.net, **kwargs)

    def test_reboot_names_resolve_by_serial_and_preserve_arm_offsets(self):
        self.device('can0', 'RIGHT_USB')
        self.device('can1', 'LEFT_USB')
        before = copy.deepcopy(self.config)
        resolved = self.resolve()
        self.assertEqual((resolved['left_can'], resolved['right_can']), ('can1', 'can0'))
        self.assertEqual(resolved['feedback_id_offsets'], {'can1': 0, 'can0': 255})
        self.assertEqual(self.config, before)
        self.assertEqual(self.resolve(resolved), resolved)

    def test_swapped_names_do_not_swap_the_physical_arms(self):
        self.device('can_arm1', 'RIGHT_USB')
        self.device('can_arm2', 'LEFT_USB')
        resolved = self.resolve()
        self.assertEqual((resolved['left_can'], resolved['right_can']), ('can_arm2', 'can_arm1'))
        self.assertEqual(resolved['feedback_id_offsets'], {'can_arm2': 0, 'can_arm1': 255})
        self.assertEqual(self.resolve(resolved), resolved)

    def test_down_link_is_reported_before_any_sdk_or_model_is_needed(self):
        self.device('can0', 'LEFT_USB', up=False)
        self.device('can1', 'RIGHT_USB', up=False)
        with self.assertRaisesRegex(RuntimeError, 'DOWN.*can0.*can1'):
            self.resolve()
        self.assertEqual(self.resolve(require_up=False)['left_can'], 'can0')
        self.assertEqual((self.net / 'can0/flags').read_text(), '0x80')

    def test_missing_or_duplicate_identity_never_falls_back_to_a_name(self):
        self.device('can_arm1', 'UNRELATED_USB')
        self.device('can_arm2', 'RIGHT_USB')
        with self.assertRaisesRegex(RuntimeError, 'LEFT_USB has 0 matches'):
            self.resolve()
        self.device('can3', 'LEFT_USB')
        self.device('can4', 'LEFT_USB')
        with self.assertRaisesRegex(RuntimeError, 'LEFT_USB has 2 matches'):
            self.resolve()

    def test_legacy_name_config_and_non_can_devices(self):
        self.device('can_arm1', 'LEFT_USB')
        self.device('can_arm2', 'RIGHT_USB')
        (self.net / 'eth0').mkdir()
        (self.net / 'eth0/type').write_text('1')
        self.assertEqual(set(discover_can_interfaces(self.net)), {'can_arm1', 'can_arm2'})
        cfg = {key: value for key, value in self.config.items() if key != 'can_usb_serials'}
        self.assertEqual(self.resolve(cfg), cfg)

    def test_invalid_serial_mapping(self):
        for serials in ({}, [], {'left': 'a'}, {'left': 'a', 'right': 'a'},
                        {'left': 'a', 'right': ''}):
            with self.subTest(serials=serials), self.assertRaisesRegex(ValueError, 'can_usb_serials'):
                self.resolve({**self.config, 'can_usb_serials': serials})


if __name__ == '__main__':
    unittest.main()
