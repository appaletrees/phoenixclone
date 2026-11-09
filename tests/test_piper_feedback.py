"""Receive-ID compatibility without opening CAN ports or sending commands."""
from types import SimpleNamespace
import unittest

from wam.inference.piper_feedback import FeedbackIdAdapter, feedback_id_offsets, gripper_diagnostics


def frame(can_id, **kwargs):
    return SimpleNamespace(arbitration_id=can_id, data=bytearray(range(8)),
                           timestamp=123.5, is_extended_id=False,
                           is_remote_frame=False, is_error_frame=False, **kwargs)


class PiperFeedbackTests(unittest.TestCase):
    def test_cached_gripper_diagnostics_units_faults_and_missing_sdk(self):
        state = SimpleNamespace(grippers_angle=69100, grippers_effort=-123,
                                status_code=0b11010100)
        message = SimpleNamespace(time_stamp=123.5, Hz=200., gripper_state=state)
        controller = SimpleNamespace(piper=SimpleNamespace(GetArmGripperMsgs=lambda: message))
        info = gripper_diagnostics(controller)
        self.assertTrue(info['available'])
        self.assertEqual(info['stroke_mm'], 69.1)
        self.assertEqual(info['effort_nm'], -.123)
        for field in ('driver_overcurrent', 'sensor_abnormal', 'driver_enabled', 'homed'):
            self.assertTrue(info[field])
        for field in ('voltage_low', 'motor_overheating', 'driver_overheating', 'driver_error'):
            self.assertFalse(info[field])
        state.status_code = 0  # diagnostics must not retain a mutable SDK object
        self.assertEqual(info['status_code'], 212)
        self.assertFalse(gripper_diagnostics(object())['available'])
        message.Hz = float('nan')
        self.assertIsNone(gripper_diagnostics(controller)['hz'])

    def test_all_eight_shifted_feedback_frames_reach_original_decoder(self):
        received = []
        adapter = FeedbackIdAdapter(received.append, 255)
        for wire, expected in ((0x3A0, 0x2A1), (0x3A1, 0x2A2), (0x3A2, 0x2A3),
                               (0x3A3, 0x2A4), (0x3A4, 0x2A5), (0x3A5, 0x2A6),
                               (0x3A6, 0x2A7), (0x3A7, 0x2A8)):
            original = frame(wire)
            adapter(original)
            decoded = received[-1]
            self.assertIsNot(decoded, original)
            self.assertEqual(decoded.arbitration_id, expected)
            self.assertEqual(original.arbitration_id, wire)
            self.assertEqual(decoded.data, original.data)
            self.assertEqual(decoded.timestamp, original.timestamp)
        self.assertEqual(adapter.mapped_frames, 8)

    def test_motor_queries_control_and_default_ids_are_not_translated(self):
        received = []
        adapter = FeedbackIdAdapter(received.append, 255)
        for can_id in (0x155, 0x212, 0x251, 0x266, 0x2A5, 0x39F, 0x3A8, 0x470, 0x4AF):
            original = frame(can_id)
            adapter(original)
            self.assertIs(received[-1], original)
            self.assertEqual(original.arbitration_id, can_id)
        self.assertEqual(adapter.mapped_frames, 0)

    def test_only_standard_complete_data_frames_are_adapted(self):
        received = []
        adapter = FeedbackIdAdapter(received.append, 255)
        for field in ('is_extended_id', 'is_remote_frame', 'is_error_frame'):
            original = frame(0x3A4)
            setattr(original, field, True)
            adapter(original)
            self.assertIs(received[-1], original)
        original = frame(0x3A4)
        original.data = bytearray(7)
        adapter(original)
        self.assertIs(received[-1], original)
        self.assertEqual(adapter.mapped_frames, 0)

    def test_per_interface_configuration_does_not_change_other_arm(self):
        self.assertEqual(feedback_id_offsets({}, ('left', 'right')), {'left': 0, 'right': 0})
        offsets = feedback_id_offsets({'feedback_id_offsets': {'right': 255}}, ('left', 'right'))
        results = {}
        for name, offset in offsets.items():
            adapter = FeedbackIdAdapter(lambda f: f.arbitration_id, offset)
            results[name] = adapter(frame(0x3A4))
        self.assertEqual(results, {'left': 0x3A4, 'right': 0x2A5})

    def test_invalid_offsets_fail_before_hardware_connection(self):
        for offsets in (None, [], {'unknown': 255}, {'right': '255'},
                        {'right': True}, {'right': -1}, {'right': 256}):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                feedback_id_offsets({'feedback_id_offsets': offsets}, ('left', 'right'))
        for offset in (-1, 256, True, '255'):
            with self.assertRaises(ValueError):
                FeedbackIdAdapter(lambda f: f, offset)


if __name__ == '__main__':
    unittest.main()
