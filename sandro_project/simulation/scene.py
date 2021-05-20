from incoming import Incoming
from sample import Sample
from detector import Detector


class Scene(object):
    def __init__(self,
                 incoming_props,
                 sample_props,
                 detector_props,
                 misalignment,
                 theta,
                 apply_noise=True,
                 detection_distance=10.0):
        """

        :param incoming_props: Tuple(length, width, shape: Tuple[str, float, float], inc_intensity, max_amplitude, sigma, ray_numbers)
        :param sample_props: Tuple(length, width, thickness)
        :param detector_props: Tuple(length, width)
        :param misalignment: Tuple(chi, omega, x, y, z)
        """

        self.incoming = Incoming(*incoming_props, det_distance=detection_distance)
        self.sample = Sample(*sample_props, det_angle=theta)
        self.sample.set_position(misalignment)
        self.sample.get_intensity(self.incoming, apply_noise=apply_noise)
        self.detector = Detector(*detector_props, det_angle=theta, det_distance=detection_distance)

        if theta == 0:
            self.detector.shadow_scan(self.incoming, self.sample, apply_noise=apply_noise)
        else:
            self.detector.get_intensity(self.sample, apply_noise=apply_noise)
