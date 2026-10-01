"""
Vision: camera, human detection, face verification -- REAL FLIGHT BUILD.

Target hardware: Raspberry Pi 5 8 GB, CPU only, Pi Camera Module 3 (IMX708).

Colour order note: picamera2 format names are byte order, reversed from
numpy array order, so "RGB888" hands you a BGR array. That is what YOLO
and InsightFace both expect. No conversion needed.

What changed from the simulation build:

  1. Outdoor exposure. Indoors the AE picks a 20-30 ms shutter and nobody
     notices. At 12 m doing 2.5 m/s that smears a person across several
     pixels, and a 29 px figure does not survive that. AeExposureMode is
     forced Short and the frame duration is capped, which bounds shutter
     time. Daylight AWB stops the white balance hunting as the drone
     passes over grass, then tarmac, then grass.
  2. Focus is fixed, not continuous. Continuous AF hunts every time the
     scene changes, and every hunt is a run of blurred frames. Patrol
     locks at infinity; the identify step switches to ~3 m and back.
  3. AsyncDetector runs inference on its own thread and records the
     vehicle state at frame-grab time, so patrol.py projects a detection
     with the attitude the picture was taken with rather than the
     attitude 200 ms later.
  4. The detector times itself on load and publishes .steady_ms, which
     patrol.py uses to pick a ground speed and a confirmation count the
     model can actually keep up with. A PyTorch .pt flies fine at 1.5
     fps; it does not fly fine at 1.5 fps and 2.5 m/s.

Two constraints the rest of the program depends on:

  - Detection.u and .v are in 640x480 coordinates. patrol.py projects
    them with CAMERA_MATRIX, calibrated for that resolution. Coordinates
    from any other size give plausible-looking but wrong GPS.
  - identify() uses the main 2028x1520 stream. A face at 3 m is about
    96 px wide there and about 30 px at 640. Recognition needs 60+.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger('vision')

# ============================================================
# CAMERA
# ============================================================

LORES_SIZE = (640, 480)      # detection stream
MAIN_SIZE = (2028, 1520)     # face stream, binned IMX708 mode
WARMUP = 2.0

# Frame duration bounds in microseconds. The upper bound is the shutter
# ceiling: 10 ms at 2.5 m/s and 12 m altitude is about 0.6 px of smear
# on a 29 px figure. Raise the ceiling only if the field is genuinely
# too dark, and expect blur when you do.
FRAME_DURATION_LIMITS = (10000, 10000)   # 100 fps cap -> shutter <= 10 ms
EXPOSURE_VALUE = -0.3        # bias down; bright ground blows out otherwise

LENS_INFINITY = 0.0          # dioptres
LENS_INSPECT = 1.0 / 3.0     # focus at 3 m for the face step
FOCUS_SETTLE = 0.6           # s, lens travel time

# ============================================================
# DETECTION
# ============================================================

# Reporting floor only. patrol.py applies the real thresholds: a box
# needs CONF_SEED to open a track and only CONF_KEEP to sustain one, so
# the weak boxes that appear when a person is flickering in and out still
# count as evidence instead of being dropped here.
DETECT_CONF = 0.25
DETECT_IOU = 0.45            # NMS IoU
DETECT_IMGSZ = 640
DETECT_THREADS = 3           # leave one core for MAVLink and the camera

# VisDrone splits humans into two classes:
#     pedestrian -- standing or walking
#     people     -- any other posture (sitting, lying)
# Filtering on class 0 alone drops every seated or lying person, which
# for search and rescue is the case that matters most.
HUMAN_CLASS_NAMES = ('pedestrian', 'people', 'person')

# ============================================================
# FACE
# ============================================================

FACE_MODEL = 'buffalo_s'     # small ArcFace pack, right size for a Pi 5
FACE_DET_SIZE = (640, 640)
FACE_MIN_PIXELS = 40


@dataclass
class Detection:
    u: float                 # box centre x, pixels, 640x480 frame
    v: float                 # box centre y, pixels, 640x480 frame
    width: float
    height: float
    confidence: float


@dataclass
class FaceResult:
    face_found: bool
    distance: float = None   # cosine distance to reference, lower is closer
    face_pixels: int = 0


# ============================================================
# CAMERA -- Pi Camera Module 3 (IMX708) on the Pi 5 CSI port
# ============================================================

class Camera:

    def __init__(self):
        self._cam = None
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()   # serialises capture_array calls
        self._stop = threading.Event()
        self._frame = None
        self._stamp = 0.0
        self._count = 0
        self._t0 = 0.0

        self.mode = None
        self._convert = False
        self._controls_ok = False

    def _configure(self):
        from picamera2 import Picamera2

        attempts = [
            ('dual-rgb', dict(
                main={'size': MAIN_SIZE, 'format': 'RGB888'},
                lores={'size': LORES_SIZE, 'format': 'RGB888'},
                display=None, buffer_count=4)),
            ('dual-yuv', dict(
                main={'size': MAIN_SIZE, 'format': 'RGB888'},
                lores={'size': LORES_SIZE, 'format': 'YUV420'},
                display=None, buffer_count=4)),
            ('single', dict(
                main={'size': MAIN_SIZE, 'format': 'RGB888'},
                display=None, buffer_count=4)),
        ]

        last = None
        for mode, kwargs in attempts:
            try:
                self._cam = Picamera2()
                self._cam.configure(self._cam.create_video_configuration(**kwargs))
                self._cam.start()

                probe = (self._cam.capture_array('main') if mode == 'single'
                         else self._cam.capture_array('lores'))
                if probe is None:
                    raise RuntimeError("configured but returned no frame")

                self.mode = mode
                self._convert = (mode == 'dual-yuv')

                if mode == 'single':
                    log.warning("Dual stream unavailable. Single %s stream, "
                                "downscaled for detection. Costs CPU.",
                                MAIN_SIZE)
                else:
                    log.info("Camera mode: %s", mode)
                return

            except Exception as e:
                last = e
                log.warning("Camera config '%s' failed: %s", mode, e)
                try:
                    self._cam.close()
                except Exception:
                    pass
                self._cam = None

        raise RuntimeError(f"No working camera configuration. Last error: {last}")

    # ---- controls ----

    def _apply_flight_controls(self):
        """
        Daylight settings for a moving airframe. Indoors these look dark;
        that is expected and correct.
        """
        try:
            from libcamera import controls
        except Exception as e:
            log.warning("libcamera controls unavailable: %s. Running on "
                        "auto settings -- expect motion blur in flight.", e)
            return

        wanted = {
            'AeEnable': True,
            'AeExposureMode': controls.AeExposureModeEnum.Short,
            'AeMeteringMode': controls.AeMeteringModeEnum.CentreWeighted,
            'AwbEnable': True,
            'AwbMode': controls.AwbModeEnum.Daylight,
            'ExposureValue': EXPOSURE_VALUE,
            'FrameDurationLimits': FRAME_DURATION_LIMITS,
            'NoiseReductionMode': controls.draft.NoiseReductionModeEnum.Fast,
        }

        applied = {}
        for key, value in wanted.items():
            try:
                self._cam.set_controls({key: value})
                applied[key] = value
            except Exception as e:
                log.warning("Control %s rejected: %s", key, e)

        self._controls_ok = bool(applied)
        log.info("Flight camera controls applied: shutter capped at %.0f ms, "
                 "daylight AWB, short-exposure AE",
                 FRAME_DURATION_LIMITS[1] / 1000.0)

    def _set_lens(self, position, label):
        try:
            from libcamera import controls
            self._cam.set_controls({'AfMode': controls.AfModeEnum.Manual,
                                    'LensPosition': position})
            log.info("Focus -> %s (%.2f dioptres)", label, position)
            return True
        except Exception as e:
            log.warning("Manual focus unavailable (%s). If this is a Camera "
                        "Module 2 there is no focus motor and that is fine.", e)
            return False

    def focus_far(self):
        """Infinity. Used for the whole patrol."""
        if self._cam is not None:
            self._set_lens(LENS_INFINITY, 'infinity')
            time.sleep(FOCUS_SETTLE)

    def focus_near(self):
        """About 3 m. Used only while identifying a face."""
        if self._cam is not None:
            self._set_lens(LENS_INSPECT, '3 m')
            time.sleep(FOCUS_SETTLE)

    # ---- lifecycle ----

    def start(self):
        self._configure()
        self._apply_flight_controls()
        self.focus_far()

        time.sleep(WARMUP)
        self._t0 = time.time()
        threading.Thread(target=self._loop, daemon=True).start()
        log.info("Camera running")

    def _grab_lores(self):
        with self._cap_lock:
            if self.mode == 'single':
                import cv2
                full = self._cam.capture_array('main')
                return cv2.resize(full, LORES_SIZE, interpolation=cv2.INTER_AREA)

            frame = self._cam.capture_array('lores')

        if self._convert:
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV420p2BGR)
        return frame

    def _loop(self):
        """Keep only the newest frame. Stale frames are worse than
        dropped ones when you are steering from them."""
        while not self._stop.is_set():
            try:
                f = self._grab_lores()
            except Exception as e:
                log.error("Capture error: %s", e)
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame, self._stamp = f, time.time()
                self._count += 1

    def read_detection(self, max_age=0.5):
        with self._lock:
            if self._frame is None or time.time() - self._stamp > max_age:
                return None
            return self._frame.copy()

    def read_detection_stamped(self, max_age=0.5):
        """(frame, capture_time) or (None, 0.0). Used by AsyncDetector so
        the projection can be tied to when the shutter actually fired."""
        with self._lock:
            if self._frame is None or time.time() - self._stamp > max_age:
                return None, 0.0
            return self._frame.copy(), self._stamp

    def read_face(self):
        """Full-resolution BGR frame. ~300 ms. Identify step only."""
        try:
            with self._cap_lock:
                return self._cam.capture_array('main')
        except Exception as e:
            log.error("Full-res capture failed: %s", e)
            return None

    @property
    def fps(self):
        return self._count / max(time.time() - self._t0, 0.001)

    def self_test(self):
        if self._cam is None:
            return False, "camera not started"

        f = self.read_detection(max_age=3.0)
        if f is None:
            return False, "no frames arriving from the sensor"

        h, w = f.shape[:2]
        if (w, h) != LORES_SIZE:
            return False, f"detection frame is {w}x{h}, expected {LORES_SIZE}"
        if f.ndim != 3 or f.shape[2] != 3:
            return False, f"frame shape {f.shape}, expected 3 channels"

        mean = float(np.mean(f))
        if mean < 5:
            return False, f"frames are black (mean {mean:.0f}) -- lens cap on?"
        if mean > 250:
            return False, f"frames are saturated (mean {mean:.0f})"

        if mean < 25:
            log.warning("Frames are dark (mean %.0f). Outdoors this is fine "
                        "with the shutter cap; indoors it means you are "
                        "testing in light the flight settings are not for.",
                        mean)

        rate = self.fps
        if rate < 8:
            log.warning("Only %.1f fps on the detection stream. Detections "
                        "will be sparse at patrol speed.", rate)

        full = self.read_face()
        if full is None:
            return False, "full-resolution stream not working"

        fh, fw = full.shape[:2]
        if (fw, fh) != MAIN_SIZE:
            return False, f"main stream is {fw}x{fh}, expected {MAIN_SIZE}"

        return True, (f"{self.mode}, detection {w}x{h} @ {rate:.1f} fps, "
                      f"face {fw}x{fh}, brightness {mean:.0f}")

    def stop(self):
        self._stop.set()
        time.sleep(0.2)
        if self._cam:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception:
                pass


class NullCamera:
    """Used with --no-vision so the flight path runs with no camera."""
    mode = 'none'
    fps = 0.0

    def read_detection(self, max_age=0.5):
        return None

    def read_detection_stamped(self, max_age=0.5):
        return None, 0.0

    def read_face(self):
        return None

    def focus_near(self):
        pass

    def focus_far(self):
        pass

    def self_test(self):
        return True, "no camera (--no-vision)"

    def stop(self):
        pass


# ============================================================
# HUMAN DETECTOR -- VisDrone-trained YOLOv8n
# ============================================================

class HumanDetector:
    """
    Person detection from the 640x480 stream.

    Class mapping is resolved from the model's own names dict rather than
    hardcoded, because VisDrone weights from different sources order
    their classes differently. load() logs exactly which class ids it
    decided are human -- read that line.

    A .pt checkpoint works, it is just slow: roughly 600-900 ms per frame
    on the Pi 5 CPU against 100-160 ms for the same weights exported to
    NCNN. load() measures the real figure and publishes it as
    .steady_ms, and patrol.py slows the airframe down to match. Flying a
    .pt is a valid configuration; flying a .pt at NCNN ground speed is
    not, because the drone covers too much ground between inferences to
    ever confirm a person.

    When you want the speed back, export on the PC, not the Pi:

        yolo export model=best.pt format=ncnn

    then pass the resulting best_ncnn_model/ folder as --weights.
    """

    def __init__(self, weights=None, imgsz=None):
        self.weights = weights
        self.imgsz = imgsz or DETECT_IMGSZ
        self.model = None
        self.human_ids = []
        self.names = {}
        self.is_ncnn = False
        self.steady_ms = 0.0
        self._last_ms = 0.0

    # ---- loading ----

    def load(self):
        if not self.weights:
            raise RuntimeError("No --weights given for the detector")
        if not os.path.exists(self.weights):
            raise RuntimeError(f"Weights not found: {self.weights}")

        self._limit_threads()

        from ultralytics import YOLO

        self.is_ncnn = (self.weights.rstrip('/').endswith('_ncnn_model')
                        or os.path.isdir(self.weights))

        if not self.is_ncnn:
            log.warning("PyTorch checkpoint on a Pi 5 CPU. Expect roughly "
                        "600-900 ms per frame. The mission will run, but "
                        "patrol.py will cut ground speed to compensate, so "
                        "the square takes longer and the battery has to "
                        "cover it.")

        self.model = YOLO(self.weights, task='detect')
        self.names = dict(self.model.names)

        log.info("Model classes: %s", self.names)
        self.human_ids = self._resolve_human_classes(self.names)

        if not self.human_ids:
            raise RuntimeError(
                f"No human class found in model names {self.names}. "
                f"Expected one of {HUMAN_CLASS_NAMES}. Set human_ids "
                f"manually if your weights use different labels.")

        chosen = {i: self.names[i] for i in self.human_ids}
        log.info("Human classes -> %s", chosen)

        if len(self.human_ids) == 1 and self.names[self.human_ids[0]] == 'pedestrian':
            log.warning("Only 'pedestrian' found, no 'people' class. Seated "
                        "and lying people may be missed.")

        self._warmup()

    @staticmethod
    def _limit_threads():
        """Leave a core for MAVLink, the setpoint stream and the camera."""
        os.environ.setdefault('OMP_NUM_THREADS', str(DETECT_THREADS))
        try:
            import cv2
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            import torch
            torch.set_num_threads(DETECT_THREADS)
            log.info("Torch threads limited to %d", DETECT_THREADS)
        except Exception:
            pass

    @staticmethod
    def _resolve_human_classes(names):
        return sorted(i for i, n in names.items()
                      if str(n).strip().lower() in HUMAN_CLASS_NAMES)

    def _warmup(self):
        """
        First inference is always slow. Do it now, not in flight.

        Timed on noise rather than a blank frame: a black image gives
        zero candidate boxes, NMS does nothing, and the .pt comes out
        looking faster than it is in the air.
        """
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (LORES_SIZE[1], LORES_SIZE[0], 3),
                             dtype=np.uint8)

        t = time.time()
        self.model.predict(noise, imgsz=self.imgsz, verbose=False)
        first = (time.time() - t) * 1000

        t = time.time()
        for _ in range(5):
            self.model.predict(noise, imgsz=self.imgsz, verbose=False)
        steady = (time.time() - t) * 1000 / 5

        self.steady_ms = steady
        log.info("Detector: %s at imgsz %d -- first %.0f ms, steady %.0f ms "
                 "(%.1f fps)",
                 'NCNN' if self.is_ncnn else 'PyTorch', self.imgsz,
                 first, steady, 1000.0 / max(steady, 1))

        if steady > 400 and not self.is_ncnn:
            log.warning("At %.0f ms per frame, dropping imgsz to 480 buys "
                        "roughly a third back if you need it "
                        "(--imgsz 480).", steady)

    # ---- inference ----

    def detect(self, frame):
        """BGR frame at LORES_SIZE -> list of Detection, best first."""
        t = time.time()

        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            classes=self.human_ids,
            verbose=False,
        )

        self._last_ms = (time.time() - t) * 1000

        out = []
        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(Detection(
                u=(x1 + x2) / 2,
                v=(y1 + y2) / 2,
                width=x2 - x1,
                height=y2 - y1,
                confidence=float(box.conf[0]),
            ))

        out.sort(key=lambda d: d.confidence, reverse=True)
        return out

    @property
    def last_ms(self):
        return self._last_ms

    def is_ready(self):
        return self.model is not None


# ============================================================
# ASYNC DETECTOR
# ============================================================

class AsyncDetector:
    """
    Runs the detector on its own thread and publishes the newest result
    together with the vehicle state sampled at frame-grab time.

    This exists for two reasons. Inference on the Pi takes long enough
    that calling it from the flight loop starves the GUIDED setpoint
    stream, and the vehicle attitude 200 ms after the shutter is not the
    attitude the picture was taken with -- projecting with the later one
    puts the target metres off.

    state_fn is Vehicle.snap.
    """

    def __init__(self, camera, detector, state_fn):
        self.camera = camera
        self.detector = detector
        self.state_fn = state_fn

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._result = None
        self._thread = None
        self.processed = 0

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Detector thread running")

    def pause(self):
        """Stop inferring, e.g. while the face model needs the CPU."""
        self._paused.set()
        with self._lock:
            self._result = None

    def resume(self):
        self._paused.clear()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def take(self):
        """
        (detections, state, stamp) if a new result is waiting, else None.
        Consuming clears it, so each frame is counted once by the
        consecutive-hit filter.
        """
        with self._lock:
            r, self._result = self._result, None
        return r

    def _loop(self):
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            frame, stamp = self.camera.read_detection_stamped()
            if frame is None:
                time.sleep(0.02)
                continue

            state = self.state_fn()

            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                log.error("Detector error: %s", e)
                time.sleep(0.2)
                continue

            self.processed += 1
            with self._lock:
                self._result = (detections, state, stamp)


# ============================================================
# FACE IDENTIFIER -- InsightFace ArcFace
# ============================================================

class FaceIdentifier:
    """
    Verification against a reference image, or against a folder of
    reference images averaged into one embedding.

    Several references from different angles is meaningfully more robust
    than one. If --reference points at a directory, every image in it is
    used.

    Detection runs at FACE_DET_SIZE, but the recognition crop is taken
    from the full 2028x1520 frame, so you keep the pixels that matter.
    """

    def __init__(self, reference_path=None):
        self.reference_path = reference_path
        self.reference_embedding = None
        self.model = None
        self.n_references = 0

    # ---- loading ----

    def load(self):
        if not self.reference_path:
            raise RuntimeError("No --reference given for face identification")
        if not os.path.exists(self.reference_path):
            raise RuntimeError(f"Reference not found: {self.reference_path}")

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise RuntimeError(
                "insightface is not installed. On the Pi:\n"
                "  python3 -m pip install insightface onnxruntime "
                "--break-system-packages\n"
                f"(original error: {e})")

        self.model = FaceAnalysis(name=FACE_MODEL,
                                  providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        log.info("Face model loaded: %s", FACE_MODEL)

        self._build_reference()

    def _build_reference(self):
        import cv2

        if os.path.isdir(self.reference_path):
            paths = sorted(
                os.path.join(self.reference_path, f)
                for f in os.listdir(self.reference_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if not paths:
                raise RuntimeError(f"No images in {self.reference_path}")
        else:
            paths = [self.reference_path]

        embeddings = []
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                log.warning("Could not read %s, skipping", path)
                continue

            faces = self.model.get(image)
            if not faces:
                log.warning("No face found in %s, skipping", path)
                continue

            face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
            width = int(face.bbox[2] - face.bbox[0])

            if width < 80:
                log.warning("Face in %s is only %d px wide. A low-resolution "
                            "reference weakens every comparison.", path, width)

            embeddings.append(self._normalise(face.embedding))
            log.info("Reference: %s (face %d px)", os.path.basename(path), width)

        if not embeddings:
            raise RuntimeError(
                f"No usable face in any reference image at "
                f"{self.reference_path}. Use a clear, front-facing photo.")

        mean = np.mean(embeddings, axis=0)
        self.reference_embedding = mean / np.linalg.norm(mean)
        self.n_references = len(embeddings)

        if self.n_references == 1:
            log.warning("1 reference embedding. Outdoors, in the light and "
                        "at the angle the drone actually sees, one photo is "
                        "thin. Use 4-6 from different angles.")
        else:
            log.info("%d reference embeddings averaged", self.n_references)
            self._report_reference_spread(embeddings)

    @staticmethod
    def _report_reference_spread(embeddings):
        """If the references disagree with each other, say so -- it
        usually means one of them is a different person."""
        worst = 0.0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                d = 1.0 - float(np.dot(embeddings[i], embeddings[j]))
                worst = max(worst, d)

        log.info("Reference spread: worst pair %.3f", worst)
        if worst > 0.55:
            log.warning("Reference images disagree with each other (%.3f). "
                        "Is one of them a different person?", worst)

    @staticmethod
    def _normalise(vec):
        v = np.asarray(vec, dtype=float)
        return v / np.linalg.norm(v)

    # ---- inference ----

    def identify(self, frame):
        """BGR frame at MAIN_SIZE -> FaceResult."""
        if frame is None:
            return FaceResult(face_found=False)

        faces = self.model.get(frame)
        if not faces:
            return FaceResult(face_found=False)

        face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        width = int(face.bbox[2] - face.bbox[0])

        if width < FACE_MIN_PIXELS:
            # Too small to trust. Report as no face rather than feeding a
            # bad embedding into the median.
            return FaceResult(face_found=False, face_pixels=width)

        embedding = self._normalise(face.embedding)
        distance = 1.0 - float(np.dot(embedding, self.reference_embedding))

        return FaceResult(face_found=True, distance=distance,
                          face_pixels=width)

    def is_ready(self):
        return self.model is not None and self.reference_embedding is not None
