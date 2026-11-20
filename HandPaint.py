import math
import os
import tempfile
import time

import cv2

from HandLandMarker import HandLandMarker
from HandGestures import CommandGesture, assign_hands
from Ui import UI

THUMB_TIP, INDEX_TIP = 4, 8
WRIST, MIDDLE_MCP = 0, 9


class HandPaint(HandLandMarker):
    """Paint controlled by two hands: one draws (pinching), the other controls (laying)."""

    PINCH_ON = 0.20    # ratio en dessous duquel on commence à dessiner
    PINCH_OFF = 0.30   # ratio au dessus duquel on s'arrête
    SMOOTHING = 0.5    # 1 = aucun lissage, plus bas = plus lisse mais plus lent

    DRAW_SIDE = "right"   # "right" : ta main droite dessine, la gauche commande ("left" pour l'inverse)

    # Name displayed on the screen for each placement of the control hand
    COMMAND_NAMES = {"one": "Undo", "two": "Restore", "three": "Brush", "four": "Eraser", "ok": "Panel"}

    def __init__(self, model_path, ui=None):
        super().__init__(model_path, num_hands=2)
        self.ui = ui or UI()
        self.pinching = False
        self.prev_point = None
        self.smooth_point = None
        self.cursor = None
        self.camera_hovered = None
        self._recording_writer = None
        self._recording_temp_path = None
        self._recording_size = None
        self._recording_fps = 30.0
        self._recording_frame_count = 0
        self._camera_record_requested = False
        self._camera_play_requested = False

        self.was_pinching = False
        self.is_clicking_button = False
        self.can_register_action = True

        # Control hand: which pose triggers which interface action
        self.command_gesture = CommandGesture()
        self.commands = {
            "one": self.ui.pop_last_action,
            "two": self.ui.restore_state,
            "three": self.ui.use_brush,
            "four": self.ui.use_eraser,
            "ok": self.ui.toggle_panel,
        }

    # ---------- Moveset (drawing hand) ----------

    @staticmethod
    def pinch_ratio(points):
        """Distance pouce-index divisée par la taille de la main.

        La division rend la mesure indépendante de la distance à la caméra.
        """
        hand_size = math.dist(points[WRIST], points[MIDDLE_MCP])
        if hand_size == 0:
            return 1.0
        return math.dist(points[THUMB_TIP], points[INDEX_TIP]) / hand_size

    def update_pinch_state(self, ratio):
        # Hystérésis : deux seuils différents évitent le clignotement
        if self.pinching:
            self.pinching = ratio < self.PINCH_OFF
        else:
            self.pinching = ratio < self.PINCH_ON

    # ---------- Drawing Point ----------

    def brush_position(self, points):
        """Middle of thumb and index finger: stable even when the fingers touch each other."""
        tx, ty = points[THUMB_TIP]
        ix, iy = points[INDEX_TIP]
        raw = ((tx + ix) / 2, (ty + iy) / 2)

        if self.smooth_point is None:
            self.smooth_point = raw
        else:
            a = self.SMOOTHING
            self.smooth_point = (a * raw[0] + (1 - a) * self.smooth_point[0],
                                 a * raw[1] + (1 - a) * self.smooth_point[1])
        return int(self.smooth_point[0]), int(self.smooth_point[1])

    def release(self):
        """Stop current line."""
        self.pinching = False
        self.prev_point = None
        self.smooth_point = None
        self.is_clicking_button = False
        self.can_register_action = True
        self.ui.on_release()

    # ---------- Control Hand ----------

    def update_commands(self, frame, command_points):
        # During a stroke, the control hand is ignored and disarmed:
        # An undo in the middle of a line would be absurd, and the hand must not trigger anything at the end of the stroke.
        if self.pinching:
            self.command_gesture.disarm()
            command_points = None

        pose = self.command_gesture.update(command_points)
        if pose is not None:
            action = self.commands.get(pose)
            if action is not None:
                action()

        self.draw_command_feedback(frame, command_points)

    def draw_command_feedback(self, frame, command_points):
        """Write near the wrist what the control hand is doing."""
        if command_points is None:
            return
        gesture = self.command_gesture
        if not gesture.armed:
            text = "make a fist"
        elif gesture.pose in self.COMMAND_NAMES:
            text = f"{self.COMMAND_NAMES[gesture.pose]} {int(gesture.progress * 100)}%"
        else:
            text = "ready"
        x, y = command_points[WRIST]
        cv2.putText(frame, text, (x - 20, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ---------- Hooks ----------

    def on_frame(self, frame, result):
        h, w = frame.shape[:2]
        self.ui.ensure_canvas(w, h)
        self.cursor = None
        self.camera_hovered = None

        # What hand is drawing, what hand is controlling?
        hands = [self.to_pixels(hand, w, h) for hand in result.hand_landmarks]
        draw_idx, command_idx = assign_hands([points[WRIST][0] for points in hands], self.DRAW_SIDE)
        draw_points = hands[draw_idx] if draw_idx is not None else None
        command_points = hands[command_idx] if command_idx is not None else None

        if command_points is not None:
            self.draw_hand(frame, command_points, "commands")
        self.update_commands(frame, command_points)

        if draw_points is None:
            self.release()
            return

        points = draw_points
        self.draw_hand(frame, points, "draw")

        self.was_pinching = self.pinching

        self.update_pinch_state(self.pinch_ratio(points))

        start_pinching = self.pinching and not self.was_pinching

        self.cursor = self.brush_position(points)
        self.camera_hovered = self.ui.camera_action_at(self.cursor, frame.shape)

        if start_pinching:
            print(f"Pinch detected at {self.cursor}")
            if self.camera_hovered is not None:
                print(f"Camera button pinch click detected: {self.camera_hovered}")
                if self.camera_hovered == "record":
                    self._camera_record_requested = True
                elif self.camera_hovered == "play":
                    self._camera_play_requested = True
                self.is_clicking_button = True
            else:
                self.is_clicking_button = self.ui.handle_click(self.cursor)

        if not self.pinching:
            self.is_clicking_button = False
            self.can_register_action = True
            self.ui.on_release()

        if self.pinching and not self.is_clicking_button:
            if self.can_register_action:
                self.ui.register_state()
                self.can_register_action = False  # Empêche d'enregistrer plusieurs fois pour le même trait
            if self.prev_point is not None:
                self.ui.draw_line(self.prev_point, self.cursor)
            self.prev_point = self.cursor

        elif self.pinching and self.is_clicking_button:
            self.ui.on_drag(self.cursor)

        else:
            self.prev_point = None

    def render(self, frame):
        self.ui.show(frame, self.cursor, self.pinching, self.camera_hovered)

    def on_key(self, key):
        if key == ord('s'):
            self.ui.save()

        self.ui.handle_tutorial_key(key)
        return super().on_key(key)

    @staticmethod
    def _select_recording_path():
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise RuntimeError(
                "The recording file picker requires Tkinter; install Python with Tk support."
            ) from error

        root = tk.Tk()
        root.withdraw()
        try:
            return filedialog.asksaveasfilename(
                parent=root,
                title="Save camera recording",
                defaultextension=".mp4",
                filetypes=(("MP4 video", "*.mp4"), ("AVI video", "*.avi")),
            )
        finally:
            root.destroy()

    def toggle_recording(self, frame=None):
        if self._recording_writer is None:
            self._start_recording(frame)
        else:
            self._stop_recording()

    def _start_recording(self, frame=None):
        if frame is None:
            height, width = self.ui.frame_h, self.ui.frame_w
        else:
            height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise RuntimeError("The camera frame is not ready for recording.")

        descriptor, temp_path = tempfile.mkstemp(prefix="handtracking-recording-", suffix=".avi")
        os.close(descriptor)
        writer = cv2.VideoWriter(
            temp_path, cv2.VideoWriter_fourcc(*"MJPG"), self._recording_fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            os.remove(temp_path)
            raise RuntimeError("Unable to start the temporary camera recording.")

        self._recording_writer = writer
        self._recording_temp_path = temp_path
        self._recording_size = (width, height)
        self._recording_frame_count = 0
        self.ui.set_recording(True)
        print("Camera recording started. Pinch Stop to save.")

    def _write_recording_frame(self, frame):
        if self._recording_writer is None:
            return

        if (frame.shape[1], frame.shape[0]) != self._recording_size:
            frame = cv2.resize(frame, self._recording_size, interpolation=cv2.INTER_AREA)
        self._recording_writer.write(frame)
        self._recording_frame_count += 1

    def _stop_recording(self):
        writer = self._recording_writer
        if writer is None:
            return
        temp_path = self._recording_temp_path
        self._recording_writer = None
        self._recording_temp_path = None
        self.ui.set_recording(False)
        writer.release()
        print("Camera recording stopped. Choose where to save it.")

        try:
            if self._recording_frame_count == 0:
                raise RuntimeError("The camera recording contains no frames.")
            destination = self._select_recording_path()
            if not destination:
                return
            extension = os.path.splitext(destination)[1].lower()
            if extension not in (".mp4", ".avi"):
                raise ValueError("Choose an MP4 or AVI format for the recording.")
            self._save_recording(temp_path, destination, extension)
            print(f"Camera recording saved: {destination}")
        finally:
            self._recording_frame_count = 0
            self._recording_size = None
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _save_recording(source_path, destination, extension):
        source = cv2.VideoCapture(source_path)
        if not source.isOpened():
            source.release()
            raise RuntimeError("Unable to read the temporary camera recording.")

        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            source.release()
            raise RuntimeError("The temporary camera recording has invalid dimensions.")

        codec = "mp4v" if extension == ".mp4" else "MJPG"
        fps = source.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        writer = cv2.VideoWriter(
            destination, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if not writer.isOpened():
            source.release()
            writer.release()
            raise RuntimeError(f"Unable to create the recording at: {destination}")

        frame_count = 0
        try:
            while True:
                success, frame = source.read()
                if not success:
                    break
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                frame_count += 1
        finally:
            source.release()
            writer.release()
        if frame_count == 0:
            raise RuntimeError("No frames could be read from the temporary recording.")

    def _discard_recording(self):
        writer = self._recording_writer
        temp_path = self._recording_temp_path
        self._recording_writer = None
        self._recording_temp_path = None
        self._recording_frame_count = 0
        self._recording_size = None
        self.ui.set_recording(False)
        if writer is not None:
            writer.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    def _on_camera_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return
        action = self.ui.camera_action_at((x, y))
        if action is not None:
            print(f"Mouse click detected: {action}")
        if action == "play":
            self._camera_play_requested = True
        elif action == "record":
            self._camera_record_requested = True

    @staticmethod
    def _select_video():
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as error:
            raise RuntimeError(
                "The video file picker requires Tkinter; install Python with Tk support."
            ) from error

        root = tk.Tk()
        root.withdraw()
        try:
            return filedialog.askopenfilename(
                parent=root,
                title="Choose a video to draw",
                filetypes=(
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.MP4 *.AVI *.MOV *.MKV *.WEBM *.M4V"),
                    ("All files", "*.*"),
                ),
            )
        finally:
            root.destroy()

    def _play_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, int(1000 / fps)) if math.isfinite(fps) and fps > 0 else 33
        previous_recording_fps = self._recording_fps
        self._recording_fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.ui.show_tutorial = False
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                recording_frame = frame.copy()
                frame = cv2.flip(frame, 1)
                self.on_frame(frame, self.detect(frame))
                self._write_recording_frame(recording_frame)
                self.render(frame)

                key = cv2.waitKey(delay) & 0xFF
                if key in (ord("h"), ord("H")):
                    continue
                if self._camera_record_requested:
                    self._camera_record_requested = False
                    self.toggle_recording()
                if self._camera_play_requested:
                    break
                if key == ord("q") and self._recording_writer is not None:
                    self._stop_recording()
                if not self.on_key(key):
                    return False
        finally:
            self._recording_fps = previous_recording_fps
            cap.release()
            self.release()
        return True

    def run(self):
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(
                f"Unable to open camera index {self.camera_index}; check camera permissions."
            )

        camera_fps = camera.get(cv2.CAP_PROP_FPS)
        self._recording_fps = camera_fps if math.isfinite(camera_fps) and camera_fps > 0 else 30.0
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        camera_window_sized = False
        cv2.setMouseCallback("Camera", self._on_camera_mouse)
        self._start = time.time()
        self._last_ts = -1

        try:
            while True:
                if self._camera_record_requested:
                    self._camera_record_requested = False
                    self.toggle_recording()
                if self._camera_play_requested:
                    self._camera_play_requested = False
                    if self._recording_writer is not None:
                        self._stop_recording()
                    video_path = self._select_video()
                    if video_path and not self._play_video(video_path):
                        break
                    continue

                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Unable to read a frame from the camera")

                if not camera_window_sized:
                    height, width = frame.shape[:2]
                    cv2.resizeWindow("Camera", int(width * 1.2), int(height * 1.2))
                    camera_window_sized = True

                recording_frame = frame.copy()
                frame = cv2.flip(frame, 1)
                self.on_frame(frame, self.detect(frame))
                self._write_recording_frame(recording_frame)
                self.render(frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") and self._recording_writer is not None:
                    self._stop_recording()
                if not self.on_key(key):
                    break
        finally:
            self._discard_recording()
            camera.release()
            cv2.destroyAllWindows()
