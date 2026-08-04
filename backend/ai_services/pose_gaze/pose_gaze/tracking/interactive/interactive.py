"""Non-blocking keyboard and student-ID controls for webcam demos."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import os
import sys
from time import monotonic
from typing import Any, Literal

from ..manager import AssignmentError, TrackingManager
from ..schemas import TrackPacket
from ..webcam import ProcessingRateController


PromptMode = Literal["student_id", "select_assign", "select_unassign"]


class _ConsoleKeyReader:
    """Poll terminal keys without blocking the webcam loop."""

    def __init__(self) -> None:
        self._msvcrt: Any = None
        if os.name == "nt":
            try:
                import msvcrt

                self._msvcrt = msvcrt
            except ImportError:
                pass

    def poll(self) -> tuple[int, ...]:
        if self._msvcrt is not None:
            keys: list[int] = []
            while self._msvcrt.kbhit():
                character = self._msvcrt.getwch()
                if character in ("\x00", "\xe0"):
                    if self._msvcrt.kbhit():
                        self._msvcrt.getwch()
                    continue
                keys.append(ord(character))
            return tuple(keys)

        # OpenCV-window keys remain available on non-Windows systems. Avoid
        # changing terminal mode merely for the fallback path.
        return ()


@dataclass
class WebcamInteractionController:
    """Manage FPS controls and manual track-to-student assignment."""

    manager: TrackingManager
    session_id: str
    rate: ProcessingRateController

    def __post_init__(self) -> None:
        self.ignored_tracks: set[int] = set()
        self.prompt_mode: PromptMode | None = None
        self.prompt_track_id: int | None = None
        self.input_buffer = ""
        self.status_message = "Waiting for tracks"
        self.quit_requested = False
        self.finalize_requested = False
        self._latest_packet: TrackPacket | None = None
        self._packet_update: TrackPacket | None = None
        self._console = _ConsoleKeyReader()

    def update_tracks(self, packet: TrackPacket) -> None:
        self._latest_packet = packet
        visible_by_id = {
            track.track_id: track for track in packet.tracks if track.is_present
        }

        if (
            self.prompt_mode == "student_id"
            and self.prompt_track_id not in visible_by_id
        ):
            self._cancel_prompt("Prompt cancelled because the track disappeared")

        if self.prompt_mode is not None:
            return

        unassigned = [
            track
            for track in packet.tracks
            if (
                track.is_present
                and not track.student_id
                and track.track_id not in self.ignored_tracks
            )
        ]
        if unassigned:
            self._begin_student_prompt(unassigned[0].track_id, automatic=True)

    def poll_console_keys(self) -> None:
        for key in self._console.poll():
            self.handle_key(key)

    def handle_key(self, raw_key: int) -> None:
        if raw_key < 0:
            return
        key = raw_key & 0xFF

        if self.prompt_mode is not None:
            self._handle_prompt_key(key)
            return

        if key in (ord("q"), ord("Q")):
            self.quit_requested = True
        elif key in (ord("+"), ord("=")):
            self.rate.increase()
            self.status_message = f"FPS limit increased to {self.rate.target_fps}"
            print(f"\n{self.status_message}")
        elif key in (ord("-"), ord("_")):
            self.rate.decrease()
            self.status_message = f"FPS limit decreased to {self.rate.target_fps}"
            print(f"\n{self.status_message}")
        elif key in (ord("a"), ord("A")):
            self._begin_track_selection("select_assign")
        elif key in (ord("u"), ord("U")):
            self._begin_track_selection("select_unassign")
        elif key in (ord("r"), ord("R")):
            self.ignored_tracks.clear()
            self.status_message = "Ignored tracks reset; unassigned tracks can prompt again"
            print(f"\n{self.status_message}")
        elif key in (ord("f"), ord("F")):
            self.finalize_requested = True
            self.quit_requested = True

    def consume_packet_update(self) -> TrackPacket | None:
        packet = self._packet_update
        self._packet_update = None
        return packet

    def overlay_lines(self) -> tuple[str, str]:
        controls = (
            "A assign | U unassign | R retry | F save+quit | "
            "+/- FPS | Q quit"
        )
        if self.prompt_mode is None:
            return controls, self.status_message

        if self.prompt_mode == "student_id":
            prompt = (
                f"Student ID for Track {self.prompt_track_id}: "
                f"{self.input_buffer}_"
            )
        elif self.prompt_mode == "select_assign":
            prompt = f"Track ID to assign/reassign: {self.input_buffer}_"
        else:
            prompt = f"Track ID to unassign: {self.input_buffer}_"
        return controls, prompt

    def _handle_prompt_key(self, key: int) -> None:
        if key in (10, 13):
            self._submit_prompt()
            return
        if key == 27:
            self._cancel_prompt("Input cancelled")
            return
        if key in (8, 127):
            if self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            return

        # Keep global controls available from the terminal when the automatic
        # ID prompt has appeared but nothing has been typed yet.
        if not self.input_buffer:
            if key in (ord("q"), ord("Q")):
                self._cancel_prompt("Input cancelled")
                self.quit_requested = True
                return
            if key in (ord("+"), ord("=")):
                self.rate.increase()
                self.status_message = (
                    f"FPS limit increased to {self.rate.target_fps}"
                )
                print(f"\n{self.status_message}")
                self._print_current_prompt()
                return
            if key in (ord("-"), ord("_")):
                self.rate.decrease()
                self.status_message = (
                    f"FPS limit decreased to {self.rate.target_fps}"
                )
                print(f"\n{self.status_message}")
                self._print_current_prompt()
                return

        character = chr(key)
        if character.isprintable() and len(self.input_buffer) < 128:
            self.input_buffer += character
            sys.stdout.write(character)
            sys.stdout.flush()

    def _submit_prompt(self) -> None:
        value = self.input_buffer.strip()
        print()
        if self.prompt_mode == "student_id":
            self._submit_student_id(value)
        else:
            self._submit_track_selection(value)

    def _submit_student_id(self, value: str) -> None:
        track_id = self.prompt_track_id
        if track_id is None:
            self._cancel_prompt("No track selected")
            return

        command = value.lower()
        if command == "full":
            self._clear_prompt()
            self.finalize_requested = True
            self.quit_requested = True
            return
        if command == "no":
            self.ignored_tracks.add(track_id)
            self._clear_prompt()
            self.status_message = f"Track {track_id} ignored"
            return
        if not value:
            self.status_message = "Student ID cannot be empty"
            self.input_buffer = ""
            self._print_current_prompt()
            return

        try:
            self._packet_update = self.manager.assign_student(
                self.session_id,
                track_id=track_id,
                student_id=value,
            )
        except AssignmentError as error:
            self.status_message = f"Assignment failed: {error}"
            print(self.status_message)
            self.input_buffer = ""
            self._print_current_prompt()
            return

        self._clear_prompt()
        self.status_message = f"Assigned/remapped Track {track_id} to {value}"
        print(self.status_message)

    def _submit_track_selection(self, value: str) -> None:
        try:
            track_id = int(value)
        except ValueError:
            self.status_message = "Track ID must be an integer"
            self.input_buffer = ""
            self._print_current_prompt()
            return

        visible_ids = {
            track.track_id
            for track in (self._latest_packet.tracks if self._latest_packet else ())
            if track.is_present
        }
        if track_id not in visible_ids:
            self.status_message = (
                f"Track {track_id} is not currently visible; choose {sorted(visible_ids)}"
            )
            print(self.status_message)
            self.input_buffer = ""
            self._print_current_prompt()
            return

        if self.prompt_mode == "select_assign":
            self._begin_student_prompt(track_id, automatic=False)
            return

        self._packet_update = self.manager.unassign_student(
            self.session_id,
            track_id=track_id,
        )
        self.ignored_tracks.discard(track_id)
        self._clear_prompt()
        self.status_message = f"Unassigned Track {track_id}"
        print(self.status_message)

    def _begin_student_prompt(self, track_id: int, *, automatic: bool) -> None:
        self.prompt_mode = "student_id"
        self.prompt_track_id = track_id
        self.input_buffer = ""
        reason = "new unassigned track" if automatic else "manual assignment"
        self.status_message = f"Track {track_id}: {reason}"
        self._print_current_prompt()

    def _begin_track_selection(self, mode: PromptMode) -> None:
        visible_ids = [
            track.track_id
            for track in (self._latest_packet.tracks if self._latest_packet else ())
            if track.is_present
        ]
        if not visible_ids:
            self.status_message = "No visible track is available"
            print(f"\n{self.status_message}")
            return
        self.prompt_mode = mode
        self.prompt_track_id = None
        self.input_buffer = ""
        self.status_message = f"Visible tracks: {visible_ids}"
        self._print_current_prompt()

    def _print_current_prompt(self) -> None:
        if self.prompt_mode == "student_id":
            text = (
                f"\nTrack {self.prompt_track_id} - enter student ID "
                "('no' ignore, 'full' save+quit): "
            )
        elif self.prompt_mode == "select_assign":
            text = "\nEnter visible track ID to assign/reassign: "
        else:
            text = "\nEnter visible track ID to unassign: "
        sys.stdout.write(text + self.input_buffer)
        sys.stdout.flush()

    def _cancel_prompt(self, message: str) -> None:
        if self.prompt_mode is not None:
            print()
        self._clear_prompt()
        self.status_message = message

    def _clear_prompt(self) -> None:
        self.prompt_mode = None
        self.prompt_track_id = None
        self.input_buffer = ""


def pump_keyboard_until_frame_deadline(
    cv2: Any,
    *,
    frame_started_at: float,
    rate: ProcessingRateController,
    interaction: WebcamInteractionController,
) -> None:
    """Pump OpenCV/terminal keys while enforcing the full-loop FPS limit."""

    wait_key = getattr(cv2, "waitKeyEx", cv2.waitKey)
    while True:
        remaining = rate.remaining_seconds(frame_started_at)
        delay_ms = 1 if remaining <= 0 else max(1, min(10, ceil(remaining * 1000)))
        window_key = wait_key(delay_ms)
        if window_key != -1:
            interaction.handle_key(window_key)
        interaction.poll_console_keys()

        if interaction.quit_requested or monotonic() - frame_started_at >= rate.frame_interval:
            return
